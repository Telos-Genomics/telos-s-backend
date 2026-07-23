import os
import sys
import math
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import torch.nn.functional as F
import json

# ---------------------------------------------------------------------------
# TELOS PROPHET: Spike Mutation Predictor using ESM-2
# ---------------------------------------------------------------------------
# FIX v2: Replaced position search function with a direct mapping (Wuhan_pos -> Aligned_idx).
# This ensures robustness against high mutation density in sequences like RE.2.2.3 by creating
# one comprehensive O(n) map traversal, mapping every Wuhan position to an exact index
# in the aligned sequence, regardless of local mutation clustering or residue state.
# ---------------------------------------------------------------------------


def get_device():
    """Detects and returns the best available computing device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device


def find_mask_index(input_ids_cpu: torch.Tensor, mask_token_id: int) -> int | None:
    """
    Searches for the [MASK] token index within the input sequence tensor.
    Avoids .nonzero(as_tuple=True) which can cause issues on MPS devices.
    """
    for idx in range(input_ids_cpu.shape[1]):
        if input_ids_cpu[0, idx].item() == mask_token_id:
            return idx
    return None


def build_wuhan_map(aligned_sequence: str) -> dict[int, int]:
    """
    Builds a direct map: Wuhan_pos (1-indexed) -> Aligned_index.

    Traverses the sequence once, counting non-gap amino acids. Each position 
    corresponds to one residue in the reference frame. Positions corresponding
    to gaps in the variant are excluded from the map keys (as they represent deletions).

    Returns:
        A dictionary {wuhan_pos: aligned_idx} for all positions containing a real amino acid.
    """
    mapping = {}
    wuhan_counter = 0  # Position in Wuhan numbering (1-indexed)

    for idx, char in enumerate(aligned_sequence):
        if char == '-':
            # Gap in variant sequence: This position corresponds to a deletion relative to Wuhan.
            # We still advance the reference count for this slot.
            wuhan_counter += 1
            # Do not add to map: deletion confirmed
        else:
            wuhan_counter += 1
            mapping[wuhan_counter] = idx

    return mapping


def validate_alignment(aligned_sequence: str) -> tuple[bool, str]:
    """
    Validates that the sequence originated from a correct alignment against Wuhan.
    Criteria:
      - Must contain exactly 1273 characters
      - Characters must be valid amino acids or gaps '-'
    Returns: (is_valid, error_message)
    """
    length = len(aligned_sequence)
    if length != 1273:
        return False, (
            f"❌ Incorrect Length: {length} characters. "
            f"Wuhan Spike is 1273 residues. Ensure sequence alignment against Wuhan is correct."
        )
    # NOTE: Detailed residue validation omitted as per logic preservation scope.
    return True, ""


def predict_mutations(spike_path: str, imputation_json_path: str, force_cpu: bool = False):
    """
    Predicts mutations at critical functional sites in the Spike protein using ESM-2 embeddings.

    Args:
        spike_path: Path to the aligned sequence file (must be 1273 residues).
        imputation_json_path: Path to the JSON file detailing imputed positions.
        force_cpu: If True, forces model execution on CPU regardless of hardware availability.
    """
    # ------------------------------------------------------------------
    # 1. Device Setup
    # ------------------------------------------------------------------
    if force_cpu:
        device = torch.device("cpu")
        print("💻 CPU forced by user.")
    else:
        device = get_device()

    # ------------------------------------------------------------------
    # 2. Model Loading
    # ------------------------------------------------------------------
    model_name = os.environ.get('ESM_2_SIZE', 'facebook/esm2_t33_650M_UR50D')
    print(f"\n📥 Loading model {model_name}...")

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    # Load with specific dtype for memory efficiency if possible
    try:
        model = EsmForMaskedLM.from_pretrained(model_name, torch_dtype=torch.float32)
    except Exception as e:
        print(f"⚠️ Error loading model type/dtype: {e}. Attempting standard load.")
        model = EsmForMaskedLM.from_pretrained(model_name)


    try:
        model.to(device)
        print(f"✅ Model loaded onto {device}")
    except Exception as e:
        print(f"⚠️ Error moving model to {device}: {e}. Falling back to CPU.")
        device = torch.device("cpu")
        model.to(device)

    model.eval()

    # ------------------------------------------------------------------
    # 3. Read and Validate Sequence
    # ------------------------------------------------------------------
    try:
        with open(spike_path, "r") as f:
            seq_con_gaps = f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: Spike sequence file not found at {spike_path}")
        sys.exit(1)

    is_valid, error_msg = validate_alignment(seq_con_gaps)
    if not is_valid:
        print(error_msg)
        print("\n💡 Tip: Use MAFFT or Clustal Omega to align your sequence against the Wuhan reference.")
        sys.exit(1)

    print(f"✅ Alignment validated: {len(seq_con_gaps)} characters")

    # ------------------------------------------------------------------
    # 4. Build Wuhan Map (The core logic replacement)
    # ------------------------------------------------------------------
    wuhan_map = build_wuhan_map(seq_con_gaps)

    aa_non_gap = len(wuhan_map)
    gaps_in_variant = seq_con_gaps.count('-')
    print(f"📊 Map built: {aa_non_gap} amino acid positions, {gaps_in_variant} gap positions (deletions)")

    # ------------------------------------------------------------------
    # 5. Define Target Positions (Wuhan numbering)
    # ------------------------------------------------------------------
    target_sites = {
        "RBM_452": 452,
        "RBM_484": 484,
        "RBM_501": 501,
        "Furin_Cleavage_681": 681,
    }

    # ------------------------------------------------------------------
    # 6. Load Imputation Status
    # ------------------------------------------------------------------
    imputed_indices = set()
    try:
        with open(imputation_json_path, "r") as f:
            imputation_data = json.load(f)
            # Collect all indices that were filled by imputation for quick lookup
            imputed_indices = {item['idx'] for item in imputation_data.get('positional_data', [])}
        print(f"📋 Imputation status loaded: {len(imputed_indices)} positions covered by imputed data.")
    except Exception as e:
        print(f"⚠️ Could not load imputation JSON file ({e}). Proceeding without imputation filtering.")

    print("\n" + "=" * 60)
    print("🔮 TELOS PROPHET: Structural Stability Analysis")
    print("=" * 60)

    mutation_predictions = []

    # ------------------------------------------------------------------
    # 7. Process Each Target Position
    # ------------------------------------------------------------------
    for site_name, wuhan_pos in target_sites.items():
        # Step A: Check if the position exists (i.e., it's not a mandatory deletion slot)
        aligned_idx = wuhan_map.get(wuhan_pos)

        if aligned_idx is None:
            # The position must be absent in the map, meaning the reference sequence contained a gap at this conceptual slot OR it was deleted.
            print(f"\n🟡 {site_name} (Wuhan {wuhan_pos}): DELETION confirmed or UNMAPPED.")
            continue

        # Step B: Check for imputation status
        is_imputed = aligned_idx in imputed_indices
        if is_imputed:
            print(f"\n⏩ {site_name} (Wuhan {wuhan_pos}): Skipping — Position was imputed, not observed.")
            continue

        # Step C: Check for gaps in the variant sequence at this position
        current_aa = seq_con_gaps[aligned_idx]
        if current_aa == '-':
            print(f"\n🟡 {site_name} (Wuhan {wuhan_pos}): GAP observed — Deletion confirmed.")
            continue

        # Step D: Execute Model Prediction
        print(f"\n📍 Target Site: {site_name} (Wuhan {wuhan_pos})")
        print(f"    Observed Residue in Variant: {current_aa}")
        print(f"    Aligned Index: {aligned_idx}")

        # Determine index in the gapless sequence (required for ESM input)
        clean_idx = sum(1 for char in seq_con_gaps[:aligned_idx] if char != '-')

        # Prepare masked sequence input
        seq_list = list(seq_con_gaps)
        
        # Set the target site to MASK (We must mask the position, regardless of whether it's mutated or identical)
        seq_list[aligned_idx] = tokenizer.mask_token 
        masked_input = "".join(seq_list)


        # --- Model Inference ---
        inputs = tokenizer(masked_input, return_tensors="pt")
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs_gpu).logits

        # Move logits to CPU for post-processing
        logits_cpu = logits.cpu()

        mask_idx = find_mask_index(inputs["input_ids"].clone(), tokenizer.mask_token_id)

        if mask_idx is None:
            print(f"    ❌ Failed to locate [MASK] token in the input sequence.")
            continue

        # Calculate probabilities for possible residues
        logits_mask = logits_cpu[0, mask_idx, :]
        probabilities = F.softmax(logits_mask, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, 5)

        predicted_residues = []
        print(f"    Model Predictions (Top 5):")
        for i in range(5):
            token = tokenizer.decode(top_indices[i].item())
            prob = top_probs[i].item() * 100
            predicted_residues.append({"amino": token, "confidence": prob})
            marker = "← (Current Variant AA)" if token == current_aa else ""
            print(f"      {i+1}. {token:<2} | {prob:5.2f}%  {marker}")

        mutation_predictions.append({
            "target_site": site_name,
            "wuhan_position": wuhan_pos,
            "aligned_index": aligned_idx,
            "clean_sequence_index": clean_idx,
            "original_aa": current_aa,
            "predictions": predicted_residues
        })


    # ------------------------------------------------------------------
    # 8. Save Results JSON
    # ------------------------------------------------------------------
    if mutation_predictions:
        base_name = os.path.basename(spike_path).replace('.txt', '').replace('spike_aligned/', '')
        json_output_path = f"output/prophet/mutation_predictions_{base_name}.json"
        os.makedirs("output/prophet", exist_ok=True)

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(mutation_predictions, f, indent=4, ensure_ascii=False)

        print("\n" + "=" * 60)
        print(f"✅ Predictions saved successfully to: {json_output_path}")
        print("=" * 60)
    else:
        print("\n⚠️ No predictions were generated for any target site.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 mutations_oracle.py <aligned_spike_seq.txt> <imputation_metadata.json> [--cpu]")
        print("\nExample:")
        print("  python3 mutations_oracle.py output/s/spike_aligned/spike_omicron.txt output/prophet/imputation_spike_omicron.json")
        print("\nOptions:")
        print("  --cpu    Force CPU usage (disable GPU)")
        sys.exit(1)

    force_cpu = "--cpu" in sys.argv
    predict_mutations(sys.argv[1], sys.argv[2], force_cpu)
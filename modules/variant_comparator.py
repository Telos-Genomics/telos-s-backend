import os
import sys
import math
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import torch.nn.functional as F
import csv
import time

# ---------------------------------------------------------------------------
# DEVICE STRATEGY:
#   - The model and inputs reside on the GPU (MPS or CUDA) during inference.
#   - Once logits are obtained, they are moved to CPU using .cpu(). All subsequent
#     post-processing (nonzero, softmax, topk, log probability calculations) is done on CPU
#     to avoid MPS trace traps related to advanced indexing/tensor operations.
# ---------------------------------------------------------------------------

def get_device():
    """Detects and returns the best available computing device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 CUDA detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("🍎 MPS detected (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device


def analyze_context(position: int) -> tuple[str, float]:
    """
    Analyzes the biological context of a residue based on its position.

    Args:
        position: The 1-indexed position in the sequence.

    Returns:
        A tuple containing (Context Label, Context Weight Multiplier).
    """
    if 437 <= position <= 508:
        # Receptor Binding Motif (RBM) - Direct Contact
        return "CRITICAL (RBM - Direct Contact)", 3.0
    elif 319 <= position <= 541:
        # Receptor Binding Domain (RBD)
        return "HIGH (RBD - Binding Domain)", 2.0
    elif 681 <= position <= 685:
        # Furin Cleavage Site
        return "MEDIUM (Furin Site)", 1.5
    else:
        # Structural Body
        return "NORMAL (Structural Region)", 1.0


def compare_with_intelligence(ref_path: str, var_path: str, force_cpu: bool = False):
    """
    Compares the variant sequence against the reference using ESM-2 model embeddings to assess mutation impact.

    Args:
        ref_path: Path to the reference sequence file.
        var_path: Path to the variant sequence file.
        force_cpu: If True, forces CPU usage regardless of hardware availability.
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
    model_name = os.environ.get('ESM_2_SIZE', 'facebook/esm2_t33_650M_UR50D') # Fallback name
    print(f"\n📥 Loading model {model_name}...")

    try:
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name, torch_dtype=torch.float32)
    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        sys.exit(1)

    try:
        model = model.to(device)
        print(f"✅ Model loaded onto {device}")
    except Exception as e:
        print(f"⚠️ Failed to move model to designated device: {e}. Falling back to CPU.")
        device = torch.device("cpu")
        model.to(device)

    model.eval()

    # ------------------------------------------------------------------
    # 3. Read Sequences
    # ------------------------------------------------------------------
    try:
        with open(ref_path, "r") as f:
            ref_seq = f.read().strip()
        with open(var_path, "r") as f:
            var_seq = f.read().strip()
    except FileNotFoundError as e:
        print(f"❌ Error reading sequence files: {e}")
        sys.exit(1)

    if len(ref_seq) != len(var_seq):
        print("❌ Error: Sequences have different lengths. They must be properly aligned.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Mask Index Finding Helper (CPU Safe)
    # ------------------------------------------------------------------
    def find_mask_index(input_ids_cpu: torch.Tensor, mask_token_id: int) -> int | None:
        """Returns the index (int) of the [MASK] token in the sequence."""
        for idx in range(input_ids_cpu.shape[1]):
            if input_ids_cpu[0, idx].item() == mask_token_id:
                return idx
        return None


    # ------------------------------------------------------------------
    # 5. Main Loop Execution
    # ------------------------------------------------------------------
    accumulated_results = []
    start_time = time.time()

    print("\n" + "=" * 60)
    print("  GENOMIC SURVEILLANCE REPORT")
    print("=" * 60)

    for i in range(len(ref_seq)):
        # --- Insertions and Deletions (No model inference needed) ---
        if ref_seq[i] == "-":
            print(f"\n🟠 Insertion at conceptual position ~{i + 1}")
            continue

        if var_seq[i] == "-":
            print(f"\n🟡 Deletion at position {i + 1}")
            continue

        # --- Only proceed if there is a substitution ---
        if ref_seq[i] == var_seq[i]:
            continue # Wildtype, skip inference

        pos = i + 1
        orig_aa, mut_aa = ref_seq[i], var_seq[i]
        context, context_weight = analyze_context(pos)

        # --- Prepare Input with [MASK] ---
        # Create a placeholder sequence for the model where only one residue is masked.
        temp_sequence = list(ref_seq)
        temp_sequence[i] = tokenizer.mask_token
        masked_input = "".join(temp_sequence)

        # Get tensor input IDs (Must be cloned to CPU for safe indexing later)
        inputs = tokenizer(masked_input, return_tensors="pt")
        input_ids_cpu = inputs["input_ids"].clone()

        # Move required tensors to GPU for inference
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

        # --- Inference on GPU ---
        with torch.no_grad():
            logits = model(**inputs_gpu).logits

        # --- Move Logits back to CPU for Safe Post-processing ---
        logits_cpu = logits.cpu()

        # --- CPU Post-processing ---
        mask_idx = find_mask_index(input_ids_cpu, tokenizer.mask_token_id)
        if mask_idx is None:
            print(f"\n⚠️ Could not locate [MASK] token at position {pos}, skipping inference.")
            continue

        # Extract logits for the masked position
        logits_mask = logits_cpu[0, mask_idx, :]  # shape: [vocab_size]
        probs = F.softmax(logits_mask, dim=-1)  # shape: [vocab_size]

        # Calculate probabilities for the original and mutant residues
        orig_id = tokenizer.convert_tokens_to_ids(orig_aa)
        mut_id = tokenizer.convert_tokens_to_ids(mut_aa)

        p_original = probs[orig_id].item()
        p_mutant = probs[mut_id].item()

        # Calculate Likelihood Ratio (LLR) with protection against log(0)
        if p_original > 0 and p_mutant > 0:
            llr = math.log(p_mutant / p_original)
        else:
            llr = -10.0  # Highly unlikely shift

        # Determine model's prediction suggestion
        top_prob, top_idx = torch.topk(probs, 1)
        model_suggestion = tokenizer.decode(top_idx[0].item())
        p_suggestion = top_prob[0].item()

        # --- Scoring ---
        # Score combines LLR difference and biological context weighting
        score_final  = (1 - abs(llr)) * context_weight
        risk_score   = (context_weight * 20) + (llr * 10) # Higher risk if high context/high mutation likelihood

        # Status determination based on model output thresholds
        is_threat = score_final > 1.5 and llr > -0.5
        status_text = "🔴 THREAT" if is_threat else "⚪ OBSERVATION"

        # --- Accumulate Result ---
        accumulated_results.append({
            "Mutation": f"{orig_aa}{pos}{mut_aa}",
            "Context": context,
            "LLR": round(llr, 4),
            "Status": status_text,
            "Score": round(risk_score, 1),
            "Suggestion_AI": f"{model_suggestion} ({p_suggestion:.4f})",
            "P_Original": round(p_original, 6),
            "P_Mutant": round(p_mutant, 6),
        })

        # --- Print Detailed Output (Simplified for readability) ---
        print(f"\n--- MUTATION ANALYSIS ---")
        print(f"Position: {pos} | Context: {context}")
        print(f"LLR: {llr:.4f}")
        print(f"P(Original): {p_original:.6f} | P(Mutant): {p_mutant:.6f}")
        print(f"Status: {status_text} (Score: {risk_score:.1f})")
        print(f"AI Suggestion: {model_suggestion} ({p_suggestion:.4f})")

    # ------------------------------------------------------------------
    # 6. Summary and Reporting
    # ------------------------------------------------------------------
    total_time = time.time() - start_time
    n_mutations = len(accumulated_results)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"Device Used: {device}")
    print(f"Total Mutations Analyzed: {n_mutations}")
    if n_mutations > 0:
        avg_time = total_time / n_mutations
        print(f"Avg Time per Mutation Check: {avg_time:.2f}s")

    if accumulated_results:
        report_filename = f"report_{os.path.basename(var_path).replace('.txt', '')}.csv"
        save_csv_report(accumulated_results, report_filename)
    else:
        print("\n✅ No mutations were detected between the two sequences.")


def save_csv_report(results: list[dict], filename: str):
    """Saves the structured results dictionary to a CSV file."""
    folder_path   = "output/s/reports"
    full_path = os.path.join(folder_path, filename)

    try:
        os.makedirs(folder_path, exist_ok=True)

        # Define fieldnames based on the structure of the results list
        fieldnames = ["Mutation", "Context", "LLR", "Status", "Score", 
                      "Suggestion_AI", "P_Original", "P_Mutant"]

        with open(full_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✅ Report successfully saved to: {full_path}")

    except OSError as e:
        print(f"❌ Error saving report file: {e}")


if __name__ == "__main__":
    # Explicit CLI call matching the user's request structure
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 variant_comparator.py <reference.txt> <variant.txt> [--cpu]")
        print("\nExample:")
        print("  python3 variant_comparator.py output/s/spike_aligned/spike_NC_0455122.txt output/s/spike_aligned/spike_variante.txt")
        print("\nOptions:")
        print("  --cpu   Force CPU usage (disable GPU)")
        sys.exit(1)

    force_cpu = "--cpu" in sys.argv
    compare_with_intelligence(sys.argv[1], sys.argv[2], force_cpu)
import os
import sys
import math
import csv
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
try:
    from telos_config import C1_ZONE, C2_LLR, DEFAULT_BATCH_SIZE, ESM_DEFAULT_MODEL, get_zone_weight
    from common import find_mask_index, load_esm, resolve_device
except ImportError:  # package-style import
    from modules.telos_config import C1_ZONE, C2_LLR, DEFAULT_BATCH_SIZE, ESM_DEFAULT_MODEL, get_zone_weight
    from modules.common import find_mask_index, load_esm, resolve_device

# ---------------------------------------------------------------------------
# DEVICE STRATEGY:
#   - The model and inputs reside on the GPU (MPS or CUDA) during inference.
#   - Once logits are obtained, they are moved to CPU using .cpu(). All subsequent
#     post-processing (nonzero, softmax, topk, log probability calculations) is done on CPU
#     to avoid MPS trace traps related to advanced indexing/tensor operations.
#   - Helpers live in common.py; wrappers below preserve the legacy import path.
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Legacy wrapper — use common.resolve_device()."""
    from common import get_device as _get_device
    try:
        return _get_device()
    except ImportError:
        from modules.common import get_device as _get_device_pkg
        return _get_device_pkg()


def analyze_context(position: int) -> tuple[str, float]:
    """Zone classification §2.4.2 — delegates to telos_config.get_zone_weight."""
    return get_zone_weight(position)


def compare_with_intelligence(
    ref_path: str, 
    var_path: str, 
    force_cpu: bool = False, 
    batch_size: int | None = None
):
    """
    Compares the variant sequence against the reference using ESM-2 model embeddings to assess mutation impact.

    Args:
        ref_path: Path to the reference sequence file.
        var_path: Path to the variant sequence file.
        force_cpu: If True, forces CPU usage regardless of hardware availability.
        batch_size: Number of masked sequences to pass simultaneously through the model.
            None → DEFAULT_BATCH_SIZE (8). Explicit 4 still honored for callers.
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    # ------------------------------------------------------------------
    # 1. Device Setup
    # ------------------------------------------------------------------
    device = resolve_device(force_cpu)

    # ------------------------------------------------------------------
    # 2. Model Loading (shared helper, same fallback to CPU)
    # ------------------------------------------------------------------
    model_name = os.environ.get('ESM_2_SIZE', ESM_DEFAULT_MODEL)
    try:
        tokenizer, model, device = load_esm(model_name, device)
    except Exception:
        sys.exit(1)

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
    # 4. Phase 1: Pre-scanning Sequence (Indels & Substitution Capture)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  GENOMIC SURVEILLANCE REPORT")
    print("=" * 60)

    substitution_indices = []

    for i in range(len(ref_seq)):
        # --- Insertions and Deletions (No model inference needed) ---
        if ref_seq[i] == "-":
            print(f"\n🟠 Insertion at conceptual position ~{i + 1}")
            continue

        if var_seq[i] == "-":
            print(f"\n🟡 Deletion at position {i + 1}")
            continue

        # --- Filter out Wildtypes; save indices of true Substitutions ---
        if ref_seq[i] != var_seq[i]:
            substitution_indices.append(i)

    # ------------------------------------------------------------------
    # 5. Phase 2 & 3: Batch Processing & Post-processing
    # ------------------------------------------------------------------
    accumulated_results = []
    start_time = time.time()

    for batch_start in range(0, len(substitution_indices), batch_size):
        batch_positions = substitution_indices[batch_start : batch_start + batch_size]

        # --- Prepare Masked Sequences Batch ---
        masked_sequences = []
        for idx in batch_positions:
            temp_seq = list(ref_seq)
            temp_seq[idx] = tokenizer.mask_token
            masked_sequences.append("".join(temp_seq))

        # Tokenize with padding for uniform batch shape
        inputs = tokenizer(masked_sequences, return_tensors="pt", padding=True)
        inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

        # --- Model Inference on GPU ---
        with torch.no_grad():
            logits = model(**inputs_gpu).logits

        # --- Move Tensors to CPU for Safe Post-processing ---
        logits_cpu = logits.cpu()
        input_ids_cpu = inputs["input_ids"].cpu()

        # --- Post-process Each Element in Batch ---
        for idx_in_batch, seq_idx in enumerate(batch_positions):
            pos = seq_idx + 1
            orig_aa, mut_aa = ref_seq[seq_idx], var_seq[seq_idx]
            
            # Recalculate context & weight specifically for this mutation's position
            context, context_weight = analyze_context(pos)

            # Find [MASK] token in this batch row
            mask_idx = find_mask_index(input_ids_cpu[idx_in_batch], tokenizer.mask_token_id)
            if mask_idx is None:
                print(f"\n⚠️ Could not locate [MASK] token at position {pos}, skipping inference.")
                continue

            # Extract logits for masked position
            logits_mask = logits_cpu[idx_in_batch, mask_idx, :]
            probs = F.softmax(logits_mask, dim=-1)

            # Extract probabilities
            orig_id = tokenizer.convert_tokens_to_ids(orig_aa)
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)

            p_original = probs[orig_id].item()
            p_mutant = probs[mut_id].item()

            # Calculate LLR with safety guard
            if p_original > 0 and p_mutant > 0:
                llr = math.log(p_mutant / p_original)
            else:
                llr = -10.0

            # AI Suggestion
            top_prob, top_idx = torch.topk(probs, 1)
            model_suggestion = tokenizer.decode(top_idx[0].item())
            p_suggestion = top_prob[0].item()

            # Scoring (§2.4.1: per-mutation term |c1·ω + c2·ζ|, c1=20, c2=10
            # heuristic §4.4; abs applied at aggregation in final_analyzer).
            # score_final/THREAT below is a legacy operational flag, NOT part
            # of the paper — it only drives Status coloring, never the Score.
            score_final = (1 - abs(llr)) * context_weight
            risk_score = (context_weight * C1_ZONE) + (llr * C2_LLR)

            # Threat Assessment
            is_threat = score_final > 1.5 and llr > -0.5
            status_text = "🔴 THREAT" if is_threat else "⚪ OBSERVATION"

            # Accumulate Result
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

            # Real-time Console Logging
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
    print(f"Batch Size Used: {batch_size}")
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
    """Saves the structured results dictionary to a CSV file (contract: output/s/reports/)."""
    try:
        from telos_config import REPORTS_DIR
    except ImportError:
        from modules.telos_config import REPORTS_DIR
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    full_path = REPORTS_DIR / filename

    try:
        fieldnames = [
            "Mutation", "Context", "LLR", "Status", "Score", 
            "Suggestion_AI", "P_Original", "P_Mutant"
        ]

        with open(full_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✅ Report successfully saved to: {full_path}")

    except OSError as e:
        print(f"❌ Error saving report file: {e}")


def _parse_batch_size(argv: list[str]) -> int:
    """CLI --batch-size N, default DEFAULT_BATCH_SIZE (same contract as before)."""
    if "--batch-size" in argv:
        try:
            return int(argv[argv.index("--batch-size") + 1])
        except (IndexError, ValueError):
            print(f"⚠️ Invalid batch size provided. Falling back to default ({DEFAULT_BATCH_SIZE}).")
    return DEFAULT_BATCH_SIZE


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 variant_comparator.py <reference.txt> <variant.txt> [--cpu] [--batch-size N]")
        print("\nExample:")
        print("  python3 variant_comparator.py ref.txt var.txt --batch-size 8")
        sys.exit(1)

    force_cpu = "--cpu" in sys.argv
    batch_size = _parse_batch_size(sys.argv)

    compare_with_intelligence(sys.argv[1], sys.argv[2], force_cpu=force_cpu, batch_size=batch_size)
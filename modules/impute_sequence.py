import os
import sys
import json

# ---------------------------------------------------------------------------
# SEQUENCE IMPUTER (VERSION 2.0)
# ---------------------------------------------------------------------------

LARGE_BLOCK_THRESHOLD = 5  # Threshold for defining a "large" block (in Xs)

def detect_x_blocks(sequence: str) -> list[tuple[int, int, int]]:
    """
    Identifies contiguous blocks of 'X' in the input sequence.

    Args:
        sequence: The sequence string containing 'X', '-', or amino acids.

    Returns:
        A list of tuples, where each tuple contains (start_index, end_index, length) 
        for a block of 'X'.
    """
    blocks = []
    in_block = False
    start = None

    for i, aa in enumerate(sequence):
        if aa == 'X':
            if not in_block:
                # Start of a new block
                in_block = True
                start = i
        else:
            if in_block:
                # End of the block
                end = i
                blocks.append((start, end, end - start))
                in_block = False

    # If sequence ends while still in a block
    if in_block:
        blocks.append((start, len(sequence), len(sequence) - start))
    return blocks

def impute_sequence(variant_path: str, reference_path: str, threshold: int = LARGE_BLOCK_THRESHOLD):
    """
    Imputes large 'X' blocks in the variant sequence using information from a reference sequence.

    The imputation process follows strict rules based on block size and reference conservation.

    Args:
        variant_path: Path to the sequence containing unknown positions ('X').
        reference_path: Path to the consensus or reference sequence (e.g., Wuhan).
        threshold: Minimum length of an 'X' block required for imputation consideration (default: 5).

    Output:
        - Imputed sequence saved in output/s/spike_aligned/.
        - Metadata JSON report saved in output/prophet/.
    """
    # ---------------------------------------------------------------------------
    # 1. Read sequences
    # ---------------------------------------------------------------------------
    try:
        with open(variant_path, "r") as f:
            var_sequence = f.read().strip()
        with open(reference_path, "r") as f:
            ref_sequence = f.read().strip()
    except FileNotFoundError as e:
        print(f"❌ Error reading input file: {e}")
        sys.exit(1)

    # CRITICAL ALIGNMENT VALIDATION
    if len(var_sequence) != len(ref_sequence):
        print(f"❌ FATAL ERROR: Sequences are not aligned.")
        print(f"Variant Length: {len(var_sequence)} | Reference Length: {len(ref_sequence)}")
        sys.exit(1)

    total_length = len(var_sequence)

    # ---------------------------------------------------------------------------
    # 2. Detect 'X' blocks
    # ---------------------------------------------------------------------------
    all_blocks = detect_x_blocks(var_sequence)
    total_x_count = sum(1 for aa in var_sequence if aa == 'X')

    print("\n🔍 X Position Analysis:")
    print(f"   Total X count: {total_x_count}")
    print(f"   Blocks detected: {len(all_blocks)}")

    # Classify blocks
    large_blocks = [b for b in all_blocks if b[2] >= threshold]
    small_blocks = [b for b in all_blocks if b[2] < threshold]

    x_in_small_blocks = sum(b[2] for b in small_blocks)
    x_in_large_blocks = sum(b[2] for b in large_blocks)

    print(f"\n   Small blocks (<{threshold} X): {len(small_blocks)}  ({x_in_small_blocks} positions)")
    print(f"   Large blocks (≥{threshold} X): {len(large_blocks)}  ({x_in_large_blocks} positions)")

    if large_blocks:
        print("\n   Details of Large Blocks:")
        for start, end, length in large_blocks:
            # Calculate Wuhan position (1-indexed, counting non-gap/non-X residues)
            positions = [i for i in range(start, end) if var_sequence[i] not in ['-', 'X']]
            pos_start_wuhan = sum(1 for c in var_sequence[:start] if c not in ['-', 'X']) + 1
            pos_end_wuhan = sum(1 for c in var_sequence[:end] if c not in ['-', 'X'])
            print(f"      Indices {start}-{end} ({length} X) -> Wuhan positions ~{pos_start_wuhan}-{pos_end_wuhan}")

    # ---------------------------------------------------------------------------
    # 3. Imputation ("Mirroring")
    # ---------------------------------------------------------------------------
    # Use a list to allow in-place modification of the sequence data structure
    imputed_sequence_list = list(var_sequence)
    imputed_positions_metadata = []

    # Pre-calculate Wuhan position map for accurate reporting: 
    # Index i maps to its corresponding count in the consensus/reference sequence.
    wuhan_position_map = []
    aa_counter = 0
    for i, char in enumerate(var_sequence):
        if char != '-':
            aa_counter += 1
            wuhan_position_map.append(aa_counter)
        else:
            wuhan_position_map.append(None)  # Gap position

    for start, end, length in large_blocks:
        for i in range(start, end):
            if var_sequence[i] == 'X':
                # RULE OF GOLD: Only impute if the reference sequence has a definitive character (not '-' or 'X')
                ref_char = ref_sequence[i]
                if ref_char not in ['-', 'X']:
                    imputed_sequence_list[i] = ref_char
                    # Record metadata for this specific imputation event
                    imputed_positions_metadata.append({
                        "index": i,
                        "wuhan_pos": wuhan_position_map[i],
                        "total_blocks": len(all_blocks),
                        "small_block_count": len(small_blocks),
                        "large_block_count": len(large_blocks),
                        "residue": ref_char
                    })

    # ---------------------------------------------------------------------------
    # 4. Reconstruct final sequence
    # ---------------------------------------------------------------------------
    final_sequence = "".join(imputed_sequence_list)

    if len(final_sequence) != total_length:
        raise ValueError(f"Length mismatch detected: {len(final_sequence)} vs {total_length}")

    # ---------------------------------------------------------------------------
    # 5. Save results
    # ---------------------------------------------------------------------------
    base_name = os.path.basename(variant_path).replace('.txt', '')
    output_dir = "output/s/spike_aligned"
    json_report_dir = "output/prophet"
    
    final_seq_path = f"{output_dir}/{base_name}_imputed.txt"
    json_meta_path = f"{json_report_dir}/imputation_{base_name}.json"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_report_dir, exist_ok=True)

    with open(final_seq_path, "w") as f:
        f.write(final_sequence)

    # Generate JSON metadata for the downstream Aligner/Comparator
    metadata = {
        "method": "Reference-based Imputation",
        "total_imputed_sites": len(imputed_positions_metadata),
        "positional_data": imputed_positions_metadata
    }

    with open(json_meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("✅ Process successful.")
    print(f"📄 Imputed sequence saved to: {final_seq_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 impute_sequence.py <variant.txt> <reference.txt>")
        print("\nExample:")
        print("  python3 impute_sequence.py variant_aligned.txt reference_aligned.txt")
        sys.exit(1)

    impute_sequence(sys.argv[1], sys.argv[2])
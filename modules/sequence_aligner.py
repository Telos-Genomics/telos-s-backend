import os
import sys
from Bio import Align

# ---------------------------------------------------------------------------
# SPIKE SEQUENCE ALIGNER
# ---------------------------------------------------------------------------
# Strategy: Template-based alignment
#
# The reference (Wuhan) defines the canonical numbering of 1273 positions.
# The variant is aligned against this reference as a template.
#
# Rules:
#   1. ref_aligned ALWAYS has 1273 characters (the original sequence without gaps)
#   2. var_aligned has 1273 characters with gaps '-' where deletions occur
#   3. Insertions in the variant (extra residues vs Wuhan) are DISCARDED
#      because they do not have a canonical position in the Wuhan numbering
#
# This ensures that:
#   - Both sequences have exactly 1273 characters
#   - The numbering is consistent with Wuhan
#   - The oracle and comparator function correctly
# ---------------------------------------------------------------------------


def validate_reference(seq, expected=1273):
    """
    Verify that the reference sequence has the expected length.
    
    Args:
        seq: Reference sequence (without gaps)
        expected: Expected length (1273 for the Wuhan Spike)
    
    Returns:
        (is_valid, message)
    """
    length = len(seq)
    
    if length != expected:
        return False, (
            f"❌ The reference have {length} amino acids, "
            f"but were expected to be {expected}.\n"
            f"   Make sure you use the complete Spike protein from the Wuhan-Hu-1 strain. "
            f"(GenBank: QHD43416.1 or UniProt: P0DTC2)."
        )
    
    # Verify that it doesn't contain gaps
    if '-' in seq:
        return False, (
            f"❌ The reference sequence contains gaps. '-'.\n"
            f"   The reference should be the original sequence, not an aligned version."
        )
    
    # Verify that it contains only valid amino acids.
    aa_valid = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = set(seq) - aa_valid
    
    if invalid:
        return False, (
            f"❌ The reference contains invalid characters: {invalid}.\n"
            f"   Only the 20 standard amino acids are allowed."
        )
    
    return True, ""


def align_template_based(ref_seq, var_seq):
    """
    Align the variant against the reference using template-based alignment.
    
    The reference defines the canonical numbering. The variant may have:
        - DELETIONS: gaps '-' are inserted into var_aligned
        - SUBSTITUTIONS: appear as differences without gaps
        - INSERTIONS: are discarded (do not have a canonical position in Wuhan)
    
    Args:
        ref_seq: Reference sequence (Wuhan, 1273 aa)
        var_seq: Variant sequence (variable length)
    
    Returns:
        (ref_aligned, var_aligned): Tuple of aligned sequences
        Both have exactly len(ref_seq) characters.
    """
    print(f"\n🔄 Aligning sequences...")
    print(f"   Reference: {len(ref_seq)} aa")
    print(f"   Variant:   {len(var_seq)} aa")
    
    # ---------------------------------------------------------------------------
    # 1. Global alignment with optimized penalties
    # ---------------------------------------------------------------------------
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    
    # Penalties for Gaps:
    #   - open_gap_score: penalty for CREATING a gap (first position)
    #   - extend_gap_score: penalty for EXTENDING a gap (consecutive positions)
    #
    # We want to heavily penalize gaps in order to prevent the creation of unusual alignments, 
    # but still allow for known deletions.
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    
    # Match/mismatch scores
    aligner.match_score = 2
    aligner.mismatch_score = -1
    
    alignments = aligner.align(ref_seq, var_seq)
    
    if len(alignments) == 0:
        raise ValueError("It was not possible to generate any alignment. Please ensure that the sequences are comparable.")
    
    mejor = alignments[0]
    
    # These are the aligned sequences that Biopython returns.
    ref_con_gaps = str(mejor[0])
    var_con_gaps = str(mejor[1])
    
    print(f"   Initial alignment: {len(ref_con_gaps)} characters")
    
    # ---------------------------------------------------------------------------
    # 2. Post-processing: Removing gaps from the reference
    # ---------------------------------------------------------------------------
    # If Biopython inserted gaps in the reference (because the variant has insertions), 
    # we remove them and also discard the corresponding positions in the variant.
    
    ref_final = []
    var_final = []
    
    for i in range(len(ref_con_gaps)):
        ref_char = ref_con_gaps[i]
        var_char = var_con_gaps[i]
        
        if ref_char == '-':
            # Position relative to reference → insertion in alternative
            # We are discarding this complete position
            print(f"   ⚠️  Insertion in specific variant position ~{i+1} discarded: '{var_char}'")
            continue
        
        # If we reach here, "ref_char" is a valid amino acid
        ref_final.append(ref_char)
        var_final.append(var_char)  # It can be an "aa" or a "-" (representing deletion)
    
    ref_aligned = "".join(ref_final)
    var_aligned = "".join(var_final)
    
    # ---------------------------------------------------------------------------
    # 3. Final validation
    # ---------------------------------------------------------------------------
    length_final = len(ref_aligned)
    
    if length_final != len(ref_seq):
        # This should not happen if we remove all the gaps in the reflector.
        raise AssertionError(
            f"Internal error: ref_aligned has {length_final} characters, "
            f"but it should have {len(ref_seq)}. "
            f"This indicates a problem in the post-processing stage."
        )
    
    if len(var_aligned) != length_final:
        raise AssertionError(
            f"Internal error: The aligned sequences have different lengths."
            f"(ref: {length_final}, var: {len(var_aligned)})"
        )
    
    # Counting deletions and insertions
    num_deletions = var_aligned.count('-')
    num_insertions = len(ref_con_gaps) - length_final  # Gaps we removed from the reference
    num_substitutions = sum(1 for r, v in zip(ref_aligned, var_aligned)
                           if r != v and v != '-')
    
    print(f"\n✅ Alignment complete:")
    print(f"   final length: {length_final} positions")
    print(f"   Substitutions:  {num_substitutions}")
    print(f"   Deletions:     {num_deletions}")
    print(f"   Insertions:    {num_insertions} (discarded)")
    
    return ref_aligned, var_aligned


def align_synchronize(ref_path, var_path):
    """
    Read the reference and variant sequences, align them, and save the result.
    
    Args:
        ref_path: Path to the file containing the reference sequence (Wuhan)
        var_path: Path to the file containing the variant sequence
    """
    # ---------------------------------------------------------------------------
    # 1. Read sequences
    # ---------------------------------------------------------------------------
    with open(ref_path, "r") as f:
        ref_seq = f.read().strip()
    
    with open(var_path, "r") as f:
        var_seq = f.read().strip()
    
    # ---------------------------------------------------------------------------
    # 2. Verify reference
    # ---------------------------------------------------------------------------
    is_valid, message = validate_reference(ref_seq, expected=1273)
    if not is_valid:
        print(message)
        print("\n💡 Tip: Download the Wuhan-Hu-1 Spike sequence:")
        print("   GenBank: QHD43416.1")
        print("   UniProt: P0DTC2")
        sys.exit(1)
    
    print("✅ Validated reference: Wuhan Spike, 1273 amino acids")
    
    # ---------------------------------------------------------------------------
    # 3. Align
    # ---------------------------------------------------------------------------
    try:
        ref_aligned, var_aligned = align_template_based(ref_seq, var_seq)
    except Exception as e:
        print(f"\n❌ Error during alignment: {e}")
        sys.exit(1)
    
    # ---------------------------------------------------------------------------
    # 4. Save results
    # ---------------------------------------------------------------------------
    folder_path = "output/s/spike_aligned"
    
    # Generar nombres de archivo basados en los paths de entrada
    name_ref_base = os.path.basename(ref_path).replace('.txt', '')
    name_var_base = os.path.basename(var_path).replace('.txt', '')
    
    path_ref_final = os.path.join(folder_path, f"{name_ref_base}.txt")
    path_var_final = os.path.join(folder_path, f"{name_var_base}.txt")
    
    try:
        os.makedirs(folder_path, exist_ok=True)
        
        with open(path_ref_final, "w", encoding="utf-8") as f:
            f.write(ref_aligned)
        
        with open(path_var_final, "w", encoding="utf-8") as f:
            f.write(var_aligned)
        
        print(f"\n📁 Aligned sequences saved:")
        print(f"   Reference: {path_ref_final}")
        print(f"   Variant:   {path_var_final}")
        
    except OSError as e:
        print(f"❌ Error saving files: {e}")
        sys.exit(1)
    
    # ---------------------------------------------------------------------------
    # 5. Final verification (paranoia)
    # ---------------------------------------------------------------------------
    with open(path_ref_final, "r") as f:
        test_ref = f.read().strip()
    with open(path_var_final, "r") as f:
        test_var = f.read().strip()
    
    assert len(test_ref) == 1273, f"Error: ref_aligned have {len(test_ref)} chars"
    assert len(test_var) == 1273, f"Error: var_aligned have {len(test_var)} chars"
    assert '-' not in test_ref, "Error: The aligned reference have gaps"
    
    print("✅ Successful verification: both sequences have 1273 characters.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Use: python3 sequence_aligner.py <reference.txt> <variant.txt>")
        print("\nExample:")
        print("  python3 sequence_aligner.py spike_wuhan.txt spike_omicron.txt")
        print("\nNote:")
        print("  The reference should be the complete Spike protein from Wuhan-Hu-1 (1273 amino acids).")
        print("  The variation can have any length.")
        sys.exit(1)
    
    align_synchronize(sys.argv[1], sys.argv[2])
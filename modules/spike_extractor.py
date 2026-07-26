import os
import sys
from Bio import SeqIO
from Bio.Seq import Seq
import json # Import json here since it's used in the function body

# Validation constants (Converted to English and Pythonic naming)
EXPECTED_SPIKE_LENGTH_AA = 1273  # Wuhan Spike length
AMINO_ACID_TOLERANCE = 10        # +/- 10 aa tolerance (small indels)
MIN_GENOME_LENGTH = 25000         # Full SARS-CoV-2 genome is ~30kb
WUHAN_SPIKE_RANGE = (21563, 25384)  # Positions in NC_045512.2

def process_reference(fasta_file):
    """
    Extracts the Spike protein from a SARS-CoV-2 genome.

    Supports:
      - Full genome sequence
      - Spike region only
      - Variants with small mutations at the N-terminus
    """
    # ---------------------------------------------------------------------------
    # 1. Load sequence
    # ---------------------------------------------------------------------------
    record = None

    # Try standard FASTA format first
    try:
        record = SeqIO.read(fasta_file, "fasta")
    except ValueError as e:
        # Multiple sequences in the file
        error_msg = str(e)

        if "multiple records" in error_msg.lower():
            print(f"⚠️  The file contains multiple sequences.")
            print(f"   Processing only the first sequence...")

            # Try reading all and take the first
            try:
                records = list(SeqIO.parse(fasta_file, "fasta"))
            except Exception:
                # If 'fasta' fails, try 'fasta-pearson' (supports comments)
                print(f"⚠️  Standard FASTA format failed, trying Pearson format...")
                try:
                    records = list(SeqIO.parse(fasta_file, "fasta-pearson"))
                except Exception as e2:
                    print(f"❌ Error reading file: {e2}")
                    return None

            if not records:
                print(f"❌ Could not read any sequences from the file.")
                print(f"   The file might be empty or have an incorrect format.")
                return None

            record = records[0]
            print(f"   ✓ Processing: {record.id}")

            if len(records) > 1:
                print(f"   ℹ️  Other {len(records)-1} sequences ignored:")
                for r in records[1:6]:  # Show up to 5
                    print(f"      - {r.id}")
                if len(records) > 6:
                    print(f"      ... and {len(records)-6} more")
        else:
            # Other ValueError type
            print(f"❌ Error reading file: {e}")
            return None

    except Exception as e:
        print(f"❌ Unexpected error reading the file: {e}")
        return None

    if record is None:
        print("❌ Could not load any sequence.")
        return None

    full_sequence = record.seq
    input_length = len(full_sequence)

    print(f"--- Analysis of {record.id} ---")
    print(f"Total length: {input_length:,} nucleotides")

    # ---------------------------------------------------------------------------
    # 2. Detect input type
    # ---------------------------------------------------------------------------
    is_full_genome = input_length >= MIN_GENOME_LENGTH

    if input_length < 3000:
        print(f"\n{'='*60}")
        print(f"❌ WARNING: Sequence is very short")
        print(f"{'='*60}")
        print(f"  Length: {input_length} nt")
        print(f"  Minimum for complete Spike: ~3,800 nt")
        print(f"\n  This sequence is too short to contain")
        print(f"  the full SARS-CoV-2 Spike protein.")
        print(f"\n  Possible causes:")
        print(f"    - Fragmented sequence")
        print(f"    - Only a partial region of S gene")
        print(f"    - Incorrect file")
        print(f"{'='*60}\n")

    if is_full_genome:
        print("✓ Detected: Full genome")
    else:
        print("✓ Detected: Partial sequence (likely Spike region)")

    # ---------------------------------------------------------------------------
    # 3. Search for Spike start
    # ---------------------------------------------------------------------------
    # Multi-level strategy:
    #   1. Search for exact "ATGTTTGTTTTT" (codes MFVF...)
    #   2. If fails and it's a full genome, use known Wuhan position
    #   3. If partial, assume it starts at position 0

    sequence_str = str(full_sequence).upper()  # Convert to string for search
    start_pos = sequence_str.find("ATGTTTGTTTTT")
    search_method = "Exact match"

    if start_pos == -1:
        if is_full_genome:
            # Fallback: use known Wuhan position
            start_pos = WUHAN_SPIKE_RANGE[0]
            search_method = "Reference position (Wuhan)"
            print("⚠️ Exact start sequence not found")
            print(f"   Using Wuhan reference position: {start_pos}")
        else:
            # If partial, assume it's the Spike region already
            start_pos = 0
            search_method = "Partial sequence start assumption"
            print("⚠️ Assuming the entire sequence is the Spike region")

    # ---------------------------------------------------------------------------
    # 4. Search for STOP codon
    # ---------------------------------------------------------------------------
    sub_seq = full_sequence[start_pos:]
    sub_seq_str = str(sub_seq).upper()  # String for search
    end_pos_relative = -1

    # Search for STOP in the correct reading frame
    for i in range(0, len(sub_seq_str), 3):
        codon = sub_seq_str[i:i+3]

        if codon in ["TAA", "TAG", "TGA"]:
            # Validate it's within the expected range for Spike (approx)
            if 3700 <= i <= 3900:
                end_pos_relative = i + 3  # Include the STOP codon length
                break

    # If no stop found in the expected range, search for the first available one
    if end_pos_relative == -1:
        print("⚠️ STOP codon not found in the expected range (3700-3900 nt)")
        print("   Searching for the first available STOP...")

        for i in range(3700, len(sub_seq_str), 3):
            if sub_seq_str[i:i+3] in ["TAA", "TAG", "TGA"]:
                end_pos_relative = i + 3
                print(f"   ✓ STOP found at position {i} (unusual)")
                break

    if end_pos_relative == -1:
        print("❌ No stop codon found in the reading frame.")
        print("   Possible frameshift or incomplete sequence.")
        return None

    # ---------------------------------------------------------------------------
    # 5. Extract and translate
    # ---------------------------------------------------------------------------
    gene_sequence = sub_seq[:end_pos_relative]

    # Validate gene length is a multiple of 3
    if len(gene_sequence) % 3 != 0:
        print(f"⚠️ Warning: Gene length ({len(gene_sequence)} nt) is not divisible by 3")
        print("   Possible frameshift. Translation may be incorrect.")

    # Translate
    try:
        protein_sequence = gene_sequence.translate(to_stop=True)
    except Exception as e:
        print(f"❌ Error during translation: {e}")
        return None

    protein_length = len(protein_sequence)

    # ---------------------------------------------------------------------------
    # 6. Validate protein length
    # ---------------------------------------------------------------------------
    difference = abs(protein_length - EXPECTED_SPIKE_LENGTH_AA)

    if difference <= AMINO_ACID_TOLERANCE:
        status = "✓ Normal"
    elif protein_length < EXPECTED_SPIKE_LENGTH_AA - AMINO_ACID_TOLERANCE:
        status = "⚠️ TRUNCATED"
        print(f"\n{'='*60}")
        print("WARNING: Spike Protein Truncated")
        print(f"{'='*60}")
        print(f"Observed length: {protein_length} aa")
        print(f"Expected length:  {EXPECTED_SPIKE_LENGTH_AA} aa")
        print(f"Difference:      {-difference} aa")
        print("\nPossible causes:")
        print("  - Premature nonsense mutation (STOP)")
        print("  - Frameshift due to non-triplet indel")
        print("  - Incomplete sequence")
        print(f"{'='*60}\n")
    else:
        status = "⚠️ EXTENDED"
        print(f"\n{'='*60}")
        print("WARNING: Spike Protein Longer Than Expected")
        print(f"{'='*60}")
        print(f"Observed length: {protein_length} aa")
        print(f"Expected length:  {EXPECTED_SPIKE_LENGTH_AA} aa")
        print(f"Difference:      +{difference} aa")
        print("\nPossible causes:")
        print("  - Large insertion")
        print("  - Reading past the correct STOP")
        print(f"{'='*60}\n")

    # ---------------------------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print("📊 EXTRACTION SUMMARY")
    print(f"{'─'*60}")
    print(f"  Search method:    {search_method}")
    print(f"  Genome position:   {start_pos:,} - {start_pos + end_pos_relative:,}")
    print(f"  Gene length:      {len(gene_sequence):,} nt")
    print(f"  Protein length:   {protein_length} aa")
    print(f"  Status:           {status}")
    print(f"{'─'*60}")

    # ---------------------------------------------------------------------------
    # 8. Save results
    # ---------------------------------------------------------------------------
    output_folder = "output/s/spike"
    base_name = os.path.basename(fasta_file).replace('.fasta', '').replace('.fa', '')
    protein_file_path = os.path.join(output_folder, f"spike_{base_name}.txt")

    try:
        os.makedirs(output_folder, exist_ok=True)

        with open(protein_file_path, "w", encoding="utf-8") as f:
            f.write(str(protein_sequence))

        print(f"\n✅ Protein saved to: {protein_file_path}")

        # Save metadata
        metadata_path = protein_file_path.replace(".txt", "_metadata.json")

        metadata = {
            "source_file": fasta_file,
            "record_id": record.id,
            "input_length_nt": input_length,
            "input_type": "full_genome" if is_full_genome else "partial_sequence",
            "search_method": search_method,
            "start_position": start_pos,
            "end_position": start_pos + end_pos_relative,
            "gene_length_nt": len(gene_sequence),
            "protein_length_aa": protein_length,
            "status": status,
            "difference_vs_expected": protein_length - EXPECTED_SPIKE_LENGTH_AA
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"📋 Metadata saved to: {metadata_path}")

    except OSError as e:
        print(f"❌ Error saving files: {e}")
        return None

    return protein_sequence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 spike_extractor.py <genome.fasta>") # Changed usage name to reflect function focus
        print("\nExample:")
        print("  python3 spike_extractor.py wuhan.fasta")
        print("  python3 spike_extractor.py omicron_BA1.fasta")
        print("\nSupports:")
        print("  - Full SARS-CoV-2 genome (~30kb)")
        print("  - Partial sequence (Spike region only)")
        print("\nOutput:")
        print("  - spike_name.txt          (amino acid sequence)")
        print("  - spike_name_metadata.json (process information)")
        sys.exit(1)

    # Use a more generic name for the script execution context if it's being refactored universally
    result = process_reference(sys.argv[1])

    if result is None:
        sys.exit(1)
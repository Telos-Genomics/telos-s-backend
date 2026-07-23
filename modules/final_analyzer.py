import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # <--- Force matplotlib to render without a display
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.lines import Line2D
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch

# ---------------------------------------------------------------------------
# RELIABILITY SYSTEM
#
# The problem:
#   ESM-2 predicts the most likely amino acid at a position based on the
#   context of neighboring residues. If a neighbor is X (indeterminate),
#   the context is corrupted and the prediction -- and therefore the LLR and
#   Score -- are not trustworthy.
#
# The solution:
#   1. Identify every position that contains an X in the CSV.
#   2. Build an "exclusion zone" around each X (a +-5 residue window, which
#      is roughly the immediate context radius that most affects ESM-2's
#      prediction at this model scale).
#   3. Classify every mutation into three tiers:
#        RELIABLE    -> outside any exclusion zone
#        SUSPECT     -> inside an exclusion zone
#        INVALID     -> contains X directly
#   4. Alerts, scoring, and lineage calls are computed only on RELIABLE data.
#      SUSPECT entries appear on the heatmap (in grey) and in the report as
#      a warning, but never trigger an alert.
# ---------------------------------------------------------------------------

CONTEXT_WINDOW = 5  # residues on each side of an X

# NOTE ON DATA CONTRACT:
# The DataFrame column names below ('Mutation', 'Score', 'Context', 'LLR',
# 'Pos', 'Reliability', 'Status') and the classification labels
# ('RELIABLE', 'SUSPECT', 'INVALID') are the schema of the upstream
# CSV files and are consumed by other backend components. They are left
# untranslated intentionally, since renaming them would change the on-disk
# data contract rather than just the code style.


def load_prophet_predictions(csv_path):
    # Extract just the file name: report_spike_job_xyz.csv
    filename = os.path.basename(csv_path)

    # Strip it down to just the ID: job_xyz
    # Assuming the format "report_spike_{variant_name}.csv"
    job_id_part = filename.replace('report_spike_', '').replace('.csv', '')

    # Build the path to the JSON based on the folder structure
    # Go up from the 'report' folder to 'output' and then down to 'prophet'
    base_dir = Path(csv_path).resolve().parents[2]  # From report/ to s/ to output/
    json_path = base_dir / "prophet" / f"mutation_predictions_spike_{job_id_part}.json"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"⚠️ Oracle JSON not found at: {json_path}")
    return None


def classify_reliability(df):
    """
    Adds the 'Reliability' column to each row of the DataFrame.

    Logic:
      - First find every position where there is an X (invalid).
      - Expand each one into a +-CONTEXT_WINDOW zone (suspect).
      - Everything else is reliable.
    """
    # Extract the numeric position
    df['Pos'] = df['Mutation'].str.extract(r'(\d+)').astype('Int64')  # nullable Int64

    # 1. Positions that contain X directly
    invalid_mask = df['Mutation'].str.contains('X', na=False)
    x_positions = set(df.loc[invalid_mask, 'Pos'].dropna().astype(int).tolist())

    # 2. Expand the exclusion zones
    suspect_positions = set()
    for x_pos in x_positions:
        for offset in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
            suspect_positions.add(x_pos + offset)
    # Remove the X positions themselves (those are INVALID, not SUSPECT)
    suspect_positions -= x_positions

    # 3. Classify each row (vectorized, no apply)
    #
    #   Order matters: each condition overwrites the previous one.
    #   We start by assuming RELIABLE and mark exceptions from there.
    #
    df['Reliability'] = 'RELIABLE'

    # Suspect: position inside the exclusion zone
    df.loc[df['Pos'].isin(suspect_positions), 'Reliability'] = 'SUSPECT'

    # Invalid (overwrites SUSPECT if applicable):
    #   - Pos is NaN (could not extract a number)
    #   - The mutation text contains X
    #   - The position exactly matches a known X position
    df.loc[df['Pos'].isna(),                                          'Reliability'] = 'INVALID'
    df.loc[df['Mutation'].str.contains('X', na=False),                'Reliability'] = 'INVALID'
    df.loc[df['Pos'].isin(x_positions),                               'Reliability'] = 'INVALID'
    return df, x_positions


def analyze_strain(csv_path):
    df = pd.read_csv(csv_path)

    # --- Reliability classification ---
    df, x_positions = classify_reliability(df)

    # IMPORTANT: Save the CSV with the Reliability column for the backend
    df.to_csv(csv_path, index=False)

    # --- Subsets by reliability tier ---
    df_reliable = df[df['Reliability'] == 'RELIABLE'].copy()
    df_suspect = df[df['Reliability'] == 'SUSPECT'].copy()
    df_invalid = df[df['Reliability'] == 'INVALID'].copy()

    # Extra filter: only biological Spike positions (1-1273)
    df_reliable = df_reliable[
        (df_reliable['Pos'] >= 1) &
        (df_reliable['Pos'] <= 1273) &
        (~df_reliable['Mutation'].str.startswith('-', na=False))
    ].copy()

    # ---------------------------------------------------------------------------
    # 1. Sequence quality
    # ---------------------------------------------------------------------------
    total = len(df)
    n_invalid = len(df_invalid)
    n_suspect = len(df_suspect)
    quality = ((total - n_invalid) / total) * 100 if total > 0 else 0.0

    # ---------------------------------------------------------------------------
    # 2. Aggression Score (reliable data only)
    # ---------------------------------------------------------------------------
    aggression_score = df_reliable['Score'].abs().sum()

    # ---------------------------------------------------------------------------
    # 3. Lineage identification (reliable data only)
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # LINEAGE SIGNATURES
    #
    # Each entry contains DIFFERENTIAL mutations: the ones that distinguish
    # that lineage from the others. Ancestral mutations shared by nearly all
    # post-2020 lineages (like D614G) are not included, since they carry no
    # discriminative power.
    #
    # Inclusion criteria:
    #   - Mutations reported as "defining" in peer-reviewed literature or in
    #     CoV-Lineages/Nextstrain.
    #   - Markers whose target is X are excluded (they never match reliable
    #     data), as are mutations that appear in more than 3 distinct
    #     lineages.
    #
    # Sources:
    #   - WHO VOC/VOI tracking (2024-2025)
    #   - Nature Communications 2024 (XBB/BA.2.86/JN.1 characterization)
    #   - PMC 10661123 (BA.2.12.1 and BA.4/5 functional impact)
    #   - PMC 10782855 (Evolution of Omicron spike)
    #   - mBio 2024 (BA.2.86 -> JN.1 -> KP.2/KP.3 comparative)
    #   - Lancet Infect Dis 2024 (JN.1 virological characterization)
    #   - bioRxiv 2024.12.10 (KP.3.1.1 structural analysis)
    # ---------------------------------------------------------------------------
    lineage_signatures = {
        # --- Original VOCs (pre-Omicron) ---
        # Signatures unique to each, not shared with the other VOCs in this list.
        "Alpha (B.1.1.7)":              ["A570D", "P681H", "T716I", "S982A", "N501Y"],
        "Beta (B.1.351)":               ["K417N", "E484K", "N501Y", "A701V"],
        "Gamma (P.1)":                  ["K417T", "E484K", "N501Y", "H655Y"],
        "Delta (B.1.617.2)":            ["T19R",  "L452R", "T478K", "P681R", "D950N"],

        # --- Omicron: BA.1 lineage ---
        # BA.1 diverged strongly from Wuhan in the RBD.
        # S371L is specific to BA.1 (BA.2 and descendants have S371F).
        "Omicron BA.1 (B.1.1.529)":     ["G339D", "S371L", "S373P", "S375F",
                                          "K417N", "N440K", "G446S", "S477N",
                                          "E484A", "Q493R", "G498R", "N501Y", "Y505H"],

        # --- Omicron: BA.2 lineage ---
        # BA.2 replaced BA.1. Key differences vs BA.1:
        #   S371F (vs S371L), L212I, V213G, T376A, D405N, R408S, S704L.
        "Omicron BA.2 (B.1.1.529.2)":   ["S371F", "L212I", "V213G", "T376A",
                                          "D405N", "R408S", "S477N", "E484A",
                                          "N501Y", "S704L"],

        # --- Omicron: BA.4/BA.5 lineage ---
        # BA.4 and BA.5 share an identical spike. Differences vs BA.2:
        #   del69-70, L452R, F486V, R493Q (reversion).
        # L452R and F486V are the key signatures.
        "Omicron BA.4/BA.5":            ["L452R", "F486V", "R493Q",
                                          "N440K", "G446S", "S477N", "N501Y"],

        # --- Omicron: BQ.1.1 ("Cerberus") ---
        # Descendant of BA.5. Added R346T and K526M on top of BA.5.
        # L452R and F486V are inherited from BA.5.
        "Omicron BQ.1.1":               ["R346T", "L452R", "F486V", "K526M",
                                          "N440K", "G446S", "N501Y"],

        # --- Omicron: XBB.1.5 ("Kraken") ---
        # BA.2 recombinant. Key RBM signatures: V445P, F486P, F490S.
        # F486P (vs F486V in BA.5) is the most important signature.
        "Omicron XBB.1.5":              ["V445P", "F486P", "F490S",
                                          "S477N", "N501Y", "G446S"],

        # --- Omicron: EG.5 ("Eris") ---
        # Descendant of XBB.1.9.2. The defining signature is F456L,
        # which appears alongside F486P inherited from XBB.
        "Omicron EG.5 (Eris)":          ["F456L", "F486P", "F490S",
                                          "V445P", "S477N", "N501Y"],

        # --- Omicron: BA.2.86 ("Pirola") ---
        # Major evolutionary jump: >30 mutations vs BA.2.
        # Signatures absolutely unique relative to every other lineage here:
        #   D339H, N394K, A484K (reversion of E484A back to K, not E484K),
        #   V483A; N501Y is retained.
        # Q493 reverts to its original form (reversion).
        "Omicron BA.2.86 (Pirola)":     ["D339H", "N394K", "V483A", "A484K",
                                          "N501Y", "V445P", "S477N", "N440K"],

        # --- Omicron: JN.1 ("Juno") ---
        # Direct descendant of BA.2.86. One additional RBD mutation:
        # L455S. This is the defining signature.
        # Was globally dominant in winter 2023-2024.
        "Omicron JN.1 (Juno)":          ["L455S", "D339H", "N394K", "V483A",
                                          "A484K", "N501Y", "V445P", "S477N"],

        # --- Omicron: KP.2 / KP.3 ("FLiRT") ---
        # Descendants of JN.1. Added R346T and F456L on top of JN.1.
        # The R346T + L455S + F456L combination is the "FLiRT" signature.
        "Omicron KP.2/KP.3 (FLiRT)":   ["R346T", "L455S", "F456L",
                                          "D339H", "N394K", "A484K", "N501Y"],

        # --- Omicron: KP.3.1.1 ---
        # Added Q493E on top of KP.3. Was dominant in the second half of 2024.
        # The L455S + F456L + Q493E combination has positive epistasis
        # that keeps ACE2 affinity high.
        "Omicron KP.3.1.1":            ["Q493E", "R346T", "L455S", "F456L",
                                          "D339H", "N394K", "A484K", "N501Y"],

        # --- Omicron: XFG ---
        # Globally dominant in the second half of 2025 per WHO.
        # Descendant of JN.1 with convergent signatures similar to FLiRT.
        # Shares L455S and F456L; differentiator: L452R (reversion/
        # convergence with Delta and BA.5).
        "Omicron XFG":                  ["L452R", "L455S", "F456L",
                                          "D339H", "N394K", "A484K", "N501Y"],

        # --- Omicron: BA.3.2 ("Cicada") ---
        # VUM declared by WHO in December 2025. >70 mutations vs Wuhan.
        # Descendant of the ancestral BA.3 (not in circulation since 2022).
        # Differentiating signatures: recovered G496S (present in BA.1,
        # absent since then), has A435S, R403S, P681H.
        # Does NOT have L455S (unlike the entire JN.1 line).
        "Omicron BA.3.2 (Cicada)":      ["G496S", "A435S", "R403S", "P681H",
                                          "N501Y", "R493Q", "N440K"],
    }

    predicted_lineage, max_match_pct = identify_lineage(df_reliable, lineage_signatures)

    # ---------------------------------------------------------------------------
    # 4. Load Prophet predictions
    # ---------------------------------------------------------------------------
    prophet_data = load_prophet_predictions(csv_path)

    # ---------------------------------------------------------------------------
    # 5. Console report
    # ---------------------------------------------------------------------------
    print("=" * 50)
    print("  VARIANT EXECUTIVE SUMMARY")
    print("=" * 50)
    print(f"  Sequencing quality:        {quality:.2f}%")
    print(f"  Reliable mutations:        {len(df_reliable)}")
    print(f"  Suspect mutations:         {n_suspect}  (context corrupted by X)")
    print(f"  Invalid mutations:         {n_invalid}  (contain X)")
    print(f"  Aggression Score:          {aggression_score:.1f}  (reliable data only)")
    print(f"  Probable lineage:          {predicted_lineage} ({max_match_pct:.1f}%)")
    print("=" * 50)

    # ---------------------------------------------------------------------------
    # 6. Heatmap and report
    # ---------------------------------------------------------------------------
    generate_heatmap(df_reliable, df_suspect, x_positions,
                      aggression_score, predicted_lineage, csv_path, prophet_data)

    generate_txt_report(df_reliable, df_suspect, df_invalid,
                         aggression_score, predicted_lineage, max_match_pct, quality,
                         csv_path, prophet_data)

    generate_pdf_report(df_reliable, df_suspect, df_invalid,
                         aggression_score, predicted_lineage, max_match_pct, quality,
                         csv_path, prophet_data)


def identify_lineage(df_reliable, lineage_signatures):
    """Looks for lineage markers only among reliable mutations."""
    predicted_lineage = "Unknown"
    max_match_pct = 0.0

    for lineage, markers in lineage_signatures.items():
        matches = 0
        for marker in markers:
            # Split the marker into its components (e.g. "K417N" -> K, 417, N)
            mutated_residue = marker[-1]
            marker_pos = int(marker[1:-1])

            # Ignore markers whose target is X (not a real data point)
            if mutated_residue == 'X':
                continue

            # Search with position tolerance (+-5, for possible indels)
            pos_range = range(marker_pos - 5, marker_pos + 6)
            match = df_reliable[
                (df_reliable['Pos'].isin(pos_range)) &
                (df_reliable['Mutation'].str.endswith(mutated_residue, na=False))
            ]
            if not match.empty:
                matches += 1

        # Compute the percentage, excluding X markers from the denominator
        valid_markers = sum(1 for m in markers if m[-1] != 'X')
        if valid_markers > 0:
            percentage = (matches / valid_markers) * 100
        else:
            percentage = 0.0

        if percentage > max_match_pct:
            max_match_pct = percentage
            predicted_lineage = lineage

    return predicted_lineage, max_match_pct


def generate_heatmap(df_reliable, df_suspect, x_positions,
                      total_score, lineage, csv_path, prophet_data=None):
    fig, ax = plt.subplots(figsize=(15, 7))

    # ---------------------------------------------------------------------------
    # Background and critical zones
    # ---------------------------------------------------------------------------
    ax.axhline(0, color='lightgrey', linewidth=20, alpha=0.3, zorder=1)
    ax.axvspan(319, 541, color='blue',   alpha=0.08, label='RBD Domain')
    ax.axvspan(437, 508, color='cyan',   alpha=0.15, label='RBM Motif')
    ax.axvspan(681, 685, color='purple', alpha=0.20, label='Furin Site')

    # ---------------------------------------------------------------------------
    # X exclusion zones (faint orange background)
    # ---------------------------------------------------------------------------
    if x_positions:
        # Group contiguous X positions into ranges to draw a single block
        sorted_x_positions = sorted(x_positions)
        blocks = []
        start = sorted_x_positions[0]
        end = sorted_x_positions[0]
        for pos in sorted_x_positions[1:]:
            if pos <= end + CONTEXT_WINDOW * 2 + 1:
                end = pos  # Merge nearby blocks
            else:
                blocks.append((start - CONTEXT_WINDOW, end + CONTEXT_WINDOW))
                start = pos
                end = pos
        blocks.append((start - CONTEXT_WINDOW, end + CONTEXT_WINDOW))

        for (x0, x1) in blocks:
            ax.axvspan(x0, x1, color='orange', alpha=0.12, zorder=2)

    # ---------------------------------------------------------------------------
    # Dynamic vertical scale
    # ---------------------------------------------------------------------------
    max_reliable_llr = df_reliable['LLR'].abs().max() if not df_reliable.empty else 1
    prediction_height = max(max_reliable_llr + 3, 8)
    upper_limit = max(max_reliable_llr, prediction_height) + 5

    # ---------------------------------------------------------------------------
    # Draw RELIABLE mutations (solid, colored by risk)
    # ---------------------------------------------------------------------------
    for _, row in df_reliable.iterrows():
        if "🔴" in str(row['Status']):
            color = "red"
        elif row['Score'] > 30:
            color = "orange"
        else:
            color = "gold"

        height = abs(row['LLR'])
        ax.scatter(row['Pos'], height, color=color, s=100,
                   edgecolor='black', linewidth=0.8, zorder=5)
        ax.text(row['Pos'], height + 0.3, row['Mutation'],
                fontsize=7.5, rotation=45, ha='left', color='black')

    # ---------------------------------------------------------------------------
    # Draw SUSPECT mutations (grey, no alert label)
    # ---------------------------------------------------------------------------
    for _, row in df_suspect.iterrows():
        if pd.isna(row['Pos']):
            continue
        height = abs(row['LLR']) if pd.notna(row['LLR']) else 0
        ax.scatter(row['Pos'], height, color='grey', s=60, alpha=0.5,
                   edgecolor='grey', linewidth=0.8, zorder=4, marker='x')
        ax.text(row['Pos'], height + 0.2, row['Mutation'],
                fontsize=6.5, rotation=45, ha='left', color='grey', style='italic')

    # ---------------------------------------------------------------------------
    # Prophet predictions (reliable positions only)
    # ---------------------------------------------------------------------------
    if prophet_data:
        reliable_positions = set(df_reliable['Pos'].dropna().astype(int).tolist())

        for target in prophet_data:
            pos = target['wuhan_position']

            # Only draw if the position is in a reliable zone
            if pos not in reliable_positions:
                continue

            original = target['original_aa']
            candidates = [p for p in target['predictions'] if p['amino'] != original]

            if candidates:
                top_candidate = candidates[0]
                confidence = top_candidate['confidence']

                if confidence > 5:
                    ax.scatter(pos, prediction_height, facecolors='none',
                               edgecolors='magenta', s=250, linestyles='--',
                               linewidth=2, zorder=6)
                    ai_label = f"{original}{pos}{top_candidate['amino']}\n{confidence:.1f}%"
                    ax.text(pos, prediction_height + 0.8, ai_label,
                            color='darkmagenta', fontsize=8.5, fontweight='bold',
                            ha='center',
                            bbox=dict(facecolor='white', alpha=0.75,
                                      edgecolor='none', boxstyle='round'))

    # ---------------------------------------------------------------------------
    # Styling
    # ---------------------------------------------------------------------------
    ax.set_title(
        f"TELOS-S: Variant Intelligence {lineage} | Score: {total_score:.1f}  "
        f"(reliable data only)",
        fontsize=13
    )
    ax.set_xlabel("Position in the Spike Protein (Residues)")
    ax.set_ylabel("Structural Impact (|LLR|)")
    ax.set_xlim(0, 1273)
    ax.set_ylim(-1, upper_limit)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Critical Threat',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Variant of Interest',
               markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Low Risk',
               markerfacecolor='gold', markersize=10),
        Line2D([0], [0], marker='x', color='grey', label='Suspect (X context)',
               markersize=9, linestyle='None'),
        Line2D([0], [0], marker='o', color='w', label='Evolutionary Path (AI)',
               markeredgecolor='magenta', markerfacecolor='none', markersize=12),
        Line2D([0], [0], color='orange', lw=6, alpha=0.2, label='Exclusion Zone (X)'),
        Line2D([0], [0], color='blue',   lw=4, alpha=0.3, label='RBD Zone'),
        Line2D([0], [0], color='cyan',   lw=4, alpha=0.3, label='RBM Zone'),
        Line2D([0], [0], color='purple', lw=4, alpha=0.3, label='Furin Zone'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')

    plt.tight_layout()

    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "output" / "s" / "report"
    filename = Path(csv_path).name

    # Save
    folder_path = "output/s/reports"
    base_name = (filename
                 .replace('.csv', '')
                 .replace('report_', ''))

    image_path = os.path.join(folder_path, f"heatmap_{base_name}.svg")

    try:
        plt.savefig(image_path, dpi=150)
        print(f"🎨 Heatmap generated: {image_path}")
    except OSError as e:
        print(f"❌ Error saving heatmap: {e}")
    finally:
        plt.close(fig)


def generate_txt_report(df_reliable, df_suspect, df_invalid,
                         score, lineage, lineage_prob, quality, csv_path, prophet_data):
    """Generates the .txt report with sections split by reliability tier."""

    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "output" / "s" / "report"
    filename = Path(csv_path).name

    # Save
    folder_path = "output/s/report"
    base_name = (filename
                 .replace('.csv', '')
                 .replace('report_', ''))

    report_path = f"output/s/reports/executive_report_{base_name}.txt"

    # Top 3 threats (reliable only)
    top_threats = df_reliable.sort_values(by='Score', ascending=False).head(3)

    # Verdict based on reliable data
    if score > 1200:
        verdict = "🔴 MAXIMUM ALERT"
        risk_level = "CRITICAL"
    elif score > 600:
        verdict = "🟠 ACTIVE MONITORING"
        risk_level = "HIGH"
    else:
        verdict = "🟡 OBSERVATION"
        risk_level = "MODERATE"

    # Wuhan reference for sites of interest
    wuhan_ref_map = {
        "RBM_452": "L",
        "RBM_484": "E",
        "RBM_501": "N",
        "Furin_Cleavage_681": "P",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==============================================\n")
        f.write("   GENOMIC INTELLIGENCE REPORT\n")
        f.write("   Telos-S -- Variant Analysis\n")
        f.write("==============================================\n\n")

        f.write(f"SAMPLE:                {base_name}\n")
        f.write(f"VERDICT:               {verdict}\n")
        f.write(f"AGGRESSION SCORE:      {score:.1f}\n")
        f.write(f"PROBABLE LINEAGE:      {lineage}\n")
        f.write(f"LINEAGE PROBABILITY:   {lineage_prob}\n\n")

        # --- Quality and reliability ---
        f.write("--- DATA QUALITY ---\n")
        f.write(f"  Sequencing quality:         {quality:.2f}%\n")
        f.write(f"  Reliable mutations:         {len(df_reliable)}\n")
        f.write(f"  Suspect mutations:          {len(df_suspect)}  "
                f"(within +-{CONTEXT_WINDOW} residues of an X)\n")
        f.write(f"  Invalid mutations:          {len(df_invalid)}  "
                f"(contain X directly)\n\n")

        # --- Risk analysis ---
        f.write("--- RISK ANALYSIS ---\n")
        f.write(f"  Risk level: {risk_level}\n")
        f.write(f"  An accumulation of mutations is observed in the RBD/RBM, "
                f"suggesting immune escape capability.\n\n")

        # --- Top 3 ---
        f.write("--- TOP 3 CRITICAL MUTATIONS (reliable data) ---\n")
        if not top_threats.empty:
            for _, row in top_threats.iterrows():
                f.write(f"  • {row['Mutation']}: Zone {row['Context']} | Score: {row['Score']:.1f}\n")
        else:
            f.write("  No reliable mutations were detected.\n")
        f.write("\n")

        # --- Suspect mutations (warning, no alert) ---
        if not df_suspect.empty:
            f.write("--- SUSPECT MUTATIONS (no alert triggered) ---\n")
            f.write(f"  These {len(df_suspect)} mutations fall within the exclusion "
                    f"zone of an X residue. Their LLR and Score values are not "
                    f"reliable because the context ESM-2 used to predict them was "
                    f"corrupted. They are listed below for reference, but do NOT "
                    f"contribute to the Score or the verdict.\n\n")

            df_suspect_sorted = df_suspect.sort_values(by='Score', ascending=False)
            for _, row in df_suspect_sorted.iterrows():
                if pd.isna(row['Pos']):
                    continue
                f.write(f"  ⚠ {row['Mutation']}: Zone {row['Context']} | "
                        f"LLR: {row['LLR']:.4f} | Score: {row['Score']:.1f} "
                        f"(NOT RELIABLE)\n")
            f.write("\n")

        # --- Prophet evolutionary forecast ---
        if prophet_data:
            reliable_positions = set(df_reliable['Pos'].dropna().astype(int).tolist())

            f.write("--- EVOLUTIONARY FORECAST (TELOS PROPHET) ---\n")
            f.write("  Structural stability analysis via AI (ESM-2).\n")
            f.write("  Only predictions for reliable positions are included.\n\n")

            for target in prophet_data:
                pos = target['wuhan_position']
                name = target['target_site']
                current = target['original_aa']

                # Check whether the position is reliable
                is_reliable = pos in reliable_positions

                # Look it up in the reliable DataFrame
                csv_match = df_reliable[
                    (df_reliable['Pos'] >= pos - 2) &
                    (df_reliable['Pos'] <= pos + 2)
                ].sort_values(by='Score', ascending=False)

                if not csv_match.empty:
                    current_aa = csv_match.iloc[0]['Mutation'][-1]
                    wuhan_ref = csv_match.iloc[0]['Mutation'][0]
                else:
                    current_aa = current
                    wuhan_ref = wuhan_ref_map.get(name, "?")

                if not is_reliable:
                    f.write(f"  • {name} ⚠ EXCLUDED -- "
                            f"position within an X exclusion zone. "
                            f"No reliable prediction can be issued.\n")
                    continue

                f.write(f"  • {name} (Wuhan Ref: {wuhan_ref} | "
                        f"Current: {current_aa}):\n")

                best_mutation = next(
                    (p for p in target['predictions'] if p['amino'] != current), None
                )

                if best_mutation and best_mutation['confidence'] > 20:
                    f.write(f"    [!] ALERT: Path toward {best_mutation['amino']} "
                            f"with {best_mutation['confidence']:.1f}% structural "
                            f"probability.\n")
                elif best_mutation:
                    f.write(f"    [✓] Stable. Best detected path: "
                            f"{best_mutation['amino']} with {best_mutation['confidence']:.1f}% "
                            f"structural probability.\n")
                else:
                    f.write(f"    [✓] Stable. No mutation paths detected.\n")

            f.write("\n")

        # --- Closing note ---
        f.write("--- METHODOLOGY NOTE ---\n")
        f.write("  The Aggression Score and verdict are computed exclusively on\n")
        f.write("  mutations classified as RELIABLE. Mutations in exclusion\n")
        f.write(f"  zones (+-{CONTEXT_WINDOW} residues of an X) are classified as\n")
        f.write("  SUSPECT and do not contribute to any alert indicator.\n")
        f.write("  Lab artifacts (His-tags/Linkers) have been filtered out.\n\n")

        f.write("==============================================\n")
        f.write("  Generated by: Telos-S -- Genomic Analyzer\n")
        f.write("==============================================\n")

    print(f"📄 Executive report: {report_path}")


def generate_pdf_report(df_reliable, df_suspect, df_invalid,
                         score, lineage, lineage_prob, quality, csv_path, prophet_data):

    # --- File configuration ---
    base_name = Path(csv_path).name.replace('.csv', '').replace('report_', '')
    pdf_path = f"output/s/reports/executive_report_{base_name}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)

    styles = getSampleStyleSheet()
    elements = []

    # --- Custom styles ---
    style_title = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=10, textColor=colors.HexColor("#1A237E"))
    style_header = ParagraphStyle('SubTitle', parent=styles['Heading2'], fontSize=12, spaceAfter=5, textColor=colors.grey)
    style_body = styles["BodyText"]
    style_warning = ParagraphStyle('Warning', parent=style_body, textColor=colors.red, fontSize=9)

    # --- 1. Header ---
    elements.append(Paragraph("GENOMIC INTELLIGENCE REPORT", style_title))
    elements.append(Paragraph(f"Telos-S Analysis Platform | ID: {base_name}", style_header))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=20))

    # --- 2. Summary box (verdict) ---
    verdict_color = colors.orange if score > 600 else colors.green
    if score > 1200: verdict_color = colors.red

    summary_data = [
        ["AGGRESSION SCORE", f"{score:.1f}"],
        ["VERDICT", "MAXIMUM ALERT" if score > 1200 else ("ACTIVE MONITORING" if score > 600 else "OBSERVATION")],
        ["PROBABLE LINEAGE", f"{lineage} ({lineage_prob:.1f}%)"],
        ["SEQUENCING QUALITY", f"{quality:.2f}%"],
        ["RELIABLE MUTATIONS", f"{len(df_reliable)}"],
        ["SUSPECT MUTATIONS", f"{len(df_suspect)} (within +-{CONTEXT_WINDOW} residues of an X)"],
        ["INVALID MUTATIONS", f"{len(df_invalid)} (contain X directly)"]
    ]

    t = Table(summary_data, colWidths=[3*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))

    # 1. Color logic (using more professional tones)
    if score > 1200:
        verdict = "🔴 MAXIMUM ALERT"
        risk_level = "CRITICAL"
        dynamic_color = colors.HexColor("#D32F2F")  # Deep Red
    elif score > 600:
        verdict = "🟠 ACTIVE MONITORING"
        risk_level = "HIGH"
        dynamic_color = colors.HexColor("#F57C00")  # Intense Orange
    else:
        verdict = "🟡 OBSERVATION"
        risk_level = "MODERATE"
        dynamic_color = colors.HexColor("#388E3C")  # Forest Green

    # 2. Create a UNIQUE style for the risk line
    # Important: the name 'RiskLevelStyle' must not be reused in the script
    risk_style = ParagraphStyle(
        'RiskLevelStyle',
        parent=styles['Heading4'],
        textColor=dynamic_color,
        fontSize=11,
        spaceBefore=10
    )

    # 3. Build the section
    elements.append(Paragraph("RISK ANALYSIS", styles['Heading3']))

    # Bullet point with the dynamic color
    risk_text = f"• <b>Risk level: {risk_level}</b>"
    elements.append(Paragraph(risk_text, risk_style))

    # Description in normal style
    elements.append(Paragraph(
        "An accumulation of mutations is observed in the RBD/RBM, suggesting immune escape capability.",
        styles["Normal"]
    ))

    # --- 3. Top mutations (clean table) ---
    elements.append(Paragraph("TOP 3 CRITICAL MUTATIONS", styles['Heading3']))
    if not df_reliable.empty:
        top_3 = df_reliable.sort_values(by='Score', ascending=False).head(3)
        mutation_data = [["Mutation", "Zone", "Score"]]
        for _, r in top_3.iterrows():
            mutation_data.append([r['Mutation'], r['Context'], f"{r['Score']:.1f}"])

        mutation_table = Table(mutation_data, colWidths=[1.5*inch, 2.5*inch, 1*inch])
        mutation_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E3F2FD")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (2,0), (2,-1), 'CENTER'),
        ]))
        elements.append(mutation_table)

    # --- 4. Prophet section (visual alerts) ---
    elements.append(Paragraph("EVOLUTIONARY FORECAST (TELOS PROPHET)", styles['Heading3']))

    if prophet_data:
        reliable_positions = set(df_reliable['Pos'].dropna().astype(int).tolist())

        for target in (prophet_data or []):
            # Check whether the position is reliable
            pos = target['wuhan_position']
            is_reliable = pos in reliable_positions

            if is_reliable:

                best = next((p for p in target['predictions'] if p['amino'] != target['original_aa']), None)
                alert_color = "#D32F2F" if best and best['confidence'] > 20 else "#2E7D32"

                text = f"<b>{target['target_site']}</b>: "
                if best:
                    text += f"Possible path toward <b>{best['amino']}</b> ({best['confidence']:.1f}% structural prob.)."
                else:
                    text += "Stable structure."

                paragraph_style = ParagraphStyle('Prophet', parent=style_body, leftIndent=10, textColor=colors.HexColor(alert_color))
                elements.append(Paragraph(f"• {text}", paragraph_style))

            else:
                text = f"<b>{target['target_site']}</b>: EXCLUDED -- position within an X exclusion zone. No reliable prediction can be issued."

                paragraph_style = ParagraphStyle('Prophet', parent=style_body, leftIndent=10, textColor=colors.HexColor('#757575'))
                elements.append(Paragraph(f"• {text}", paragraph_style))

    # --- 5. Methodology (footer) ---
    elements.append(Spacer(1, 40))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Paragraph("Methodology: the score is computed on RELIABLE mutations (outside the +-5 exclusion zone around an X residue).", styles['Italic']))

    # Build the PDF
    doc.build(elements)
    print(f"📄 PDF generated: {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 final_analyzer.py <report.csv>")
        print("\nExample:")
        print("  python3 final_analyzer.py output/s/reports/report_spike_omicron.csv")
        sys.exit(1)

    analyze_strain(sys.argv[1])
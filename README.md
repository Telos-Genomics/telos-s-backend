# TELOS-S v0.1.1

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21380916.svg)](https://doi.org/10.5281/zenodo.21380916)

### Powered by Telos Genomics

**Predictive Intelligence Engine for Protein Evolution.**

Telos-S represents the first deployment of the Telos Genomics pipeline,
utilizing Large Protein Language Models (ESM-2 650M) to quantify
mutational impact on the SARS-CoV-2 Spike protein.

## 📊 Workflow

Use only confirmed sequencing data. Positions marked with `X` are excluded from analysis.

**Result**: Aggression Score includes predictions. The report clearly indicates which data are actual versus imputed.

---

## 🛠️ System Components

### 1. `extraer_spike.py`

Extract spike protein from SARS-CoV-2 genome

```bash
python3 extraer_spike.py <reference.txt>
```

**Output**

- `output/s/spike/spike_name.txt`

---

### 2. `alineador_secuencias.py`

Align the variant spike against Wuhan spike using template-based alignment.

**Guarantees:**

- Aligned sequences of exactly 1273 characters
- Gaps only in the variant (deletions)
- Discarded insertions (they do not have a canonical position)

```bash
python3 alineador_secuencias.py <referencia.txt> <variante.txt>
```

**Output**:

- `output/s/spike_aligned/spike_wuhan_ref.txt`
- `output/s/spike_aligned/spike_variant_name.txt`

---

### 3. `imputar_secuencia.py` ⭐ NEW

Try filling in the spaces with 'X' using the reference (Wuhan) as a template; the imputed positions will not be used in subsequent calculations.

**Estrategia**:

- Verify that the reference and variant are correctly aligned.
- Make sure the reference doesn't have a gap (-) or an 'X' to mirror the sequences
- It performs the operation on data loss of 5 or more blocks

```bash
python3 imputar_secuencia.py <output/s/spike_aligned/spike_wuhan_ref.txt> <output/s/spike_aligned/spike_variant_name.txt>
```

**Output**:

- `output/s/spike_aligned/spike_name_imputed.txt` - Imputed sequence (for debuging purposes only)
- `output/prophet/imputation_spike_variant_name.json` - Mirrored secuences

**JSON includes**:

```json
{
    "metodo": "Imputaci\u00f3n por Referencia",
    "total_imputados": 20,
    "posiciones": [
        {
            "idx": 784,
            "wuhan_pos": 780,
            "res": "V"
        },
        {
            "idx": 785,
            "wuhan_pos": 781,
            "res": "K"
        },
        {
            "idx": 786,
            "wuhan_pos": 782,
            "res": "Q"
        },
        {
            "idx": 787,
            "wuhan_pos": 783,
            "res": "I"
        },
        ...
    ]
}
```

---

### 4. `oraculo_mutaciones.py`

Predicts future evolution in 4 critical positions: 452, 484, 501, 681 only if the positions are real (not imputed).
Predicts the top 5 most probable mutations at each position with % confidence based on structural stability.

```bash
python3 oraculo_mutaciones.py <spike_aligned.txt> [--cpu]
```

**Output**: `output/prophet/mutation_predictions_spike_variant_name.json`

**JSON includes**:

```json
[
    {
        "target": "Sitio_RBM_452",
        "detected_position": 452,
        "aligned_index": 451,
        "clean_index": 451,
        "original": "R",
        "predictions": [
            {
                "amino": "G",
                "confidence": 8.666696399450302
            },
            {
                "amino": "T",
                "confidence": 8.07761624455452
            },
            {
                "amino": "Y",
                "confidence": 7.56719708442688
            },
            {
                "amino": "N",
                "confidence": 7.294604927301407
            },
            {
                "amino": "V",
                "confidence": 6.8845875561237335
            }
        ]
    },
    ...
}
```

---

### 5. `comparador_inteligente.py`

Compare reference vs variant, calculate LLR and Risk Score.

```bash
python3 comparador_inteligente.py <ref_aligned.txt> <var_aligned.txt> [--cpu]
```

**Output**: `output/s/report/reporte_spike_variante.csv`

Columns:

- `Mutation`: "E484K" format
- `Zone`: CRITICAL/HIGH/Medium/Normal
- `LLR`: Log-likelihood ratio (structural stability)
- `Score`: Combined biological risk
- `P_Original`, `P_Mutant`: Model probabilities

---

### 6. `analizador_final.py`

Comprehensive analysis: lineage, scoring, heatmap, executive report.

```bash
python3 analizador_final.py <reporte.csv>
```

**Reliability System**:

- **RELIABLE**: Outside of areas with X, contributes to the Score
- **SUSPICIOUS**: Within ±5 positions of an X, does NOT contribute to the Score
- **INVALID**: Contains X directly, excluded from the analysis

**Output**:

- PNG Heatmap with mutations and predictions
- TXT Executive Summary with verdict and recommendations
- Updated CSV with reliability classification

**Lineage Signatures** (16 variants):

- VOCs: Alpha, Beta, Gamma, Delta
- Omicron: BA.1, BA.2, BA.4/5, BQ.1.1, XBB.1.5, EG.5
- JN.1 Lineage: BA.2.86, JN.1, KP.2/3, KP.3.1.1
- Emerging: XFG, BA.3.2 (Cicada)

---

## ⚙️ Configurable Parameters

### Context Window (`analizador_final.py`)

Defined as `CONTEXT_WINDOW = 5` on line 35.

Radius around each X where mutations are marked as SUSPECTED.

- ±5: Default (immediate context of ESM-2)
- ±3: More permissive
- ±7: More conservative

---

## 📈 Interpreting Results

### Aggression Score

- **>1200**: MAXIMUM ALERT - High immune evasion capacity
- **600-1200**: ACTIVE MONITORING - Variant of interest
- **<600**: OBSERVATION - Moderate mutations

### Lineage Classification

The percentage indicates what fraction of the lineage signatures are present:

- **>80%**: Strong match
- **60-80%**: Probable match (may be sublineage or recombinant)
- **<60%**: Weak match or unknown lineage

---

## 🚨 Troubleshooting

### Error: "Incorrect length: expected 1273"

**Cause**: The sequence is not aligned correctly.

**Solution**: Use `sequence_aligner.py` with Wuhan as a reference.

### Error: "Trace trap" on Mac

**Cause**: Operations not supported by MPS.

**Solution**: Already fixed in all scripts. If it persists, use `--cpu`.

---

## 📚 Scientific References

**Lineage Signatures**:

- WHO VOC/VOI Tracking (2024-2025)
- Nature Communications 2024: XBB/BA.2.86/JN.1
- mBio 2024: JN.1 → KP.2/KP.3
- Lancet Infect Dis 2024: JN.1 characterization

**Model ESM-2**:

-Lin et al. (2023): "Evolutionary-scale prediction of atomic-level protein structure"

- Science 379.6637 (2023): 1123-1130

---

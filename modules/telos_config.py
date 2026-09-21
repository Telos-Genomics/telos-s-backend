"""
Telos-S central configuration — single source of truth for paper constants.

Paper mapping:
  §2.2  CONTEXT_WINDOW = 5 (±5 exclusion around 'X')
  §2.3  SPIKE_LENGTH = 1273, Wuhan-Hu-1 NC_045512.2, gap open -10 / extend -0.5
  §2.4  C1 = 20, C2 = 10; zone weights RBM 3.0 / RBD 2.0 / Furin 1.5 / Other 1.0
  §2.5  PROPHET_TARGETS 452/484/501/681, alert threshold τ = 20% (heuristic §4.4)
  §3.1  Score tiers 600 / 1200 (heuristic §4.4, retrospective only)
  §4.5  Epi/SIM outputs are exploratory, excluded from paper results.

Do NOT tune these values to fit data without updating the paper (§4.4).
"""

from pathlib import Path

# --- Biological reference ---
SPIKE_LENGTH = 1273
WUHAN_ACCESSION = "NC_045512.2"
WUHAN_SPIKE_RANGE = (21563, 25384)  # nt positions in NC_045512.2
EXPECTED_SPIKE_LENGTH_AA = SPIKE_LENGTH
AMINO_ACID_TOLERANCE = 10
MIN_GENOME_LENGTH = 25000

# --- Alignment (§2.3) ---
GAP_OPEN_SCORE = -10
GAP_EXTEND_SCORE = -0.5
ALIGN_MATCH_SCORE = 2
ALIGN_MISMATCH_SCORE = -1

# --- QC (§2.2) ---
CONTEXT_WINDOW = 5
# Reliability labels are the on-disk data contract (CSV 'Reliability' column).
# Paper terms in parentheses: RELIABLE (Trusted MT), SUSPECT (Suspicious MS),
# INVALID (Invalid MI). Do NOT rename without migrating stored jobs.
RELIABLE = "RELIABLE"
SUSPECT = "SUSPECT"
INVALID = "INVALID"

# --- Aggression Score (§2.4.1, heuristic §4.4) ---
C1_ZONE = 20.0
C2_LLR = 10.0

# --- Zones (§2.4.2, Table 1) ---
RBM_RANGE = (437, 508)
RBD_RANGE = (319, 541)
FURIN_RANGE = (681, 685)
WEIGHT_RBM = 3.0
WEIGHT_RBD = 2.0
WEIGHT_FURIN = 1.5
WEIGHT_OTHER = 1.0

# --- Prophet (§2.5.2-2.5.3) ---
PROPHET_TARGETS = {
    "RBM_452": 452,
    "RBM_484": 484,
    "RBM_501": 501,
    "Furin_Cleavage_681": 681,
}
ALERT_THRESHOLD_PCT = 20.0
PROPHET_TOP_K = 5

# --- Score tiers (§3.1 Table 2, heuristic §4.4) ---
SCORE_MID = 600.0
SCORE_HIGH = 1200.0

# --- Imputation (NOT part of the paper; pre-processing only) ---
IMPUTATION_BLOCK_THRESHOLD = 5

# --- Model / runtime ---
ESM_DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
DEFAULT_BATCH_SIZE = 8
STEP_TIMEOUT_S = 1200  # 20 min per pipeline step

# --- Output layout (relative to backend root) ---
BACKEND_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BACKEND_ROOT / "output"
SPIKE_DIR = OUTPUT_DIR / "s" / "spike"
ALIGNED_DIR = OUTPUT_DIR / "s" / "spike_aligned"
REPORTS_DIR = OUTPUT_DIR / "s" / "reports"
PROPHET_DIR = OUTPUT_DIR / "prophet"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
JOBS_DIR = OUTPUT_DIR / "jobs"


def get_zone_weight(position: int) -> tuple[str, float]:
    """Zone label + weight per §2.4.2. Order matters: RBM inside RBD."""
    if RBM_RANGE[0] <= position <= RBM_RANGE[1]:
        return "CRITICAL (RBM - Direct Contact)", WEIGHT_RBM
    if RBD_RANGE[0] <= position <= RBD_RANGE[1]:
        return "HIGH (RBD - Binding Domain)", WEIGHT_RBD
    if FURIN_RANGE[0] <= position <= FURIN_RANGE[1]:
        return "MEDIUM (Furin Site)", WEIGHT_FURIN
    return "NORMAL (Structural Region)", WEIGHT_OTHER


def risk_tier(score: float) -> tuple[str, str]:
    """(verdict, level) per §3.1 tiers; cutoffs heuristic §4.4."""
    if score > SCORE_HIGH:
        return "🔴 MAXIMUM ALERT — Critical Alteration", "CRITICAL"
    if score > SCORE_MID:
        return "🟠 ACTIVE MONITORING — Elevated Risk", "HIGH"
    return "🟡 OBSERVATION — Baseline", "MODERATE"


def is_clean_context(wuhan_pos, x_positions, window: int = CONTEXT_WINDOW) -> bool:
    """
    §2.5.3 exclusion AFTER localization: True iff no Invalid ('X')
    at the target nor within ±window.
    """
    try:
        p = int(wuhan_pos)
    except (TypeError, ValueError):
        return False
    return all(abs(int(x) - p) > window for x in x_positions)

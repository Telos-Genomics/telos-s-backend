"""
Telos-S prospective evaluation harness — prediction registry (Fase A).

Purpose: turn "retrospectively consistent" into a prospectively testable
claim. Every prediction is recorded with a server-generated UTC timestamp
BEFORE its epidemiological outcome is observed, together with a hash of
the exact pipeline configuration that produced it. No outcome peeking.

Protocol (see paper §4.4, §5):
  1. Freeze the pipeline (config hash + code git SHA + ESM model id).
  2. Run analysis on a newly released isolate.
  3. Register the prediction via this module (timestamp is generated here,
     never accepted from the caller).
  4. Only afterwards annotate the outcome (separate step, not this module).

Registry format: JSONL, one object per line, append-only. The canonical
registry lives at <backend>/eval/prospective_registry.jsonl (created on
first registration). Field contract is pinned in
eval/registry_schema.json.

This module is stdlib-only and changes no existing pipeline behavior.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
try:
    import telos_config as cfg
except ImportError:  # package-style import (pytest / modules.*)
    from modules import telos_config as cfg

SCHEMA_VERSION = "1"
REGISTRY_FILENAME = "prospective_registry.jsonl"

# Config keys covered by the freeze hash. Anything that changes a score,
# a zone, a threshold or a Prophet forecast must be listed here.
FROZEN_CONFIG_KEYS = [
    "SPIKE_LENGTH",
    "CONTEXT_WINDOW",
    "C1_ZONE",
    "C2_LLR",
    "RBM_RANGE",
    "RBD_RANGE",
    "FURIN_RANGE",
    "WEIGHT_RBM",
    "WEIGHT_RBD",
    "WEIGHT_FURIN",
    "WEIGHT_OTHER",
    "GAP_OPEN_SCORE",
    "GAP_EXTEND_SCORE",
    "ALIGN_MATCH_SCORE",
    "ALIGN_MISMATCH_SCORE",
    "PROPHET_TARGETS",
    "ALERT_THRESHOLD_PCT",
    "PROPHET_TOP_K",
    "SCORE_MID",
    "SCORE_HIGH",
    "IMPUTATION_BLOCK_THRESHOLD",
    "ESM_DEFAULT_MODEL",
]

REQUIRED_FIELDS = [
    "accession",
    "job_id",
    "aggression_score",
    "sequence_quality_Qr",
    "reliable_mutations_n",
    "lineage",
    "prophet",
]


def default_registry_path() -> Path:
    """Canonical registry location (robust to cwd)."""
    return cfg.BACKEND_ROOT / "eval" / REGISTRY_FILENAME


def code_git_sha() -> str:
    """Short git SHA of the backend repo, or 'unknown' outside git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cfg.BACKEND_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "unknown"
    except Exception:
        return "unknown"


def freeze_config() -> dict:
    """Snapshot of every scoring/threshold constant. Order-independent."""
    frozen = {}
    for key in FROZEN_CONFIG_KEYS:
        value = getattr(cfg, key)
        # JSON-canonicalize tuples (ranges, targets stay readable).
        if isinstance(value, tuple):
            value = list(value)
        frozen[key] = value
    return frozen


def config_hash(frozen: dict | None = None) -> str:
    """SHA-256 over the canonical JSON of the frozen config."""
    canonical = json.dumps(
        frozen if frozen is not None else freeze_config(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_prophet_entry(item: dict) -> None:
    for field in ("wuhan_position", "top_aa", "pst_pct", "alert"):
        if field not in item:
            raise ValueError(f"prophet entry missing '{field}': {item!r}")
    pos = int(item["wuhan_position"])
    if pos not in cfg.PROPHET_TARGETS.values():
        raise ValueError(f"prophet position {pos} not in PROPHET_TARGETS")
    pst = float(item["pst_pct"])
    if not 0.0 <= pst <= 100.0:
        raise ValueError(f"pst_pct out of range [0, 100]: {pst}")


def validate_fields(fields: dict) -> dict:
    """Type/range-check caller-supplied fields. Returns a clean copy."""
    clean = dict(fields)
    for field in REQUIRED_FIELDS:
        if field not in clean or clean[field] is None:
            raise ValueError(f"missing required field '{field}'")
    if not isinstance(clean["accession"], str) or not clean["accession"].strip():
        raise ValueError("accession must be a non-empty string")
    float(clean["aggression_score"])
    qr = float(clean["sequence_quality_Qr"])
    if not 0.0 <= qr <= 100.0:
        raise ValueError(f"sequence_quality_Qr out of range [0, 100]: {qr}")
    int(clean["reliable_mutations_n"])
    if not isinstance(clean["prophet"], list):
        raise ValueError("prophet must be a list")
    for item in clean["prophet"]:
        _validate_prophet_entry(item)
    return clean


def register(fields: dict, registry_path: str | Path | None = None) -> dict:
    """
    Append a prediction to the registry. The timestamp, schema version,
    config snapshot hash, code SHA and model id are generated here —
    caller-supplied values for these keys are ignored/overwritten.

    Returns the stored entry.
    """
    clean = validate_fields(fields)
    frozen = freeze_config()
    entry = {
        "schema_version": SCHEMA_VERSION,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "accession": clean["accession"].strip(),
        "isolate_label": clean.get("isolate_label"),
        "job_id": clean["job_id"],
        "pipeline": "telos-s-backend",
        "code_git_sha": code_git_sha(),
        "config_hash": config_hash(frozen),
        "config_snapshot": frozen,
        "esm_model": os.environ.get("ESM_2_SIZE", cfg.ESM_DEFAULT_MODEL),
        "aggression_score": float(clean["aggression_score"]),
        "sequence_quality_Qr": float(clean["sequence_quality_Qr"]),
        "reliable_mutations_n": int(clean["reliable_mutations_n"]),
        "lineage": clean["lineage"],
        "lineage_confidence": clean.get("lineage_confidence"),
        "verdict": clean.get("verdict"),
        "prophet": clean["prophet"],
        "notes": clean.get("notes"),
    }

    path = Path(registry_path) if registry_path else default_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Warn (not fail) on duplicate accession: re-analysis happens, but the
    # first registration is the prospective one.
    for existing in list_entries(path):
        if existing.get("accession") == entry["accession"]:
            print(
                f"⚠️ Accession {entry['accession']} already registered at "
                f"{existing.get('registered_at')} — appending anyway."
            )
            break

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Registered {entry['accession']} at {entry['registered_at']}")
    print(f"   config_hash: {entry['config_hash'][:12]}…  registry: {path}")
    return entry


def list_entries(registry_path: str | Path | None = None) -> list[dict]:
    """Read all registry entries ([] if the registry does not exist yet)."""
    path = Path(registry_path) if registry_path else default_registry_path()
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}")
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Telos-S prospective prediction registry (append-only)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("freeze", help="Print the frozen config hash.")

    p_reg = sub.add_parser("register", help="Register a prediction (timestamp generated here).")
    p_reg.add_argument("--json", required=True, help="Prediction fields as a JSON object string.")
    p_reg.add_argument("--registry", default=None, help="Registry path (default: eval/prospective_registry.jsonl).")

    p_list = sub.add_parser("list", help="List registered predictions.")
    p_list.add_argument("--registry", default=None)

    args = parser.parse_args(argv)

    if args.command == "freeze":
        frozen = freeze_config()
        print(json.dumps(frozen, indent=2, sort_keys=True))
        print(f"config_hash: {config_hash(frozen)}")
        print(f"code_git_sha: {code_git_sha()}")
        return 0

    if args.command == "register":
        try:
            fields = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid --json: {e}")
            return 1
        try:
            entry = register(fields, args.registry)
        except ValueError as e:
            print(f"❌ Invalid prediction: {e}")
            return 1
        print(json.dumps({"accession": entry["accession"], "registered_at": entry["registered_at"]}))
        return 0

    if args.command == "list":
        for entry in list_entries(args.registry):
            print(
                f"{entry.get('registered_at')}  {entry.get('accession')}  "
                f"A={entry.get('aggression_score')}  {entry.get('lineage')}  "
                f"{entry.get('config_hash', '')[:12]}"
            )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

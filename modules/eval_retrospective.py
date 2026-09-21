"""
Telos-S retrospective set — inclusion criteria as executable code (Fase A).

The 4 paper isolates (n=4) are insufficient for fitting anything (§4.4):
any calibration needs a broader retrospective set first (target: 20-40
isolates with independently documented trajectories). This module does NOT
pick isolates for you — it enforces the acceptance rules for every row in
eval/retrospective_set.csv so the set stays honest as it grows.

Inclusion criteria (curator checklist, enforced below):
  1. Human clinical isolate with a full Spike retrievable from GenBank
     (accession with version, e.g. OK091006.1). No lab constructs.
  2. Pipeline must run end-to-end (aligned to 1273). Low Qr is recorded,
     not hidden — quality is a column of the analysis, not an excuse to
     drop inconvenient rows. Exclusion requires a reason in writing.
  3. Mix of lineage-dictionary coverage (yes/no) and evolutionary periods,
     mirroring the paper's two-stage rationale (§2.6).
  4. outcome_tier describes the OBSERVED trajectory, never the Score.
     Scoring an isolate and then labeling its outcome from the same score
     is circular — every non-pending tier needs external evidence cited
     (WHO report, peer-reviewed literature) in outcome_evidence.
  5. Anti-leakage: a retrospective accession must NEVER enter the
     prospective registry. summary() flags overlaps as leakage.

Outcome tiers (observed trajectory, cf. paper Table 2 vocabulary):
  baseline | elevated | critical | pending
`pending` means the trajectory is not yet established — allowed without
evidence, and the only honest value for freshly released isolates.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

REQUIRED_COLUMNS = [
    "accession",
    "isolate_label",
    "lineage",
    "origin",
    "collection_date",
    "release_date",
    "dictionary_coverage",
    "outcome_tier",
    "outcome_evidence",
    "outcome_annotated_at",
    "annotator",
    "include",
    "exclusion_reason",
    "notes",
]

VALID_TIERS = ("baseline", "elevated", "critical", "pending")
VALID_YESNO = ("yes", "no")

# Accession with version (OK091006.1) or versionless (OK091006).
ACCESSION_RE = re.compile(r"^[A-Z]{1,3}\d{5,8}(\.\d+)?$")
# ISO date prefixes: YYYY, YYYY-MM or YYYY-MM-DD.
DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def default_set_path() -> Path:
    """Canonical retrospective set location (robust to cwd)."""
    try:
        import telos_config as cfg
    except ImportError:
        from modules import telos_config as cfg
    return cfg.BACKEND_ROOT / "eval" / "retrospective_set.csv"


def load_rows(set_path: str | Path | None = None) -> list[dict]:
    """Read the CSV as a list of row dicts (all values stripped)."""
    path = Path(set_path) if set_path else default_set_path()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(
                f"unexpected header: {reader.fieldnames!r} "
                f"(expected {REQUIRED_COLUMNS!r})"
            )
        return [
            {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            for row in reader
        ]


def validate_row(row: dict, lineno: int) -> list[str]:
    """Acceptance rules for one row. Returns a list of error strings."""
    errors: list[str] = []
    tag = f"row {lineno} ({row.get('accession') or 'no-accession'})"

    if not ACCESSION_RE.match(row.get("accession") or ""):
        errors.append(f"{tag}: bad accession {row.get('accession')!r}")
    if not (row.get("lineage") or "").strip():
        errors.append(f"{tag}: lineage is required")
    if (row.get("dictionary_coverage") or "") not in VALID_YESNO:
        errors.append(
            f"{tag}: dictionary_coverage must be yes/no, "
            f"got {row.get('dictionary_coverage')!r}"
        )
    tier = row.get("outcome_tier") or ""
    if tier not in VALID_TIERS:
        errors.append(f"{tag}: outcome_tier must be one of {VALID_TIERS}, got {tier!r}")
    include = row.get("include") or ""
    if include not in VALID_YESNO:
        errors.append(f"{tag}: include must be yes/no, got {include!r}")

    # Circularity guard: a claimed trajectory needs external evidence.
    if include == "yes" and tier != "pending" and not (row.get("outcome_evidence") or "").strip():
        errors.append(
            f"{tag}: outcome_tier={tier!r} without outcome_evidence "
            f"(scores cannot justify their own outcome)"
        )
    # Exclusions must be justified in writing.
    if include == "no" and not (row.get("exclusion_reason") or "").strip():
        errors.append(f"{tag}: include=no without exclusion_reason")

    for col in ("collection_date", "release_date", "outcome_annotated_at"):
        value = (row.get(col) or "").strip()
        if value and not DATE_RE.match(value):
            errors.append(f"{tag}: bad {col} {value!r} (use YYYY[-MM[-DD]])")

    return errors


def validate(set_path: str | Path | None = None) -> list[str]:
    """Validate the whole set, including duplicate accessions."""
    rows = load_rows(set_path)
    errors: list[str] = []
    seen: dict[str, int] = {}
    for i, row in enumerate(rows, start=2):  # line 1 is the header
        errors.extend(validate_row(row, i))
        acc = row.get("accession") or ""
        if acc in seen:
            errors.append(f"row {i}: duplicate accession {acc!r} (first at row {seen[acc]})")
        else:
            seen[acc] = i
    return errors


def summary(set_path: str | Path | None = None, registry_path: str | Path | None = None) -> dict:
    """
    Counts by tier/coverage for included rows. If a prospective registry
    is given, accessions present in both are reported as leakage.
    """
    rows = [r for r in load_rows(set_path) if r.get("include") == "yes"]
    by_tier: dict[str, int] = {t: 0 for t in VALID_TIERS}
    uncovered_elevated_or_critical = 0
    for row in rows:
        by_tier[row["outcome_tier"]] += 1
        if row["dictionary_coverage"] == "no" and row["outcome_tier"] in ("elevated", "critical"):
            uncovered_elevated_or_critical += 1

    leakage: list[str] = []
    if registry_path:
        try:
            import eval_harness
        except ImportError:
            from modules import eval_harness
        registered = {e.get("accession") for e in eval_harness.list_entries(registry_path)}
        leakage = sorted({r["accession"] for r in rows} & registered)

    return {
        "n_included": len(rows),
        "by_tier": by_tier,
        "uncovered_elevated_or_critical": uncovered_elevated_or_critical,
        "pending_annotation": by_tier["pending"],
        "leakage": leakage,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Telos-S retrospective set validator.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Check every row against the acceptance rules.")
    p_val.add_argument("--set", default=None)

    p_sum = sub.add_parser("summary", help="Counts by tier/coverage + leakage check.")
    p_sum.add_argument("--set", default=None)
    p_sum.add_argument("--registry", default=None, help="Prospective registry to check for leakage.")

    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            errors = validate(args.set)
        except (ValueError, OSError) as e:
            print(f"❌ {e}")
            return 1
        if errors:
            for error in errors:
                print(f"❌ {error}")
            return 1
        print("✅ Retrospective set valid.")
        return 0

    if args.command == "summary":
        try:
            stats = summary(args.set, args.registry)
        except (ValueError, OSError) as e:
            print(f"❌ {e}")
            return 1
        print(f"Included isolates: {stats['n_included']}")
        for tier, count in stats["by_tier"].items():
            print(f"  {tier}: {count}")
        print(f"Uncovered elevated/critical (dictionary-blind signal): "
              f"{stats['uncovered_elevated_or_critical']}")
        print(f"Pending annotation: {stats['pending_annotation']}")
        if stats["leakage"]:
            print(f"⚠️ LEAKAGE — in both sets (must never happen): {stats['leakage']}")
            return 1
        print("✅ No leakage vs prospective registry.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

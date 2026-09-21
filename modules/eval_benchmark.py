"""
Telos-S benchmark protocol — same set, same outcomes, same metrics (Fase A).

Compares the Aggression Score against external structural/fitness methods
(CoVFit: supervised ESM-2 fitness estimate; EVEscape: immune-escape score;
see paper §4.1 for the methodological niche each occupies). Protocol rules:

  1. Same isolates, same outcomes. Only retrospective rows with annotated
     (non-pending) tiers count. `pending` rows never enter a statistic.
  2. Same metric for everyone: Spearman rank correlation between method
     score and outcome-tier rank (baseline=0 < elevated=1 < critical=2).
     No method gets a home-team metric.
  3. Scores are monotone-increasing in risk by convention: higher score =
     higher predicted risk. CoVFit's relative-R0 and EVEscape's escape
     score already point this way; if you import an inverted method,
     transform it BEFORE import and say so in notes.
  4. Provenance per score row: model/version + run timestamp. Telos rows
     additionally pin the frozen config hash (eval_harness.config_hash),
     so a benchmark always says WHICH Telos it beat or lost to.
  5. Honesty guardrails: `table` (per-isolate listing) always works;
     `compare` (statistics) refuses below MIN_SHARED_ISOLATES shared
     annotated isolates spanning MIN_TIERS tiers. Pairwise-complete
     observations per method pair, with n reported — never silently
     mixing different denominators.

This module runs no external model: CoVFit/EVEscape scores come from
their own pipelines (see their repos/docs) and enter here via
`import-scores` with model_version cited. What it guarantees is that
once scores exist, the comparison is symmetric and auditable.
"""

import argparse
import csv
import itertools
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
try:
    import telos_config as cfg
    import eval_retrospective as retro
    import eval_calibrate as calib
    import eval_harness as harness
except ImportError:
    from modules import telos_config as cfg
    from modules import eval_retrospective as retro
    from modules import eval_calibrate as calib
    from modules import eval_harness as harness

# Refusal thresholds for `compare` (statistics, not listings).
MIN_SHARED_ISOLATES = 4
MIN_TIERS = 2

SCORES_COLUMNS = ["accession", "method", "score", "model_version", "run_at", "notes"]

# Canonical method ids. Others are allowed (e.g. ablations) but these three
# are the paper's comparison frame (§4.1).
KNOWN_METHODS = ("telos", "covfit", "evescape")


def default_scores_path() -> Path:
    """Canonical method-scores location (robust to cwd)."""
    return cfg.BACKEND_ROOT / "eval" / "method_scores.csv"


def normalize_method(name: str) -> str:
    """Lowercase, stripped. Empty names rejected."""
    method = (name or "").strip().lower()
    if not method:
        raise ValueError("method must be a non-empty string")
    return method


def load_scores(scores_path: str | Path | None = None) -> list[dict]:
    """All score rows ([] if the file does not exist yet)."""
    path = Path(scores_path) if scores_path else default_scores_path()
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != SCORES_COLUMNS:
            raise ValueError(f"unexpected scores header: {reader.fieldnames!r}")
        return [dict(row) for row in reader]


def _append_rows(rows: list[dict], scores_path: str | Path | None = None) -> None:
    """Append rows, refusing (accession, method) duplicates (they corrupt metrics)."""
    existing = {(r["accession"], r["method"]) for r in load_scores(scores_path)}
    for row in rows:
        if (row["accession"], row["method"]) in existing:
            raise ValueError(
                f"duplicate score for ({row['accession']}, {row['method']}) — "
                f"delete the row to re-import, never double-count"
            )
        existing.add((row["accession"], row["method"]))
    path = Path(scores_path) if scores_path else default_scores_path()
    fresh = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORES_COLUMNS)
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


def import_telos(
    set_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    scores_path: str | Path | None = None,
    notes: str = "",
) -> int:
    """
    Score every cache-covered isolate with the CURRENT config and register
    the results as method=telos, pinning the frozen config hash in
    model_version. Only isolates present in the retrospective set are
    imported (stray cache rows stay out of the benchmark).
    """
    dataset = calib.build_dataset(set_path, cache_path)
    if not dataset:
        raise ValueError("no isolates with cached mutations (run `import-report` first)")
    params = calib.current_params()
    scores = calib.score_all(dataset, params)
    stamp = datetime.now(timezone.utc).isoformat()
    short_hash = harness.config_hash()[:12]
    rows = [{
        "accession": d["accession"],
        "method": "telos",
        "score": f"{s:.4f}",
        "model_version": f"telos:config-{short_hash}+{cfg.ESM_DEFAULT_MODEL}",
        "run_at": stamp,
        "notes": notes,
    } for d, s in zip(dataset, scores)]
    _append_rows(rows, scores_path)
    print(f"✅ Imported {len(rows)} telos scores (config-{short_hash}).")
    return len(rows)


def import_scores(
    csv_path: str | Path,
    scores_path: str | Path | None = None,
    notes: str = "",
) -> int:
    """
    Import EXTERNAL method scores (CoVFit/EVEscape runs from their own
    pipelines). Expected columns: accession, method, score, model_version.
    run_at is stamped here (import time), never trusted from the file.
    Unknown accessions import with a warning (only shared ones compare).
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for col in ("accession", "method", "score", "model_version"):
            if col not in (reader.fieldnames or []):
                raise ValueError(f"{csv_path} missing column {col!r}")
        incoming = list(reader)
    try:
        known = {r["accession"] for r in retro.load_rows()}
    except (ValueError, OSError):
        known = set()
    stamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for lineno, row in enumerate(incoming, start=2):
        acc = (row["accession"] or "").strip()
        if not acc:
            raise ValueError(f"{csv_path} line {lineno}: empty accession")
        if known and acc not in known:
            print(f"⚠️ {acc}: not in retrospective set — imports, but never compares.")
        rows.append({
            "accession": acc,
            "method": normalize_method(row["method"]),
            "score": str(float(row["score"])),
            "model_version": (row["model_version"] or "").strip() or "unknown",
            "run_at": stamp,
            "notes": notes or (row.get("notes") or "").strip(),
        })
    _append_rows(rows, scores_path)
    print(f"✅ Imported {len(rows)} external scores → {scores_path or default_scores_path()}.")
    return len(rows)


def comparison_frame(
    set_path: str | Path | None = None,
    scores_path: str | Path | None = None,
    methods: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Join annotated set rows with scores.
    Returns ([{accession, tier, rank, scores: {method: float}}], sorted_methods).
    Isolates/rows outside the annotated set are excluded (with a notice).
    """
    rows = [r for r in retro.load_rows(set_path) if r.get("include") == "yes"]
    annotated = {r["accession"]: r for r in rows if r.get("outcome_tier") in calib.TIER_RANK}
    wanted = {normalize_method(m) for m in methods} if methods else None
    frame: dict[str, dict] = {}
    skipped = 0
    for row in load_scores(scores_path):
        acc, method = row["accession"], normalize_method(row["method"])
        if wanted and method not in wanted:
            continue
        if acc not in annotated:
            skipped += 1
            continue
        frame.setdefault(acc, {"tier": annotated[acc]["outcome_tier"],
                               "rank": calib.TIER_RANK[annotated[acc]["outcome_tier"]],
                               "scores": {}})[ "scores"][method] = float(row["score"])
    if skipped:
        print(f"ℹ️ {skipped} score rows outside the annotated set — excluded.")
    methods_sorted = sorted({m for v in frame.values() for m in v["scores"]})
    return [{"accession": acc, **v} for acc, v in sorted(frame.items())], methods_sorted


def per_method_stats(frame: list[dict], methods: list[str]) -> dict[str, dict]:
    """Spearman per method on its own annotated coverage (n reported)."""
    stats = {}
    for method in methods:
        paired = [(v["scores"][method], v["rank"]) for v in frame if method in v["scores"]]
        tiers = {v["rank"] for v in frame if method in v["scores"]}
        if len(paired) < MIN_SHARED_ISOLATES or len(tiers) < MIN_TIERS:
            stats[method] = {"n": len(paired), "tiers": len(tiers),
                             "spearman": None, "insufficient": True}
            continue
        try:
            rho = calib.spearman([s for s, _ in paired], [t for _, t in paired])
        except ValueError:
            rho = None  # constant scores — undefined, not zero
        stats[method] = {"n": len(paired), "tiers": len(tiers),
                         "spearman": rho, "insufficient": False}
    return stats


def pairwise_deltas(frame: list[dict], methods: list[str]) -> list[dict]:
    """
    Head-to-head Spearman deltas on pairwise-complete isolates.
    Pairs below thresholds are reported as insufficient, never silently
    compared on mismatched denominators.
    """
    out = []
    for a, b in itertools.combinations(methods, 2):
        shared = [v for v in frame if a in v["scores"] and b in v["scores"]]
        tiers = {v["rank"] for v in shared}
        if len(shared) < MIN_SHARED_ISOLATES or len(tiers) < MIN_TIERS:
            out.append({"pair": (a, b), "n": len(shared), "insufficient": True})
            continue
        try:
            rho_a = calib.spearman([v["scores"][a] for v in shared], [v["rank"] for v in shared])
        except ValueError:
            rho_a = None
        try:
            rho_b = calib.spearman([v["scores"][b] for v in shared], [v["rank"] for v in shared])
        except ValueError:
            rho_b = None
        delta = None if rho_a is None or rho_b is None else rho_a - rho_b
        out.append({"pair": (a, b), "n": len(shared), "rho_a": rho_a, "rho_b": rho_b,
                    "delta": delta, "insufficient": False})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Telos-S benchmark protocol (Fase A).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-scores", help="Create the scores file with header only.")
    p_init.add_argument("--out", default=None)

    p_telos = sub.add_parser("import-telos", help="Score cache-covered isolates with current config.")
    p_telos.add_argument("--set", default=None)
    p_telos.add_argument("--cache", default=None)
    p_telos.add_argument("--scores", default=None)
    p_telos.add_argument("--notes", default="")

    p_imp = sub.add_parser("import-scores", help="Import EXTERNAL method scores CSV.")
    p_imp.add_argument("--csv", required=True, help="Columns: accession,method,score,model_version.")
    p_imp.add_argument("--scores", default=None)
    p_imp.add_argument("--notes", default="")

    p_tab = sub.add_parser("table", help="Per-isolate score listing (descriptive, always allowed).")
    p_tab.add_argument("--set", default=None)
    p_tab.add_argument("--scores", default=None)
    p_tab.add_argument("--methods", nargs="*", default=None)

    p_cmp = sub.add_parser("compare", help="Spearman per method + pairwise deltas (guardrailed).")
    p_cmp.add_argument("--set", default=None)
    p_cmp.add_argument("--scores", default=None)
    p_cmp.add_argument("--methods", nargs="*", default=None)

    args = parser.parse_args(argv)

    if args.command == "init-scores":
        path = Path(args.out) if args.out else default_scores_path()
        if path.exists() and path.stat().st_size > 0:
            print(f"ℹ️ Scores file already exists: {path}")
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SCORES_COLUMNS).writeheader()
        print(f"✅ Scores file initialized: {path} (import telos + external runs next)")
        return 0

    if args.command == "import-telos":
        try:
            import_telos(args.set, args.cache, args.scores, args.notes)
        except (ValueError, OSError) as e:
            print(f"❌ {e}")
            return 1
        return 0

    if args.command == "import-scores":
        try:
            import_scores(args.csv, args.scores, args.notes)
        except (ValueError, OSError) as e:
            print(f"❌ {e}")
            return 1
        return 0

    if args.command == "table":
        try:
            frame, methods = comparison_frame(args.set, args.scores, args.methods)
        except (ValueError, OSError) as e:
            print(f"❌ {e}")
            return 1
        if not frame:
            print("ℹ️ No shared annotated isolates with scores yet.")
            return 0
        header = f"{'accession':<14}{'tier':<10}" + "".join(f"{m:>12}" for m in methods)
        print(header)
        for v in frame:
            line = f"{v['accession']:<14}{v['tier']:<10}"
            line += "".join(f"{v['scores'].get(m, float('nan')):>12.2f}" for m in methods)
            print(line)
        return 0

    if args.command == "compare":
        try:
            frame, methods = comparison_frame(args.set, args.scores, args.methods)
        except (ValueError, OSError) as e:
            print(f"❌ {e}")
            return 1
        if len(methods) < 2:
            print(f"❌ Need ≥2 methods with scores (have: {methods or 'none'}). "
                  f"Import telos + at least one external run first.")
            return 1
        stats = per_method_stats(frame, methods)
        print(f"{'method':<12}{'n':>4}{'tiers':>7}{'spearman':>10}")
        blocked = 0
        for method in methods:
            s = stats[method]
            rho = "insufficient" if s["insufficient"] else (
                "undefined" if s["spearman"] is None else f"{s['spearman']:.3f}")
            if s["insufficient"]:
                blocked += 1
            print(f"{method:<12}{s['n']:>4}{s['tiers']:>7}{rho:>10}")
        print("pairwise deltas (shared isolates, Δ = ρ_row − ρ_col):")
        for d in pairwise_deltas(frame, methods):
            a, b = d["pair"]
            if d["insufficient"]:
                print(f"  {a} vs {b}: insufficient (n={d['n']})")
                blocked += 1
                continue
            da = "undef" if d["rho_a"] is None else f"{d['rho_a']:.3f}"
            db = "undef" if d["rho_b"] is None else f"{d['rho_b']:.3f}"
            dd = "undef" if d["delta"] is None else f"{d['delta']:+.3f}"
            print(f"  {a} vs {b}: ρ={da}/{db} Δ={dd} (n={d['n']})")
        if blocked:
            print(f"❌ {blocked} statistic(s) below thresholds "
                  f"(≥{MIN_SHARED_ISOLATES} shared isolates, ≥{MIN_TIERS} tiers).")
            return 1
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

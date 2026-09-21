"""
Telos-S calibration scaffold — heuristic constants under test (Fase A).

Problem (§4.4): c1/c2, zone weights and τ were set heuristically, never
fitted. This module provides the fitting machinery WITHOUT pretending the
current data suffices: `search` refuses to run until the retrospective set
holds MIN_ANNOTATED_ISOLATES annotated outcomes spanning MIN_TIERS tiers.

Key design points (read before touching):
  1. No ESM-2 re-runs. Aggression is Σ|c1·w + c2·ζ| over Trusted mutations
     (§2.4.1), so any (c1, c2, weights) scores recompute exactly from cached
     per-mutation (zone, LLR) rows. A full grid costs milliseconds, not GPU
     hours. Cache lives at eval/per_mutation_cache.csv (built from REAL
     pipeline reports via `import-report`, never hand-written).
  2. Biology constrains the fit. Zone weights must preserve the paper's
     ordering RBM ≥ RBD ≥ Furin ≥ Other (§2.4.2); grid points violating it
     are discarded, not fitted. w_other stays anchored at 1.0 (baseline).
  3. τ is NOT optimizable here. Alert-threshold ROC needs position-level
     truth labels, which do not exist. `tau-scan` is descriptive only
     (alert counts per τ on current data) and says so in its output.
  4. Objective is rank-based (Spearman score↔tier), never absolute-score
     matching: with retrospective n in the dozens, fitting magnitudes
     would be pure overfit. LOO reports parameter STABILITY across folds
     (which params keep winning), not fake precision estimates.

Outcome ranks: baseline=0, elevated=1, critical=2. `pending` rows excluded.
"""

import argparse
import csv
import itertools
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
try:
    import telos_config as cfg
    import eval_retrospective as retro
except ImportError:
    from modules import telos_config as cfg
    from modules import eval_retrospective as retro

# Refusal thresholds: below these, `search` exits non-zero with guidance.
MIN_ANNOTATED_ISOLATES = 4
MIN_TIERS = 2

TIER_RANK = {"baseline": 0, "elevated": 1, "critical": 2}

CACHE_COLUMNS = ["accession", "mutation", "pos", "zone", "llr", "reliability"]
VALID_ZONES = ("RBM", "RBD", "FURIN", "OTHER")

DEFAULT_GRID = {
    "c1": [10.0, 20.0, 30.0],
    "c2": [5.0, 10.0, 15.0],
    "w_rbm": [2.0, 3.0, 4.0],
    "w_rbd": [1.5, 2.0, 2.5],
    "w_furin": [1.0, 1.5, 2.0],
    "tau": [10.0, 15.0, 20.0, 25.0, 30.0],
}


def default_cache_path() -> Path:
    """Canonical per-mutation cache location (robust to cwd)."""
    return cfg.BACKEND_ROOT / "eval" / "per_mutation_cache.csv"


def recompute_aggression(mutations: list[tuple[float, float]], c1: float, c2: float) -> float:
    """
    Σ|c1·w + c2·ζ| over Trusted mutations (§2.4.1, abs on the combined term).
    mutations: [(zone_weight, signed_llr), ...]. Callers pass Trusted only.
    """
    return sum(abs(c1 * w + c2 * zeta) for w, zeta in mutations)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks for ties (1-based)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (stdlib; n>=2, non-constant inputs)."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("spearman needs ≥2 paired values")
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        raise ValueError("spearman undefined for constant input")
    return cov / math.sqrt(vx * vy)


def respects_ordering(w_rbm: float, w_rbd: float, w_furin: float, w_other: float = 1.0) -> bool:
    """Paper ordering RBM ≥ RBD ≥ Furin ≥ Other (§2.4.2)."""
    return w_rbm >= w_rbd >= w_furin >= w_other


def current_params() -> dict:
    """Live pipeline constants — the baseline every grid point competes with."""
    return {
        "c1": cfg.C1_ZONE,
        "c2": cfg.C2_LLR,
        "w_rbm": cfg.WEIGHT_RBM,
        "w_rbd": cfg.WEIGHT_RBD,
        "w_furin": cfg.WEIGHT_FURIN,
        "w_other": cfg.WEIGHT_OTHER,
        "tau": cfg.ALERT_THRESHOLD_PCT,
    }


def load_cache(cache_path: str | Path | None = None) -> dict[str, list[tuple[str, float, str]]]:
    """
    Per-mutation cache → {accession: [(zone_label, llr, reliability), ...]}.
    Zone weights are NOT frozen here: cache stores zone labels, weights
    re-resolve per grid point via weights_map().
    """
    path = Path(cache_path) if cache_path else default_cache_path()
    if not path.exists():
        raise ValueError(f"empty cache: {path} does not exist (see `init-cache`)")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CACHE_COLUMNS:
            raise ValueError(f"unexpected cache header: {reader.fieldnames!r}")
        out: dict[str, list[tuple[str, float, str]]] = {}
        for lineno, row in enumerate(reader, start=2):
            try:
                zone = (row["zone"] or "").strip()
                if zone not in VALID_ZONES:
                    raise ValueError(f"bad zone {zone!r}")
                out.setdefault(row["accession"].strip(), []).append(
                    (zone, float(row["llr"]), (row["reliability"] or "").strip())
                )
            except (ValueError, KeyError) as e:
                raise ValueError(f"cache line {lineno}: {e}")
    if not out:
        raise ValueError(f"empty cache: {path} has no rows")
    return out


def weights_map(params: dict) -> dict[str, float]:
    """Grid params → zone-label weight lookup."""
    return {
        "RBM": params["w_rbm"],
        "RBD": params["w_rbd"],
        "FURIN": params["w_furin"],
        "OTHER": params.get("w_other", 1.0),
    }


def build_dataset(
    set_path: str | Path | None = None,
    cache_path: str | Path | None = None,
) -> list[dict]:
    """
    Join annotated retrospective rows with cached Trusted mutations.
    Returns [{accession, tier, rank, mutations: [(weight_label, llr), ...]}].
    Annotated isolates missing from the cache are reported and skipped
    (their pipeline reports were never imported — run `import-report`).
    """
    rows = [r for r in retro.load_rows(set_path) if r.get("include") == "yes"]
    annotated = [r for r in rows if r.get("outcome_tier") in TIER_RANK]
    cache = load_cache(cache_path)
    dataset = []
    for row in annotated:
        acc = row["accession"]
        if acc not in cache:
            print(f"⚠️ {acc}: annotated but no cached mutations — skipping (import its report).")
            continue
        trusted = [(zone, llr) for zone, llr, rel in cache[acc] if rel == cfg.RELIABLE]
        dataset.append({
            "accession": acc,
            "tier": row["outcome_tier"],
            "rank": TIER_RANK[row["outcome_tier"]],
            "mutations": trusted,
        })
    return dataset


def check_sufficient(dataset: list[dict]) -> None:
    """Refuse to optimize on anecdotal data. Raises ValueError if unfit."""
    tiers = {d["tier"] for d in dataset}
    if len(dataset) < MIN_ANNOTATED_ISOLATES or len(tiers) < MIN_TIERS:
        raise ValueError(
            f"insufficient annotated outcomes: have {len(dataset)} isolates / "
            f"{len(tiers)} tiers, need ≥{MIN_ANNOTATED_ISOLATES} isolates spanning "
            f"≥{MIN_TIERS} tiers. Annotate more of eval/retrospective_set.csv first."
        )


def score_all(dataset: list[dict], params: dict) -> list[float]:
    """Aggression per isolate under a candidate parameter set."""
    wm = weights_map(params)
    return [recompute_aggression([(wm[z], llr) for z, llr in d["mutations"]], params["c1"], params["c2"])
            for d in dataset]


def grid_points(grid: dict | None = None):
    """Yield ordering-respecting (c1, c2, weights) combos. τ handled separately."""
    grid = grid or DEFAULT_GRID
    for c1, c2, w_rbm, w_rbd, w_furin in itertools.product(
        grid["c1"], grid["c2"], grid["w_rbm"], grid["w_rbd"], grid["w_furin"]
    ):
        if respects_ordering(w_rbm, w_rbd, w_furin):
            yield {"c1": c1, "c2": c2, "w_rbm": w_rbm, "w_rbd": w_rbd, "w_furin": w_furin}


def search(dataset: list[dict], grid: dict | None = None) -> dict:
    """
    Best grid params by Spearman on full data + LOO stability: for each
    held-out isolate, which params win on the rest. With small n the LOO
    table is a STABILITY readout, not a performance estimate.
    """
    check_sufficient(dataset)
    ranks = [d["rank"] for d in dataset]
    results = []
    for params in grid_points(grid):
        try:
            rho = spearman(score_all(dataset, params), ranks)
        except ValueError:
            continue  # degenerate grid point (constant scores) — skip
        results.append((rho, params))
    if not results:
        raise ValueError("no valid grid point (all scored constant — check cache LLRs)")
    results.sort(key=lambda r: r[0], reverse=True)

    # LOO stability: winner params per held-out isolate.
    stability: dict[str, int] = {}
    for held in range(len(dataset)):
        rest = dataset[:held] + dataset[held + 1:]
        rest_ranks = [d["rank"] for d in rest]
        best_key, best_rho = None, float("-inf")
        for rho_full, params in results:
            try:
                rho = spearman(score_all(rest, params), rest_ranks)
            except ValueError:
                continue
            if rho > best_rho:
                best_rho, best_key = rho, param_key(params)
        if best_key:
            stability[best_key] = stability.get(best_key, 0) + 1

    best_rho, best_params = results[0]
    return {
        "n": len(dataset),
        "best_rho": best_rho,
        "best_params": best_params,
        "current_rho": _current_rho(dataset),
        "loo_stability": stability,
        "loo_folds": len(dataset),
    }


def _current_rho(dataset: list[dict]) -> float | None:
    try:
        return spearman(score_all(dataset, current_params()), [d["rank"] for d in dataset])
    except ValueError:
        return None


def param_key(params: dict) -> str:
    """Stable string id for a parameter set (for stability tables)."""
    return (f"c1={params['c1']},c2={params['c2']},"
            f"rbm={params['w_rbm']},rbd={params['w_rbd']},furin={params['w_furin']}")


def context_to_zone(context: str) -> str:
    """Report Context label → canonical zone (inverse of get_zone_weight)."""
    c = (context or "").upper()
    if "RBM" in c:
        return "RBM"
    if "RBD" in c:
        return "RBD"
    if "FURIN" in c:
        return "FURIN"
    return "OTHER"


def import_report(report_csv: str | Path, accession: str, cache_path: str | Path | None = None) -> int:
    """
    Append per-mutation rows from a REAL pipeline report CSV into the cache.
    Refuses duplicates for the same accession (re-import would double-count).
    Returns the number of rows appended.
    """
    cache_p = Path(cache_path) if cache_path else default_cache_path()
    existing: set[str] = set()
    if cache_p.exists():
        with open(cache_p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("accession") == accession:
                    existing.add(row.get("mutation", ""))
    with open(report_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for col in ("Mutation", "Context", "LLR", "Reliability"):
            if col not in (reader.fieldnames or []):
                raise ValueError(f"report {report_csv} missing column {col!r}")
        new_rows = []
        for row in reader:
            mut = (row["Mutation"] or "").strip()
            if not mut or mut in existing:
                continue
            m = __import__("re").search(r"(\d+)", mut)
            new_rows.append({
                "accession": accession,
                "mutation": mut,
                "pos": m.group(1) if m else "",
                "zone": context_to_zone(row["Context"]),
                "llr": str(float(row["LLR"])),
                "reliability": (row["Reliability"] or "").strip(),
            })
    if not new_rows:
        print(f"ℹ️ No new mutations for {accession} (already imported or empty report).")
        return 0
    fresh = not cache_p.exists() or cache_p.stat().st_size == 0
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_COLUMNS)
        if fresh:
            writer.writeheader()
        writer.writerows(new_rows)
    print(f"✅ Imported {len(new_rows)} mutations for {accession} → {cache_p}")
    return len(new_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Telos-S calibration scaffold (Fase A).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-cache", help="Create the cache file with header only.")
    p_init.add_argument("--out", default=None)

    p_imp = sub.add_parser("import-report", help="Import a REAL pipeline report CSV into the cache.")
    p_imp.add_argument("--report", required=True)
    p_imp.add_argument("--accession", required=True)
    p_imp.add_argument("--cache", default=None)

    p_eval = sub.add_parser("evaluate", help="Score/outcome table + Spearman under CURRENT config.")
    p_eval.add_argument("--set", default=None)
    p_eval.add_argument("--cache", default=None)

    p_search = sub.add_parser("search", help="Grid search (refuses without sufficient outcomes).")
    p_search.add_argument("--set", default=None)
    p_search.add_argument("--cache", default=None)

    p_tau = sub.add_parser(
        "tau-scan",
        help="DESCRIPTIVE alert counts per τ on cached prophet data — not an optimization.",
    )
    p_tau.add_argument("--cache", default=None, help="(reserved: needs prophet cache — not yet built)")

    args = parser.parse_args(argv)

    if args.command == "init-cache":
        path = Path(args.out) if args.out else default_cache_path()
        if path.exists() and path.stat().st_size > 0:
            print(f"ℹ️ Cache already exists: {path}")
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CACHE_COLUMNS).writeheader()
        print(f"✅ Cache initialized: {path} (import real reports next)")
        return 0

    if args.command == "import-report":
        try:
            import_report(args.report, args.accession, args.cache)
        except (ValueError, OSError) as e:
            print(f"❌ {e}")
            return 1
        return 0

    if args.command == "evaluate":
        try:
            dataset = build_dataset(args.set, args.cache)
        except (ValueError, OSError) as e:
            print(f"❌ {e}")
            return 1
        if not dataset:
            print("ℹ️ No annotated isolates with cached mutations yet.")
            return 0
        params = current_params()
        scores = score_all(dataset, params)
        print(f"{'accession':<14}{'tier':<10}{'score':>10}")
        for d, s in sorted(zip(dataset, scores), key=lambda t: t[1]):
            print(f"{d['accession']:<14}{d['tier']:<10}{s:>10.1f}")
        try:
            print(f"Spearman (current config): {spearman(scores, [d['rank'] for d in dataset]):.3f} "
                  f"(n={len(dataset)} — anecdotal below thresholds)")
        except ValueError as e:
            print(f"Spearman undefined: {e}")
        return 0

    if args.command == "search":
        try:
            dataset = build_dataset(args.set, args.cache)
            result = search(dataset)
        except (ValueError, OSError) as e:
            print(f"❌ {e}")
            return 1
        print(f"Best Spearman: {result['best_rho']:.3f} (n={result['n']})")
        print(f"Best params:   {param_key(result['best_params'])}")
        if result["current_rho"] is not None:
            print(f"Current config:{result['current_rho']:.3f} ({param_key(current_params())})")
        print(f"LOO stability over {result['loo_folds']} folds (wins per params):")
        for key, wins in sorted(result["loo_stability"].items(), key=lambda t: -t[1]):
            print(f"  {wins}x  {key}")
        if result["n"] < 10:
            print("⚠️ n<10: treat best params as exploratory, not fitted (wide CIs).")
        return 0

    if args.command == "tau-scan":
        print("ℹ️ tau-scan is descriptive-only: τ ROC needs position-level truth labels,")
        print("   which do not exist. It will count alerts per τ once a prophet cache")
        print("   exists. Not built yet — τ stays heuristic (§2.5.2, §4.4).")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

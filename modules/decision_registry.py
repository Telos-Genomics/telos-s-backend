"""
Telos-S decision seal registry (W3 prospective watchlist).

Purpose: seal each exposure-monitoring recommendation with a server-generated
UTC timestamp BEFORE its outcome is observed, together with hashes of the
exact evidence and configuration that produced it. Cherry-picking is
impossible by construction: the timestamp, config hash and chain hash are
generated here, never accepted from the caller.

Design notes:
  - Reuses the freeze/config-hash machinery from eval_harness.py (same
    heuristic constants from paper §4.4, same code SHA). No duplicated logic.
  - Storage is JSONL under the shared output volume
    (<backend>/output/registry/decision_seals.jsonl) so the read-only MCP
    server can verify seals without write access.
  - Tamper-evidence via prev_hash chain: every seal embeds the SHA-256 of
    the previous line. Verification recomputes the chain.
  - Sealing is explicit: the decide tool only recommends, a human or the
    pipeline seals. This module never calls a model.

This module is stdlib-only and changes no existing pipeline behavior.
"""

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
try:
    import telos_config as cfg
    from eval_harness import config_hash, freeze_config, code_git_sha
except ImportError:  # package-style import (pytest / modules.*)
    from modules import telos_config as cfg
    from modules.eval_harness import config_hash, freeze_config, code_git_sha

SCHEMA_VERSION = "1"
REGISTRY_DIRNAME = "registry"
REGISTRY_FILENAME = "decision_seals.jsonl"

ALLOWED_ACTIONS = ("no_action", "monitor", "alert", "human_review")


def default_registry_path() -> Path:
    """Canonical registry location inside the shared output volume."""
    return cfg.BACKEND_ROOT / "output" / REGISTRY_DIRNAME / REGISTRY_FILENAME


def _canonical(obj: dict) -> str:
    """Order-independent canonical JSON for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def list_entries(registry_path: str | Path | None = None) -> list:
    """Read all seals in order. Returns [] when the registry is empty."""
    path = Path(registry_path) if registry_path else default_registry_path()
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def seal(
    job_id: str,
    recommended_action: str,
    evidence: dict,
    nimble: dict | None = None,
    prospect: str | None = None,
    exposure_context: str | None = None,
    notes: str | None = None,
    registry_path: str | Path | None = None,
) -> dict:
    """
    Append a decision seal. Timestamp, config hash, input hash and chain
    hash are generated here — caller-supplied values for these keys are
    ignored. Returns the stored entry.
    """
    if not job_id or not str(job_id).strip():
        raise ValueError("job_id must be a non-empty string")
    if recommended_action not in ALLOWED_ACTIONS:
        raise ValueError(f"recommended_action must be one of {ALLOWED_ACTIONS}")
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be a dict")

    frozen = freeze_config()
    entry_core = {
        "job_id": str(job_id).strip(),
        "recommended_action": recommended_action,
        "evidence": evidence,
        "nimble": nimble or {},
        "prospect": prospect,
        "exposure_context": exposure_context,
        "notes": notes,
    }
    previous = list_entries(registry_path)
    prev_hash = sha256_hex(_canonical(previous[-1])) if previous else "GENESIS"
    entry = {
        "schema_version": SCHEMA_VERSION,
        "seal_id": f"seal_{uuid.uuid4().hex[:12]}",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "telos-s-backend",
        "code_git_sha": code_git_sha(),
        "config_hash": config_hash(frozen),
        "input_hash": sha256_hex(_canonical(entry_core)),
        "prev_hash": prev_hash,
        **entry_core,
    }

    path = Path(registry_path) if registry_path else default_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def find_by_job(job_id: str, registry_path: str | Path | None = None) -> list:
    """All seals for a job_id, oldest first."""
    return [e for e in list_entries(registry_path) if e.get("job_id") == job_id]


def verify(registry_path: str | Path | None = None) -> dict:
    """
    Recompute input hashes and the prev_hash chain over the whole registry.
    Returns {"ok": bool, "seals": n, "failures": [...]}.
    """
    entries = list_entries(registry_path)
    failures = []
    prev = "GENESIS"
    for i, entry in enumerate(entries):
        core = {k: entry.get(k) for k in (
            "job_id", "recommended_action", "evidence",
            "nimble", "prospect", "exposure_context", "notes",
        )}
        if sha256_hex(_canonical(core)) != entry.get("input_hash"):
            failures.append({"index": i, "seal_id": entry.get("seal_id"),
                             "reason": "input_hash mismatch"})
        if entry.get("prev_hash") != prev:
            failures.append({"index": i, "seal_id": entry.get("seal_id"),
                             "reason": "prev_hash chain break"})
        prev = sha256_hex(_canonical(entry))
    return {"ok": not failures, "seals": len(entries), "failures": failures}

"""Where things live in the work directory.

Data products are stored in folders whose names carry a short hash of the
settings that produced them (``ingest-3fa2c1d0``).  Changing a setting that
affects the result therefore starts a fresh folder instead of silently mixing
old and new outputs; the old folder stays available for comparison.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .config import experiment_config, get_path
from .state import ensure_dir, stable_hash

INGEST_VERSION = 2
FEATURIZE_VERSION = 1


def work(cfg) -> Path:
    return ensure_dir(get_path(cfg, "paths.work_dir"))


def reports(cfg, *sub) -> Path:
    return ensure_dir(work(cfg).joinpath("reports", *sub))


def state_dir(cfg) -> Path:
    return ensure_dir(work(cfg) / "state")


def inventory_path(cfg) -> Path:
    return reports(cfg) / "inventory.json"


def ingest_spec(cfg) -> Dict[str, Any]:
    ing = cfg.get("ingest", {})
    return {
        "v": INGEST_VERSION,
        "label_column": ing.get("label_column", "auto"),
        "drop_columns": sorted(ing.get("drop_columns", [])),
        "context": bool(ing.get("context_features", True)),
        "ctx_short_s": float(ing.get("ctx_short_s", 1.0)),
        "ctx_long_s": float(ing.get("ctx_long_s", 10.0)),
        "max_carry_rows": int(ing.get("max_carry_rows", 2_000_000)),
        "max_gap_s": float(ing.get("max_gap_s", 300.0)),
        "backstep_s": float(ing.get("segment_backstep_s", 1.0)),
    }


def ingest_dir(cfg) -> Path:
    return ensure_dir(work(cfg) / "data" / f"ingest-{stable_hash(ingest_spec(cfg))}")


def feature_list(cfg, fs: str):
    sets = get_path(cfg, "features.sets", {}) or {}
    if fs not in sets:
        raise KeyError(f"Unknown feature set '{fs}'. Known: {sorted(sets)}")
    return list(sets[fs])


def compact_spec(cfg, fs: str) -> Dict[str, Any]:
    comp = cfg.get("compact", {})
    dedupe = fs in (comp.get("dedupe_sets") or [])
    return {
        "v": FEATURIZE_VERSION,
        "ingest": stable_hash(ingest_spec(cfg)),
        "fs": fs,
        "features": feature_list(cfg, fs),
        "split_key": feature_list(cfg, get_path(cfg, "features.split_key_set", "header")),
        "split": {k: get_path(cfg, f"split.{k}") for k in ("fractions", "seed")},
        "dedupe": dedupe,
        "buckets": int(comp.get("buckets", 64)) if dedupe else None,
    }


def compact_dir(cfg, fs: str) -> Path:
    spec = compact_spec(cfg, fs)
    return ensure_dir(work(cfg) / "data" / f"compact-{fs}-{stable_hash(spec)}")


def experiment_spec(cfg, name: str) -> Dict[str, Any]:
    exp = experiment_config(cfg, name)
    exp_clean = {k: v for k, v in exp.items() if k not in ("enabled", "notes", "name")}
    fs = exp.get("feature_set", "header")
    return {
        "exp": exp_clean,
        "compact": stable_hash(compact_spec(cfg, fs)),
        "labels": cfg.get("labels", {}),
        "split_mode": get_path(cfg, "split.mode", "vector"),
        "device_holdout": get_path(cfg, "split.device_holdout", {}),
    }


def model_dir(cfg, name: str) -> Path:
    return ensure_dir(work(cfg) / "models" / f"{name}-{stable_hash(experiment_spec(cfg, name))}")


def exp_reports(cfg, name: str) -> Path:
    return ensure_dir(model_dir(cfg, name) / "reports")

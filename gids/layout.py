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


def dataset_spec(cfg) -> Dict[str, Any]:
    """Settings that shape the sampled training/validation tables shared by all experiments."""
    from .prepare import prepare_spec

    ds = cfg.get("dataset", {})
    return {"prepare": stable_hash(prepare_spec(cfg)), "max_rows_per_class": int(ds.get("max_rows_per_class", 2_000_000)),
            "val_max_rows_per_class": int(ds.get("val_max_rows_per_class", 1_000_000)),
            "merge_cross_file": bool(ds.get("merge_cross_file", True)), "seed": int(ds.get("seed", 0))}


def dataset_dir(cfg) -> Path:
    return ensure_dir(work(cfg) / "data" / f"dataset-{stable_hash(dataset_spec(cfg))}")


def experiment_spec(cfg, name: str) -> Dict[str, Any]:
    exp = experiment_config(cfg, name)
    exp_clean = {k: v for k, v in exp.items() if k not in ("enabled", "notes", "name")}
    model = exp_clean.get("model", "xgboost")
    exp_clean = {k: v for k, v in exp_clean.items() if not isinstance(v, dict) or k == model}
    return {"exp": exp_clean, "dataset": stable_hash(dataset_spec(cfg)),
            "features": get_path(cfg, f"features.sets.{exp.get('feature_set', 'full')}")}


def model_dir(cfg, name: str) -> Path:
    return ensure_dir(work(cfg) / "models" / f"{name}-{stable_hash(experiment_spec(cfg, name))}")

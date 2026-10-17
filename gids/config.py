"""Layered configuration.

Order (later wins):
1. ``configs/default.yaml``      (in the repo)
2. ``configs/experiments.yaml``  (in the repo)
3. ``<work_dir>/config.override.yaml`` (on Drive, survives new sessions)
4. environment variables GIDS_WORK_DIR / GIDS_ZIP_PATH / GIDS_DATA_DIR
5. ``--set a.b.c=value`` arguments
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def set_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_path(cfg: Dict[str, Any], dotted: str, default=None):
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(work_dir: Optional[str] = None, overrides: Iterable[str] = (),
                extra_files: Iterable[str] = ()) -> Dict[str, Any]:
    cfg = _load_yaml(CONFIG_DIR / "default.yaml")
    deep_merge(cfg, _load_yaml(CONFIG_DIR / "experiments.yaml"))
    for f in extra_files:
        deep_merge(cfg, _load_yaml(Path(f)))

    env_map = {"GIDS_WORK_DIR": "paths.work_dir", "GIDS_ZIP_PATH": "paths.zip_path",
               "GIDS_DATA_DIR": "paths.data_dir"}
    for env, key in env_map.items():
        if os.environ.get(env):
            set_path(cfg, key, os.environ[env])
    if work_dir:
        set_path(cfg, "paths.work_dir", work_dir)

    wd = Path(get_path(cfg, "paths.work_dir"))
    override_file = wd / "config.override.yaml"
    if override_file.exists():
        deep_merge(cfg, _load_yaml(override_file))
        cfg.setdefault("_meta", {})["override_file"] = str(override_file)
        # an explicit --work-dir or env var still wins over the override file
        for env, key in env_map.items():
            if os.environ.get(env):
                set_path(cfg, key, os.environ[env])
        if work_dir:
            set_path(cfg, "paths.work_dir", work_dir)

    applied: List[str] = []
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got '{item}'")
        key, _, raw = item.partition("=")
        set_path(cfg, key.strip(), yaml.safe_load(raw))
        applied.append(item)
    cfg.setdefault("_meta", {})["cli_overrides"] = applied
    return cfg


def experiment_names(cfg: Dict[str, Any]) -> List[str]:
    exps = cfg.get("experiments") or {}
    return [k for k, v in exps.items() if isinstance(v, dict) and v.get("enabled", True)]


def experiment_config(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    exps = cfg.get("experiments") or {}
    if name not in exps:
        raise KeyError(f"Unknown experiment '{name}'. Known: {sorted(exps)}")
    out = copy.deepcopy(cfg.get("train_defaults") or {})
    deep_merge(out, exps[name])
    out["name"] = name
    return out

"""Environment report: hardware, storage, GPU and library versions."""
from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PACKAGES = ["numpy", "pandas", "pyarrow", "sklearn", "xgboost", "lightgbm", "imblearn", "matplotlib",
            "yaml", "psutil"]


def _gpu() -> Optional[str]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def _code_version() -> Optional[Dict[str, Any]]:
    """Git commit of the checked-out code, so every result can be traced to the code that made it."""
    repo = Path(__file__).resolve().parent.parent

    def git(*args) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              timeout=10).stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        if not commit:
            return None
        return {"commit": commit, "date": git("log", "-1", "--format=%cI"),
                "subject": git("log", "-1", "--format=%s"), "dirty": bool(git("status", "--porcelain"))}
    except Exception:  # noqa: BLE001
        return None


def _disk(path) -> Optional[Dict[str, float]]:
    try:
        du = shutil.disk_usage(path)
        return {"free_gb": round(du.free / 2**30, 1), "total_gb": round(du.total / 2**30, 1)}
    except OSError:
        return None


def collect(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(),
                            "cpus": os.cpu_count(), "colab": importlib.util.find_spec("google.colab") is not None}
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_gb"] = round(vm.total / 2**30, 1)
        info["ram_available_gb"] = round(vm.available / 2**30, 1)
    except Exception:  # noqa: BLE001
        pass
    info["local_disk"] = _disk("/content") or _disk("/")
    info["drive_mounted"] = os.path.isdir("/content/drive/MyDrive")
    info["gpu"] = _gpu()
    info["code"] = _code_version()
    vers = {}
    for p in PACKAGES:
        try:
            m = importlib.import_module(p)
            vers[p] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            vers[p] = "missing"
    info["packages"] = vers
    if cfg:
        paths = cfg.get("paths", {})
        wd = paths.get("work_dir")
        info["work_dir"] = wd
        info["work_dir_disk"] = _disk(wd) if wd and os.path.exists(wd) else None
        zp = paths.get("zip_path")
        info["zip_path"] = zp
        info["zip_found"] = bool(zp and os.path.exists(zp))
        if info["zip_found"]:
            info["zip_gb"] = round(os.path.getsize(zp) / 2**30, 2)
    return info


def short_env(cfg: Optional[Dict[str, Any]] = None) -> str:
    i = collect(cfg)
    pk = ", ".join(f"{k} {v}" for k, v in i["packages"].items())
    disk = i.get("local_disk") or {}
    return (f"python {i['python']} | cpus {i['cpus']} | RAM {i.get('ram_gb')} GB "
            f"(free {i.get('ram_available_gb')}) | local disk free {disk.get('free_gb')} GB | "
            f"GPU {i.get('gpu') or 'none'} | colab {i['colab']}\n{pk}")


def print_report(i: Dict[str, Any]) -> None:
    print("Environment")
    print(f"  python {i['python']} on {i['platform']}  (colab: {i['colab']})")
    print(f"  CPUs: {i['cpus']}   RAM: {i.get('ram_gb')} GB total, {i.get('ram_available_gb')} GB available")
    d = i.get("local_disk") or {}
    print(f"  local disk: {d.get('free_gb')} GB free of {d.get('total_gb')} GB")
    print(f"  GPU: {i.get('gpu') or 'none (CPU runtime)'}")
    c = i.get("code")
    print("  code: " + (f"{c['commit']}{' (+ local changes)' if c['dirty'] else ''} {c['date']} {c['subject']}"
                        if c else "unknown (not a git checkout)"))
    print(f"  Google Drive mounted: {i['drive_mounted']}")
    print("  packages: " + ", ".join(f"{k}={v}" for k, v in i["packages"].items()))
    if "work_dir" in i:
        wd = i.get("work_dir_disk") or {}
        print(f"  work dir: {i['work_dir']} (free space reported: {wd.get('free_gb')} GB)")
        print(f"  dataset zip: {i['zip_path']} -> " + (f"found, {i.get('zip_gb')} GB" if i["zip_found"]
                                                      else "NOT FOUND (the code will search My Drive)"))

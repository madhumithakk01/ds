"""Structured tracing for every command.

Each command invocation is a *run* with its own id.  It produces:

* ``logs/<run_id>.jsonl`` - one JSON object per event (stage, event name,
  message, data, timing, process memory, free RAM).  This is what the
  ``diagnose`` and ``trace`` commands read.
* ``logs/<run_id>.log``   - the same events as readable text.
* ``runs/<run_id>/manifest.json`` - command, arguments, resolved config,
  environment, final status.  A manifest that still says ``running`` means the
  process was killed (runtime disconnect, out-of-memory, ...).
* ``state/heartbeat.json`` - what the most recent run was doing and when.
"""
from __future__ import annotations

import contextlib
import os
import platform
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from .state import SESSION_ID, GracefulStop, atomic_write_json, dumps, ensure_dir, now_iso, read_json

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}


def _mem() -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        import psutil

        p = psutil.Process(os.getpid())
        out["rss_mb"] = round(p.memory_info().rss / 2**20, 1)
        vm = psutil.virtual_memory()
        out["avail_mb"] = round(vm.available / 2**20, 1)
    except Exception:
        pass
    return out


def disk_free_gb(path) -> Optional[float]:
    try:
        return round(shutil.disk_usage(path).free / 2**30, 2)
    except OSError:
        return None


class Tracer:
    def __init__(self, work_dir, command: str, echo_level: str = "info", argv=None):
        self.work = Path(work_dir)
        self.command = command
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{SESSION_ID[:6]}"
        self.t0 = time.time()
        self.echo_level = LEVELS.get(echo_level, 20)
        self.logs = ensure_dir(self.work / "logs")
        self.jsonl = self.logs / f"{self.run_id}.jsonl"
        self.text = self.logs / f"{self.run_id}.log"
        self.run_dir = ensure_dir(self.work / "runs" / self.run_id)
        self.heartbeat_path = self.work / "state" / "heartbeat.json"
        ensure_dir(self.heartbeat_path.parent)
        self.counts = {"warn": 0, "error": 0}
        self._lock = threading.Lock()
        self._monitor: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        self._hb_extra: Dict[str, Any] = {}
        self._last_progress_print = 0.0
        self.manifest: Dict[str, Any] = {
            "run_id": self.run_id, "command": command, "argv": list(argv or sys.argv),
            "session": SESSION_ID, "started": now_iso(), "status": "running",
            "python": sys.version.split()[0], "platform": platform.platform(),
        }
        try:
            (self.logs / "latest_run.txt").write_text(self.run_id)
        except OSError:
            pass

    # ------------------------------------------------------------ core
    def event(self, level: str, stage: str, event: str, msg: str = "", **data) -> None:
        rec = {
            "ts": now_iso(), "t": round(time.time() - self.t0, 3), "run": self.run_id,
            "level": level, "stage": stage, "event": event, "msg": msg,
        }
        if data:
            rec["data"] = data
        rec.update(_mem())
        line = dumps(rec)
        text = f"{rec['ts']} {level.upper():5s} [{stage}] {event}: {msg}"
        if data and level in ("warn", "error"):
            short = dumps(data)
            text += f" | {short[:800]}"
        with self._lock:
            if level in self.counts:
                self.counts[level] += 1
            try:
                with open(self.jsonl, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                with open(self.text, "a", encoding="utf-8") as f:
                    f.write(text + "\n")
            except OSError:
                pass
        if LEVELS.get(level, 20) >= self.echo_level:
            print(text, flush=True)

    def debug(self, stage, event, msg="", **data):
        self.event("debug", stage, event, msg, **data)

    def info(self, stage, event, msg="", **data):
        self.event("info", stage, event, msg, **data)

    def warn(self, stage, event, msg="", **data):
        self.event("warn", stage, event, msg, **data)

    def error(self, stage, event, msg="", **data):
        self.event("error", stage, event, msg, **data)

    def exception(self, stage: str, event: str, exc: BaseException, **data) -> str:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.error(stage, event, f"{type(exc).__name__}: {exc}", traceback=tb[-6000:], **data)
        err_dir = ensure_dir(self.work / "state" / "errors")
        name = f"{self.run_id}-{stage}-{event}.txt".replace("/", "_")
        try:
            (err_dir / name).write_text(f"{now_iso()}\n{dumps(data)}\n\n{tb}")
        except OSError:
            pass
        return tb

    @contextlib.contextmanager
    def span(self, stage: str, name: str, **data):
        t = time.time()
        self.debug(stage, f"{name}.start", **data)
        try:
            yield
        except BaseException as exc:
            soft = isinstance(exc, (KeyboardInterrupt, GracefulStop))
            self.event("warn" if soft else "error", stage,
                       f"{name}.fail", f"{type(exc).__name__}: {exc}",
                       seconds=round(time.time() - t, 2), **data)
            raise
        self.debug(stage, f"{name}.end", seconds=round(time.time() - t, 2), **data)

    # ------------------------------------------------------------ progress
    def heartbeat(self, stage: str, **info) -> None:
        self._hb_extra = {"stage": stage, **info}
        self._write_heartbeat()

    def _write_heartbeat(self) -> None:
        hb = {"run": self.run_id, "command": self.command, "ts": now_iso(),
              "elapsed_s": round(time.time() - self.t0, 1), **_mem(), **self._hb_extra}
        try:
            atomic_write_json(self.heartbeat_path, hb)
        except OSError:
            pass

    def progress(self, stage: str, done: int, total: int, what: str = "units",
                 every_s: float = 20.0, **extra) -> None:
        self.heartbeat(stage, done=done, total=total, what=what, **extra)
        now = time.time()
        if now - self._last_progress_print >= every_s or done >= total:
            self._last_progress_print = now
            pct = 100.0 * done / max(total, 1)
            el = now - self.t0
            eta = (el / max(done, 1)) * (total - done) if done else float("nan")
            extra_txt = " ".join(f"{k}={v}" for k, v in extra.items())
            print(f"  [{stage}] {done}/{total} {what} ({pct:.1f}%), elapsed {el/60:.1f} min,"
                  f" eta {eta/60:.1f} min {extra_txt}", flush=True)

    # ------------------------------------------------------------ monitor
    def start_monitor(self, lock=None, every_s: float = 60.0, gpu: bool = False) -> None:
        def loop():
            n = 0
            while not self._stop_monitor.wait(every_s):
                n += 1
                data = {"disk_content_gb": disk_free_gb("/content") or disk_free_gb("/"),
                        "disk_work_gb": disk_free_gb(self.work)}
                if gpu and n % 5 == 1:
                    data["gpu"] = gpu_memory()
                self.debug("resource", "sample", **data)
                self._write_heartbeat()
                if lock is not None:
                    lock.heartbeat()

        self._monitor = threading.Thread(target=loop, name="gids-monitor", daemon=True)
        self._monitor.start()

    def stop_monitor(self) -> None:
        self._stop_monitor.set()

    # ------------------------------------------------------------ manifest
    def write_manifest(self, **extra) -> None:
        self.manifest.update(extra)
        self.manifest["updated"] = now_iso()
        self.manifest["elapsed_s"] = round(time.time() - self.t0, 1)
        self.manifest["warnings"] = self.counts["warn"]
        self.manifest["errors"] = self.counts["error"]
        try:
            atomic_write_json(self.run_dir / "manifest.json", self.manifest)
        except OSError:
            pass

    def finish(self, status: str, **extra) -> None:
        self.stop_monitor()
        self.info("run", "finish", f"status={status}", seconds=round(time.time() - self.t0, 1),
                  warnings=self.counts["warn"], errors=self.counts["error"])
        self.write_manifest(status=status, finished=now_iso(), **extra)
        self._hb_extra["status"] = status
        self._write_heartbeat()


def gpu_memory() -> Optional[list]:
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        rows = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                rows.append({"name": parts[0], "used_mb": float(parts[1]),
                             "total_mb": float(parts[2]), "util_pct": float(parts[3])})
        return rows or None
    except Exception:
        return None


def read_events(work_dir, run_id: Optional[str] = None, last_runs: int = 1):
    """Load events of the latest run(s) for inspection."""
    import json

    logs = Path(work_dir) / "logs"
    files = sorted(logs.glob("*.jsonl"))
    if run_id:
        files = [f for f in files if f.stem == run_id]
    else:
        files = files[-last_runs:]
    events = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def list_manifests(work_dir):
    runs = Path(work_dir) / "runs"
    out = []
    for d in sorted(runs.glob("*")):
        m = read_json(d / "manifest.json")
        if m:
            out.append(m)
    return out

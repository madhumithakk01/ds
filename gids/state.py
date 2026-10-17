"""Persistence primitives that make every stage safe to interrupt and resume.

Everything that matters is written to the work directory (normally on Google
Drive) using write-to-temp-then-rename, and progress is recorded in
append-only JSONL ledgers.  A stage never trusts a ledger entry blindly: the
output file named in the entry must exist as well, otherwise the unit is redone.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import signal
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

SESSION_ID = uuid.uuid4().hex[:12]


class GracefulStop(Exception):
    """Raised at a safe point after the user asked the process to stop."""


class LockHeld(RuntimeError):
    pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj):
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if v != v else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except Exception:  # pragma: no cover - numpy always present in practice
        pass
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def dumps(obj, indent=None) -> str:
    return json.dumps(obj, default=_json_default, indent=indent, sort_keys=False)


def stable_hash(obj, n: int = 8) -> str:
    text = json.dumps(obj, default=_json_default, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _tmp_name(path: Path) -> Path:
    return path.with_name(f".{path.name}.{SESSION_ID}.tmp")


def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = _tmp_name(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def atomic_write_text(path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, dumps(obj, indent=2))


def atomic_copy(src, dst) -> None:
    """Copy a finished local file into place on the work directory."""
    dst = Path(dst)
    ensure_dir(dst.parent)
    tmp = _tmp_name(dst)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def retry(fn: Callable, *, tries: int = 4, base_delay: float = 2.0,
          exceptions=(OSError,), on_error: Optional[Callable] = None):
    """Call fn() and retry transient failures with exponential backoff."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except exceptions as exc:  # noqa: PERF203
            if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
                raise
            last = exc
            if on_error is not None:
                on_error(attempt, exc)
            if attempt < tries:
                time.sleep(base_delay * (2 ** (attempt - 1)))
    raise last


# ---------------------------------------------------------------- ledgers

class Ledger:
    """Append-only JSONL progress log.

    Each record carries a ``unit`` id and a ``status`` (start/done/failed).
    Lines that cannot be parsed (for example a write cut short by a crash)
    are ignored when reading.
    """

    def __init__(self, path):
        self.path = Path(path)
        ensure_dir(self.path.parent)

    def append(self, unit: str, status: str, **data) -> Dict[str, Any]:
        rec = {"ts": now_iso(), "session": SESSION_ID, "unit": unit, "status": status}
        rec.update(data)
        line = dumps(rec) + "\n"

        def _write():
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

        retry(_write, tries=4, base_delay=1.0)
        return rec

    def records(self) -> list:
        out = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return out

    def latest(self) -> Dict[str, Dict[str, Any]]:
        last: Dict[str, Dict[str, Any]] = {}
        for rec in self.records():
            last[rec.get("unit")] = rec
        return last

    def done(self) -> Dict[str, Dict[str, Any]]:
        return {u: r for u, r in self.latest().items() if r.get("status") == "done"}

    def failed(self) -> Dict[str, Dict[str, Any]]:
        return {u: r for u, r in self.latest().items() if r.get("status") == "failed"}

    def start_count(self, unit: str) -> int:
        """How many times a unit was started without finishing afterwards."""
        n = 0
        for rec in self.records():
            if rec.get("unit") != unit:
                continue
            if rec.get("status") == "start":
                n += 1
            elif rec.get("status") == "done":
                n = 0
        return n


def output_ok(path, min_bytes: int = 1) -> bool:
    try:
        return Path(path).stat().st_size >= min_bytes
    except OSError:
        return False


# ---------------------------------------------------------------- lock

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


class RunLock:
    """Prevents two sessions from writing to the same work directory at once.

    A lock whose owner process is gone (same machine) or whose heartbeat is
    older than ``stale_after_s`` (different machine, e.g. a recycled Colab VM)
    is taken over automatically.
    """

    def __init__(self, path, stale_after_s: int = 300):
        self.path = Path(path)
        self.stale_after_s = stale_after_s
        self.owned = False

    def info(self):
        return read_json(self.path)

    def acquire(self, command: str, force: bool = False) -> None:
        cur = self.info()
        if cur and not force:
            same_host = cur.get("host") == socket.gethostname()
            age = time.time() - float(cur.get("heartbeat", 0))
            if same_host and _pid_alive(cur.get("pid")) and int(cur.get("pid", -1)) != os.getpid():
                raise LockHeld(
                    f"Another command is still running on this machine (pid {cur.get('pid')}, "
                    f"'{cur.get('command')}'). Wait for it, or stop it before starting a new one."
                )
            if not same_host and age < self.stale_after_s:
                raise LockHeld(
                    f"Work directory is in use by another machine ({cur.get('host')}, "
                    f"heartbeat {int(age)}s ago, '{cur.get('command')}'). If that session is dead, "
                    f"wait {self.stale_after_s - int(age)}s or pass --force."
                )
        self._write(command)
        self.owned = True

    def _write(self, command: str) -> None:
        atomic_write_json(self.path, {
            "host": socket.gethostname(), "pid": os.getpid(), "session": SESSION_ID,
            "command": command, "since": now_iso(), "heartbeat": time.time(),
        })

    def heartbeat(self) -> None:
        if not self.owned:
            return
        cur = self.info() or {}
        if cur.get("session") not in (None, SESSION_ID):
            return
        cur["heartbeat"] = time.time()
        try:
            atomic_write_json(self.path, cur)
        except OSError:
            pass

    def release(self) -> None:
        if not self.owned:
            return
        cur = self.info() or {}
        if cur.get("session") == SESSION_ID:
            try:
                self.path.unlink()
            except OSError:
                pass
        self.owned = False


# ---------------------------------------------------------------- stopping

class _StopState:
    requested = False
    reason = ""


def install_signal_handlers(log: Optional[Callable[[str], None]] = None) -> None:
    def handler(signum, frame):  # noqa: ARG001
        if _StopState.requested:
            raise KeyboardInterrupt
        _StopState.requested = True
        _StopState.reason = signal.Signals(signum).name
        if log:
            log(f"{_StopState.reason} received: finishing the current unit, then stopping. "
                f"Send it again to stop immediately.")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # not in main thread
            pass


def request_stop(reason: str = "time budget") -> None:
    _StopState.requested = True
    _StopState.reason = reason


def stop_requested() -> bool:
    return _StopState.requested


def check_stop() -> None:
    if _StopState.requested:
        raise GracefulStop(_StopState.reason or "stop requested")


class Deadline:
    """Optional wall-clock budget for a command (``--max-minutes``)."""

    def __init__(self, minutes: Optional[float]):
        self.t_end = time.time() + 60.0 * minutes if minutes else None

    def check(self) -> None:
        if self.t_end is not None and time.time() >= self.t_end:
            request_stop("time budget reached")
        check_stop()


# ---------------------------------------------------------------- testing hook

_crash_counter: Dict[str, int] = {}


def crash_point(stage: str) -> None:
    """Simulate a hard kill for the self-test (GIDS_CRASH_AFTER=stage:n)."""
    spec = os.environ.get("GIDS_CRASH_AFTER", "")
    if not spec or ":" not in spec:
        return
    name, _, n = spec.partition(":")
    if name != stage:
        return
    _crash_counter[stage] = _crash_counter.get(stage, 0) + 1
    if _crash_counter[stage] >= int(n):
        os._exit(17)


def iter_existing(paths: Iterable) -> Iterable[Path]:
    for p in paths:
        if output_ok(p):
            yield Path(p)

"""Command line entry point: ``python -m gids <command>``.

Commands that change the work directory take a lock, write a run manifest
and a structured event log, and stop cleanly on Ctrl-C / "stop cell" (the
current unit finishes, progress is kept).  Running the same command again
continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import load_config
from .state import (Deadline, GracefulStop, LockHeld, RunLock, ensure_dir, install_signal_handlers, read_json)
from .trace import Tracer, list_manifests, read_events

WRITE_CMDS = {"inventory", "explore", "ingest", "prepare", "train"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gids", description="Gotham IoT intrusion-detection pipeline")
    p.add_argument("--work-dir", help="folder for all outputs (default: paths.work_dir)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set explore.block_mb=16")
    p.add_argument("--config", action="append", default=[], help="extra YAML file merged into the config")
    p.add_argument("--quiet", action="store_true", help="print warnings and errors only")
    p.add_argument("--force", action="store_true", help="take over a lock held by another session")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env", help="hardware, storage, GPU and library report")
    s = sub.add_parser("inventory", help="list the archive, peek at CSV headers, measure read speed")
    s.add_argument("--max-minutes", type=float)
    s = sub.add_parser("explore", help="scan every CSV (resumable) and write reports/explore/SUMMARY.md")
    s.add_argument("--max-files", type=int, help="only the N smallest files (quick look)")
    s.add_argument("--refresh-inventory", action="store_true")
    s.add_argument("--max-minutes", type=float, help="stop cleanly after this many minutes")
    s.add_argument("--block-mb", type=float)
    s = sub.add_parser("ingest", help="CSV -> Parquet parts with context features, then the split (resumable)")
    s.add_argument("--max-files", type=int, help="only the N smallest files (quick look)")
    s.add_argument("--max-minutes", type=float, help="stop cleanly after this many minutes")
    s = sub.add_parser("prepare", help="labels, split, per-segment direction, merged rows, checks (resumable)")
    s.add_argument("--max-files", type=int, help="only the N smallest files (quick look)")
    s.add_argument("--max-minutes", type=float, help="stop cleanly after this many minutes")
    s = sub.add_parser("train", help="build the dataset, train every experiment, compare on validation (resumable)")
    s.add_argument("--only", help="comma-separated experiment names (default: all enabled, in order)")
    s.add_argument("--max-minutes", type=float, help="stop cleanly after this many minutes")
    sub.add_parser("status", help="progress of every stage and recent runs")
    s = sub.add_parser("trace", help="show recent log events")
    s.add_argument("--tail", type=int, default=40)
    s.add_argument("--errors", action="store_true", help="warnings and errors only")
    s.add_argument("--run", help="run id (default: latest)")
    s = sub.add_parser("selftest", help="end-to-end check on a small synthetic archive (~1 min)")
    s.add_argument("--keep", action="store_true")
    s = sub.add_parser("unlock", help="remove the work-directory lock")
    s.add_argument("--yes", action="store_true")
    return p


def _ago(ts: float) -> str:
    s = max(0, time.time() - ts)
    return f"{s:.0f}s" if s < 120 else (f"{s / 60:.0f} min" if s < 7200 else f"{s / 3600:.1f} h")


def print_status(cfg) -> None:
    work = Path(cfg["paths"]["work_dir"])
    print(f"work dir: {work}")
    lock = read_json(work / "state" / "lock.json")
    if lock:
        print(f"lock: held by '{lock.get('command')}' on {lock.get('host')} (pid {lock.get('pid')}), "
              f"heartbeat {_ago(float(lock.get('heartbeat', 0)))} ago")
    else:
        print("lock: free")
    hb = read_json(work / "state" / "heartbeat.json")
    if hb:
        extra = {k: v for k, v in hb.items() if k not in ("run", "command", "ts", "elapsed_s")}
        print(f"last activity: {hb.get('ts')} run {hb.get('run')} '{hb.get('command')}' {extra}")
    inv = read_json(work / "reports" / "explore" / "inventory.json")
    if inv:
        csvs = [m for m in inv["members"] if m["kind"] == "processed_csv"]
        led = work / "data" / "explore" / "_ledger.jsonl"
        done, failed, rows = 0, 0, 0
        if led.exists():
            last = {}
            for line in led.read_text().splitlines():
                try:
                    r = json.loads(line)
                    last[r["unit"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue
            done = sum(1 for r in last.values() if r["status"] == "done")
            failed = sum(1 for r in last.values() if r["status"] == "failed")
            rows = sum(int(r.get("rows", 0)) for r in last.values() if r["status"] == "done")
        print(f"explore: inventory done ({len(csvs)} CSV files); scanned {done}/{len(csvs)} files, "
              f"{rows:,} rows" + (f", {failed} failed" if failed else ""))
        if (work / "reports" / "explore" / "SUMMARY.md").exists():
            print(f"         summary: {work / 'reports' / 'explore' / 'SUMMARY.md'}")
    else:
        print("explore: inventory not run yet")
    from .layout import ingest_dir

    led = ingest_dir(cfg) / "_ledger.jsonl"
    if led.exists():
        last = {}
        for line in led.read_text().splitlines():
            try:
                r = json.loads(line)
                last[r["unit"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
        done = [r for r in last.values() if r["status"] == "done"]
        n_csv = len([m for m in inv["members"] if m["kind"] == "processed_csv"]) if inv else "?"
        print(f"ingest: {len(done)}/{n_csv} files, {sum(int(r.get('rows', 0)) for r in done):,} rows "
              f"({led.parent.name})")
        partial = sorted((led.parent / "_state").glob("*.json")) if (led.parent / "_state").exists() else []
        for f in partial:
            st = read_json(f) or {}
            if st.get("member") not in {r["unit"] for r in done}:
                print(f"         in progress: {Path(st.get('member', '?')).name} - {len(st.get('parts', []))} "
                      f"part(s), byte {st.get('offset', 0):,}")
        if (work / "reports" / "ingest" / "INGEST.md").exists():
            print(f"         report: {work / 'reports' / 'ingest' / 'INGEST.md'}")
    else:
        print("ingest: not run yet")
    from .prepare import prepare_dir

    led = prepare_dir(cfg) / "_ledger.jsonl"
    if led.exists():
        last = {}
        for line in led.read_text().splitlines():
            try:
                r = json.loads(line)
                last[r["unit"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
        done = [r for r in last.values() if r["status"] == "done"]
        print(f"prepare: {len(done)} files, {sum(int(r.get('unique', 0)) for r in done):,} unique rows "
              f"({led.parent.name})")
        if (work / "reports" / "prepare" / "PREPARE.md").exists():
            print(f"         report: {work / 'reports' / 'prepare' / 'PREPARE.md'}")
    else:
        print("prepare: not run yet")
    try:
        from .config import experiment_names
        from .layout import experiment_spec
        from .state import stable_hash

        def model_dir_name(cfg, n):  # same name as layout.model_dir, without creating the folder
            return f"{n}-{stable_hash(experiment_spec(cfg, n))}"

        names = experiment_names(cfg)
        dirs = {}
        for n in names:  # the folder for the current settings, else the latest one trained with other settings
            cur = work / "models" / model_dir_name(cfg, n)
            others = sorted((work / "models").glob(f"{n}-*"), key=lambda p: p.stat().st_mtime) \
                if (work / "models").exists() else []
            dirs[n] = (cur, "") if (cur / "state.json").exists() else \
                ((others[-1], " (other settings)") if others else (None, ""))
        states = {n: read_json(d / "state.json") if d else None for n, (d, _) in dirs.items()}
        if any(states.values()):
            print("train:")
            for n, st in states.items():
                d, note = dirs[n]
                m = read_json(d / "metrics.json") if st else None
                txt = (f"done, validation macro-F1 {m['macro_f1']:.4f}" if m and st.get("status") == "done"
                       else f"{st['iterations']} rounds so far" if st else "not started")
                print(f"  {n:22s} {txt}{note}")
            if (work / "reports" / "train" / "TRAIN.md").exists():
                print(f"  report: {work / 'reports' / 'train' / 'TRAIN.md'}")
        else:
            print("train: not run yet")
    except Exception as exc:  # noqa: BLE001 - status must never fail
        print(f"train: (status unavailable: {exc})")
    runs = list_manifests(work)[-8:]
    if runs:
        print("recent runs:")
        for m in runs:
            st = m.get("status")
            if st == "running":
                st = "running (or killed - no clean finish)"
            print(f"  {m['run_id']}  {m.get('command'):10s} {st:38s} {m.get('elapsed_s', 0) / 60:6.1f} min "
                  f"warnings={m.get('warnings', 0)} errors={m.get('errors', 0)}")


def print_trace(cfg, args) -> None:
    work = Path(cfg["paths"]["work_dir"])
    ev = read_events(work, run_id=args.run)
    if args.errors:
        ev = [e for e in ev if e.get("level") in ("warn", "error")]
    for e in ev[-args.tail:]:
        line = f"{e['ts']} {e['level'].upper():5s} [{e['stage']}] {e['event']}: {e.get('msg', '')}"
        if e.get("data"):
            d = dict(e["data"])
            tb = d.pop("traceback", None)
            line += " | " + json.dumps(d, default=str)[:600]
            if tb and args.errors:
                line += "\n" + tb
        print(line)
    errs = sorted((work / "state" / "errors").glob("*.txt"))[-5:] if (work / "state" / "errors").exists() else []
    if errs:
        print("\nlatest error files:")
        for f in errs:
            print(f"  {f}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "selftest":
        from .synth import selftest

        return 0 if selftest(keep=args.keep) else 1
    cfg = load_config(args.work_dir, args.set, args.config)
    if getattr(args, "block_mb", None):
        cfg.setdefault("explore", {})["block_mb"] = args.block_mb
    work = ensure_dir(cfg["paths"]["work_dir"])
    if args.cmd == "env":
        from .envinfo import collect, print_report

        print_report(collect(cfg))
        return 0
    if args.cmd == "status":
        print_status(cfg)
        return 0
    if args.cmd == "trace":
        print_trace(cfg, args)
        return 0
    if args.cmd == "unlock":
        lock = RunLock(work / "state" / "lock.json")
        print(f"current lock: {lock.info()}")
        if args.yes:
            (work / "state" / "lock.json").unlink(missing_ok=True)
            print("lock removed")
        else:
            print("add --yes to remove it")
        return 0

    tracer = Tracer(work, args.cmd, echo_level="warn" if args.quiet else "info")
    lock = RunLock(work / "state" / "lock.json")
    try:
        lock.acquire(args.cmd, force=args.force)
    except LockHeld as exc:
        print(f"ERROR: {exc}")
        tracer.finish("refused")
        return 2
    install_signal_handlers(lambda m: tracer.warn("run", "signal", m))
    tracer.start_monitor(lock=lock)
    from .envinfo import collect

    tracer.write_manifest(version=__version__, config={k: v for k, v in cfg.items() if k != "_meta"},
                          overrides=cfg.get("_meta", {}), env=collect(cfg))
    deadline = Deadline(getattr(args, "max_minutes", None))
    status, code = "ok", 0
    try:
        if args.cmd == "inventory":
            from .explore import run_inventory

            run_inventory(cfg, tracer)
        elif args.cmd == "explore":
            from .explore import run_explore

            run_explore(cfg, tracer, deadline, max_files=args.max_files,
                        refresh_inventory=args.refresh_inventory)
        elif args.cmd == "ingest":
            from .ingest import run_ingest

            run_ingest(cfg, tracer, deadline, max_files=args.max_files)
        elif args.cmd == "prepare":
            from .prepare import run_prepare

            run_prepare(cfg, tracer, deadline, max_files=args.max_files)
        elif args.cmd == "train":
            from .train import run_train

            run_train(cfg, tracer, deadline, only=[x.strip() for x in args.only.split(",")] if args.only else None)
    except GracefulStop as exc:
        status, code = "stopped", 3
        print(f"\nStopped ({exc}). Progress is saved - run the same command again to continue.")
    except KeyboardInterrupt:
        status, code = "interrupted", 130
        print("\nInterrupted. Finished units are saved - run the same command again to continue.")
    except Exception as exc:  # noqa: BLE001
        tracer.exception("run", "failed", exc)
        status, code = "failed", 1
        print(f"\nFAILED: {type(exc).__name__}: {exc}\nDetails: python -m gids trace --errors "
              f"(log: {tracer.text})")
    finally:
        tracer.finish(status)
        lock.release()
    return code


if __name__ == "__main__":
    sys.exit(main())

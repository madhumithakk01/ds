"""A small archive with the same layout and field formats as the Gotham
dataset (tshark field names, hex flags, repeated values, stitched captures),
used by ``python -m gids selftest`` to check the code on a new machine in a
few seconds before the real archive is touched.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HEADER = ["frame.time", "frame.len", "frame.protocols", "eth.src", "eth.dst", "ip.dst", "ip.src", "ip.flags",
          "ip.ttl", "ip.proto", "ip.checksum", "ip.tos", "tcp.srcport", "tcp.dstport", "tcp.flags",
          "tcp.window_size_value", "tcp.window_size_scalefactor", "tcp.checksum", "tcp.options", "tcp.pdu.size",
          "udp.srcport", "udp.dstport", "label"]
DEVICES = [("air-quality", 1, "mqtt"), ("building-monitor", 2, "mqtt"), ("city-power", 1, "coap"),
           ("ip-camera-street", 1, "rtsp"), ("cooler-motor", 2, "mqtt")]
ATTACKS = ["Network Scanning", "Telnet Brute Forcing", "Reporting", "Ingress Tool Transfer",
           "Remote Code Execution", "Periodic C&C Communication", "Remote Command Execution", "DoS Attack"]
T0 = 1733407512.0
STITCHED_IP = "10.0.1.251"  # cooler-motor-2's attack segment comes from this (other) device


def _ts(t: float) -> str:
    d = dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
    frac = int(round((t - int(t)) * 1e9)) % 1_000_000_000
    return f"{d.strftime('%b')} {d.day:>2}, {d.year} {d.strftime('%H:%M:%S')}.{frac:09d} GMT"


def _row(rng, t, kind, dev_ip, label) -> List[str]:
    mac_d, mac_g = "02:42:0a:00:01:05", "02:42:0a:00:01:01"
    server, attacker = "10.0.2.10", "10.0.9.9"
    r = {h: "" for h in HEADER}
    r["frame.time"] = _ts(t)
    r["eth.src"], r["eth.dst"] = mac_d, mac_g
    r["ip.flags"] = "0x40"
    r["ip.ttl"] = "64"
    r["ip.checksum"] = f"0x{rng.integers(0, 65535):04x}"
    r["ip.tos"] = "0x00"
    r["label"] = label
    if kind in ("mqtt", "rtsp", "telnet", "syn", "http"):
        r["ip.proto"] = "6"
        r["tcp.checksum"] = f"0x{rng.integers(0, 65535):04x}"
        r["tcp.window_size_value"] = str(int(rng.choice([502, 509, 64240])))
        r["tcp.window_size_scalefactor"] = "128" if rng.random() < 0.7 else "-1"
        if kind == "syn":
            r["ip.src"], r["ip.dst"] = attacker, dev_ip
            r["tcp.srcport"], r["tcp.dstport"] = str(int(rng.integers(40000, 60000))), "23"
            r["tcp.flags"] = "0x0002"
            r["frame.len"] = "60"
            r["frame.protocols"] = "eth:ethertype:ip:tcp"
            r["tcp.window_size_value"] = "14600"
            r["tcp.window_size_scalefactor"] = "-1"
        else:
            port = {"mqtt": "1883", "rtsp": "554", "telnet": "23", "http": "80"}[kind]
            out = rng.random() < 0.5
            r["ip.src"], r["ip.dst"] = (dev_ip, server) if out else (server, dev_ip)
            eph = str(int(rng.integers(40000, 41000)))
            r["tcp.srcport"], r["tcp.dstport"] = (eph, port) if out else (port, eph)
            r["tcp.flags"] = str(rng.choice(["0x0018", "0x0010", "0x0011"]))
            r["frame.len"] = str(int(rng.integers(66, 200) if kind != "rtsp" else rng.integers(900, 1450)))
            r["frame.protocols"] = f"eth:ethertype:ip:tcp:{kind}" if r["tcp.flags"] == "0x0018" else "eth:ethertype:ip:tcp"
            r["tcp.options"] = f"0101080a{rng.integers(0, 2**32):08x}{rng.integers(0, 2**32):08x}"
            if rng.random() < 0.1:
                r["tcp.pdu.size"] = str(int(rng.integers(2, 120)))
    else:
        r["ip.proto"] = "17"
        if kind == "coap":
            r["ip.src"], r["ip.dst"] = dev_ip, server
            r["udp.srcport"], r["udp.dstport"] = str(int(rng.integers(50000, 50010))), "5683"
            r["frame.len"] = str(int(rng.integers(60, 110)))
            r["frame.protocols"] = "eth:ethertype:ip:udp:coap"
        else:  # flood packet: identical every time
            r["ip.src"], r["ip.dst"] = attacker, dev_ip
            r["udp.srcport"], r["udp.dstport"] = "4444", "53"
            r["frame.len"] = "554"
            r["frame.protocols"] = "eth:ethertype:ip:udp:data"
    if rng.random() < 0.003:  # ICMP error quoting an inner header -> repeated fields
        r["ip.ttl"] = "64,63"
        r["ip.proto"] = "1,17"
        r["frame.protocols"] = "eth:ethertype:ip:icmp:ip:udp:dns"
    return [r[h] for h in HEADER]


def _device_csv(dev: str, proto: str, n: int, rng, malformed: int = 0,
                stitched_ip: str = "") -> Tuple[str, Counter]:
    """One device's CSV; with ``stitched_ip`` the attack capture belongs to another device (two captures joined)."""
    dev_ip = f"10.0.1.{int(rng.integers(2, 250))}"
    rows, labels = [], Counter()
    n_benign = int(n * 0.6)
    t = T0
    for _ in range(n_benign):
        t += float(rng.exponential(0.5))
        rows.append(_row(rng, t, proto, dev_ip, "Normal"))
    t = T0 - 3600.0  # attack capture recorded earlier -> time jumps back at the join
    dev_ip = stitched_ip or dev_ip
    phases = ATTACKS + (["CoAP Amplification Attack"] if proto == "coap" else [])
    per = max(1, (n - n_benign) // (len(phases) * 2))
    for ph in phases:
        for _ in range(per):  # benign traffic between attacks
            t += float(rng.exponential(0.5))
            rows.append(_row(rng, t, proto, dev_ip, "Normal"))
        for _ in range(per):
            t += float(rng.exponential(0.01))
            kind = {"Network Scanning": "syn", "Telnet Brute Forcing": "telnet", "Reporting": "telnet",
                    "Ingress Tool Transfer": "http", "Remote Code Execution": "telnet",
                    "Periodic C&C Communication": "http", "Remote Command Execution": "telnet",
                    "DoS Attack": "flood", "CoAP Amplification Attack": "coap"}[ph]
            if rng.random() < 0.15:  # benign telemetry inside an attack window keeps the attack label
                kind = proto
            rows.append(_row(rng, t, kind, dev_ip, ph))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(HEADER)
    for r in rows:
        w.writerow(r)
        labels[r[-1]] += 1
    text = buf.getvalue()
    if malformed:
        lines = text.splitlines(keepends=True)
        for k in range(malformed):
            lines.insert(10 + k, "this,line,has,too,many,fields," + ",".join(["x"] * 30) + "\n")
        text = "".join(lines)
    return text, labels


def make_archive(path, rows_per_device: int = 3000, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    truth: Dict = {"files": {}, "labels": Counter(), "rows": 0, "bad_lines": 0}
    root = "GothamDataset2025"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{root}/metadata/normal.json", json.dumps({"scenario": "normal", "start": T0, "end": T0 + 7200}))
        zf.writestr(f"{root}/metadata/mirai-dos.json", json.dumps(
            {"attacks": [{"type": "udp", "start": T0 - 3000, "end": T0 - 2990, "attacker_ip": "10.0.9.9"}]}))
        first = True
        for dtype, count, proto in DEVICES:
            for i in range(1, count + 1):
                dev = f"{dtype}-{i}"
                zf.writestr(f"{root}/raw/normal/iotsim-{dev}.pcap", rng.bytes(4096))
                zf.writestr(f"{root}/raw/malicious/mirai-dos/iotsim-{dev}.pcap", rng.bytes(2048))
                n = int(rows_per_device * (3 if proto == "rtsp" else 1))
                bad = 2 if first else 0
                text, labels = _device_csv(dev, proto, n, rng, malformed=bad,
                                           stitched_ip=STITCHED_IP if dev == "cooler-motor-2" else "")
                if first:
                    text = text.rstrip("\n")  # last line without a newline
                first = False
                zf.writestr(f"{root}/processed/iotsim-{dev}.csv", text)
                rows = sum(labels.values())
                truth["files"][f"iotsim-{dev}.csv"] = {"rows": rows, "labels": dict(labels)}
                truth["labels"].update(labels)
                truth["rows"] += rows
                truth["bad_lines"] += bad
    truth["labels"] = dict(truth["labels"])
    return truth


def _same_ingest(a: Path, b: Path, truth: Dict) -> Tuple[bool, str]:
    """Compare the ingested rows of two work directories file by file."""
    import pandas as pd

    def load(work: Path) -> Dict[str, "pd.DataFrame"]:
        d = next((work / "data").glob("ingest-*"))
        out = {}
        for f in d.glob("*.json"):
            meta = json.loads(f.read_text())
            if "parts" not in meta:
                continue
            df = pd.concat([pd.read_parquet(d / p) for p in meta["parts"]], ignore_index=True)
            out[meta["member"]] = df.astype({c: str for c in df.columns if df[c].dtype.name == "category"})
        return out

    x, y = load(a), load(b)
    if sorted(x) != sorted(y) or len(x) != len(truth["files"]):
        return False, f"files {len(x)} vs {len(y)}"
    for k in x:
        if not x[k].equals(y[k]):
            diff = [c for c in x[k].columns if c not in y[k] or not x[k][c].equals(y[k][c])]
            return False, f"{k}: columns differ {diff[:5]}"
    return True, ""


def experiment_names(cfg):
    from .config import experiment_names as names

    return names(cfg)


def load_config(work_dir: str):
    from .config import load_config as load

    return load(work_dir)


def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _same_prepare(a: Path, b: Path) -> Tuple[bool, str]:
    import pandas as pd

    def load(work: Path):
        d = next((work / "data").glob("prepare-*")) / "rows"
        return {f.name: pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))}

    x, y = load(a), load(b)
    if sorted(x) != sorted(y) or not x:
        return False, f"files {len(x)} vs {len(y)}"
    for k in x:
        if not x[k].equals(y[k]):
            return False, f"{k} differs"
    return True, ""


def _same_model(a: Path, b: Path, name: str) -> Tuple[bool, str]:
    import pandas as pd
    import xgboost as xgb

    def load(work: Path):
        d = next((work / "models").glob(f"{name}-*"))
        ds = next((work / "data").glob("dataset-*"))
        return xgb.Booster(model_file=str(d / "model.bin")), json.loads((d / "metrics.json").read_text()), ds

    (ma, ja, ds), (mb, jb, _) = load(a), load(b)
    va = pd.read_parquet(ds / "val.parquet")
    X = xgb.DMatrix(va[ja["features"]].to_numpy("float32"))
    pa_, pb_ = ma.predict(X), mb.predict(X)
    if ma.num_boosted_rounds() != mb.num_boosted_rounds():
        return False, f"rounds {ma.num_boosted_rounds()} vs {mb.num_boosted_rounds()}"
    if not np.array_equal(pa_, pb_):
        return False, f"max prediction difference {np.abs(pa_ - pb_).max()}"
    return ja["confusion"] == jb["confusion"], "confusion matrices differ"


def selftest(keep: bool = False) -> bool:
    repo = Path(__file__).resolve().parents[1]
    tmp = Path(tempfile.mkdtemp(prefix="gids_selftest_"))
    zp = tmp / "GothamDataset2025.zip"
    truth = make_archive(zp)
    work = tmp / "work"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    for k in ("GIDS_CRASH_AFTER", "GIDS_WORK_DIR", "GIDS_ZIP_PATH", "GIDS_DATA_DIR"):
        env.pop(k, None)
    base = [sys.executable, "-m", "gids", "--work-dir", str(work), "--set", f"paths.zip_path={zp}",
            "--set", "explore.block_mb=0.05", "--set", "explore.throughput_mb=1",
            "--set", f"paths.local_tmp={tmp / 'local'}", "--quiet"]
    checks: List[Tuple[str, bool, str]] = []

    def run(args, extra_env=None):
        e = dict(env, **(extra_env or {}))
        return subprocess.run(base + args, cwd=repo, env=e, capture_output=True, text=True)

    print(f"self-test in {tmp}")
    r = run(["inventory"])
    checks.append(("inventory runs", r.returncode == 0, r.stderr[-600:] or r.stdout[-300:]))
    r = run(["explore"], {"GIDS_CRASH_AFTER": "explore:3"})
    checks.append(("simulated crash after 3 files", r.returncode == 17, f"exit code {r.returncode} {r.stderr[-400:]}"))
    r = run(["explore"])
    checks.append(("explore resumes and finishes", r.returncode == 0, r.stderr[-800:] or r.stdout[-400:]))
    s = json.loads((work / "reports/explore/summary.json").read_text()) if \
        (work / "reports/explore/summary.json").exists() else {}
    checks.append(("all files scanned", s.get("files_scanned") == len(truth["files"]),
                   f"{s.get('files_scanned')} vs {len(truth['files'])}"))
    checks.append(("row count exact (nothing read twice)", s.get("rows") == truth["rows"],
                   f"{s.get('rows')} vs {truth['rows']}"))
    checks.append(("label counts exact", s.get("labels") == truth["labels"], ""))
    checks.append(("malformed lines detected", s.get("bad_lines") == truth["bad_lines"],
                   f"{s.get('bad_lines')} vs {truth['bad_lines']}"))
    checks.append(("time parsed", s.get("time_unparsed") == 0 and bool(s.get("time")), str(s.get("time_methods"))))
    checks.append(("capture joins found", "segments" in (work / "reports/explore/files.csv").read_text()
                   if (work / "reports/explore/files.csv").exists() else False, ""))
    ledger = (work / "data/explore/_ledger.jsonl").read_text().splitlines() if \
        (work / "data/explore/_ledger.jsonl").exists() else []
    sessions = {json.loads(x)["session"] for x in ledger if x.strip()}
    checks.append(("work split across two sessions", len(sessions) >= 2, f"{len(sessions)} sessions"))
    checks.append(("summary written", (work / "reports/explore/SUMMARY.md").exists(), ""))
    ing = base + ["--set", "ingest.block_mb=0.05", "--set", "ingest.part_mb=0.2"]
    r = subprocess.run(ing + ["ingest"], cwd=repo, env=dict(env, GIDS_CRASH_AFTER="ingest_part:5"),
                       capture_output=True, text=True)
    checks.append(("ingest: simulated crash in the middle of a file", r.returncode == 17,
                   f"exit code {r.returncode} {r.stderr[-400:]}"))
    r = subprocess.run(ing + ["ingest"], cwd=repo, env=env, capture_output=True, text=True)
    checks.append(("ingest resumes and finishes", r.returncode == 0, r.stderr[-800:] or r.stdout[-400:]))
    si = json.loads((work / "reports/ingest/summary.json").read_text()) if \
        (work / "reports/ingest/summary.json").exists() else {}
    checks.append(("ingest: row count exact", si.get("rows") == truth["rows"], f"{si.get('rows')} vs {truth['rows']}"))
    checks.append(("ingest: rows equal to explore", si.get("rows_match_explore") is True, ""))
    checks.append(("split: every label in train, val and test", not any("has no packets" in c for c in
                                                                         si.get("checks", ["?has no packets"])),
                   str(si.get("checks"))))
    ref = tmp / "work_ref"
    r = subprocess.run([a if a != str(work) else str(ref) for a in base]
                       + ["--set", "ingest.block_mb=0.3", "--set", "ingest.part_mb=1", "ingest"],
                       cwd=repo, env=env, capture_output=True, text=True)
    same, why = _same_ingest(work, ref, truth) if r.returncode == 0 else (False, r.stderr[-600:])
    checks.append(("ingest: resumed output identical to an uninterrupted run with other block sizes", same, why))
    r = subprocess.run(base + ["prepare"], cwd=repo, env=dict(env, GIDS_CRASH_AFTER="prepare:3"),
                       capture_output=True, text=True)
    checks.append(("prepare: simulated crash after 3 files", r.returncode == 17, f"exit code {r.returncode}"))
    r = subprocess.run(base + ["prepare"], cwd=repo, env=env, capture_output=True, text=True)
    checks.append(("prepare resumes and finishes", r.returncode == 0, r.stderr[-800:] or r.stdout[-400:]))
    sp = json.loads((work / "reports/prepare/summary.json").read_text()) if \
        (work / "reports/prepare/summary.json").exists() else {}
    checks.append(("prepare: packets per label and split equal to step 1", sp.get("accounting_ok") is True,
                   str(sp.get("checks"))))
    segs = _read_csv(work / "reports/prepare/segments.csv")
    st = segs[segs["file"] == "iotsim-cooler-motor-2.csv"] if len(segs) else segs
    checks.append(("prepare: joined captures get one device IP per segment",
                   len(st) >= 2 and STITCHED_IP in set(st["device_ip"]) and st["device_ip"].nunique() == 2
                   and bool((segs["share"] > 0.99).all()), st.to_string()[:600] if len(st) else "no segments"))
    r = subprocess.run([a if a != str(work) else str(ref) for a in base] + ["prepare"], cwd=repo, env=env,
                       capture_output=True, text=True)
    same, why = _same_prepare(work, ref) if r.returncode == 0 else (False, r.stderr[-600:])
    checks.append(("prepare: resumed output identical to an uninterrupted run", same, why))
    tr_args = ["--set", "train_defaults.xgboost.n_estimators=30", "--set", "train_defaults.xgboost.checkpoint_every=5",
               "--set", "train_defaults.xgboost.subsample=1", "--set", "train_defaults.xgboost.colsample_bytree=1",
               "--set", "train_defaults.xgboost.device=cpu",
               "--set", "train_defaults.lightgbm.n_estimators=15", "--set", "train_defaults.random_forest.n_estimators=6",
               "--set", "train_defaults.random_forest.batch=3"]
    r = subprocess.run(base + tr_args + ["train"], cwd=repo, env=dict(env, GIDS_CRASH_AFTER="train_ckpt:3"),
                       capture_output=True, text=True)
    checks.append(("train: simulated crash in the middle of training", r.returncode == 17,
                   f"exit code {r.returncode} {r.stderr[-400:]}"))
    r = subprocess.run(base + tr_args + ["train"], cwd=repo, env=env, capture_output=True, text=True)
    logs = "".join(p.read_text() for p in (work / "logs").glob("*.jsonl"))
    n_models = len(list((work / "models").glob("*/metrics.json")))
    checks.append(("train resumes and finishes every experiment", r.returncode == 0 and
                   "continuing after round 15" in logs and n_models == len(experiment_names(load_config(str(work)))),
                   f"exit {r.returncode}, {n_models} models; {r.stderr[-600:]}"))
    checks.append(("train: report written", (work / "reports/train/TRAIN.md").exists(), ""))
    r = subprocess.run([a if a != str(work) else str(ref) for a in base] + tr_args + ["train", "--only", "xgb_full"],
                       cwd=repo, env=env, capture_output=True, text=True)
    same, why = _same_model(work, ref, "xgb_full") if r.returncode == 0 else (False, r.stderr[-600:])
    checks.append(("train: resumed model predicts exactly like an uninterrupted one", same, why))
    ev_args = tr_args + ["--set", "experiments.xgb_portable_mirai_holdout.holdout.label_regex=^DoS",
                         "--set", "experiments.xgb_portable_mirai_holdout.holdout.episode=0",
                         "--set", "evaluate.shap_rows_per_class=200"]
    r = subprocess.run(base + ev_args + ["evaluate"], cwd=repo, env=dict(env, GIDS_CRASH_AFTER="evaluate:2"),
                       capture_output=True, text=True)
    checks.append(("evaluate: simulated crash after two models", r.returncode == 17,
                   f"exit code {r.returncode} {r.stderr[-400:]}"))
    r = subprocess.run(base + ev_args + ["evaluate"], cwd=repo, env=env, capture_output=True, text=True)
    ev = work / "reports/evaluate"
    checks.append(("evaluate resumes and writes the report", r.returncode == 0 and (ev / "EVALUATE.md").exists()
                   and (ev / "unseen_run.json").exists() and (ev / "explain.json").exists()
                   and len(list(ev.glob("test_*.json"))) == 5, r.stderr[-800:] or r.stdout[-400:]))
    tm = json.loads(next((work / "data").glob("dataset-*/test.json")).read_text()) if \
        list((work / "data").glob("dataset-*/test.json")) else {}
    lab = _read_csv(work / "reports/prepare/rows_by_label.csv")
    want = float(lab["test"].sum()) if "test" in lab else -1
    checks.append(("evaluate: every test packet scored exactly once", abs(tm.get("packets", -2) - want) < 0.5,
                   f"{tm.get('packets')} vs {want}"))
    r = run(["status"])
    checks.append(("status command", r.returncode == 0 and "xgb_full" in r.stdout, r.stderr[-300:]))
    ok = all(c[1] for c in checks)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + ("" if passed else f"  -> {detail}"))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    if not keep and ok:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok

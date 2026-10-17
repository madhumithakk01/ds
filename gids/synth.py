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


def _device_csv(dev: str, proto: str, n: int, rng, malformed: int = 0) -> Tuple[str, Counter]:
    dev_ip = f"10.0.1.{int(rng.integers(2, 250))}"
    rows, labels = [], Counter()
    n_benign = int(n * 0.6)
    t = T0
    for _ in range(n_benign):
        t += float(rng.exponential(0.5))
        rows.append(_row(rng, t, proto, dev_ip, "Normal"))
    t = T0 - 3600.0  # attack capture recorded earlier -> time jumps back at the join
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
                text, labels = _device_csv(dev, proto, n, rng, malformed=bad)
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
    r = run(["status"])
    checks.append(("status command", r.returncode == 0, r.stderr[-300:]))
    ok = all(c[1] for c in checks)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + ("" if passed else f"  -> {detail}"))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    if not keep and ok:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok

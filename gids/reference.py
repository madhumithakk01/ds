"""Step 6, E1: the published evaluation protocol, reproduced on the same packets.

The scores published for this dataset (macro-F1 0.97-0.99 for tree models) rest on the dataset
authors' preparation notebook: the packets of every file are split at random (stratified,
``random_state=42``) into 60 % train, 20 % validation and 20 % test, and the model inputs include the
packet's absolute arrival time (``timestamp``), ``ip.ttl`` and the IP/TCP checksums.  ``reference``
measures how much of such a score comes from the protocol rather than from recognising attacks:

1. *stream* (resumable, one unit per CSV, checkpointed inside a file): the processed CSVs are read once
   more from the zip and the authors' preprocessing is applied - protocol and port extraction, the
   port groups, hex checksums and TCP options as integers, missing values, label grouping, unpacked
   flag bits and the one-hot encoding (70 inputs).  Packets are sampled per raw label (each kept row
   stands for 1/p packets) and also tagged with the time-blocked split of step 1;
2. *ours*: the prepared rows of step 2 (our features) are sampled per class and split at random packet
   by packet: a row that stands for n identical packets sends a multinomial share to every part;
3. protocols P1-P9 (inputs x split) are trained with the recipe of step 3 (class-balanced weights,
   early stopping on validation) and scored on packets.  All protocols use the authors' six classes
   (``Reporting`` counts as Infection, as in their grouping).

Results: ``reports/paper/reference/<protocol>_<model>.json`` and ``reference.csv``.
"""
from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import layout
from .config import get_path
from .explore import TIME_COLS, BlockParser
from .ingest import _read_parquet, _to_parquet_bytes
from .labels import LabelMapper
from .parse import clean_label, detect_label_column, device_from_member, normalize_names, parse_numeric, parse_time
from .split import assign, split_name
from .state import (Deadline, GracefulStop, Ledger, atomic_write_bytes, atomic_write_json, crash_point, ensure_dir,
                    output_ok, read_json, retry, stable_hash)
from .zipsrc import MemberStream, iter_blocks, read_header, skip_bytes

STAGE = "reference"
REF_VERSION = 2

# Port groups of the dataset authors' notebook (PORT_HIERARCHY_MAP_IOT in "GothamDataset2025 -- Data
# Preparation.ipynb", github.com/othmbela/gotham-network-packet-labeller, Apache License 2.0).
# The first group that contains a port wins.
PORT_MAP: List[Tuple[Any, str]] = [
    ([1883, 8883], "mqttPorts"),
    ([5683, 5684], "coapPorts"),
    ([8554, 8322, 8000, 8001, 8002, 8003, 1935, 8888], "rtspPorts"),
    ([80, 280, 443, 591, 593, 777, 488, 1183, 1184, 2069, 2301, 2381, 8008, 8080], "httpPorts"),
    ([24, 25, 50, 58, 61, 109, 110, 143, 158, 174, 209, 220, 406, 512, 585, 993, 995], "mailPorts"),
    ([42, 53, 81, 101, 105, 261], "dnsPorts"),
    ([20, 21, 47, 69, 115, 152, 189, 349, 574, 662, 989, 990], "ftpPorts"),
    ([22, 23, 59, 87, 89, 107, 211, 221, 222, 513, 614, 759, 992], "shellPorts"),
    ([512, 514], "remoteExecPorts"),
    ([13, 56, 113, 316, 353, 370, 749, 750], "authPorts"),
    ([229, 464, 586, 774], "passwordPorts"),
    ([114, 119, 532, 563], "newsPorts"),
    ([194, 258, 531, 994], "chatPorts"),
    ([35, 92, 170, 515, 631], "printPorts"),
    ([13, 37, 52, 123, 519, 525], "timePorts"),
    ([65, 66, 118, 150, 156, 217], "dbmsPorts"),
    ([546, 547, 647, 847], "dhcpPorts"),
    ([43, 63], "whoisPorts"),
    (range(137, 139 + 1), "netbiosPorts"),
    ([88, 748, 750], "kerberosPorts"),
    ([111, 121, 369, 530, 567, 593, 602], "RPCPorts"),
    ([161, 162, 391], "snmpPorts"),
    (range(0, 1024), "PRIVILEGED_PORTS"),
    (range(1024, 65536), "NONPRIVILEGED_PORTS"),
]
PORT_CATS = sorted([name for _, name in PORT_MAP] + [""])   # the one-hot encoder's order; "" = no port
PROTOS = ["ICMP", "TCP", "UDP"]                               # same; the first category of each is dropped
# group_labels of the authors' code ("Benign" in the repository, "Normal" in the notebook)
REF_GROUP = {"Benign": "Normal", "Normal": "Normal", "TCP Scan": "Network Scanning", "UDP Scan": "Network Scanning",
             "Telnet Brute Force": "Brute Force", "Reporting": "Infection", "Ingress Tool Transfer": "Infection",
             "File Download": "Infection", "CoAP Amplification": "DoS", "Merlin TCP Flooding": "DoS",
             "Merlin UDP Flooding": "DoS", "Merlin ICMP Flooding": "DoS",
             "Merlin C&C Communication": "C&C Communication", "Mirai TCP Flooding": "DoS",
             "Mirai UDP Flooding": "DoS", "Mirai GRE Flooding": "DoS",
             "Mirai C&C Communication": "C&C Communication"}
REF_CLASSES = ["Normal", "Brute Force", "C&C Communication", "DoS", "Infection", "Network Scanning"]
# numeric inputs of the reference notebook -> column in the sampled rows
REF_NUMERIC = [("timestamp", "ts"), ("frame.len", "frame_len"), ("ip.ttl", "ip_ttl"), ("ip.checksum", "ip_checksum"),
               ("tcp.window_size_value", "tcp_window_size_value"),
               ("tcp.window_size_scalefactor", "tcp_window_size_scalefactor"), ("tcp.checksum", "tcp_checksum"),
               ("tcp.options", "tcp_options"), ("tcp.pdu.size", "tcp_pdu_size")]
# protocol: (inputs, split, description)
PROTOCOLS = {
    "P1": ("reference", "random", "reference inputs (70, as published), random split"),
    "P2": ("reference_no_time", "random", "reference inputs without the timestamp, random split"),
    "P3": ("time_only", "random", "the timestamp alone, random split"),
    "P4": ("reference", "temporal", "reference inputs (70), time-blocked split"),
    "P5": ("reference_no_time", "temporal", "reference inputs without the timestamp, time-blocked split"),
    "P6": ("portable", "random", "our portable features, random split"),
    "P7": ("full", "random", "our full features, random split"),
    "P8": ("portable", "temporal", "our portable features, time-blocked split"),
    "P9": ("full", "temporal", "our full features, time-blocked split"),
}
MODEL_TAG = {"xgboost": "xgb", "random_forest": "rf"}


def reference_spec(cfg) -> Dict[str, Any]:
    r = get_path(cfg, "paper.reference", {}) or {}
    return {"v": REF_VERSION, "ingest": stable_hash(layout.ingest_spec(cfg)), "split": split_name(cfg),
            "max_rows_per_label": int(r.get("max_rows_per_label", 250_000)), "seed": int(r.get("seed", 42)),
            "fractions": [float(x) for x in r.get("fractions", [0.6, 0.2, 0.2])]}


def reference_dir(cfg) -> Path:
    return ensure_dir(layout.work(cfg) / "data" / f"reference-{stable_hash(reference_spec(cfg))}")


def _rep(cfg) -> Path:
    return layout.base_reports(cfg, "paper", "reference")


def ref_class(raw: str, mapper: LabelMapper) -> Tuple[int, str]:
    """Class index in REF_CLASSES (-1 = left out) and why."""
    r = re.sub(r"^C&C Communication", "Mirai C&C Communication", str(raw))   # rename_labels of the reference
    if r in REF_GROUP:
        return REF_CLASSES.index(REF_GROUP[r]), "reference grouping"
    if r.lower() == "unknown":
        return -1, "left out (Unknown, as in the reference)"
    c = mapper.map_one(raw)
    if c in REF_CLASSES:
        return REF_CLASSES.index(c), "not in the reference grouping: our rules"
    return -1, "not in the reference grouping: left out"


# =====================================================================
# the reference preprocessing, vectorised

def _port_lut() -> np.ndarray:
    lut = np.full(65536, PORT_CATS.index(""), dtype=np.int8)
    taken = np.zeros(65536, dtype=bool)
    for ports, name in PORT_MAP:
        idx = np.array(list(ports), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < 65536)]
        new = idx[~taken[idx]]
        lut[new] = PORT_CATS.index(name)
        taken[new] = True
    return lut


_PORT_LUT = _port_lut()


def port_category(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    out = np.full(len(p), PORT_CATS.index(""), dtype=np.int8)
    ok = np.isfinite(p) & (p >= 0) & (p < 65536) & (p == np.floor(p))
    out[ok] = _PORT_LUT[p[ok].astype(np.int64)]
    return out


def _hex_one(u, missing: float) -> float:
    v = str(u).strip()
    if "," in v:
        v = v.split(",", 1)[0].strip()
    v = v.replace(":", "")
    if not v or v.lower() in ("nan", "none", "null", "-"):
        return missing
    try:
        return float(int(v, 16))
    except (ValueError, OverflowError):
        return np.nan


def hex_values(s: Optional[pd.Series], n: int, missing: float) -> np.ndarray:
    """``int(x, 16)`` of every value as float64 (first of repeated values); empty -> ``missing``."""
    if s is None:
        return np.full(n, missing)
    codes, uniq = pd.factorize(s, use_na_sentinel=True)
    vals = np.array([_hex_one(u, missing) for u in uniq], dtype=np.float64)
    out = np.full(n, missing)
    ok = codes >= 0
    out[ok] = vals[codes[ok]]
    return out


def reference_features(sub: pd.DataFrame, time_col: Optional[str], hint: Optional[str]):
    """The reference notebook's per-packet fields (before one-hot encoding and scaling)."""
    n = len(sub)

    def col(name):
        return sub[name] if name in sub.columns else None

    def num(name, missing):
        c = col(name)
        if c is None:
            return np.full(n, missing)
        v, _ = parse_numeric(c)
        return np.where(np.isnan(v), missing, v)

    if time_col:
        t, _, hint = parse_time(sub[time_col], hint)
    else:
        t = np.full(n, np.nan)
    out: Dict[str, Any] = {"t": t, "ts": np.floor(t)}   # convert_time: whole seconds
    for nm in ("frame_len", "ip_ttl", "tcp_window_size_value", "tcp_window_size_scalefactor", "tcp_pdu_size"):
        out[nm] = num(nm, -1.0)                           # fill_missing_values: -1
    for nm in ("ip_checksum", "tcp_checksum", "tcp_options"):
        out[nm] = hex_values(col(nm), n, 0.0)             # convert_checksums: int(x, 16), missing 0
    for nm in ("ip_flags", "tcp_flags"):                  # unpack_flags reads the low byte
        v = np.nan_to_num(hex_values(col(nm), n, 0.0), nan=0.0, posinf=0.0)
        out[nm] = (v.astype(np.int64) & 0xFF).astype(np.uint8)
    # extract_protocols_and_ports: first match of ":tcp", ":udp", ":icmp" in frame.protocols
    fp = col("frame_protocols")
    fps = fp if fp is not None else pd.Series([""] * n)

    def has(s):
        return fps.str.contains(s, regex=False).fillna(False).to_numpy(dtype=bool)

    is_tcp = has(":tcp")
    is_udp = ~is_tcp & has(":udp")
    is_icmp = ~is_tcp & ~is_udp & has(":icmp")
    proto = np.select([is_tcp, is_udp, is_icmp], [1, 2, 0], -1).astype(np.int64)
    nan = np.nan
    sport = np.where(is_tcp, num("tcp_srcport", nan), np.where(is_udp, num("udp_srcport", nan), nan))
    dport = np.where(is_tcp, num("tcp_dstport", nan), np.where(is_udp, num("udp_dstport", nan), nan))
    matched = proto >= 0
    if not matched.all():  # the reference loop keeps the previous packet's values (its variables are not reset)
        prev = np.maximum.accumulate(np.where(matched, np.arange(n), -1))
        um = ~matched & (prev >= 0)
        proto[um], sport[um], dport[um] = proto[prev[um]], sport[prev[um]], dport[prev[um]]
        proto[~matched & (prev < 0)] = 0
    out["proto"] = proto.astype(np.int8)
    out["sport_cat"] = port_category(sport)
    out["dport_cat"] = port_category(dport)
    return pd.DataFrame(out), hint


def reference_input_names(which: str) -> List[str]:
    if which == "time_only":
        return ["timestamp"]
    names = [f"ip.flags bit{b}" for b in range(3)] + [f"tcp.flags bit{b}" for b in range(8)]
    names += [f"ip.protocol={p}" for p in PROTOS[1:]]
    names += [f"src.port={c}" for c in PORT_CATS[1:]] + [f"dst.port={c}" for c in PORT_CATS[1:]]
    return names + [nm for nm, _ in REF_NUMERIC if not (which == "reference_no_time" and nm == "timestamp")]


def numeric_stats(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """StandardScaler of the reference, fitted on the training rows (one scaler for all files)."""
    out = {}
    for nm, c in REF_NUMERIC:
        x = df[c].to_numpy(np.float64)
        x = x[np.isfinite(x)]
        out[nm] = (float(x.mean()) if len(x) else 0.0, float(x.std()) if len(x) else 1.0)
    return out


def reference_inputs(df: pd.DataFrame, which: str, stats: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Model inputs in the order of the reference ColumnTransformer: flag bits, one-hot categories, scaled numbers.

    Numbers are scaled in float64 before the float32 cast, which keeps the timestamp's one-second resolution.
    """
    cols = []
    if which != "time_only":
        ipf, tcpf = df["ip_flags"].to_numpy().astype(np.uint8), df["tcp_flags"].to_numpy().astype(np.uint8)
        cols += [((ipf >> b) & 1).astype(np.float32) for b in range(3)]
        cols += [((tcpf >> b) & 1).astype(np.float32) for b in range(8)]
        pr = df["proto"].to_numpy()
        cols += [(pr == k).astype(np.float32) for k in range(1, len(PROTOS))]
        for c in ("sport_cat", "dport_cat"):
            v = df[c].to_numpy()
            cols += [(v == k).astype(np.float32) for k in range(1, len(PORT_CATS))]
    for nm, c in REF_NUMERIC:
        if (which == "time_only" and nm != "timestamp") or (which == "reference_no_time" and nm == "timestamp"):
            continue
        mu, sd = stats[nm]
        z = (df[c].to_numpy(np.float64) - mu) / (sd if sd > 0 else 1.0)
        cols.append(np.clip(z, -1e38, 1e38).astype(np.float32))   # TCP options as integers reach 1e96
    return np.column_stack(cols) if cols else np.zeros((len(df), 0), np.float32)


# =====================================================================
# 1. stream the CSVs

def _members(cfg) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Inventory entries of the CSVs that step 1 ingested, and step 1's metadata of each."""
    from .relabel import _ingest_metas
    from .zipsrc import build_inventory

    _, metas = _ingest_metas(cfg)
    inv = read_json(layout.base_reports(cfg, "explore") / "inventory.json") or build_inventory(cfg)
    by_id = {m["id"]: m for m in inv["members"]}
    missing = [m["member"] for m in metas if m["id"] not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} ingested file(s) are not in the inventory, e.g. {missing[0]}")
    return [by_id[m["id"]] for m in metas], {m["id"]: m for m in metas}


def row_uniform(key: int, start: int, n: int) -> np.ndarray:
    """Uniform numbers in [0, 1) for rows start ... start + n - 1 of a file (splitmix64 of the row number).

    A row's number does not depend on how the file is cut into blocks, so the sample is the same whether a run
    was interrupted or not.
    """
    with np.errstate(over="ignore"):
        z = np.uint64(key) + (np.arange(start, start + n, dtype=np.uint64) + np.uint64(1)) * \
            np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def stream_member(m: Dict[str, Any], cfg, tracer, deadline: Deadline, out: Path, probs: Dict[str, float],
                  classes: Dict[str, int], table: pd.DataFrame, progress) -> Dict[str, Any]:
    spec = reference_spec(cfg)
    ing = cfg.get("ingest", {})
    block = int(float(ing.get("block_mb", 32)) * 2**20)
    part_bytes = int(float(ing.get("part_mb", 128)) * 2**20)
    fid = stable_hash(m["id"], 16)
    sdir, pdir = ensure_dir(out / "_state"), ensure_dir(out / "parts")
    spath = sdir / f"{fid}.json"
    _, dtype = device_from_member(m["name"])
    st = read_json(spath)
    if st and st.get("member") == m["id"] and all(output_ok(pdir / p) for p in st["parts"]):
        tracer.info(STAGE, "resume", f"{Path(m['name']).name}: continuing at byte {st['offset']:,}")
    else:
        st = {"member": m["id"], "offset": None, "parts": [], "rows": 0, "bad": 0, "seen": {}, "kept": {},
              "hint": None, "seconds": 0.0}
    seen, kept = pd.Series(st["seen"], dtype=np.int64), pd.Series(st["kept"], dtype=np.int64)
    with MemberStream(m, cfg) as fh:
        header, hdr_len = read_header(fh)
        htxt = header.decode("utf-8", errors="replace").strip("\r\n")
        delim = "\t" if htxt.count("\t") > htxt.count(",") else ","
        names = normalize_names(next(csv.reader([htxt], delimiter=delim)) if htxt else [])
        label_col = detect_label_column(names, ing.get("label_column", "auto"))
        time_col = next((c for c in TIME_COLS if c in names), None)
        start = st["offset"] or hdr_len
        if start > hdr_len:
            skip_bytes(fh, start - hdr_len)
        parser = BlockParser(names, delim, tracer, prefer=str(get_path(cfg, "explore.parser", "auto")))
        pending: List[pd.DataFrame] = []
        pending_from, t0 = start, time.time()

        def flush(end: int) -> None:
            nonlocal pending, pending_from, t0
            df = pd.concat(pending, ignore_index=True) if pending else pd.DataFrame(columns=["label", "t"])
            sp, ep = assign(df["label"], df["t"].to_numpy(np.float64), table, with_episode=True)
            df["split_t"], df["episode"] = sp.astype(np.int8), ep
            df["device_type"] = dtype
            name = f"{fid}-{len(st['parts']):05d}.parquet"
            atomic_write_bytes(pdir / name, _to_parquet_bytes(df))
            st["parts"].append(name)
            st.update(offset=end, seen=seen.to_dict(), kept=kept.to_dict(),
                      seconds=st["seconds"] + time.time() - t0)
            atomic_write_json(spath, st)
            pending, pending_from, t0 = [], end, time.time()
            crash_point("reference_part")

        end = start
        key = int(stable_hash([spec["seed"], m["id"]], 16), 16)
        for s, end, blk in iter_blocks(fh, start, block):
            deadline.check()
            raw, n_lines = parser.parse(blk)
            del blk
            n = len(raw)
            u = row_uniform(key, st["rows"], n)
            st["rows"] += n
            st["bad"] += max(n_lines - n, 0)
            lab = clean_label(raw[label_col]) if label_col else pd.Series(["UNKNOWN"] * n)
            codes, uniq = pd.factorize(lab)
            uniq = [str(u) for u in uniq]
            cls_u = np.array([classes.get(u, -1) for u in uniq] + [-1], dtype=np.int64)
            p_u = np.array([probs.get(u, 1.0) for u in uniq] + [0.0])
            codes = np.where(codes < 0, len(uniq), codes)
            keep = (cls_u[codes] >= 0) & (u < p_u[codes])
            seen = seen.add(pd.Series(np.bincount(codes, minlength=len(uniq) + 1)[:len(uniq)], index=uniq),
                            fill_value=0).astype(np.int64)
            kept = kept.add(pd.Series(np.bincount(codes[keep], minlength=len(uniq) + 1)[:len(uniq)], index=uniq),
                            fill_value=0).astype(np.int64)
            if keep.any():
                sub = raw[keep].reset_index(drop=True)
                df, st["hint"] = reference_features(sub, time_col, st["hint"])
                df["label"] = np.asarray(lab, dtype=object)[keep]
                df["ref_cls"] = cls_u[codes[keep]].astype(np.int8)
                df["w"] = 1.0 / p_u[codes[keep]]
                pending.append(df)
            del raw
            progress(end)
            if end - pending_from >= part_bytes:
                flush(end)
        if pending or not st["parts"] or end > pending_from:
            flush(end)
    return {"member": m["name"], "id": m["id"], "device_type": dtype, "parts": st["parts"], "rows": st["rows"],
            "bad": st["bad"], "seen": st["seen"], "kept": st["kept"], "seconds": round(st["seconds"], 1)}


def run_stream(cfg, tracer, deadline: Deadline) -> pd.DataFrame:
    out = reference_dir(cfg)
    rows_path = out / "rows.parquet"
    if output_ok(rows_path) and read_json(out / "rows.json"):
        return _read_parquet(rows_path)
    spec = reference_spec(cfg)
    atomic_write_json(out / "_spec.json", spec)
    members, metas = _members(cfg)
    sp = read_json(layout.ingest_dir(cfg) / f"{split_name(cfg)}.json")
    if not sp:
        raise RuntimeError("No split table from step 1 for the current split settings; run `python -m gids ingest`.")
    table = pd.DataFrame(sp["blocks"])
    counts: Dict[str, int] = {}
    for meta in metas.values():
        for lab, v in (meta.get("labels") or {}).items():
            counts[lab] = counts.get(lab, 0) + int(v)
    mapper = LabelMapper(cfg.get("labels", {}))
    grouping = {lab: ref_class(lab, mapper) for lab in counts}
    classes = {lab: c for lab, (c, _) in grouping.items()}
    cap = spec["max_rows_per_label"]
    probs = {lab: min(1.0, cap / max(v, 1)) for lab, v in counts.items()}
    ledger = Ledger(out / "_ledger.jsonl")
    done = ledger.done()
    members = sorted(members, key=lambda m: (m["size"], m["id"]))

    def finished(m):
        r = done.get(m["id"])
        return bool(r) and output_ok(r.get("meta", "")) and all(
            output_ok(out / "parts" / p) for p in (read_json(r["meta"]) or {}).get("parts", ["?"]))

    todo = [m for m in members if not finished(m)]
    total = sum(m["size"] for m in members)
    done_bytes = total - sum(m["size"] for m in todo)
    tracer.info(STAGE, "plan", f"{len(members)} CSV files ({total / 2**30:.2f} GB) -> {out}; "
                f"{len(members) - len(todo)} already read, {len(todo)} to go; at most {cap:,} packets per raw label")
    t_start, read_bytes = time.time(), 0
    for k, m in enumerate(todo):
        deadline.check()

        def progress(pos, m=m):
            rate = (read_bytes + pos) / max(time.time() - t_start, 1e-6)
            left = total - done_bytes - read_bytes - pos
            tracer.progress(STAGE, len(members) - len(todo) + k, len(members), "files", every_s=30,
                            current=Path(m["name"]).name, file_pct=f"{100 * pos / max(m['size'], 1):.0f}%",
                            mb_s=f"{rate / 2**20:.1f}", eta_min=f"{left / max(rate, 1) / 60:.1f}")

        ledger.append(m["id"], "start")
        try:
            info = retry(lambda: stream_member(m, cfg, tracer, deadline, out, probs, classes, table, progress),
                         tries=3, base_delay=10.0,
                         on_error=lambda a, e: tracer.warn(STAGE, "io_retry", f"attempt {a}: {e}", member=m["name"]))
        except (GracefulStop, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001
            tracer.exception(STAGE, "file_failed", exc, member=m["name"])
            ledger.append(m["id"], "failed", error=f"{type(exc).__name__}: {exc}")
            continue
        mpath = out / f"{stable_hash(m['id'], 16)}.json"
        atomic_write_json(mpath, info)
        read_bytes += m["size"]
        ledger.append(m["id"], "done", meta=str(mpath))
        tracer.info(STAGE, "file", f"{Path(m['name']).name}: {info['rows']:,} packets, "
                    f"{sum(info['kept'].values()):,} sampled, {info['seconds']} s")
        crash_point(STAGE)
    failed = [m["name"] for m in members if ledger.latest().get(m["id"], {}).get("status") == "failed"]
    if failed:
        raise RuntimeError(f"{len(failed)} file(s) failed; run the reference part again")
    infos = [read_json(out / f"{stable_hash(m['id'], 16)}.json") for m in sorted(members, key=lambda m: m["id"])]
    frames = [_read_parquet(out / "parts" / p) for i in infos for p in i["parts"]]
    rows = pd.concat([f for f in frames if len(f)], ignore_index=True)
    rows["split_r"] = random_split(rows["label"], spec["seed"], spec["fractions"])
    atomic_write_bytes(rows_path, _to_parquet_bytes(rows))
    seen = pd.DataFrame([i["seen"] for i in infos]).sum().astype(np.int64)
    kept = pd.DataFrame([i["kept"] for i in infos]).sum().astype(np.int64)
    labels = [{"label": lab, "packets": int(seen.get(lab, 0)), "sampled": int(kept.get(lab, 0)),
               "keep_probability": probs.get(lab, 1.0),
               "class": REF_CLASSES[grouping[lab][0]] if grouping.get(lab, (-1,))[0] >= 0 else None,
               "grouping": grouping.get(lab, (-1, "?"))[1]} for lab in sorted(seen.index, key=lambda x: -seen[x])]
    atomic_write_json(out / "rows.json", {"spec": spec, "rows": int(len(rows)), "packets": int(seen.sum()),
                                          "labels": labels, "files": len(infos),
                                          "seconds": round(sum(i["seconds"] for i in infos), 1)})
    tracer.info(STAGE, "rows", f"{len(rows):,} sampled packets from {int(seen.sum()):,}")
    return rows


def random_split(labels: pd.Series, seed: int, fractions: List[float]) -> np.ndarray:
    """Stratified random split per raw label (train_test_split(..., stratify=labels, random_state=seed))."""
    codes = pd.Categorical(labels).codes.astype(np.int64)
    n = len(codes)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    order = perm[np.argsort(codes[perm], kind="stable")]
    cs = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1] if n else np.zeros(0, np.int64)
    size = np.diff(np.r_[starts, n])
    rank = np.arange(n) - np.repeat(starts, size)
    g = np.repeat(size, size).astype(np.float64)
    a, b = np.round(fractions[0] * g), np.round((fractions[0] + fractions[1]) * g)
    out = np.empty(n, dtype=np.int8)
    out[order] = np.where(rank < a, 0, np.where(rank < b, 1, 2))
    return out


# =====================================================================
# 2. our features, split at random packet by packet

def ours_spec(cfg) -> Dict[str, Any]:
    from .prepare import prepare_spec

    r = get_path(cfg, "paper.reference", {}) or {}
    return {"v": REF_VERSION, "prepare": stable_hash(prepare_spec(cfg)),
            "cap": int(r.get("ours_max_rows_per_class", 400_000)), "seed": reference_spec(cfg)["seed"],
            "fractions": reference_spec(cfg)["fractions"]}


def build_ours(cfg, tracer, deadline: Deadline) -> pd.DataFrame:
    from .train import _prepare_infos, _rows_path

    spec = ours_spec(cfg)
    path = reference_dir(cfg) / f"ours-{stable_hash(spec)}.parquet"
    if output_ok(path):
        return _read_parquet(path)
    pdir, infos, voc = _prepare_infos(cfg)
    cap = spec["cap"]
    mapper = LabelMapper(cfg.get("labels", {}))
    lab2ref = np.array([ref_class(lab, mapper)[0] for lab in voc["labels"]] + [-1], dtype=np.int64)
    unique = np.zeros(len(REF_CLASSES))
    for info in infos:
        for a in info.get("accounting", []):
            c = lab2ref[int(a["label"])]
            if c >= 0:
                unique[c] += int(a["size"])
    p_c = np.minimum(1.0, cap / np.maximum(unique, 1))
    frames = []
    with tracer.span(STAGE, "ours"):
        for fi, info in enumerate(infos):
            deadline.check()
            df = _read_parquet(_rows_path(pdir, info))
            c = lab2ref[df["label"].to_numpy(np.int64)]
            rng = np.random.default_rng([spec["seed"], fi])
            keep = (c >= 0) & (rng.random(len(df)) < p_c[np.maximum(c, 0)])
            df = df[keep].reset_index(drop=True)
            n = rng.multinomial(df["weight"].to_numpy(np.int64), spec["fractions"]) if len(df) else np.zeros((0, 3))
            df["n_tr"], df["n_va"], df["n_te"] = (n[:, k].astype(np.int64) for k in range(3))
            df["ref_cls"] = c[keep].astype(np.int8)
            df["p"] = p_c[c[keep]]
            df["device_type"] = df["device_type"].astype(str)
            frames.append(df)
            tracer.progress(STAGE, fi + 1, len(infos), "files (our features)", every_s=30)
    ours = pd.concat(frames, ignore_index=True)
    atomic_write_bytes(path, _to_parquet_bytes(ours))
    atomic_write_json(path.with_suffix(".json"), {"spec": spec, "rows": int(len(ours)),
                                                  "unique_rows_per_class": dict(zip(REF_CLASSES, unique.tolist())),
                                                  "keep_probability": dict(zip(REF_CLASSES, p_c.tolist()))})
    return ours


# =====================================================================
# 3. protocols

def _parts(df: pd.DataFrame, split: str, ours: bool) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Row indices and packets per row for train (0), validation (1) and test (2)."""
    if ours and split == "random":
        p = df["p"].to_numpy(np.float64)
        res = {}
        for k, col in enumerate(("n_tr", "n_va", "n_te")):
            n = df[col].to_numpy(np.float64)
            idx = np.flatnonzero(n > 0)
            res[k] = (idx, n[idx] / p[idx])
        return res
    if ours:
        s, w = df["split"].to_numpy(), df["weight"].to_numpy(np.float64) / df["p"].to_numpy(np.float64)
    else:
        s, w = df["split_r" if split == "random" else "split_t"].to_numpy(), df["w"].to_numpy(np.float64)
    return {k: (np.flatnonzero(s == k), w[s == k]) for k in (0, 1, 2)}


def _params(cfg, model: str) -> Dict[str, Any]:
    p = dict(get_path(cfg, f"train_defaults.{model}", {}) or {})
    p.update(get_path(cfg, f"paper.reference.{model}", {}) or {})
    return p


def run_protocol(cfg, P: str, model: str, ref: Optional[pd.DataFrame], ours: Optional[pd.DataFrame], tracer,
                 deadline: Deadline) -> Dict[str, Any]:
    from .evaluate import _alarm_stats
    from .train import _feature_name, balanced, evaluate, make_matrix, raw_predictor, train_on_arrays

    path = _rep(cfg) / f"{P}_{MODEL_TAG[model]}.json"
    inputs, split, desc = PROTOCOLS[P]
    is_ours = inputs in ("portable", "full")
    params = _params(cfg, model)
    spec = {"step": "reference", "v": REF_VERSION, "protocol": P, "inputs": inputs, "split": split,
            "data": stable_hash(ours_spec(cfg) if is_ours else reference_spec(cfg)), "model": model,
            "params": params, "classes": REF_CLASSES}
    assert stable_hash(spec) == protocol_spec_hash(cfg, P, model)
    res = read_json(path)
    if res and res.get("spec_hash") == stable_hash(spec):  # results of other settings are computed again
        return res
    df = ours if is_ours else ref
    parts = _parts(df, split, is_ours)
    y_all = df["ref_cls"].to_numpy(np.int64)
    (tr, w_tr), (va, w_va), (te, w_te) = parts[0], parts[1], parts[2]
    if is_ours:
        names = list(get_path(cfg, f"features.sets.{inputs}"))

        def mat(idx):
            return make_matrix(df.iloc[idx], names)
    else:
        names = reference_input_names(inputs)
        stats = numeric_stats(df.iloc[tr])

        def mat(idx):
            return reference_inputs(df.iloc[idx], inputs, stats)
    n_cls = len(REF_CLASSES)
    y, yv = y_all[tr], y_all[va]
    tracer.info(STAGE, "protocol", f"{P} ({desc}), {model}: {len(tr):,} training rows, {len(names)} inputs")
    predict, mdir = train_on_arrays(cfg, f"ref_{P.lower()}_{MODEL_TAG[model]}", spec, model, params, mat(tr), y,
                                    balanced(w_tr, y, n_cls), mat(va), yv, balanced(w_va, yv, n_cls), n_cls,
                                    tracer, deadline)
    yt = y_all[te]
    pred = np.concatenate([np.asarray(predict(mat(te[i:i + 500_000]))).argmax(axis=1)
                           for i in range(0, len(te), 500_000)]) if len(te) else np.zeros(0, np.int64)
    m = evaluate(yt, pred, w_te, REF_CLASSES)
    m.update(_alarm_stats(yt, pred, w_te))
    st = read_json(mdir / "state.json") or {}
    imp = {}
    if model == "xgboost":
        bst = raw_predictor(model, mdir, int(st.get("best_iter", 0)), n_cls)[0]
        g = pd.Series({_feature_name(k, names): v for k, v in bst.get_score(importance_type="total_gain").items()})
        imp = (g / g.sum()).sort_values(ascending=False).head(12).round(4).to_dict() if g.sum() else {}
    m.update({"protocol": P, "description": desc, "inputs": inputs, "n_inputs": len(names), "split": split,
              "model": model, "train_rows": int(len(tr)), "train_packets": float(w_tr.sum()),
              "test_rows": int(len(te)), "test_packets": float(w_te.sum()), "rounds": int(st.get("best_iter", 0)),
              "importance": imp, "model_dir": str(mdir), "spec_hash": stable_hash(spec)})
    atomic_write_json(path, m)
    tracer.info(STAGE, "result", f"{P} {model}: test macro-F1 {m['macro_f1']:.4f}")
    return m


def run_reference(cfg, tracer, deadline: Deadline) -> Dict[str, Any]:
    """E1 end to end; finished protocols are skipped on a re-run."""
    r = get_path(cfg, "paper.reference", {}) or {}
    protos = [p for p in r.get("protocols", list(PROTOCOLS)) if p in PROTOCOLS]
    rf = [p for p in r.get("rf_protocols", ["P1", "P4"]) if p in protos]
    with tracer.span(STAGE, "stream"):
        ref = run_stream(cfg, tracer, deadline)
    need_ours = any(PROTOCOLS[p][0] in ("portable", "full") for p in protos)
    ours = build_ours(cfg, tracer, deadline) if need_ours else None
    results = {}
    for P in protos:
        for model in ["xgboost"] + (["random_forest"] if P in rf else []):
            deadline.check()
            try:
                results[f"{P}_{MODEL_TAG[model]}"] = run_protocol(cfg, P, model, ref, ours, tracer, deadline)
            except (GracefulStop, KeyboardInterrupt):
                raise
            except Exception as exc:  # noqa: BLE001
                tracer.exception(STAGE, "protocol_failed", exc, protocol=P, model=model)
    table = reference_table(cfg)
    table.to_csv(_rep(cfg) / "reference.csv", index=False)
    return results


def protocol_spec_hash(cfg, P: str, model: str) -> str:
    inputs, split, _ = PROTOCOLS[P]
    is_ours = inputs in ("portable", "full")
    return stable_hash({"step": "reference", "v": REF_VERSION, "protocol": P, "inputs": inputs, "split": split,
                        "data": stable_hash(ours_spec(cfg) if is_ours else reference_spec(cfg)), "model": model,
                        "params": _params(cfg, model), "classes": REF_CLASSES})


def reference_table(cfg) -> pd.DataFrame:
    """Finished protocols of the current settings."""
    current = {(P, MODEL_TAG[m]): protocol_spec_hash(cfg, P, m) for P in PROTOCOLS for m in MODEL_TAG}
    rows = []
    for P, (inputs, split, desc) in PROTOCOLS.items():
        for tag in MODEL_TAG.values():
            m = read_json(_rep(cfg) / f"{P}_{tag}.json")
            if not m or m.get("spec_hash") != current.get((P, tag)):
                continue
            row = {"protocol": P, "model": {"xgb": "XGBoost", "rf": "random forest"}[tag], "inputs": inputs,
                   "split": split, "macro_f1": m["macro_f1"], "balanced_acc": m["balanced_accuracy"]}
            row.update({f"F1 {p['class']}": p["f1"] for p in m["per_class"]})
            row["benign flagged /1M"] = m["benign_flagged_per_million"]
            rows.append(row)
    return pd.DataFrame(rows)

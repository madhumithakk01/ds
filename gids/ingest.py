"""Step 1: CSV text -> typed Parquet parts with per-packet context features.

``ingest`` (resumable, one unit per CSV file, checkpointed every
``ingest.part_mb`` of CSV text inside a file)
    parses every processed CSV, turns the tshark text fields into numbers,
    adds context features computed from the packets that came *before* each
    packet in the same capture (rates, inter-arrival times, new destination
    ports), and writes Parquet parts to ``data/ingest-<hash>/``.  When every
    file is done the time-blocked train/val/test split is computed (see
    :mod:`gids.split`) and ``reports/ingest/INGEST.md`` is written.

Context windows only look back, never forward, so a feature never uses a
packet that arrives later - the same information a live detector would have.
Features do not depend on how the file is cut into blocks or parts: the
packets of the last 2 x ``ingest.ctx_long_s`` seconds are carried from one
block to the next and saved with every checkpoint.
"""
from __future__ import annotations

import csv
import io
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import layout
from .config import get_path
from .explore import TIME_COLS, BlockParser, md_table
from .labels import LabelMapper
from .parse import (clean_label, detect_label_column, device_from_member, first_value, mode_ip,
                    normalize_names, parse_numeric, parse_tcp_options, parse_time)
from .split import SPLITS, assign, make_split, split_name, split_spec
from .state import (Deadline, GracefulStop, Ledger, atomic_write_bytes, atomic_write_json, atomic_write_text,
                    crash_point, ensure_dir, output_ok, read_json, retry, stable_hash)
from .zipsrc import MemberStream, build_inventory, iter_blocks, peek_member, read_header, skip_bytes

STAGE = "ingest"
TEXT_KEEP = ("ip_src", "ip_dst", "frame_protocols")
CARRY_COLS = ["t", "seg", "ip_src", "ip_dst", "sport", "dport", "proto", "frame_len"]

# every column a part can hold, with what it means (also written to the report)
FEATURES: Dict[str, str] = {
    "frame_len": "frame length in bytes",
    "ip_flags": "IP flags (hex -> int)",
    "ip_ttl": "IP time-to-live (outer header)",
    "ip_proto": "IP protocol number (6 TCP, 17 UDP, 1 ICMP, 47 GRE)",
    "tcp_srcport": "TCP source port",
    "tcp_dstport": "TCP destination port",
    "tcp_flags": "TCP flags (hex -> int)",
    "tcp_window_size_value": "TCP window size",
    "tcp_window_size_scalefactor": "TCP window scale factor (-1/-2 = unknown to tshark)",
    "tcp_pdu_size": "TCP PDU size reported by tshark",
    "udp_srcport": "UDP source port",
    "udp_dstport": "UDP destination port",
    "tcp_options_len": "length of the TCP options in bytes",
    "tcp_options_kinds": "bitmask of the TCP option kinds present",
    "inner_header": "1 when fields repeat (ICMP error quoting an inner IP header)",
    "proto_layers": "number of layers in frame.protocols",
    "proto_gre": "1 when the packet is GRE-tunnelled",
    "dir_out": "1 when the packet is sent by the device itself",
    "dir_in": "1 when the packet is addressed to the device",
    "ctx_all_pkts_1s": "packets in this capture in the last 1 s (this one included)",
    "ctx_src_pkts_1s": "packets from the same source IP in the last 1 s",
    "ctx_src_pkts_10s": "packets from the same source IP in the last 10 s",
    "ctx_src_bytes_1s": "bytes from the same source IP in the last 1 s",
    "ctx_dst_pkts_1s": "packets to the same destination IP in the last 1 s",
    "ctx_pair_pkts_1s": "packets between the same source and destination IP in the last 1 s",
    "ctx_pair_iat": "seconds since the previous packet between the same IPs (capped at 10 s)",
    "ctx_flow_pkts_10s": "packets of the same 5-tuple flow in the last 10 s",
    "ctx_flow_iat": "seconds since the previous packet of the same 5-tuple flow (capped at 10 s)",
    "ctx_src_new_dports_10s": "destination ports this source started contacting in the last 10 s",
}


def _out(cfg) -> Path:
    return layout.reports(cfg, "ingest")


def _ctx_names(cfg) -> Dict[str, float]:
    return {"short": float(get_path(cfg, "ingest.ctx_short_s", 1.0)),
            "long": float(get_path(cfg, "ingest.ctx_long_s", 10.0))}


# =====================================================================
# typing one parsed block

def type_block(df: pd.DataFrame, label_col: Optional[str], time_col: Optional[str], drop: set,
               hint: Optional[str], fails: Counter) -> Tuple[pd.DataFrame, Optional[str]]:
    n = len(df)
    out: Dict[str, Any] = {}
    if time_col:
        tv, nf, hint = parse_time(df[time_col], hint)
        fails["frame_time"] += nf
    else:
        tv = np.full(n, np.nan)
    out["t"] = tv
    out["label_raw"] = clean_label(df[label_col]).to_numpy(dtype=object) if label_col else np.full(n, "UNKNOWN",
                                                                                                    dtype=object)
    for c in TEXT_KEEP:
        if c in df.columns and c not in drop:
            out[c] = first_value(df[c]).to_numpy(dtype=object)
    for c in df.columns:
        if c in TIME_COLS or c == label_col or c in TEXT_KEEP or c in drop:
            continue
        if c == "tcp_options":
            ln, mask, nf = parse_tcp_options(df[c])
            out["tcp_options_len"], out["tcp_options_kinds"] = ln, mask
        else:
            vals, nf = parse_numeric(df[c])
            out[c] = vals
        fails[c] += nf
    if "ip_proto" in df.columns:
        out["inner_header"] = df["ip_proto"].str.contains(",", regex=False).to_numpy(
            dtype=np.float64, na_value=0.0)
    if "frame_protocols" in df.columns:
        fp = df["frame_protocols"]
        out["proto_layers"] = (fp.str.count(":") + 1).to_numpy(dtype=np.float64, na_value=np.nan)
        out["proto_gre"] = fp.str.contains(":gre", regex=False).to_numpy(dtype=np.float64, na_value=0.0)
    return pd.DataFrame(out), hint


# =====================================================================
# context features (vectorised sliding windows over packets of the same key)

def _group(*cols) -> np.ndarray:
    return pd.DataFrame({i: c for i, c in enumerate(cols)}).groupby(
        list(range(len(cols))), sort=False, dropna=False).ngroup().to_numpy(np.int64)


class _Windows:
    """Packets sorted by (key, time); counts/sums over the last w microseconds per key."""

    def __init__(self, key: np.ndarray, t_us: np.ndarray):
        self.order = np.lexsort((t_us, key))
        self.k = key[self.order]
        self.t = t_us[self.order]
        span = int(self.t.max()) + 1 if len(self.t) else 1
        if (int(self.k.max()) + 1 if len(self.k) else 1) > (2**62) // span:
            raise ValueError("too many keys x time span for one window chunk; lower ingest.max_carry_rows")
        self.comp = self.k * span + self.t
        self.pos = np.arange(len(key))

    def _back(self, sorted_vals: np.ndarray) -> np.ndarray:
        out = np.empty_like(sorted_vals)
        out[self.order] = sorted_vals
        return out

    def _lower(self, w_us: int) -> np.ndarray:
        return np.searchsorted(self.comp, self.comp - w_us, side="left")

    def count(self, w_us: int) -> np.ndarray:
        return self._back((self.pos - self._lower(w_us) + 1).astype(np.float64))

    def total(self, w_us: int, vals: np.ndarray) -> np.ndarray:
        cs = np.r_[0.0, np.cumsum(vals[self.order])]
        return self._back(cs[self.pos + 1] - cs[self._lower(w_us)])

    def iat(self, cap_s: float) -> np.ndarray:
        d = np.full(len(self.k), cap_s)
        if len(self.k) > 1:
            same = self.k[1:] == self.k[:-1]
            d[1:] = np.where(same, np.minimum((self.t[1:] - self.t[:-1]) / 1e6, cap_s), cap_s)
        return self._back(d)


def context_features(c: pd.DataFrame, short_s: float, long_s: float) -> Dict[str, np.ndarray]:
    """Features for every row of ``c`` (columns CARRY_COLS) from rows at or before it."""
    n = len(c)
    if n == 0:
        return {k: np.empty(0) for k in FEATURES if k.startswith("ctx_")}
    w_s, w_l = int(round(short_s * 1e6)), int(round(long_s * 1e6))
    t = c["t"].to_numpy(dtype=np.float64)
    base = float(np.nanmin(t)) if np.isfinite(t).any() else 0.0
    t_us = np.round((np.nan_to_num(t, nan=base) - base) * 1e6).astype(np.int64) + w_l
    seg = c["seg"].to_numpy(np.int64)
    src = pd.factorize(c["ip_src"], use_na_sentinel=False)[0]
    dst = pd.factorize(c["ip_dst"], use_na_sentinel=False)[0]
    sport = c["sport"].to_numpy(np.float64)
    dport = c["dport"].to_numpy(np.float64)
    proto = c["proto"].to_numpy(np.float64)
    flen = np.nan_to_num(c["frame_len"].to_numpy(np.float64))
    out: Dict[str, np.ndarray] = {}
    out["ctx_all_pkts_1s"] = _Windows(seg, t_us).count(w_s)
    w = _Windows(_group(seg, src), t_us)
    out["ctx_src_pkts_1s"] = w.count(w_s)
    out["ctx_src_pkts_10s"] = w.count(w_l)
    out["ctx_src_bytes_1s"] = w.total(w_s, flen)
    out["ctx_dst_pkts_1s"] = _Windows(_group(seg, dst), t_us).count(w_s)
    w = _Windows(_group(seg, src, dst), t_us)
    out["ctx_pair_pkts_1s"] = w.count(w_s)
    out["ctx_pair_iat"] = w.iat(long_s)
    w = _Windows(_group(seg, src, dst, sport, dport, proto), t_us)
    out["ctx_flow_pkts_10s"] = w.count(w_l)
    out["ctx_flow_iat"] = w.iat(long_s)
    new = (_Windows(_group(seg, src, dport), t_us).iat(long_s) >= long_s).astype(np.float64)
    out["ctx_src_new_dports_10s"] = _Windows(_group(seg, src), t_us).total(w_l, new)
    return out


# =====================================================================
# one file

class FileIngest:
    """Turns the blocks of one CSV into feature rows; all state needed to resume is in ``state()``."""

    def __init__(self, cfg, device: str, device_type: str, device_ip: Optional[str], st: Optional[Dict] = None,
                 carry: Optional[pd.DataFrame] = None):
        ing = cfg.get("ingest", {})
        self.ctx = bool(ing.get("context_features", True))
        self.short_s, self.long_s = _ctx_names(cfg)["short"], _ctx_names(cfg)["long"]
        self.backstep = float(ing.get("segment_backstep_s", 1.0))
        self.max_gap = float(ing.get("max_gap_s", 300.0))
        self.max_carry = int(ing.get("max_carry_rows", 2_000_000))
        self.drop = {c for c in ing.get("drop_columns", [])}
        self.device, self.device_type, self.device_ip = device, device_type, device_ip
        st = st or {}
        self.seg = int(st.get("seg", 0))
        self.prev_t = st.get("prev_t")
        self.hint = st.get("hint")
        self.rows = int(st.get("rows", 0))
        self.bad = int(st.get("bad", 0))
        self.carry_capped = int(st.get("carry_capped", 0))
        self.fails: Counter = Counter(st.get("fails", {}))
        self.labels: Counter = Counter(st.get("labels", {}))
        self.stats: Dict[str, Dict[str, float]] = st.get("stats", {})
        self.dev_hits = int(st.get("dev_hits", 0))
        self.carry = carry if carry is not None else pd.DataFrame(columns=CARRY_COLS)

    def state(self) -> Dict[str, Any]:
        return {"seg": self.seg, "prev_t": self.prev_t, "hint": self.hint, "rows": self.rows, "bad": self.bad,
                "carry_capped": self.carry_capped, "fails": dict(self.fails), "labels": dict(self.labels),
                "stats": self.stats, "dev_hits": self.dev_hits}

    def _segments(self, t: np.ndarray) -> np.ndarray:
        n = len(t)
        tf = pd.Series(t).ffill().to_numpy()
        if self.prev_t is not None:
            tf = np.where(np.isnan(tf), self.prev_t, tf)
        prev = np.empty(n)
        prev[0] = self.prev_t if self.prev_t is not None else np.nan
        prev[1:] = tf[:-1]
        with np.errstate(invalid="ignore"):
            d = tf - prev
            new = (d < -self.backstep) | (d > self.max_gap)
        seg = self.seg + np.cumsum(new)
        if n:
            self.seg = int(seg[-1])
            if np.isfinite(tf[-1]):
                self.prev_t = float(tf[-1])
        return seg.astype(np.int32)

    def _update_stats(self, df: pd.DataFrame) -> None:
        for c in df.columns:
            if c not in FEATURES:
                continue
            v = df[c].to_numpy(dtype=np.float64)
            ok = np.isfinite(v)
            s = self.stats.setdefault(c, {"n": 0, "nonnull": 0, "min": None, "max": None, "sum": 0.0})
            s["n"] += len(v)
            s["nonnull"] += int(ok.sum())
            if ok.any():
                vv = v[ok]
                s["min"] = float(vv.min()) if s["min"] is None else min(s["min"], float(vv.min()))
                s["max"] = float(vv.max()) if s["max"] is None else max(s["max"], float(vv.max()))
                s["sum"] += float(vv.sum())

    def process(self, raw: pd.DataFrame, n_lines: int, label_col, time_col) -> pd.DataFrame:
        self.rows += len(raw)
        self.bad += max(n_lines - len(raw), 0)
        df, self.hint = type_block(raw, label_col, time_col, self.drop, self.hint, self.fails)
        n = len(df)
        if n == 0:
            return df
        df["seg"] = self._segments(df["t"].to_numpy())
        self.labels.update(pd.Series(df["label_raw"]).value_counts().to_dict())
        src = df["ip_src"] if "ip_src" in df else pd.Series([""] * n)
        dst = df["ip_dst"] if "ip_dst" in df else pd.Series([""] * n)
        if self.device_ip:
            df["dir_out"] = (src == self.device_ip).to_numpy(dtype=np.float64)
            df["dir_in"] = (dst == self.device_ip).to_numpy(dtype=np.float64)
            self.dev_hits += int(((src == self.device_ip) | (dst == self.device_ip)).sum())
        if self.ctx:
            def port(a, b):
                x = df[a] if a in df else pd.Series(np.nan, index=df.index)
                y = df[b] if b in df else pd.Series(np.nan, index=df.index)
                return x.fillna(y).fillna(-1.0)

            cur = pd.DataFrame({"t": df["t"], "seg": df["seg"], "ip_src": src.to_numpy(), "ip_dst": dst.to_numpy(),
                                "sport": port("tcp_srcport", "udp_srcport"),
                                "dport": port("tcp_dstport", "udp_dstport"),
                                "proto": df["ip_proto"].fillna(-1.0) if "ip_proto" in df else -1.0,
                                "frame_len": df["frame_len"] if "frame_len" in df else 0.0})
            chunk = pd.concat([self.carry, cur], ignore_index=True) if len(self.carry) else cur
            feats = context_features(chunk, self.short_s, self.long_s)
            k = len(chunk) - n
            for name, v in feats.items():
                df[name] = v[k:]
            last = chunk[chunk["seg"] == chunk["seg"].iloc[-1]]
            tmax = np.nanmax(last["t"].to_numpy(dtype=np.float64)) if last["t"].notna().any() else np.nan
            # 2 x long window: ctx_src_new_dports_10s sums flags that themselves look back long_s
            keep = last[last["t"] >= tmax - 2 * self.long_s] if np.isfinite(tmax) else last.iloc[0:0]
            if len(keep) > self.max_carry:
                keep = keep.iloc[-self.max_carry:]
                self.carry_capped += 1
            self.carry = keep.reset_index(drop=True)
        df["device"] = self.device
        df["device_type"] = self.device_type
        self._update_stats(df)
        return df


def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("string"):
            df[c] = df[c].astype("category")
        elif c in FEATURES:
            df[c] = df[c].astype(np.float32)
    if "seg" in df:
        df["seg"] = df["seg"].astype(np.int32)
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="zstd")
    return buf.getvalue()


def _read_parquet(path, columns=None) -> pd.DataFrame:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pandas()


def _device_ip(m: Dict[str, Any], cfg) -> Optional[str]:
    """Most common IP among the first lines of the file (decided once, independent of the block size)."""
    pk = peek_member(m, cfg, n_lines=20000, max_bytes=4 * 2**20)
    delim = "\t" if pk["delimiter"] == "tab" else ","
    names = normalize_names(next(csv.reader([pk["header"]], delimiter=delim)))
    if "ip_src" not in names or "ip_dst" not in names or not pk["lines"]:
        return None
    df = pd.DataFrame([r for r in csv.reader(pk["lines"][:-1] or pk["lines"], delimiter=delim)
                       if len(r) == len(names)], columns=names)
    return mode_ip(first_value(df["ip_src"]), first_value(df["ip_dst"])) if len(df) else None


def ingest_member(m: Dict[str, Any], cfg, tracer, deadline: Deadline, out: Path, progress) -> Dict[str, Any]:
    ing = cfg.get("ingest", {})
    block = int(float(ing.get("block_mb", 32)) * 2**20)
    part_bytes = int(float(ing.get("part_mb", 128)) * 2**20)
    fid = stable_hash(m["id"], 16)
    sdir = ensure_dir(out / "_state")
    spath = sdir / f"{fid}.json"
    dev, dtype = device_from_member(m["name"])

    st = read_json(spath)
    if st and st.get("member") == m["id"] and all(output_ok(out / p) for p in st["parts"]) \
            and output_ok(sdir / st["carry"]):
        carry = _read_parquet(sdir / st["carry"])
        tracer.info(STAGE, "resume", f"{Path(m['name']).name}: continuing at part {len(st['parts'])} "
                    f"(byte {st['offset']:,} of {m['size']:,})")
    else:
        st, carry = None, None
    if st is None:
        st = {"member": m["id"], "offset": None, "parts": [], "carry": None, "device_ip": _device_ip(m, cfg),
              "file": {}, "t_first": time.time(), "seconds": 0.0}
    fi = FileIngest(cfg, dev, dtype, st["device_ip"], st["file"], carry)
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
        pending_from = start

        def flush(end: int) -> None:
            nonlocal pending, pending_from
            idx = len(st["parts"])
            name = f"{fid}-{idx:05d}.parquet"
            df = pd.concat(pending, ignore_index=True) if pending else pd.DataFrame()
            atomic_write_bytes(out / name, _to_parquet_bytes(df))
            cname = f"{fid}.carry-{idx:05d}.parquet"
            atomic_write_bytes(sdir / cname, _to_parquet_bytes(fi.carry.assign(seg=fi.carry["seg"].astype(np.int32))))
            old = st["carry"]
            st.update(offset=end, carry=cname, file=fi.state(), seconds=st["seconds"] + time.time() - t0_part[0])
            st["parts"].append(name)
            atomic_write_json(spath, st)
            if old and old != cname:
                (sdir / old).unlink(missing_ok=True)
            tracer.debug(STAGE, "part", f"{Path(m['name']).name} part {idx}: {len(df):,} rows", bytes_to=end)
            pending, pending_from = [], end
            t0_part[0] = time.time()
            crash_point("ingest_part")

        t0_part = [time.time()]
        end = start
        for s, end, blk in iter_blocks(fh, start, block):
            deadline.check()
            raw, n_lines = parser.parse(blk)
            del blk
            pending.append(fi.process(raw, n_lines, label_col, time_col))
            del raw
            progress(end)
            if end - pending_from >= part_bytes:
                flush(end)
        if pending or not st["parts"]:
            flush(end)
    meta = {"member": m["name"], "id": m["id"], "device": dev, "device_type": dtype, "device_ip": st["device_ip"],
            "parts": st["parts"], "label_column": label_col, "time_column": time_col, "parser": parser.engine,
            "seconds": round(st["seconds"], 1), **fi.state()}
    meta["segments"] = meta.pop("seg") + 1
    meta["device_ip_share"] = fi.dev_hits / max(fi.rows - fi.bad, 1) if st["device_ip"] else None
    return meta


# =====================================================================
# command

def _explore_rows(cfg) -> Dict[str, int]:
    led = Ledger(layout.work(cfg) / "data" / "explore" / "_ledger.jsonl")
    return {u: int(r.get("rows", 0)) for u, r in led.done().items()}


def run_ingest(cfg, tracer, deadline: Deadline, max_files: Optional[int] = None) -> Dict[str, Any]:
    inv = read_json(layout.base_reports(cfg, "explore") / "inventory.json")
    if not inv:
        tracer.info(STAGE, "inventory", "no inventory from step 0; listing the archive")
        inv = build_inventory(cfg)
    csvs = sorted([m for m in inv["members"] if m["kind"] == "processed_csv"], key=lambda m: (m["size"], m["id"]))
    if not csvs:
        raise RuntimeError("No CSV files found in the archive.")
    if max_files:
        csvs = csvs[:max_files]
    out = layout.ingest_dir(cfg)
    atomic_write_json(out / "_spec.json", layout.ingest_spec(cfg))
    ledger = Ledger(out / "_ledger.jsonl")
    done = ledger.done()

    def finished(m):
        r = done.get(m["id"])
        if not r or not output_ok(r["meta"]):
            return False
        meta = read_json(r["meta"]) or {}
        return all(output_ok(out / p) for p in meta.get("parts", ["?"]))

    todo = [m for m in csvs if not finished(m)]
    total = sum(m["size"] for m in csvs)
    done_bytes = sum(m["size"] for m in csvs if finished(m))
    tracer.info(STAGE, "plan", f"{len(csvs)} CSV files ({total / 2**30:.2f} GB) -> {out}; "
                f"{len(csvs) - len(todo)} already ingested, {len(todo)} to go")
    t_start = time.time()
    read_bytes = 0
    for k, m in enumerate(todo):
        deadline.check()

        def progress(pos, m=m):
            el = time.time() - t_start
            rate = (read_bytes + pos) / max(el, 1e-6)
            left = total - done_bytes - read_bytes - pos
            tracer.progress(STAGE, len(csvs) - len(todo) + k, len(csvs), "files", every_s=30,
                            current=Path(m["name"]).name, file_pct=f"{100 * pos / max(m['size'], 1):.0f}%",
                            mb_s=f"{rate / 2**20:.1f}", eta_min=f"{left / max(rate, 1) / 60:.1f}")

        ledger.append(m["id"], "start")
        try:
            meta = retry(lambda: ingest_member(m, cfg, tracer, deadline, out, progress), tries=3, base_delay=10.0,
                         on_error=lambda a, e: tracer.warn(STAGE, "io_retry", f"attempt {a}: {e}", member=m["name"]))
        except (GracefulStop, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001
            tracer.exception(STAGE, "file_failed", exc, member=m["name"])
            ledger.append(m["id"], "failed", error=f"{type(exc).__name__}: {exc}")
            continue
        mpath = out / f"{stable_hash(m['id'], 16)}.json"
        atomic_write_json(mpath, meta)
        read_bytes += m["size"]
        ledger.append(m["id"], "done", meta=str(mpath), rows=meta["rows"], parts=len(meta["parts"]))
        tracer.info(STAGE, "file", f"{Path(m['name']).name}: {meta['rows']:,} rows, {len(meta['parts'])} part(s), "
                    f"{meta['seconds']} s", device_ip=meta["device_ip"], bad_lines=meta["bad"])
        crash_point(STAGE)
    failed = [m["name"] for m in csvs if ledger.latest().get(m["id"], {}).get("status") == "failed"]
    if failed:
        tracer.warn(STAGE, "failed_files", "some files could not be ingested; run ingest again to retry",
                    files=failed)
        raise RuntimeError(f"{len(failed)} file(s) failed; the split is made only when every file is ingested")
    metas = [read_json(out / f"{stable_hash(m['id'], 16)}.json") for m in csvs]
    return finish_ingest(cfg, tracer, out, metas, partial=bool(max_files))


# =====================================================================
# split + report

def load_columns(out: Path, metas: List[Dict[str, Any]], columns: List[str]) -> pd.DataFrame:
    """Selected columns of every part; text columns stay categorical to keep memory low."""
    import pyarrow.parquet as pq
    from pandas.api.types import union_categoricals

    parts: Dict[str, list] = {c: [] for c in columns}
    for meta in metas:
        for p in meta["parts"]:
            have = [c for c in columns if c in pq.read_schema(out / p).names]
            if len(have) < len(columns):
                continue  # empty part (file without rows)
            df = _read_parquet(out / p, columns=columns)
            for c in columns:
                parts[c].append(df[c])
    res = {}
    for c, lst in parts.items():
        if not lst:
            res[c] = pd.Series([], dtype=float)
        elif isinstance(lst[0].dtype, pd.CategoricalDtype):
            res[c] = pd.Series(union_categoricals(lst, ignore_order=True))
        else:
            res[c] = pd.Series(np.concatenate([x.to_numpy() for x in lst]))
    return pd.DataFrame(res)


def finish_ingest(cfg, tracer, out: Path, metas: List[Dict[str, Any]], partial: bool) -> Dict[str, Any]:
    rep = _out(cfg)
    spec = split_spec(cfg)
    sname = split_name(cfg)
    with tracer.span(STAGE, "split"):
        cols = load_columns(out, metas, ["t", "label_raw"])
        table = make_split(cols["label_raw"], cols["t"].to_numpy(), spec)
        atomic_write_json(out / f"{sname}.json", {"spec": spec, "blocks": table.to_dict(orient="records")})
        code = assign(cols["label_raw"], cols["t"].to_numpy(), table)
    mapper = LabelMapper(cfg.get("labels", {}))
    lab = cols["label_raw"].astype("category")
    lc = lab.cat.codes.to_numpy().astype(np.int64)
    nl, ns = len(lab.cat.categories), len(SPLITS) + 1
    counts = np.bincount(lc * ns + np.where(code < 0, len(SPLITS), code), minlength=nl * ns).reshape(nl, ns)
    by_label = pd.DataFrame(counts[:, :len(SPLITS)], columns=list(SPLITS),
                            index=pd.Index([str(c) for c in lab.cat.categories], name="label"))
    by_label["total"] = counts.sum(axis=1)
    by_label = by_label.sort_values("total", ascending=False)
    cls_map = {str(r): (mapper.map_one(r) or "(dropped)") for r in by_label.index}
    cls = pd.Series([cls_map[str(r)] for r in by_label.index], index=by_label.index)
    by_class = by_label.groupby(cls.rename("class")).sum().sort_values("total", ascending=False)
    t0 = float(np.nanmin(cols["t"])) if len(cols) else 0.0
    blocks = table.copy()
    for c in ("start", "end", "train_end", "val_start", "val_end", "test_start"):
        blocks[c + "_h"] = (blocks[c] - t0) / 3600.0
    episodes = blocks.groupby(["label", "episode"]).agg(
        start_h=("start_h", "min"), end_h=("end_h", "max"), blocks=("block", "count"),
        packets=("packets", "sum")).reset_index().sort_values("start_h")
    by_label.to_csv(rep / "split_by_label.csv")
    by_class.to_csv(rep / "split_by_class.csv")
    blocks.to_csv(rep / "split_blocks.csv", index=False)
    episodes.to_csv(rep / "split_episodes.csv", index=False)

    files = pd.DataFrame([{"file": Path(mt["member"]).name, "device_type": mt["device_type"], "rows": mt["rows"],
                           "bad_lines": mt["bad"], "parts": len(mt["parts"]), "segments": mt["segments"],
                           "device_ip": mt["device_ip"], "device_ip_share": mt["device_ip_share"],
                           "carry_capped": mt["carry_capped"], "seconds": mt["seconds"]} for mt in metas])
    ex_rows = _explore_rows(cfg)
    files["explore_rows"] = [ex_rows.get(mt["id"]) for mt in metas]
    files = files.sort_values("rows", ascending=False)
    files.to_csv(rep / "files.csv", index=False)

    stats: Dict[str, Dict[str, float]] = {}
    fails: Counter = Counter()
    for mt in metas:
        fails.update(mt["fails"])
        for c, s in mt["stats"].items():
            a = stats.setdefault(c, {"n": 0, "nonnull": 0, "min": None, "max": None, "sum": 0.0})
            a["n"] += s["n"]
            a["nonnull"] += s["nonnull"]
            a["sum"] += s["sum"]
            for k, f in (("min", min), ("max", max)):
                if s[k] is not None:
                    a[k] = s[k] if a[k] is None else f(a[k], s[k])
    feats = pd.DataFrame([{"feature": c, "description": d,
                           "non_null_pct": 100.0 * stats[c]["nonnull"] / max(stats[c]["n"], 1),
                           "min": stats[c]["min"], "max": stats[c]["max"],
                           "mean": stats[c]["sum"] / max(stats[c]["nonnull"], 1)}
                          for c, d in FEATURES.items() if c in stats])
    feats.to_csv(rep / "features.csv", index=False)

    rows = int(files["rows"].sum())
    mismatch = files[files["explore_rows"].notna() & (files["explore_rows"] != files["rows"])]
    checks: List[str] = []
    if partial:
        checks.append("Only part of the files were ingested (--max-files): the split covers those files only.")
    if len(mismatch):
        checks.append(f"Row counts differ from step 0 for {len(mismatch)} file(s): "
                      + ", ".join(mismatch["file"].head(5)))
    kept = [r for r in by_label.index if mapper.map_one(r) is not None]
    for r in kept:
        missing = [s for s in ("train", "val", "test") if by_label.loc[r, s] == 0]
        if missing:
            checks.append(f"Label '{r}' has no packets in: {', '.join(missing)}")
    low_ip = files[files["device_ip_share"].notna() & (files["device_ip_share"] < 0.9)]
    if len(low_ip):
        checks.append(f"Device IP seen in under 90% of packets for {len(low_ip)} file(s) (see files.csv); "
                      "dir_in/dir_out may be wrong there")
    if int(files["carry_capped"].sum()):
        checks.append("Context window history was capped (ingest.max_carry_rows) in some files")
    bad_parse = {c: n for c, n in fails.items() if n}
    if bad_parse:
        checks.append(f"Values that could not be parsed (set to empty): {bad_parse}")

    summary = {"ingest_dir": str(out), "split_file": str(out / f"{sname}.json"), "files": len(metas), "rows": rows,
               "bad_lines": int(files["bad_lines"].sum()), "partial": partial, "split_spec": spec,
               "split_by_class": by_class.reset_index().to_dict(orient="records"),
               "rows_match_explore": len(mismatch) == 0 and files["explore_rows"].notna().all(),
               "parse_failures": bad_parse, "checks": checks}
    atomic_write_json(rep / "summary.json", summary)
    atomic_write_text(rep / "INGEST.md", ingest_markdown(cfg, out, sname, summary, files, feats, by_label,
                                                         by_class, episodes))
    tracer.info(STAGE, "done", f"{rows:,} rows in {len(metas)} files; report {rep / 'INGEST.md'}")
    return summary


def _pct(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for s in ("train", "val", "test", "gap"):
        d[s + "_pct"] = 100.0 * d[s] / d["total"].clip(lower=1)
    return d


def ingest_markdown(cfg, out, sname, s, files, feats, by_label, by_class, episodes) -> str:
    sp = s["split_spec"]
    f = sp["fractions"]
    L = ["# Gotham dataset - step 1: ingest", ""]
    L.append(f"Files: **{s['files']}** | rows: **{s['rows']:,}** | malformed lines skipped: {s['bad_lines']:,} | "
             f"rows equal to step 0: **{'yes' if s['rows_match_explore'] else 'NO - see section 5'}**")
    L += ["", f"Parquet parts: `{out}`  ", f"Split table: `{s['split_file']}`", ""]
    L += ["## 1. Split", "",
          f"Per raw label, in time order: episodes (pause > {sp['episode_gap_s']:.0f} s starts a new one), cut "
          f"into blocks of at most {sp['block_h']:g} h. In every block the first {f[0]:.0%} of packets -> train, "
          f"next {f[1]:.0%} -> val, last {f[2]:.0%} -> test; packets in a gap of max({sp['gap_frac']:.0%} of the "
          f"block, {sp['gap_min_s']:g} s) before val and before test are left out (a gap never takes more than "
          f"{sp['gap_max_share']:.0%} of a block's packets).", "",
          "Per class (classes from `labels` in the config; `(dropped)` is not used for training):", "",
          md_table(_pct(by_class).reset_index(), max_rows=20), "",
          "Per raw label:", "", md_table(_pct(by_label).reset_index(), max_rows=40), "",
          "Episodes (hours since the first packet):", "", md_table(episodes, max_rows=40), ""]
    L += ["## 2. Features", "", "Context features (`ctx_*`) look only at earlier packets of the same capture "
          "segment.", "", md_table(feats, max_rows=60), ""]
    L += ["## 3. Files", "", md_table(files, max_rows=20), ""]
    L += ["## 4. Columns", "", "Dropped before parsing: " + ", ".join(f"`{c}`" for c in
                                                                      cfg.get("ingest", {}).get("drop_columns", [])),
          "", "Kept as text (not features): `ip_src`, `ip_dst`, `frame_protocols`, `device`, `device_type`, "
          "`label_raw`; also `t` (epoch seconds) and `seg` (capture segment).", ""]
    L += ["## 5. Things to check", ""]
    L += [f"- {c}" for c in s["checks"]] or ["- nothing flagged"]
    L += ["", "## 6. Next", "", "Step 2 reads these parts, maps labels to classes, merges identical packets "
          "within each split and checks feature distributions per class before training.", ""]
    return "\n".join(L)

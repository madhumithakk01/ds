"""Look inside the dataset archive before any modelling.

Two commands live here:

``inventory`` (a minute or two)
    lists every file in the zip, copies the small metadata JSON files, reads
    the first rows of a handful of CSVs (header, delimiter, example values,
    label column) and measures how fast the zip can be read from Drive.

``explore`` (resumable, one unit per CSV file)
    reads every processed CSV once and records, per file: rows, malformed
    lines, label counts, time format and range, capture joins (time jumping
    back or forward), time range of every label, protocol stacks per label,
    column-level statistics, and a hash of each packet's header fields.  The
    hashes are merged at the end to measure how many packets are exact
    repeats and how often identical packets carry different labels.

Everything ends up in ``reports/explore`` - ``SUMMARY.md`` is the file to read.
"""
from __future__ import annotations

import csv
import io
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import layout, plots
from .labels import LabelMapper
from .parse import (clean_label, detect_label_column, device_from_member, normalize_names, parse_numeric,
                    parse_tcp_options, parse_time)
from .state import (Deadline, GracefulStop, Ledger, atomic_write_json, atomic_write_text, crash_point,
                    ensure_dir, output_ok, read_json, retry, stable_hash)
from .zipsrc import MemberStream, build_inventory, iter_blocks, read_header

STAGE = "explore"
TIME_COLS = ("frame_time_epoch", "frame_time", "timestamp", "time")
IDENT_COLS = {"eth_src", "eth_dst", "ip_src", "ip_dst", "ipv6_src", "ipv6_dst"}
NO_KEY_COLS = IDENT_COLS | set(TIME_COLS) | {
    "ip_checksum", "tcp_checksum", "udp_checksum", "ip_id", "frame_number", "tcp_seq", "tcp_ack",
    "tcp_stream", "frame_time_relative", "frame_time_delta"}
NO_TOP = set(TIME_COLS) | {"ip_checksum", "tcp_checksum", "udp_checksum", "tcp_options", "ip_id",
                           "tcp_seq", "tcp_ack", "frame_number", "frame_time_relative", "frame_time_delta"}
TEXT_COLS = IDENT_COLS | {"frame_protocols", "tcp_options"}


# =====================================================================
# helpers

def _out(cfg) -> Path:
    return layout.reports(cfg, "explore")


def _data(cfg) -> Path:
    return ensure_dir(layout.work(cfg) / "data" / "explore")


def _bool(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=bool, na_value=False)
    return np.asarray(x, dtype=bool)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return str(n)


def md_table(df: pd.DataFrame, max_rows: int = 40, floatfmt: str = "{:,.4g}") -> str:
    if df is None or len(df) == 0:
        return "_(none)_\n"
    d = df.head(max_rows)
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        cells = []
        for v in r.tolist():
            if isinstance(v, float):
                cells.append("" if math.isnan(v) else (f"{v:,.0f}" if v.is_integer() and abs(v) >= 1000
                                                       else floatfmt.format(v)))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}")
            else:
                cells.append(str(v).replace("|", "\\|").replace("\n", " ")[:80])
        lines.append("| " + " | ".join(cells) + " |")
    more = "" if len(df) <= max_rows else f"\n_... {len(df) - max_rows} more rows in the CSV file_\n"
    return "\n".join(lines) + "\n" + more


def merge_pairs(h: np.ndarray, lab: np.ndarray, cnt: np.ndarray):
    """Collapse duplicate (hash, label) pairs, summing counts; result sorted by (hash, label)."""
    if len(h) == 0:
        return h, lab, cnt
    order = np.lexsort((lab, h))
    h, lab, cnt = h[order], lab[order], cnt[order]
    new = np.r_[True, (h[1:] != h[:-1]) | (lab[1:] != lab[:-1])]
    starts = np.flatnonzero(new)
    return h[starts], lab[starts], np.add.reduceat(cnt, starts)


# =====================================================================
# block parsing (pyarrow when it agrees with pandas, pandas otherwise)

class BlockParser:
    def __init__(self, names: List[str], delim: str, tracer, prefer: str = "auto"):
        self.names = names
        self.delim = delim
        self.tracer = tracer
        self.engine = None if prefer == "auto" else prefer

    @staticmethod
    def count_lines(block: bytes) -> int:
        return block.count(b"\n") + (0 if (not block or block.endswith(b"\n")) else 1)

    def _pandas(self, block: bytes) -> pd.DataFrame:
        text = block.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text), sep=self.delim, header=None, names=self.names, dtype=str,
                         keep_default_na=False, na_filter=False, engine="c", on_bad_lines="skip",
                         quoting=csv.QUOTE_MINIMAL, skip_blank_lines=True)
        return df.fillna("")

    def _arrow(self, block: bytes) -> pd.DataFrame:
        import pyarrow as pa
        import pyarrow.csv as pacsv

        tbl = pacsv.read_csv(
            io.BytesIO(block),
            read_options=pacsv.ReadOptions(column_names=self.names, autogenerate_column_names=False),
            parse_options=pacsv.ParseOptions(delimiter=self.delim, invalid_row_handler=lambda row: "skip"),
            convert_options=pacsv.ConvertOptions(column_types={n: pa.string() for n in self.names},
                                                 strings_can_be_null=False,
                                                 quoted_strings_can_be_null=False))
        return tbl.to_pandas(types_mapper={pa.string(): pd.StringDtype("pyarrow")}.get)

    def _decide(self, block: bytes) -> None:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.engine = "pandas"
            return
        cut = 0
        for _ in range(2000):
            nxt = block.find(b"\n", cut)
            if nxt == -1:
                break
            cut = nxt + 1
        probe = block[:cut] if cut else block
        try:
            a = self._arrow(probe)
            p = self._pandas(probe)
            same = a.shape == p.shape and a.astype(object).equals(p.astype(object))
        except Exception as exc:  # noqa: BLE001
            self.tracer.warn(STAGE, "pyarrow_parser_off", f"pyarrow CSV parser failed ({exc}); using pandas")
            self.engine = "pandas"
            return
        self.engine = "pyarrow" if same else "pandas"
        if not same:
            self.tracer.warn(STAGE, "pyarrow_parser_off", "pyarrow and pandas parsed the first rows "
                             "differently; using the pandas parser", shapes=[a.shape, p.shape])
        else:
            self.tracer.debug(STAGE, "parser", "pyarrow CSV parser verified against pandas")

    def parse(self, block: bytes) -> Tuple[pd.DataFrame, int]:
        if self.engine is None:
            self._decide(block)
        n_lines = self.count_lines(block)
        if self.engine == "pyarrow":
            try:
                return self._arrow(block), n_lines
            except Exception as exc:  # noqa: BLE001
                self.tracer.warn(STAGE, "pyarrow_block_failed", f"{exc}; parsing this block with pandas")
        return self._pandas(block), n_lines


# =====================================================================
# inventory

def _json_shape(obj: Any, depth: int = 0) -> Any:
    if depth > 2:
        return type(obj).__name__
    if isinstance(obj, dict):
        keys = list(obj.keys())
        return {"type": "dict", "n_keys": len(keys),
                "keys": {str(k): _json_shape(obj[k], depth + 1) for k in keys[:15]}}
    if isinstance(obj, list):
        return {"type": "list", "len": len(obj), "first": _json_shape(obj[0], depth + 1) if obj else None}
    if isinstance(obj, str):
        return f"str e.g. {obj[:40]!r}"
    return type(obj).__name__


def _head_profile(m: Dict[str, Any], cfg, tracer, head_mb: float) -> Dict[str, Any]:
    with MemberStream(m, cfg) as fh:
        header, _ = read_header(fh)
        raw = fh.read(int(head_mb * 2**20))
    cut = raw.rfind(b"\n")
    raw = raw[:cut + 1] if cut >= 0 else raw
    htxt = header.decode("utf-8", errors="replace").strip("\r\n")
    delim = "\t" if htxt.count("\t") > htxt.count(",") else ","
    raw_names = next(csv.reader([htxt], delimiter=delim))
    names = normalize_names(raw_names)
    df = BlockParser(names, delim, tracer, prefer="pandas").parse(raw)[0] if raw else pd.DataFrame(columns=names)
    label_col = detect_label_column(names, cfg.get("ingest", {}).get("label_column", "auto"))
    cols = []
    for raw_n, c in zip(raw_names, names):
        s = df[c].astype(str)
        nb = _bool(s != "")
        vals, nf = parse_numeric(s) if c not in TEXT_COLS and c not in TIME_COLS and c != label_col \
            else (np.full(len(s), np.nan), 0)
        ex = [str(v)[:60] for v in s[nb].unique()[:3]]
        cols.append({"column": raw_n, "name": c, "non_empty_pct": 100.0 * nb.mean() if len(s) else 0.0,
                     "numeric_pct": 100.0 * np.isfinite(vals).sum() / max(nb.sum(), 1),
                     "hex_pct": 100.0 * _bool(s.str[:2].str.lower() == "0x").sum() / max(nb.sum(), 1),
                     "multi_value_pct": 0.0 if c in TIME_COLS else
                     100.0 * _bool(s.str.contains(",", regex=False)).sum() / max(nb.sum(), 1),
                     "distinct_in_head": int(s[nb].nunique()), "examples": ex})
    tinfo = {}
    tcol = next((c for c in TIME_COLS if c in names), None)
    if tcol and len(df):
        tv, nf, method = parse_time(df[tcol])
        fin = tv[np.isfinite(tv)]
        tinfo = {"column": tcol, "method": method, "unparsed": nf,
                 "first": float(fin[0]) if len(fin) else None, "last": float(fin[-1]) if len(fin) else None}
    labels = clean_label(df[label_col]).value_counts().head(20).to_dict() if label_col and len(df) else {}
    return {"member": m["name"], "size_mb": round(m["size"] / 2**20, 1), "delimiter": "tab" if delim == "\t" else "comma",
            "header": raw_names, "label_column": label_col, "rows_in_head": len(df), "columns": cols,
            "time": tinfo, "labels_in_head": {str(k): int(v) for k, v in labels.items()},
            "first_lines": raw[:1500].decode("utf-8", errors="replace").splitlines()[:5]}


def _throughput(m: Dict[str, Any], cfg, mb: float) -> Dict[str, float]:
    t0 = time.time()
    n = 0
    with MemberStream(m, cfg) as fh:
        t_open = time.time() - t0
        target = int(mb * 2**20)
        while n < target:
            buf = fh.read(min(16 * 2**20, target - n))
            if not buf:
                break
            n += len(buf)
    dt = max(time.time() - t0 - t_open, 1e-6)
    return {"member": m["name"], "mb_read": round(n / 2**20, 1), "seconds": round(dt, 2),
            "mb_per_s": round(n / 2**20 / dt, 1), "open_seconds": round(t_open, 2)}


def run_inventory(cfg, tracer) -> Dict[str, Any]:
    ex = cfg.get("explore", {})
    t0 = time.time()
    with tracer.span(STAGE, "list_archive"):
        inv = retry(lambda: build_inventory(cfg), tries=3, base_delay=5.0)
    inv["list_seconds"] = round(time.time() - t0, 1)
    members = inv["members"]
    csvs = [m for m in members if m["kind"] == "processed_csv"]
    if not csvs:
        others = [m for m in members if m["kind"] == "other_csv"]
        if others:
            tracer.warn(STAGE, "csv_fallback", "no CSV matched inventory.include; using every CSV in the archive")
            for m in others:
                m["kind"] = "processed_csv"
            csvs = others
        else:
            tracer.error(STAGE, "no_csv", "The archive contains no CSV files. Check reports/explore/inventory.csv")
    out = _out(cfg)
    df = pd.DataFrame(members)
    df.to_csv(out / "inventory.csv", index=False)
    by_kind = df.groupby("kind").agg(files=("name", "size"), gb=("size", lambda s: s.sum() / 2**30),
                                     compressed_gb=("csize", lambda s: s.sum() / 2**30)).reset_index()
    tops = df.assign(folder=df["name"].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[0])
    by_folder = tops.groupby(["kind", "folder"]).agg(files=("name", "size"),
                                                     gb=("size", lambda s: s.sum() / 2**30)).reset_index()
    for d in (by_kind, by_folder):
        for c in ("gb", "compressed_gb"):
            if c in d.columns:
                d[c] = d[c].round(3)
    inv["by_kind"] = by_kind.to_dict(orient="records")
    inv["by_folder"] = by_folder.to_dict(orient="records")

    meta_dir = ensure_dir(out / "metadata")
    meta_info = []
    for m in [m for m in members if m["kind"] == "metadata"][:60]:
        try:
            if m["size"] > 20 * 2**20:
                meta_info.append({"file": m["name"], "size_mb": round(m["size"] / 2**20, 1), "shape": "too big to copy"})
                continue
            with MemberStream(m, cfg) as fh:
                raw = fh.read()
            (meta_dir / Path(m["name"]).name).write_bytes(raw)
            try:
                shape = _json_shape(json.loads(raw.decode("utf-8", errors="replace")))
            except json.JSONDecodeError:
                shape = "not valid JSON"
            meta_info.append({"file": m["name"], "size_kb": round(m["size"] / 1024, 1), "shape": shape})
        except Exception as exc:  # noqa: BLE001
            tracer.warn(STAGE, "metadata_failed", str(exc), file=m["name"])
    inv["metadata"] = meta_info

    heads = []
    if csvs:
        by_size = sorted(csvs, key=lambda m: m["size"])
        n_head = int(ex.get("head_files", 6))
        picks = [by_size[int(round(i))] for i in np.linspace(0, len(by_size) - 1, min(n_head, len(by_size)))]
        seen = set()
        for m in picks:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            try:
                with tracer.span(STAGE, "head", member=m["name"]):
                    heads.append(_head_profile(m, cfg, tracer, float(ex.get("head_mb", 8))))
            except Exception as exc:  # noqa: BLE001
                tracer.exception(STAGE, "head_failed", exc, member=m["name"])
        try:
            with tracer.span(STAGE, "throughput"):
                inv["throughput"] = _throughput(by_size[-1], cfg, float(ex.get("throughput_mb", 256)))
        except Exception as exc:  # noqa: BLE001
            tracer.exception(STAGE, "throughput_failed", exc)
    inv["heads"] = heads
    atomic_write_json(out / "inventory.json", inv)
    atomic_write_text(out / "INVENTORY.md", inventory_markdown(inv))
    tot_csv = sum(m["size"] for m in csvs)
    tracer.info(STAGE, "inventory", f"{len(members)} files in archive; {len(csvs)} processed CSVs "
                f"({tot_csv / 2**30:.2f} GB of text)", by_kind=inv["by_kind"])
    if heads:
        h = heads[0]
        tracer.info(STAGE, "schema", f"delimiter={h['delimiter']}, label column={h['label_column']}",
                    header=h["header"])
    if inv.get("throughput"):
        tracer.info(STAGE, "throughput", f"{inv['throughput']['mb_per_s']} MB/s reading from the archive")
    return inv


def inventory_markdown(inv: Dict[str, Any]) -> str:
    src = inv.get("source", {})
    lines = ["# Archive inventory", "",
             f"Source: `{src.get('path')}` ({_fmt_bytes(src.get('bytes', 0))}), listed in {inv.get('list_seconds')} s", "",
             "## Files by kind", "", md_table(pd.DataFrame(inv.get("by_kind", []))),
             "## Folders", "", md_table(pd.DataFrame(inv.get("by_folder", [])), 60)]
    if inv.get("throughput"):
        t = inv["throughput"]
        lines += ["## Read speed", "", f"{t['mb_read']} MB of `{t['member']}` in {t['seconds']} s "
                  f"= **{t['mb_per_s']} MB/s** (opening took {t['open_seconds']} s)", ""]
    for h in inv.get("heads", [])[:3]:
        lines += [f"## Head of `{h['member']}` ({h['size_mb']} MB)", "",
                  f"delimiter: {h['delimiter']}, label column: `{h['label_column']}`, rows read: {h['rows_in_head']:,}", "",
                  md_table(pd.DataFrame(h["columns"]).drop(columns=["name"])), "",
                  f"time: {h.get('time')}", "", f"labels in head: {h.get('labels_in_head')}", "",
                  "first lines:", "```", *h.get("first_lines", []), "```", ""]
    if inv.get("metadata"):
        lines += ["## Metadata files", ""]
        for mi in inv["metadata"]:
            lines += [f"- `{mi['file']}`: {json.dumps(mi.get('shape'))[:600]}"]
    return "\n".join(lines) + "\n"


# =====================================================================
# full scan of one file

class FileScan:
    def __init__(self, names: List[str], label_col: Optional[str], cfg):
        ex = cfg.get("explore", {})
        self.names = names
        self.label_col = label_col
        self.time_col = next((c for c in TIME_COLS if c in names), None)
        self.key_cols = [c for c in names if c not in NO_KEY_COLS and c != label_col]
        self.backstep = float(ex.get("segment_backstep_s", 1.0))
        self.max_gap = float(ex.get("max_gap_s", 300.0))
        self.rows = 0
        self.bad = 0
        self.blocks = 0
        self.deep_blocks = 0
        self.deep_rows = 0
        self.labels: Counter = Counter()
        self.lab_index: Dict[str, int] = {}
        self.cols = {c: {"nonblank": 0, "multi": 0, "hex": 0, "num_ok": 0, "num_fail": 0, "rows": 0,
                         "min": math.inf, "max": -math.inf, "top": Counter(), "high_card": False,
                         "distinct_max_block": 0} for c in names if c != label_col}
        self.t = {"methods": Counter(), "unparsed": 0, "min": math.inf, "max": -math.inf, "backward": 0,
                  "gaps": 0, "minor_reorder": 0, "joins": [], "label_min": {}, "label_max": {}}
        self.prev_t: Optional[float] = None
        self.hint: Optional[str] = None
        self.proto: Counter = Counter()
        self.optlen: Counter = Counter()
        self.optmask: Counter = Counter()
        self._h: List[np.ndarray] = []
        self._l: List[np.ndarray] = []
        self._c: List[np.ndarray] = []
        self._pending = 0

    def _lab_codes(self, lab: pd.Series) -> np.ndarray:
        codes, uniq = pd.factorize(lab, use_na_sentinel=False)
        idx = np.array([self.lab_index.setdefault(str(u), len(self.lab_index)) for u in uniq], dtype=np.int16)
        return idx[codes] if len(uniq) else np.zeros(len(lab), dtype=np.int16)

    def update(self, df: pd.DataFrame, n_lines: int, deep: bool) -> None:
        n = len(df)
        self.rows += n
        self.bad += max(n_lines - n, 0)
        self.blocks += 1
        if n == 0:
            return
        lab = clean_label(df[self.label_col]) if self.label_col else pd.Series(["UNKNOWN"] * n)
        for k, v in lab.value_counts().items():
            self.labels[str(k)] += int(v)
        lcode = self._lab_codes(lab)

        if self.time_col:
            tv, nf, method = parse_time(df[self.time_col], self.hint)
            self.hint = method
            self.t["methods"][method] += 1
            self.t["unparsed"] += nf
            fin = np.isfinite(tv)
            if fin.any():
                self.t["min"] = min(self.t["min"], float(tv[fin].min()))
                self.t["max"] = max(self.t["max"], float(tv[fin].max()))
                g = pd.DataFrame({"l": lcode[fin], "t": tv[fin]}).groupby("l")["t"].agg(["min", "max"])
                for li, r in g.iterrows():
                    self.t["label_min"][int(li)] = min(self.t["label_min"].get(int(li), math.inf), float(r["min"]))
                    self.t["label_max"][int(li)] = max(self.t["label_max"].get(int(li), -math.inf), float(r["max"]))
                tf = pd.Series(tv).ffill().to_numpy()
                prev = np.empty(n)
                prev[0] = self.prev_t if self.prev_t is not None else np.nan
                prev[1:] = tf[:-1]
                with np.errstate(invalid="ignore"):
                    d = tv - prev
                    back = fin & (d < -self.backstep)
                    gap = fin & (d > self.max_gap)
                    minor = fin & (d < 0) & ~back
                self.t["backward"] += int(back.sum())
                self.t["gaps"] += int(gap.sum())
                self.t["minor_reorder"] += int(minor.sum())
                for pos in np.flatnonzero(back | gap)[:20]:
                    if len(self.t["joins"]) < 60:
                        self.t["joins"].append({"row": int(self.rows - n + pos), "from": float(prev[pos]),
                                                "to": float(tv[pos]), "kind": "back" if back[pos] else "gap"})
                self.prev_t = float(tf[-1]) if np.isfinite(tf[-1]) else self.prev_t

        if "frame_protocols" in df.columns:
            pc = pd.DataFrame({"p": df["frame_protocols"].astype(str).to_numpy(), "l": lcode}).groupby(
                ["p", "l"]).size()
            for (p, li), v in pc.items():
                self.proto[(p, int(li))] += int(v)

        kdf = df[self.key_cols]
        if "tcp_options" in kdf.columns:
            ln, mask, _ = parse_tcp_options(kdf["tcp_options"])
            kdf = kdf.assign(tcp_options=pd.Series(ln).fillna(-1).astype(np.int64).astype(str).to_numpy())
            if deep:
                self.optlen.update(pd.Series(ln).fillna(-1).astype(np.int64).value_counts().to_dict())
                self.optmask.update(pd.Series(mask).fillna(-1).astype(np.int64).value_counts().head(30).to_dict())
        h = pd.util.hash_pandas_object(kdf, index=False, categorize=True).to_numpy(dtype=np.uint64) \
            if len(kdf.columns) else np.zeros(n, dtype=np.uint64)
        bh, bl, bc = merge_pairs(h, lcode.astype(np.int16), np.ones(n, dtype=np.int64))
        self._h.append(bh)
        self._l.append(bl)
        self._c.append(bc)
        self._pending += len(bh)
        if self._pending > 3_000_000:
            self._consolidate()

        if deep:
            self.deep_blocks += 1
            self.deep_rows += n
            for c, st in self.cols.items():
                s = df[c]
                nb = _bool(s != "")
                st["rows"] += n
                st["nonblank"] += int(nb.sum())
                if c not in TIME_COLS:
                    st["multi"] += int(_bool(s.str.contains(",", regex=False)).sum())
                st["hex"] += int(_bool(s.str[:2].str.lower() == "0x").sum())
                if c not in TEXT_COLS and c not in TIME_COLS:
                    vals, nf = parse_numeric(s)
                    ok = np.isfinite(vals)
                    st["num_ok"] += int(ok.sum())
                    st["num_fail"] += int(nf)
                    if ok.any():
                        st["min"] = min(st["min"], float(vals[ok].min()))
                        st["max"] = max(st["max"], float(vals[ok].max()))
                if c in NO_TOP or st["high_card"]:
                    continue
                vc = s.value_counts()
                st["distinct_max_block"] = max(st["distinct_max_block"], int(len(vc)))
                st["top"].update({str(k): int(v) for k, v in vc.head(300).items()})
                if len(st["top"]) > 20000:
                    st["top"] = Counter(dict(st["top"].most_common(2000)))
                    st["high_card"] = True

    def _consolidate(self) -> None:
        if len(self._h) > 1:
            h, l, c = merge_pairs(np.concatenate(self._h), np.concatenate(self._l), np.concatenate(self._c))
            self._h, self._l, self._c = [h], [l], [c]
        self._pending = len(self._h[0]) if self._h else 0

    def result(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        self._consolidate()
        inv_lab = {v: k for k, v in self.lab_index.items()}
        cols = {}
        for c, st in self.cols.items():
            cols[c] = {k: st[k] for k in ("rows", "nonblank", "multi", "hex", "num_ok", "num_fail",
                                          "high_card", "distinct_max_block")}
            cols[c]["min"] = st["min"] if math.isfinite(st["min"]) else None
            cols[c]["max"] = st["max"] if math.isfinite(st["max"]) else None
            cols[c]["top"] = dict(st["top"].most_common(40))
            cols[c]["distinct_seen"] = int(st["distinct_max_block"])
        t = self.t
        time_info = {"column": self.time_col, "methods": dict(t["methods"]), "unparsed": t["unparsed"],
                     "min": t["min"] if math.isfinite(t["min"]) else None,
                     "max": t["max"] if math.isfinite(t["max"]) else None,
                     "backward_jumps": t["backward"], "forward_gaps": t["gaps"],
                     "minor_reorders": t["minor_reorder"], "joins": t["joins"],
                     "label_ranges": {inv_lab[k]: [t["label_min"][k], t["label_max"].get(k)]
                                      for k in t["label_min"]}}
        proto = [{"protocols": p, "label": inv_lab[li], "rows": v} for (p, li), v in self.proto.most_common(400)]
        res = {"rows": self.rows, "bad_lines": self.bad, "blocks": self.blocks, "deep_blocks": self.deep_blocks,
               "deep_rows": self.deep_rows, "labels": dict(self.labels), "label_list": [inv_lab[i] for i in
                                                                                         range(len(inv_lab))],
               "columns": cols, "time": time_info, "protocols": proto, "key_columns": self.key_cols,
               "tcp_options_len": {str(k): int(v) for k, v in self.optlen.items()},
               "tcp_options_kinds": {str(k): int(v) for k, v in self.optmask.most_common(20)},
               "unique_header_vectors": int(len(np.unique(self._h[0]))) if self._h else 0}
        arrays = {"h": self._h[0] if self._h else np.empty(0, np.uint64),
                  "l": self._l[0] if self._l else np.empty(0, np.int16),
                  "c": self._c[0] if self._c else np.empty(0, np.int64)}
        return res, arrays


def scan_member(m: Dict[str, Any], cfg, tracer, deadline: Deadline, progress) -> Tuple[Dict, Dict]:
    ex = cfg.get("explore", {})
    block = int(float(ex.get("block_mb", 32)) * 2**20)
    deep_every = max(1, int(ex.get("deep_every", 3)))
    t0 = time.time()
    with MemberStream(m, cfg) as fh:
        header, hdr_len = read_header(fh)
        htxt = header.decode("utf-8", errors="replace").strip("\r\n")
        delim = "\t" if htxt.count("\t") > htxt.count(",") else ","
        raw_names = next(csv.reader([htxt], delimiter=delim)) if htxt else []
        names = normalize_names(raw_names)
        label_col = detect_label_column(names, cfg.get("ingest", {}).get("label_column", "auto"))
        parser = BlockParser(names, delim, tracer, prefer=str(ex.get("parser", "auto")))
        scan = FileScan(names, label_col, cfg)
        for i, (s, e, blk) in enumerate(iter_blocks(fh, hdr_len, block)):
            deadline.check()
            df, n_lines = parser.parse(blk)
            del blk
            scan.update(df, n_lines, deep=(i % deep_every == 0))
            del df
            progress(e)
    res, arrays = scan.result()
    res.update({"member": m["name"], "id": m["id"], "size": m["size"], "header": raw_names, "names": names,
                "delimiter": "tab" if delim == "\t" else "comma", "label_column": label_col,
                "parser": parser.engine, "seconds": round(time.time() - t0, 1)})
    dev, dtype = device_from_member(m["name"])
    res["device"], res["device_type"] = dev, dtype
    return res, arrays


def run_explore(cfg, tracer, deadline: Deadline, max_files: Optional[int] = None,
                refresh_inventory: bool = False) -> Dict[str, Any]:
    out = _out(cfg)
    inv = None if refresh_inventory else read_json(out / "inventory.json")
    if not inv:
        inv = run_inventory(cfg, tracer)
    csvs = sorted([m for m in inv["members"] if m["kind"] == "processed_csv"], key=lambda m: (m["size"], m["id"]))
    if not csvs:
        raise RuntimeError("No CSV files found in the archive (see reports/explore/INVENTORY.md).")
    if max_files:
        csvs = csvs[:max_files]
    data = _data(cfg)
    ledger = Ledger(data / "_ledger.jsonl")
    done = ledger.done()

    def finished(m):
        r = done.get(m["id"])
        return bool(r) and output_ok(r["json"]) and output_ok(r["npz"])

    todo = [m for m in csvs if not finished(m)]
    total = sum(m["size"] for m in csvs)
    done_bytes = sum(m["size"] for m in csvs if finished(m))
    tracer.info(STAGE, "plan", f"{len(csvs)} CSV files ({total / 2**30:.2f} GB); {len(csvs) - len(todo)} already "
                f"scanned, {len(todo)} to go")
    t_start = time.time()
    scanned_bytes = 0
    for k, m in enumerate(todo):
        deadline.check()
        name = stable_hash(m["id"], 16)

        def progress(pos, m=m):
            el = time.time() - t_start
            rate = (scanned_bytes + pos) / max(el, 1e-6)
            left = total - done_bytes - scanned_bytes - pos
            tracer.progress(STAGE, len(csvs) - len(todo) + k, len(csvs), "files", every_s=30,
                            current=Path(m["name"]).name, file_pct=f"{100 * pos / max(m['size'], 1):.0f}%",
                            mb_s=f"{rate / 2**20:.1f}", eta_min=f"{left / max(rate, 1) / 60:.1f}")

        ledger.append(m["id"], "start")
        try:
            res, arrays = retry(lambda: scan_member(m, cfg, tracer, deadline, progress), tries=3,
                                base_delay=10.0, on_error=lambda a, e: tracer.warn(
                                    STAGE, "io_retry", f"attempt {a}: {e}", member=m["name"]))
        except (GracefulStop, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001
            tracer.exception(STAGE, "file_failed", exc, member=m["name"])
            ledger.append(m["id"], "failed", error=f"{type(exc).__name__}: {exc}")
            continue
        jpath = data / f"{name}.json"
        npath = data / f"{name}.npz"
        atomic_write_json(jpath, res)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        from .state import atomic_write_bytes
        atomic_write_bytes(npath, buf.getvalue())
        scanned_bytes += m["size"]
        ledger.append(m["id"], "done", json=str(jpath), npz=str(npath), rows=res["rows"], seconds=res["seconds"])
        tracer.info(STAGE, "file", f"{Path(m['name']).name}: {res['rows']:,} rows in {res['seconds']} s",
                    labels=res["labels"], bad_lines=res["bad_lines"], parser=res["parser"])
        crash_point(STAGE)
    failed = [m["name"] for m in csvs if ledger.latest().get(m["id"], {}).get("status") == "failed"]
    if failed:
        tracer.warn(STAGE, "failed_files", "some files could not be read; run explore again to retry", files=failed)
    return merge_explore(cfg, tracer, inv, csvs)


# =====================================================================
# merge

def _dup_stats(h, l, c, names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    h, l, c = merge_pairs(h, l, c)
    if len(h) == 0:
        return pd.DataFrame(), pd.DataFrame(), {}
    starts = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    nlab = np.diff(np.r_[starts, len(h)])
    hmax = np.maximum.reduceat(c, starts)
    grp = np.repeat(np.arange(len(starts)), nlab)
    conflict = nlab[grp] > 1
    major = c == hmax[grp]
    K = len(names)
    cf = c.astype(np.float64)
    per = pd.DataFrame({
        "packets": np.bincount(l, weights=cf, minlength=K),
        "unique_vectors": np.bincount(l, minlength=K),
        "conflict_packets": np.bincount(l, weights=cf * conflict, minlength=K),
        "conflict_vectors": np.bincount(l, weights=conflict.astype(float), minlength=K),
        "majority_packets": np.bincount(l, weights=cf * major, minlength=K),
    }, index=names)
    per = per[per["packets"] > 0]
    per["packets_per_vector"] = per["packets"] / per["unique_vectors"].clip(lower=1)
    per["conflict_share"] = per["conflict_packets"] / per["packets"]
    per["recall_ceiling"] = per["majority_packets"] / per["packets"]
    per = per.sort_values("packets", ascending=False)
    pairs = pd.DataFrame(columns=["label_a", "label_b", "shared_vectors", "packets_a", "packets_b"])
    if conflict.any():
        cd = pd.DataFrame({"h": h[conflict], "l": l[conflict], "c": c[conflict]})
        if len(cd) > 4_000_000:
            cd = cd[(cd["h"] % np.uint64(8)) == 0]
        pr = cd.merge(cd, on="h", suffixes=("_a", "_b"))
        pr = pr[pr["l_a"] < pr["l_b"]]
        if len(pr):
            pairs = pr.groupby(["l_a", "l_b"]).agg(shared_vectors=("h", "size"), packets_a=("c_a", "sum"),
                                                   packets_b=("c_b", "sum")).reset_index()
            pairs["label_a"] = [names[i] for i in pairs["l_a"]]
            pairs["label_b"] = [names[i] for i in pairs["l_b"]]
            pairs = pairs[["label_a", "label_b", "shared_vectors", "packets_a", "packets_b"]] \
                .sort_values("packets_a", ascending=False)
    overall = {"packets": float(cf.sum()), "unique_vectors": int(len(starts)),
               "accuracy_ceiling": float(hmax.sum() / max(cf.sum(), 1)),
               "conflicting_vectors": int((nlab > 1).sum())}
    return per, pairs, overall


def merge_explore(cfg, tracer, inv: Dict[str, Any], csvs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = _out(cfg)
    data = _data(cfg)
    done = Ledger(data / "_ledger.jsonl").done()
    results, arrays_paths = [], []
    for m in csvs:
        r = done.get(m["id"])
        if r and output_ok(r["json"]) and output_ok(r["npz"]):
            results.append(read_json(r["json"]))
            arrays_paths.append(r["npz"])
    missing = len(csvs) - len(results)
    if not results:
        raise RuntimeError("No scanned files to summarise yet.")
    mapper = LabelMapper(cfg.get("labels", {}))

    files = pd.DataFrame([{
        "file": Path(r["member"]).name, "device": r["device"], "device_type": r["device_type"],
        "rows": r["rows"], "mb": round(r["size"] / 2**20, 1), "bad_lines": r["bad_lines"],
        "segments": 1 + r["time"]["backward_jumps"] + r["time"]["forward_gaps"],
        "first_time": r["time"]["min"], "last_time": r["time"]["max"],
        "unique_header_vectors": r["unique_header_vectors"], "labels": len(r["labels"]),
        "parser": r.get("parser"), "seconds": r["seconds"]} for r in results])
    files.to_csv(out / "files.csv", index=False)

    lab_tot: Counter = Counter()
    rows_lab = []
    for r in results:
        lab_tot.update(r["labels"])
        for k, v in r["labels"].items():
            rows_lab.append({"device": r["device"], "device_type": r["device_type"], "label": k, "rows": v})
    lab_df = pd.DataFrame(rows_lab)
    total_rows = int(sum(lab_tot.values()))
    labels_total = pd.DataFrame([{"label": k, "rows": v, "share_pct": 100.0 * v / max(total_rows, 1)}
                                 for k, v in lab_tot.most_common()])
    labels_total.to_csv(out / "labels_total.csv", index=False)
    by_type = lab_df.pivot_table(index="device_type", columns="label", values="rows", aggfunc="sum", fill_value=0)
    by_type["TOTAL"] = by_type.sum(axis=1)
    by_type = by_type.sort_values("TOTAL", ascending=False)
    by_type.to_csv(out / "labels_by_device_type.csv")
    lab_df.pivot_table(index="device", columns="label", values="rows", aggfunc="sum", fill_value=0) \
        .to_csv(out / "labels_by_device.csv")
    devices_with_attacks = lab_df[~lab_df["label"].str.contains("normal|benign", case=False, regex=True)] \
        .groupby("device")["rows"].sum()

    mapping = mapper.table(dict(lab_tot))
    mapping.to_csv(out / "label_mapping_preview.csv", index=False)
    cls_tot = mapping.groupby("class", dropna=False)["rows"].sum().sort_values(ascending=False)

    variants: Dict[str, List[str]] = {}
    for r in results:
        variants.setdefault(" , ".join(r["header"]), []).append(Path(r["member"]).name)
    label_cols = Counter(r["label_column"] for r in results)

    colstats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for c, st in r["columns"].items():
            a = colstats.setdefault(c, {"rows": 0, "nonblank": 0, "multi": 0, "hex": 0, "num_ok": 0, "num_fail": 0,
                                        "min": None, "max": None, "top": Counter(), "distinct": 0, "high_card": False})
            for k in ("rows", "nonblank", "multi", "hex", "num_ok", "num_fail"):
                a[k] += st[k]
            if st["min"] is not None:
                a["min"] = st["min"] if a["min"] is None else min(a["min"], st["min"])
                a["max"] = st["max"] if a["max"] is None else max(a["max"], st["max"])
            a["top"].update(st["top"])
            a["distinct"] = max(a["distinct"], st.get("distinct_seen", 0))
            a["high_card"] = a["high_card"] or st["high_card"]
    col_rows = []
    for c, a in colstats.items():
        nb = max(a["nonblank"], 1)
        col_rows.append({
            "column": c, "non_empty_pct": 100.0 * a["nonblank"] / max(a["rows"], 1),
            "numeric_pct": 100.0 * a["num_ok"] / nb if (a["num_ok"] or a["num_fail"]) else float("nan"),
            "hex_pct": 100.0 * a["hex"] / nb, "multi_value_pct": 100.0 * a["multi"] / nb,
            "min": a["min"], "max": a["max"],
            "distinct_per_file_max": "n/a" if c in NO_TOP else (">20000" if a["high_card"] else a["distinct"]),
            "top_values": ", ".join(f"{k if k != '' else '<empty>'} ({v:,})" for k, v in a["top"].most_common(5))})
    columns = pd.DataFrame(col_rows)
    columns.to_csv(out / "columns.csv", index=False)

    t_rows = []
    for r in results:
        for lab, (a, b) in r["time"]["label_ranges"].items():
            t_rows.append({"device": r["device"], "device_type": r["device_type"], "label": lab, "start": a, "end": b})
    tdf = pd.DataFrame(t_rows)
    tsum = {}
    if len(tdf):
        t0 = tdf["start"].min()
        tdf["start_h"] = (tdf["start"] - t0) / 3600
        tdf["end_h"] = (tdf["end"] - t0) / 3600
        tl = tdf.groupby(["device_type", "label"]).agg(start_h=("start_h", "min"), end_h=("end_h", "max"),
                                                       devices=("device", "nunique")).reset_index()
        tl.to_csv(out / "time_by_device_type_label.csv", index=False)
        tlab = tdf.groupby("label").agg(start_h=("start_h", "min"), end_h=("end_h", "max"),
                                        devices=("device", "nunique")).reset_index().sort_values("start_h")
        tlab["span_h"] = tlab["end_h"] - tlab["start_h"]
        tlab.to_csv(out / "time_by_label.csv", index=False)
        plots.timeline(tl.assign(row_label=tl["device_type"] + " | " + tl["label"], start=tl["start_h"],
                                 end=tl["end_h"], color_key=tl["label"]),
                       out / "timeline.png", "When each label occurs (hours from first packet)")
        tsum = {"first": float(tdf["start"].min()), "last": float(tdf["end"].max()),
                "span_h": float((tdf["end"].max() - tdf["start"].min()) / 3600)}
    methods: Counter = Counter()
    unparsed = 0
    for r in results:
        methods.update(r["time"]["methods"])
        unparsed += r["time"]["unparsed"]

    proto: Counter = Counter()
    for r in results:
        for p in r["protocols"]:
            proto[(p["protocols"], p["label"])] += p["rows"]
    pdf = pd.DataFrame([{"protocols": k[0], "label": k[1], "rows": v} for k, v in proto.items()])
    if len(pdf):
        pdf = pdf.sort_values("rows", ascending=False)
        pdf.to_csv(out / "protocols_by_label.csv", index=False)
        proto_top = pdf.groupby("protocols")["rows"].sum().sort_values(ascending=False).reset_index()
    else:
        proto_top = pd.DataFrame()

    gl = list(lab_tot.keys())
    gidx = {k: i for i, k in enumerate(gl)}
    H, L, C = [], [], []
    for r, pth in zip(results, arrays_paths):
        z = np.load(pth)
        lut = np.array([gidx[x] for x in r["label_list"]], dtype=np.int32) if r["label_list"] else np.zeros(1, np.int32)
        H.append(z["h"])
        L.append(lut[z["l"].astype(np.int64)] if len(z["l"]) else z["l"].astype(np.int32))
        C.append(z["c"])
    h = np.concatenate(H) if H else np.empty(0, np.uint64)
    lab = np.concatenate(L).astype(np.int32) if L else np.empty(0, np.int32)
    cnt = np.concatenate(C) if C else np.empty(0, np.int64)
    dup_raw, pairs_raw, over_raw = _dup_stats(h, lab, cnt, gl)
    classes = sorted({c for c in mapper.mapping(gl).values() if c is not None})
    cidx = {c: i for i, c in enumerate(classes)}
    lut_cls = np.array([cidx.get(mapper.map_one(x), -1) for x in gl], dtype=np.int32)
    keep = lut_cls[lab] >= 0 if len(lab) else np.zeros(0, bool)
    dup_cls, pairs_cls, over_cls = _dup_stats(h[keep], lut_cls[lab[keep]], cnt[keep], classes)
    for name, d in (("duplicates_by_label", dup_raw), ("duplicates_by_class", dup_cls)):
        d.to_csv(out / f"{name}.csv")
    pairs_raw.to_csv(out / "shared_vectors_label_pairs.csv", index=False)
    pairs_cls.to_csv(out / "shared_vectors_class_pairs.csv", index=False)

    if len(labels_total):
        plots.bar_log(labels_total.set_index("label")["rows"], "Packets per raw label", out / "labels.png")
    if len(cls_tot):
        plots.bar_log(cls_tot.rename(index=str), "Packets per class (preview of coarse mapping)", out / "classes.png")
    hm = by_type.drop(columns=["TOTAL"])
    if hm.size:
        plots.heatmap(hm, "Packets per device type and label", out / "device_type_by_label.png")

    scanned = files["seconds"].sum()
    scanned_mb = files["mb"].sum()
    summary = {
        "files_scanned": len(results), "files_missing": missing, "rows": total_rows,
        "csv_gb": round(sum(m["size"] for m in csvs) / 2**30, 3), "labels": dict(lab_tot.most_common()),
        "classes_preview": {str(k): int(v) for k, v in cls_tot.items()},
        "header_variants": len(variants), "label_columns": dict(label_cols), "time_methods": dict(methods),
        "time_unparsed": unparsed, "time": tsum, "bad_lines": int(files["bad_lines"].sum()),
        "duplicates_raw": over_raw, "duplicates_class": over_cls,
        "scan_mb_per_s": round(scanned_mb / max(scanned, 1e-6), 1),
        "devices": int(files["device"].nunique()), "device_types": int(files["device_type"].nunique()),
        "devices_with_attack_rows": int((devices_with_attacks > 0).sum()),
    }
    atomic_write_json(out / "summary.json", summary)
    md = summary_markdown(cfg, inv, summary, files, labels_total, mapping, by_type, variants, columns,
                          (tl if len(tdf) else pd.DataFrame()), (tlab if len(tdf) else pd.DataFrame()),
                          proto_top, pdf, dup_raw, dup_cls, pairs_cls)
    atomic_write_text(out / "SUMMARY.md", md)
    tracer.info(STAGE, "summary", f"{total_rows:,} rows in {len(results)} files; "
                f"{over_raw.get('unique_vectors', 0):,} distinct header vectors", classes=summary["classes_preview"],
                path=str(out / "SUMMARY.md"))
    if missing:
        tracer.warn(STAGE, "incomplete", f"{missing} file(s) not scanned yet - run explore again")
    return summary


def summary_markdown(cfg, inv, s, files, labels_total, mapping, by_type, variants, columns, tl, tlab,
                     proto_top, pdf, dup_raw, dup_cls, pairs_cls) -> str:
    from .envinfo import short_env

    L: List[str] = ["# Gotham dataset - exploration summary", ""]
    L += [f"Files scanned: **{s['files_scanned']}** of {s['files_scanned'] + s['files_missing']} | "
          f"rows: **{s['rows']:,}** | CSV text: {s['csv_gb']} GB | devices: {s['devices']} "
          f"({s['device_types']} types) | scan speed {s['scan_mb_per_s']} MB/s", ""]
    L += ["## 1. Environment", "", "```", short_env(), "```", ""]
    src = inv.get("source", {})
    L += ["## 2. Archive", "", f"`{src.get('path')}` - {_fmt_bytes(src.get('bytes', 0))}", "",
          md_table(pd.DataFrame(inv.get("by_kind", []))), ""]
    fold = pd.DataFrame(inv.get("by_folder", []))
    if len(fold):
        L += [md_table(fold, 25), ""]
    if inv.get("throughput"):
        L += [f"Read speed from the archive: **{inv['throughput']['mb_per_s']} MB/s**", ""]
    L += ["## 3. Files", "", md_table(files.sort_values("rows", ascending=False)
                                      [["file", "device_type", "rows", "mb", "segments", "labels", "bad_lines"]], 15),
          "", md_table(files.groupby("device_type").agg(files=("file", "size"), rows=("rows", "sum"),
                                                        mb=("mb", "sum")).sort_values("rows", ascending=False)
                       .reset_index(), 30), ""]
    L += ["## 4. Schema", "", f"Header variants across files: **{len(variants)}**; label column(s): "
          f"{s['label_columns']}", ""]
    for i, (hdr, fl) in enumerate(list(variants.items())[:4]):
        L += [f"- variant {i + 1} ({len(fl)} files, e.g. {fl[0]}): `{hdr[:900]}`"]
    L += ["", "## 5. Columns (statistics from every few blocks)", "",
          md_table(columns, 60), ""]
    L += ["## 6. Labels", "", md_table(labels_total, 30), "",
          "Preview of the coarse class mapping (edit `labels` in the config if it is wrong):", "",
          md_table(mapping[["raw_label", "class", "rule", "rows"]], 30), "",
          "Packets per device type and label:", "", md_table(by_type.reset_index(), 30), ""]
    L += ["## 7. Time and capture structure", "",
          f"Time parse methods: {s['time_methods']}; unparsed timestamps: {s['time_unparsed']:,}; "
          f"overall span: {s['time'].get('span_h', float('nan')):.2f} h", ""]
    if len(tlab):
        L += [md_table(tlab, 30), ""]
    L += ["Files whose time jumps (capture joins) are listed in files.csv (`segments`).", ""]
    L += ["## 8. Protocol stacks", "", md_table(proto_top.head(20), 20), ""]
    if len(pdf):
        top_per = pdf.sort_values("rows", ascending=False).groupby("label").head(3)
        L += ["Top stacks per label:", "", md_table(top_per, 45), ""]
    L += ["## 9. Duplicates and label ambiguity", "",
          "Header vector = every column except time, IP/MAC addresses and checksums "
          "(TCP options reduced to their length).", "",
          f"Distinct header vectors: **{s['duplicates_raw'].get('unique_vectors', 0):,}** for "
          f"{int(s['duplicates_raw'].get('packets', 0)):,} packets; best possible accuracy on these fields "
          f"(each vector predicted as its majority label): raw labels "
          f"**{s['duplicates_raw'].get('accuracy_ceiling', float('nan')):.4f}**, coarse classes "
          f"**{s['duplicates_class'].get('accuracy_ceiling', float('nan')):.4f}**", "",
          "Per raw label:", "", md_table(dup_raw.reset_index().rename(columns={"index": "label"}), 30), "",
          "Per coarse class:", "", md_table(dup_cls.reset_index().rename(columns={"index": "class"}), 20), "",
          "Classes that share identical header vectors:", "", md_table(pairs_cls, 20), ""]
    L += ["## 10. Metadata files", ""]
    for mi in inv.get("metadata", [])[:12]:
        L += [f"- `{Path(mi['file']).name}`: {json.dumps(mi.get('shape'))[:400]}"]
    L += ["", "## 11. Things to check", ""]
    warn = []
    if s["files_missing"]:
        warn.append(f"{s['files_missing']} file(s) not scanned yet - run explore again.")
    if len(variants) > 1:
        warn.append("Files do not all share the same header (see section 4).")
    if s["bad_lines"]:
        warn.append(f"{s['bad_lines']:,} malformed lines were skipped.")
    if s["time_unparsed"]:
        warn.append(f"{s['time_unparsed']:,} timestamps could not be parsed.")
    um = mapping[mapping["rule"].astype(str).str.startswith("unmapped")]
    if len(um):
        warn.append("Labels not covered by the coarse mapping: " + ", ".join(um["raw_label"].astype(str)))
    if len(dup_cls):
        low = dup_cls[dup_cls["recall_ceiling"] < 0.9]
        for c, r in low.iterrows():
            warn.append(f"Class '{c}': {r['conflict_share']:.1%} of its packets share their exact header "
                        f"vector with another class; recall on header fields alone is capped near "
                        f"{r['recall_ceiling']:.2f}.")
    if len(dup_raw):
        for lab, r in dup_raw[dup_raw["recall_ceiling"] < 0.5].iterrows():
            warn.append(f"Raw label '{lab}': only {r['recall_ceiling']:.0%} of its packets can be told apart from "
                        f"other labels using header fields alone.")
    multi = columns[(columns["multi_value_pct"] > 0.01)]
    if len(multi):
        warn.append("Columns with comma-separated repeated values: " + ", ".join(multi["column"]))
    hexc = columns[(columns["hex_pct"] > 1)]
    if len(hexc):
        warn.append("Hex-encoded columns: " + ", ".join(hexc["column"]))
    L += [f"- {w}" for w in warn] or ["- nothing unusual detected"]
    uv = s["duplicates_raw"].get("unique_vectors", 0)
    L += ["", "## 12. Planning numbers", "",
          f"- Full read of all CSVs at the measured speed: ~{s['csv_gb'] * 1024 / max(s['scan_mb_per_s'], 0.1) / 60:.0f} min",
          f"- Training matrix if every packet is kept (50 float32 features): "
          f"~{s['rows'] * 50 * 4 / 2**30:.1f} GB",
          f"- Training matrix after merging identical header vectors: ~{uv * 50 * 4 / 2**30:.2f} GB "
          f"({uv:,} rows)", ""]
    return "\n".join(L) + "\n"

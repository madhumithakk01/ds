"""Step 6: dataset audit and the behaviour-consistent (corrected) label set.

``audit`` (resumable, one unit per ingested file) reads the ingested packets and

1. measures what the official labelling rules produce for the Mirai infection
   phase: for every packet labelled TCP Scan or Telnet Brute Force, whether it
   involves the attacker host named in the metadata or other internal hosts
   (infected bots), and whether its telnet connection carries data (a login
   session) or only probes;
2. describes the packets labelled Unknown (hosts, protocols, ports);
3. finds captures of other devices copied into a file (prefix-glob bug);
4. computes corrected labels - telnet connections by behaviour, masscan
   packets recovered from Unknown, copies marked ``Duplicate copy`` - and the
   time-blocked split of the corrected labels.

Corrected labels are written as small sidecar files next to the ingested
parts; ``corrections.enabled: true`` makes step 2 onwards use them.
``reports/audit/AUDIT.md`` documents every finding.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import layout
from .config import get_path
from .explore import md_table
from .ingest import _read_parquet, _to_parquet_bytes
from .labels import DUPLICATE
from .split import SPLITS, assign, make_split, split_name, split_spec
from .state import (Deadline, GracefulStop, Ledger, atomic_write_bytes, atomic_write_json, atomic_write_text,
                    crash_point, ensure_dir, output_ok, read_json, retry, stable_hash)

STAGE = "audit"
CAND = ("TCP Scan", "Telnet Brute Force")
NEEDED = ["t", "seg", "label_raw", "ip_src", "ip_dst", "ip_proto", "tcp_srcport", "tcp_dstport", "tcp_flags",
          "frame_len", "tcp_options_len"]
REPO = "https://github.com/othmbela/gotham-network-packet-labeller"
REPO_COMMIT = "cad1575 (2026-08-23)"


def corrections_spec(cfg) -> Dict[str, Any]:
    c = cfg.get("corrections", {}) or {}
    return {"v": 1, "ingest": stable_hash(layout.ingest_spec(cfg)),
            "telnet_by_behaviour": bool(c.get("telnet_by_behaviour", True)),
            "telnet_ports": sorted(int(x) for x in c.get("telnet_ports", [23, 2323])),
            "masscan_ip": c.get("masscan_ip"), "drop_duplicate_devices": bool(c.get("drop_duplicate_devices", True))}


def relabel_dir(cfg) -> Path:
    return ensure_dir(layout.work(cfg) / "data" / f"relabel-{stable_hash(corrections_spec(cfg))}")


def corrected_cfg(cfg) -> Dict[str, Any]:
    c = copy.deepcopy(cfg)
    c.setdefault("corrections", {})["enabled"] = True
    return c


def _ingest_metas(cfg) -> Tuple[Path, List[Dict[str, Any]]]:
    ing = layout.ingest_dir(cfg)
    led = Ledger(ing / "_ledger.jsonl")
    metas = sorted([read_json(r["meta"]) for r in led.done().values() if output_ok(r.get("meta", ""))],
                   key=lambda m: m["id"])
    if not metas:
        raise RuntimeError(f"No ingested files in {ing}. Run `python -m gids ingest` first.")
    return ing, metas


# =====================================================================
# one file

def read_cols(path: Path, columns: List[str]) -> pd.DataFrame:
    """Selected columns of a Parquet part; columns the part lacks come back empty (NaN), rows are never lost."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    present = [c for c in columns if c in pf.schema_arrow.names]
    df = pf.read(columns=present).to_pandas() if present else pd.DataFrame(index=range(pf.metadata.num_rows))
    for c in columns:
        if c not in df:
            df[c] = np.nan
    return df[columns]


def _load(ing: Path, meta: Dict[str, Any], columns: List[str]) -> Tuple[pd.DataFrame, List[int]]:
    """Columns of every part of a file, text columns as one categorical per column; also the part lengths."""
    from pandas.api.types import union_categoricals

    parts, lengths = [], []
    for p in meta["parts"]:
        df = read_cols(ing / p, columns)
        lengths.append(len(df))
        parts.append(df)
    out = {}
    for c in columns:
        vals = [p[c] for p in parts if len(p)]
        if not vals:
            out[c] = pd.Series([], dtype=float)
        elif isinstance(vals[0].dtype, pd.CategoricalDtype) or vals[0].dtype == object:
            out[c] = pd.Series(union_categoricals([v.astype("category") for v in vals], ignore_order=True))
        else:
            out[c] = pd.Series(np.concatenate([v.to_numpy() for v in vals]))
    return pd.DataFrame(out), lengths


def _segment_ips(seg: np.ndarray, src: np.ndarray, dst: np.ndarray, ips: List[str]) -> pd.DataFrame:
    both = pd.DataFrame({"seg": np.r_[seg, seg], "ip": np.r_[src, dst]})
    both = both[both["ip"] >= 0]
    top = both.groupby(["seg", "ip"]).size().reset_index(name="n").sort_values(["seg", "n"], ascending=[True, False])
    top = top.drop_duplicates("seg").set_index("seg")["ip"]
    seg_ip = pd.Series(seg).map(top).fillna(-2).to_numpy(np.int64)
    hit = (src == seg_ip) | (dst == seg_ip)
    rows = []
    for s, d in pd.DataFrame({"seg": seg, "hit": hit}).groupby("seg", sort=True):
        code = int(top.get(s, -1))
        rows.append({"seg": int(s), "rows": int(len(d)), "device_ip": ips[code] if code >= 0 else None,
                     "share": float(d["hit"].mean())})
    return pd.DataFrame(rows)


def telnet_behaviour(lab: np.ndarray, seg: np.ndarray, src: np.ndarray, dst: np.ndarray, proto: np.ndarray,
                     sp: np.ndarray, dp: np.ndarray, flags: np.ndarray, flen: np.ndarray, optlen: np.ndarray,
                     ports: np.ndarray, a_code: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per packet: labelled TCP Scan / Telnet Brute Force; on a telnet port; in a telnet connection that carries
    data; to or from the Mirai attacker host.

    A connection = same capture segment, hosts and ports (IPs as integer codes); it carries data when any of its
    packets has the PSH flag or bytes beyond the TCP/IP headers.
    """
    cand = np.isin(lab, CAND)
    telnet = cand & (proto == 6) & (np.isin(sp, ports) | np.isin(dp, ports))
    payload = ((flags & 8) > 0) | ((flen - 54 - optlen) > 6)   # PSH, or bytes beyond the TCP/IP headers
    flow_data = np.zeros(len(lab), dtype=bool)
    if telnet.any():
        k = np.flatnonzero(telnet)
        a, b = np.minimum(src[k], dst[k]), np.maximum(src[k], dst[k])
        pa, pb = np.minimum(sp[k], dp[k]), np.maximum(sp[k], dp[k])
        g = pd.DataFrame({"seg": seg[k], "a": a, "b": b, "pa": pa, "pb": pb, "pl": payload[k]})
        flow_data[k] = g.groupby(["seg", "a", "b", "pa", "pb"])["pl"].transform("any").to_numpy(bool)
    return cand, telnet, flow_data, (src == a_code) | (dst == a_code)


def behaviour_names(telnet: np.ndarray, flow_data: np.ndarray, attacker: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    host = np.where(attacker, "Mirai attacker host", "other internal hosts")
    kind = np.where(~telnet, "not telnet", np.where(flow_data, "telnet data connection",
                                                    "telnet probe-only connection"))
    return host, kind


def relabel_member(meta: Dict[str, Any], ing: Path, out: Path, cfg) -> Dict[str, Any]:
    spec = corrections_spec(cfg)
    attacker = str(get_path(cfg, "audit.mirai_attacker_ip", "192.168.0.100"))
    probe_port = float(get_path(cfg, "audit.mirai_probe_port", 65148))
    df, lengths = _load(ing, meta, NEEDED)
    from pandas.api.types import union_categoricals

    n = len(df)
    info: Dict[str, Any] = {"member": meta["member"], "id": meta["id"], "device_type": meta["device_type"],
                            "parts": meta["parts"], "lengths": lengths, "rows": int(n), "sidecars": None}
    if n == 0:
        info.update(segments=[], cand=[], unknown=[], transitions=[])
        return info
    ipcat = union_categoricals([df["ip_src"].astype("category"), df["ip_dst"].astype("category")], ignore_order=True)
    ips = [str(x) for x in ipcat.categories]
    src = ipcat.codes[:n].astype(np.int64)
    dst = ipcat.codes[n:].astype(np.int64)
    seg = df["seg"].to_numpy(np.int64)
    info["segments"] = _segment_ips(seg, src, dst, ips).to_dict(orient="records")

    lab = df["label_raw"].astype(str).to_numpy(dtype=object)
    fixed = lab.copy()
    proto = df["ip_proto"].to_numpy(np.float64)
    sp, dp = df["tcp_srcport"].to_numpy(np.float64), df["tcp_dstport"].to_numpy(np.float64)
    flags = np.nan_to_num(df["tcp_flags"].to_numpy(np.float64)).astype(np.int64)
    flen = df["frame_len"].to_numpy(np.float64)
    optlen = np.nan_to_num(df["tcp_options_len"].to_numpy(np.float64))
    ip_code = {ip: i for i, ip in enumerate(ips)}
    a_code = ip_code.get(attacker, -9)

    # --- Mirai infection labels: host x behaviour
    ports = np.array(spec["telnet_ports"], dtype=np.float64)
    cand, telnet, flow_data, attacker = telnet_behaviour(lab, seg, src, dst, proto, sp, dp, flags, flen, optlen, ports,
                                                         a_code)
    if cand.any():
        k = np.flatnonzero(cand)
        host, kind = behaviour_names(telnet[k], flow_data[k], attacker[k])
        probe = (sp[k] == probe_port) | (dp[k] == probe_port)
        c = pd.DataFrame({"seg": seg[k], "label": lab[k], "host": host, "kind": kind, "probe_port": probe})
        info["cand"] = c.groupby(["seg", "label", "host", "kind", "probe_port"]).size().reset_index(
            name="packets").to_dict(orient="records")
        if spec["telnet_by_behaviour"]:
            kt = np.flatnonzero(telnet)
            fixed[kt] = np.where(flow_data[kt], "Telnet Brute Force", "TCP Scan")
    else:
        info["cand"] = []

    # --- Unknown packets: who, what
    unk = lab == "Unknown"
    if unk.any():
        k = np.flatnonzero(unk)
        mip = spec["masscan_ip"]
        m_code = ip_code.get(str(mip), -9) if mip else -9
        masscan = (src[k] == m_code) | (dst[k] == m_code)
        u = pd.DataFrame({"seg": seg[k], "src": [ips[i] if i >= 0 else "" for i in src[k]],
                          "dst": [ips[i] if i >= 0 else "" for i in dst[k]], "proto": proto[k],
                          "dport": np.where(np.isnan(dp[k]), -1, dp[k]), "masscan": masscan,
                          "syn_only": (flags[k] & 0x12) == 0x02})
        info["unknown"] = u.groupby(["seg", "src", "dst", "proto", "dport", "masscan", "syn_only"]).size().reset_index(
            name="packets").to_dict(orient="records")
        if mip:
            fixed[k[masscan]] = "TCP Scan"
    else:
        info["unknown"] = []

    tr = pd.DataFrame({"seg": seg, "from": lab, "to": fixed})
    tr = tr[tr["from"] != tr["to"]]
    info["transitions"] = tr.groupby(["seg", "from", "to"]).size().reset_index(name="packets").to_dict(
        orient="records")
    if len(tr):  # sidecars only for files whose labels change
        fid = stable_hash(meta["id"], 16)
        names, start = [], 0
        for i, ln in enumerate(lengths):
            name = f"{fid}-{i:05d}.labels.parquet"
            atomic_write_bytes(out / name, _to_parquet_bytes(pd.DataFrame(
                {"label_fixed": pd.Categorical(fixed[start:start + ln])})))
            names.append(name)
            start += ln
        info["sidecars"] = names
    return info


# =====================================================================
# duplicates, corrected split, report

def duplicate_segments(infos: List[Dict[str, Any]]) -> Tuple[Dict[str, List[int]], pd.DataFrame]:
    """Segments of a file whose device IP is the only device IP of another file whose name extends this one's.

    That is exactly what ``glob(f"{iot_device}*.csv")`` in the dataset's data_preparation.py produces: the files
    of devices -10 ... -15 are concatenated into device -1's file.
    """
    stem = {i["id"]: Path(i["member"]).name.rsplit(".", 1)[0] for i in infos}
    owners: Dict[str, List[str]] = {}
    for i in infos:
        ips = {s["device_ip"] for s in i["segments"] if s["device_ip"]}
        if len(ips) == 1:
            owners.setdefault(next(iter(ips)), []).append(i["id"])
    dup: Dict[str, List[int]] = {}
    rows = []
    for i in infos:
        for s in i["segments"]:
            others = [g for g in owners.get(s["device_ip"], []) if g != i["id"]]
            prefix = [g for g in others if stem[g].startswith(stem[i["id"]]) and len(stem[g]) > len(stem[i["id"]])]
            if prefix:
                dup.setdefault(i["id"], []).append(int(s["seg"]))
            if others:
                rows.append({"file": stem[i["id"]] + ".csv", "copy of": stem[(prefix or others)[0]] + ".csv",
                             "segment": s["seg"], "packets": s["rows"], "device_ip": s["device_ip"],
                             "prefix_glob": bool(prefix)})
    return dup, pd.DataFrame(rows, columns=["file", "copy of", "segment", "packets", "device_ip", "prefix_glob"])


def fixed_labels(rdir: Path, info: Dict[str, Any], part_index: int, raw: pd.Series) -> np.ndarray:
    """Corrected labels of one ingested part (the original ones when the file has no corrections)."""
    if info.get("sidecars"):
        return _read_parquet(rdir / info["sidecars"][part_index])["label_fixed"].astype(str).to_numpy(dtype=object)
    return raw.astype(str).to_numpy(dtype=object)


def run_audit(cfg, tracer, deadline: Deadline) -> Dict[str, Any]:
    ing, metas = _ingest_metas(cfg)
    out = relabel_dir(cfg)
    atomic_write_json(out / "_spec.json", corrections_spec(cfg))
    ledger = Ledger(out / "_ledger.jsonl")
    done = ledger.done()
    todo = [m for m in metas if not (done.get(m["id"]) and output_ok(done[m["id"]].get("info", "")))]
    tracer.info(STAGE, "plan", f"{len(metas)} files -> {out}; {len(metas) - len(todo)} already audited, "
                f"{len(todo)} to go")
    for k, m in enumerate(todo):
        deadline.check()
        tracer.progress(STAGE, len(metas) - len(todo) + k, len(metas), "files", every_s=30,
                        current=Path(m["member"]).name)
        ledger.append(m["id"], "start")
        try:
            info = retry(lambda: relabel_member(m, ing, out, cfg), tries=3, base_delay=10.0,
                         on_error=lambda a, e: tracer.warn(STAGE, "io_retry", f"attempt {a}: {e}"))
        except (GracefulStop, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001
            tracer.exception(STAGE, "file_failed", exc, member=m["member"])
            ledger.append(m["id"], "failed", error=f"{type(exc).__name__}: {exc}")
            continue
        ipath = out / f"{stable_hash(m['id'], 16)}.json"
        atomic_write_json(ipath, info)
        ledger.append(m["id"], "done", info=str(ipath))
        crash_point(STAGE)
    failed = [m["member"] for m in metas if ledger.latest().get(m["id"], {}).get("status") == "failed"]
    if failed:
        raise RuntimeError(f"{len(failed)} file(s) failed; run audit again")
    infos = [read_json(out / f"{stable_hash(m['id'], 16)}.json") for m in metas]
    return finish_audit(cfg, tracer, ing, out, infos)


def corrected_split(cfg, tracer, ing: Path, out: Path, infos: List[Dict[str, Any]],
                    dup: Dict[str, List[int]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-blocked split of the corrected labels (copies left out) and its packet counts per label and split."""
    from pandas.api.types import union_categoricals

    sname = split_name(cfg)
    spath = out / f"{sname}.json"
    cpath = out / f"{sname}.counts.csv"
    if output_ok(spath) and output_ok(cpath):
        return pd.DataFrame(read_json(spath)["blocks"]), pd.read_csv(cpath, index_col=0)
    ts, labs = [], []
    drop_dups = corrections_spec(cfg)["drop_duplicate_devices"]
    for info in infos:
        dsegs = set(dup.get(info["id"], [])) if drop_dups else set()
        for pi, p in enumerate(info["parts"]):
            df = read_cols(ing / p, ["t", "seg", "label_raw"])
            if not len(df):
                continue
            lab = fixed_labels(out, info, pi, df["label_raw"])
            keep = ~df["seg"].isin(dsegs).to_numpy()
            if not keep.any():
                continue
            ts.append(df["t"].to_numpy()[keep])
            labs.append(pd.Categorical(lab[keep].astype(str)))
    t = np.concatenate(ts) if ts else np.empty(0)
    lab = pd.Series(union_categoricals(labs, ignore_order=True)) if labs else pd.Series([], dtype="category")
    with tracer.span(STAGE, "corrected_split"):
        table = make_split(lab, t, split_spec(cfg))
        code = assign(lab, t, table)
    atomic_write_json(spath, {"spec": split_spec(cfg), "blocks": table.to_dict(orient="records")})
    lc = lab.cat.codes.to_numpy().astype(np.int64)
    nl, ns = len(lab.cat.categories), len(SPLITS) + 1
    counts = np.bincount(lc * ns + np.where(code < 0, len(SPLITS), code), minlength=nl * ns).reshape(nl, ns)
    by_label = pd.DataFrame(counts[:, :len(SPLITS)], columns=list(SPLITS),
                            index=pd.Index([str(c) for c in lab.cat.categories], name="label"))
    by_label["total"] = counts.sum(axis=1)
    by_label.to_csv(cpath)
    return table, by_label


def finish_audit(cfg, tracer, ing: Path, out: Path, infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    rep = layout.base_reports(cfg, "audit")
    spec = corrections_spec(cfg)
    dup, dup_table = duplicate_segments(infos)
    dsegs = {k: set(v) for k, v in dup.items()} if spec["drop_duplicate_devices"] else {}

    def frame(key: str) -> pd.DataFrame:
        rows = [dict(r, file=i["id"]) for i in infos for r in i.get(key, [])]
        df = pd.DataFrame(rows)
        if len(df):
            df["copy"] = [int(s) in dsegs.get(f, set()) for f, s in zip(df["file"], df["seg"])]
        return df

    cand, unk, trans = frame("cand"), frame("unknown"), frame("transitions")
    original = cand[~cand["copy"]] if len(cand) else cand
    infection = original.groupby(["label", "host", "kind"])["packets"].sum().reset_index() if len(original) \
        else pd.DataFrame(columns=["label", "host", "kind", "packets"])
    probe_pk = original[original["probe_port"]].groupby("label")["packets"].sum().to_dict() if len(original) else {}
    u = unk[~unk["copy"]] if len(unk) else unk
    unknown = {"packets": int(u["packets"].sum()) if len(u) else 0,
               "masscan_packets": int(u.loc[u["masscan"], "packets"].sum()) if len(u) else 0,
               "syn_only_packets": int(u.loc[u["syn_only"], "packets"].sum()) if len(u) else 0}
    top_pairs = u.groupby(["src", "dst"])["packets"].sum().sort_values(ascending=False).head(8).reset_index() \
        if len(u) else pd.DataFrame()
    top_ports = u.groupby("dport")["packets"].sum().sort_values(ascending=False).head(8).reset_index() \
        if len(u) else pd.DataFrame()
    tr = trans[~trans["copy"]] if len(trans) else trans
    transitions = tr.groupby(["from", "to"])["packets"].sum().reset_index().sort_values("packets", ascending=False) \
        if len(tr) else pd.DataFrame(columns=["from", "to", "packets"])

    table, by_label = corrected_split(cfg, tracer, ing, out, infos, dup)
    cdir = ensure_dir(layout.work(cfg) / "reports_corrected" / "ingest")
    by_label.to_csv(cdir / "split_by_label.csv")

    def pk(label, host=None, kind=None) -> int:
        d = infection[infection["label"] == label]
        if host:
            d = d[d["host"] == host]
        if kind:
            d = d[d["kind"] == kind]
        return int(d["packets"].sum())

    scan = pk("TCP Scan")
    bot_sessions = pk("TCP Scan", "other internal hosts", "telnet data connection")
    bf = pk("Telnet Brute Force")
    summary = {
        "relabel_dir": str(out), "files": len(infos),
        "duplicate_packets": int(dup_table.loc[dup_table["prefix_glob"], "packets"].sum()),
        "duplicate_files": sorted(dup_table.loc[dup_table["prefix_glob"], "file"].unique().tolist()),
        "shared_ip_not_prefix_packets": int(dup_table.loc[~dup_table["prefix_glob"], "packets"].sum()),
        "tcp_scan_packets": scan, "tcp_scan_bot_telnet_sessions": bot_sessions,
        "tcp_scan_bot_session_share": bot_sessions / scan if scan else None,
        "brute_force_packets": bf, "brute_force_attacker_share": pk("Telnet Brute Force", "Mirai attacker host") / bf
        if bf else None, "probe_port_packets": {k: int(v) for k, v in probe_pk.items()},
        "unknown": unknown, "transitions": transitions.to_dict(orient="records"),
        "corrected_split_counts": by_label.reset_index().to_dict(orient="records"),
        "infection_table": infection.to_dict(orient="records")}
    atomic_write_json(rep / "audit.json", summary)
    for name, d in (("duplicates", dup_table), ("infection_labels", infection), ("unknown_pairs", top_pairs),
                    ("transitions", transitions)):
        d.to_csv(rep / f"{name}.csv", index=False)
    atomic_write_text(rep / "AUDIT.md", audit_markdown(cfg, summary, dup_table, infection, top_pairs, top_ports,
                                                       transitions, by_label))
    tracer.info(STAGE, "done", f"audit written to {rep / 'AUDIT.md'}")
    return summary


def audit_markdown(cfg, s, dup_table, infection, top_pairs, top_ports, transitions, by_label) -> str:
    att = get_path(cfg, "audit.mirai_attacker_ip")
    mip = get_path(cfg, "corrections.masscan_ip")
    L = ["# Gotham Dataset 2025 - audit", "",
         f"Measured on the ingested packets of {s['files']} files. Code references are to the dataset's public "
         f"processing repository {REPO} at commit {REPO_COMMIT}.", ""]
    L += ["## 1. Captures of other devices copied into a file", "",
          "`data_preparation.py` builds `processed/<device>.csv` from every labelled file matching "
          "`glob(f\"{iot_device}*.csv\")`, so the files of devices `-10` ... `-15` are also concatenated into "
          "device `-1`'s file. Segments found here (device IP of the segment = the only device IP of a file whose "
          "name extends this one's):", "",
          f"**{s['duplicate_packets']:,} packets** in {len(s['duplicate_files'])} files: "
          + ", ".join(f"`{f}`" for f in s["duplicate_files"]), "",
          md_table(dup_table.groupby(["file", "copy of", "prefix_glob"])["packets"].sum().reset_index()
                   if len(dup_table) else dup_table, max_rows=40), "",
          f"Segments sharing another file's device IP without the prefix pattern (not changed): "
          f"{s['shared_ip_not_prefix_packets']:,} packets.", ""]
    L += ["## 2. Labels of the Mirai infection phase", "",
          f"`metadata-mirai-infection.json` labels traffic with the attacker host `{att}` as `Telnet Brute Force` "
          "(except its SYN probes from source port "
          f"{get_path(cfg, 'audit.mirai_probe_port')} to ports 23/2323, which are `TCP Scan`); its last rule labels "
          "**any other traffic between internal hosts** `TCP Scan`. `labeller.py` applies the rules in order and "
          "only to packets still `Unknown`, and uses no timestamps. The dataset paper describes bots that scan *and* "
          "brute-force telnet; by these rules the bots' login sessions fall into `TCP Scan`.", "",
          f"- `TCP Scan`: {s['tcp_scan_packets']:,} packets, of which **{s['tcp_scan_bot_telnet_sessions']:,} "
          f"({100 * (s['tcp_scan_bot_session_share'] or 0):.1f}%) belong to telnet connections that carry data "
          "between other internal hosts** (bots' login sessions).",
          f"- `Telnet Brute Force`: {s['brute_force_packets']:,} packets, "
          f"{100 * (s['brute_force_attacker_share'] or 0):.1f}% involving the attacker host.", "",
          "Packets per label, host and behaviour (a telnet connection = same hosts and ports within a capture "
          "segment; it carries data when any of its packets has the PSH flag or bytes beyond the TCP/IP headers):",
          "", md_table(infection, max_rows=30), ""]
    u = s["unknown"]
    L += ["## 3. Packets labelled Unknown", "",
          "`run_labeling.py` loads `metadata/metadata-{event}.json` for each scenario; for `network-scanning` that "
          "file does not exist in the repository (the masscan rules are in `metadata-masscan.json`), and every "
          "published pipeline drops `Unknown` packets.", "",
          f"- {u['packets']:,} packets; **{u['masscan_packets']:,} involve the masscan host `{mip}`**; "
          f"{u['syn_only_packets']:,} are bare SYN packets.", "", "Most frequent host pairs:", "",
          md_table(top_pairs, max_rows=8), "", "Most frequent destination ports:", "", md_table(top_ports,
                                                                                               max_rows=8), ""]
    L += ["## 4. Other points in the public code", "",
          "- `labeller.py` (`label_malicious_traffic_by_ip`) labels by IP / port / protocol rules only; the dataset "
          "paper describes a timing-synchronisation (time-window) labelling.",
          "- The reference preparation notebook (`GothamDataset2025 -- Data Preparation.ipynb`) splits packets "
          "with a stratified random split (`train_test_split(..., stratify=labels, random_state=42)`) and keeps "
          "the absolute `timestamp`, `ip.ttl` and the IP/TCP checksums as model inputs.", ""]
    L += ["## 5. Corrected labels", "",
          "Rules (`corrections` in `configs/default.yaml`): telnet connections labelled TCP Scan or Telnet Brute "
          "Force are relabelled by behaviour - data-carrying connections `Telnet Brute Force`, probe-only "
          f"connections `TCP Scan` - whichever host sent them; `Unknown` packets involving `{mip}` become "
          f"`TCP Scan`; copied captures become `{DUPLICATE}` and are left out.", "",
          "Packets that change label (copies excluded):", "", md_table(transitions, max_rows=20), "",
          "Corrected labels per split (time-blocked, as for the original labels):", "",
          md_table(by_label.reset_index(), max_rows=30), ""]
    return "\n".join(L) + "\n"


def audit_infos(cfg) -> List[Dict[str, Any]]:
    """The audit's per-file results (run `audit` first)."""
    led = Ledger(relabel_dir(cfg) / "_ledger.jsonl").done()
    infos = [read_json(r["info"]) for r in led.values() if output_ok(r.get("info", ""))]
    if not infos:
        raise RuntimeError("No audit results yet: run `python -m gids audit` (or `paper --parts audit`) first.")
    return sorted(infos, key=lambda i: i["id"])


def candidate_test_rows(cfg, tracer, deadline: Deadline, feats: List[str]) -> pd.DataFrame:
    """Test packets labelled TCP Scan / Telnet Brute Force under the original labels and split (copied captures
    left out), with the model inputs of step 2 and the host and behaviour group of the audit.

    Used to see what models trained on the original labels predict for each behaviour (step 6, E2b). One
    cached Parquet file per CSV, so an interrupted run continues with the next file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .prepare import prepare_spec
    from .train import _prepare_infos

    spec = corrections_spec(cfg)
    infos = audit_infos(cfg)
    dup, _ = duplicate_segments(infos)
    _, pinfos, _ = _prepare_infos(cfg)
    seg_ips = {i["id"]: {int(s["seg"]): s["device_ip"] for s in i.get("segments", [])} for i in pinfos}
    ing = layout.ingest_dir(cfg)
    sp = read_json(ing / f"{split_name(cfg)}.json")
    if not sp:
        raise RuntimeError("No split table from step 1 for the current split settings.")
    table = pd.DataFrame(sp["blocks"])
    cache = ensure_dir(relabel_dir(cfg) / f"behaviour-{stable_hash({'prepare': stable_hash(prepare_spec(cfg)), 'feats': feats})}")
    attacker = str(get_path(cfg, "audit.mirai_attacker_ip", "192.168.0.100"))
    ports = np.array(spec["telnet_ports"], dtype=np.float64)
    cols = list(dict.fromkeys(NEEDED + feats))
    todo = [i for i in infos if i.get("cand")]
    frames = []
    for k, info in enumerate(todo):
        deadline.check()
        path = cache / f"{stable_hash(info['id'], 16)}.parquet"
        if output_ok(path):
            frames.append(_read_parquet(path))
            continue
        parts = []
        for p in info["parts"]:
            pf = pq.ParquetFile(ing / p)
            if "label_raw" not in pf.schema_arrow.names or not pf.metadata.num_rows:
                continue
            mask = pf.read(columns=["label_raw"]).to_pandas()["label_raw"].isin(CAND).to_numpy()
            if not mask.any():
                continue
            present = [c for c in cols if c in pf.schema_arrow.names]
            df = pf.read(columns=present).filter(pa.array(mask)).to_pandas()
            for c in cols:
                if c not in df:
                    df[c] = np.nan
            for c in ("ip_src", "ip_dst", "label_raw"):
                df[c] = df[c].astype(str)
            parts.append(df)
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        ipcat = pd.Categorical(np.r_[df["ip_src"].to_numpy(), df["ip_dst"].to_numpy()])
        codes = ipcat.codes.astype(np.int64)
        n = len(df)
        a_code = list(ipcat.categories).index(attacker) if attacker in set(ipcat.categories) else -9
        lab = df["label_raw"].to_numpy(dtype=object)
        _, telnet, flow_data, att = telnet_behaviour(
            lab, df["seg"].to_numpy(np.int64), codes[:n], codes[n:], df["ip_proto"].to_numpy(np.float64),
            df["tcp_srcport"].to_numpy(np.float64), df["tcp_dstport"].to_numpy(np.float64),
            np.nan_to_num(df["tcp_flags"].to_numpy(np.float64)).astype(np.int64), df["frame_len"].to_numpy(np.float64),
            np.nan_to_num(df["tcp_options_len"].to_numpy(np.float64)), ports, a_code)
        host, kind = behaviour_names(telnet, flow_data, att)
        split = assign(df["label_raw"], df["t"].to_numpy(np.float64), table)
        keep = (split == 2) & ~df["seg"].isin(dup.get(info["id"], [])).to_numpy()
        ip = df["seg"].map(seg_ips.get(info["id"], {})).astype(object).to_numpy()
        df["dir_out"] = (df["ip_src"].to_numpy(dtype=object) == ip).astype(np.float32)
        df["dir_in"] = (df["ip_dst"].to_numpy(dtype=object) == ip).astype(np.float32)
        res = df.loc[keep, feats].astype(np.float32).reset_index(drop=True)
        res["label_raw"], res["host"], res["kind"] = lab[keep], host[keep], kind[keep]
        res["device_type"] = info["device_type"]
        atomic_write_bytes(path, _to_parquet_bytes(res))
        frames.append(res)
        tracer.progress(STAGE, k + 1, len(todo), "files (behaviour groups of test packets)", every_s=30)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=feats + ["label_raw", "host",
                                                                                            "kind"])


# =====================================================================
# used by step 2 when corrections are enabled

class Corrections:
    """Corrected labels and copied segments of every file, for prepare."""

    def __init__(self, cfg):
        self.dir = relabel_dir(cfg)
        led = Ledger(self.dir / "_ledger.jsonl").done()
        self.infos = {u: read_json(r["info"]) for u, r in led.items() if output_ok(r.get("info", ""))}
        sname = split_name(cfg)
        sp = read_json(self.dir / f"{sname}.json")
        if not self.infos or not sp:
            raise RuntimeError("Corrected labels are not computed yet: run `python -m gids audit` first.")
        self.table = pd.DataFrame(sp["blocks"])
        dup, _ = duplicate_segments(list(self.infos.values()))
        self.dup = dup if corrections_spec(cfg)["drop_duplicate_devices"] else {}

    def apply(self, meta: Dict[str, Any], part: str, df: pd.DataFrame) -> pd.DataFrame:
        info = self.infos.get(meta["id"])
        if info is None:
            raise RuntimeError(f"{meta['member']} has no audit result; run `python -m gids audit` again.")
        lab = fixed_labels(self.dir, info, info["parts"].index(part), df["label_raw"])
        if self.dup.get(meta["id"]):
            lab = np.where(df["seg"].isin(self.dup[meta["id"]]).to_numpy(), DUPLICATE, lab)
        return df.assign(label_raw=pd.Categorical(lab))

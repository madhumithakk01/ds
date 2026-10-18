"""Step 2: ingested packets -> deduplicated, labelled training tables + checks.

``prepare`` (resumable, one unit per ingested CSV file)
    1. finds the device IP of every *capture segment* (some files are several
       captures joined end to end, each from a different device) and
       recomputes ``dir_in`` / ``dir_out`` from it;
    2. maps raw labels to classes, drops labels left out of training and the
       packets that fall in a split gap, and tags every packet with its split
       and with its episode (the n-th separate run of its attack);
    3. merges packets that are identical in every feature, label, split and
       episode within a file, keeping the number of copies as ``weight``;
    4. writes one Parquet table per file to ``data/prepare-<hash>/rows``.

When every file is done ``reports/prepare/PREPARE.md`` is written: rows per
class and split, a *shortcut check* (how well each feature on its own
separates the classes on the validation split), value tables for
low-cardinality fields, per-class medians and the segment / device-IP table.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import layout
from .config import get_path
from .explore import md_table
from .ingest import FEATURES, _read_parquet, _to_parquet_bytes
from .labels import LabelMapper
from .split import SPLITS, assign, split_name
from .state import (Deadline, GracefulStop, Ledger, atomic_write_bytes, atomic_write_json, atomic_write_text,
                    crash_point, ensure_dir, output_ok, read_json, retry, stable_hash)

STAGE = "prepare"
PREPARE_VERSION = 1
KEYS = ["split", "cls", "label", "episode"]
LOWCARD = ["ip_ttl", "ip_flags", "ip_proto", "proto_gre", "dir_out", "dir_in", "tcp_window_size_scalefactor",
           "tcp_options_len", "proto_layers"]


def feature_columns(cfg) -> List[str]:
    drop = set(get_path(cfg, "prepare.drop_features", []) or [])
    return [f for f in FEATURES if f not in drop]


def prepare_spec(cfg) -> Dict[str, Any]:
    spec = {"v": PREPARE_VERSION, "ingest": stable_hash(layout.ingest_spec(cfg)), "split": split_name(cfg),
            "labels": cfg.get("labels", {}), "features": feature_columns(cfg)}
    if layout.corrections_enabled(cfg):  # only then, so the original folder's fingerprint is unchanged
        from .relabel import corrections_spec

        spec["corrections"] = stable_hash(corrections_spec(cfg))
    return spec


def prepare_dir(cfg) -> Path:
    return ensure_dir(layout.work(cfg) / "data" / f"prepare-{stable_hash(prepare_spec(cfg))}")


def _out(cfg) -> Path:
    return layout.reports(cfg, "prepare")


# =====================================================================
# inputs from step 1

def ingest_inputs(cfg) -> Tuple[Path, List[Dict[str, Any]], pd.DataFrame]:
    ing = layout.ingest_dir(cfg)
    led = Ledger(ing / "_ledger.jsonl")
    metas = [read_json(r["meta"]) for r in led.done().values() if output_ok(r.get("meta", ""))]
    metas = sorted([m for m in metas if m], key=lambda m: m["id"])
    if not metas:
        raise RuntimeError(f"No ingested files in {ing}. Run `python -m gids ingest` first.")
    if layout.corrections_enabled(cfg):
        from .relabel import Corrections

        return ing, metas, Corrections(cfg).table
    sfile = ing / f"{split_name(cfg)}.json"
    sp = read_json(sfile)
    if not sp:
        raise RuntimeError(f"No split table for the current split settings ({sfile.name}). Run `python -m gids "
                           "ingest` again: finished files are skipped and only the split is computed (seconds).")
    return ing, metas, pd.DataFrame(sp["blocks"])


def vocab(cfg, table: pd.DataFrame) -> Dict[str, Any]:
    mapper = LabelMapper(cfg.get("labels", {}))
    raws = sorted(table["label"].unique().tolist())
    classes = mapper.classes(raws)
    cidx = {c: i for i, c in enumerate(classes)}
    return {"labels": raws, "classes": classes,
            "label_class": [cidx.get(mapper.map_one(r), -1) if mapper.map_one(r) is not None else -1 for r in raws]}


# =====================================================================
# one file

def segment_table(ing: Path, meta: Dict[str, Any]) -> pd.DataFrame:
    """Device IP of every capture segment = the IP present in most of its packets."""
    from pandas.api.types import union_categoricals

    cols = {"seg": [], "src": [], "dst": [], "t": [], "label": []}
    for p in meta["parts"]:
        try:
            df = _read_parquet(ing / p, columns=["seg", "ip_src", "ip_dst", "t", "label_raw"])
        except Exception:  # noqa: BLE001 - empty part
            continue
        cols["seg"].append(df["seg"].to_numpy())
        cols["t"].append(df["t"].to_numpy())
        for k, c in (("src", "ip_src"), ("dst", "ip_dst"), ("label", "label_raw")):
            cols[k].append(df[c].astype("category"))
    if not cols["seg"]:
        return pd.DataFrame(columns=["seg", "rows", "device_ip", "share", "t_start", "t_end", "labels"])
    seg = np.concatenate(cols["seg"])
    t = np.concatenate(cols["t"])
    ips = union_categoricals(cols["src"] + cols["dst"], ignore_order=True)
    n = len(seg)
    src_c = ips.codes[:n].astype(np.int64)
    dst_c = ips.codes[n:].astype(np.int64)
    lab = pd.Series(union_categoricals(cols["label"], ignore_order=True))
    both = pd.DataFrame({"seg": np.r_[seg, seg], "ip": np.r_[src_c, dst_c]})
    both = both[both["ip"] >= 0]
    top = both.groupby(["seg", "ip"]).size().reset_index(name="n").sort_values(["seg", "n"], ascending=[True, False])
    top = top.drop_duplicates("seg").set_index("seg")["ip"]
    seg_ip = pd.Series(seg).map(top).fillna(-2).to_numpy(np.int64)
    hit = (src_c == seg_ip) | (dst_c == seg_ip)
    g = pd.DataFrame({"seg": seg, "hit": hit, "t": t, "label": lab})
    rows = []
    for s, d in g.groupby("seg", sort=True):
        code = int(top.get(s, -1))
        rows.append({"seg": int(s), "rows": len(d), "device_ip": str(ips.categories[code]) if code >= 0 else None,
                     "share": float(d["hit"].mean()), "t_start": float(np.nanmin(d["t"])),
                     "t_end": float(np.nanmax(d["t"])),
                     "labels": ", ".join(f"{k} {v}" for k, v in d["label"].value_counts().head(3).items() if v)})
    return pd.DataFrame(rows)


def _row_hash(X: np.ndarray, keys: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(X, dtype=np.float32).view(np.uint32)
    return pd.util.hash_pandas_object(pd.DataFrame(np.column_stack([bits, keys.astype(np.uint32)])),
                                      index=False).to_numpy(np.uint64)


def _dedupe(X: np.ndarray, keys: np.ndarray, w: np.ndarray, h: np.ndarray):
    """Merge rows with the same hash of (X, keys); weights are summed.  Result sorted by row hash."""
    if len(X) == 0:
        return X, keys, w, h
    order = np.argsort(h, kind="stable")
    hs = h[order]
    starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])
    first = order[starts]
    return X[first], keys[first], np.add.reduceat(w[order], starts), hs[starts]


def prepare_member(meta: Dict[str, Any], ing: Path, table: pd.DataFrame, voc: Dict[str, Any], feats: List[str],
                   deadline: Deadline, fix=None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    segs = segment_table(ing, meta)
    seg_ip = dict(zip(segs["seg"], segs["device_ip"]))
    lab_index = {l: i for i, l in enumerate(voc["labels"])}
    lab_cls = np.array(voc["label_class"] + [-1], dtype=np.int16)
    Xs, Ks, Ws, Hs = [], [], [], []
    rows_in = 0
    for p in meta["parts"]:
        deadline.check()
        try:
            df = _read_parquet(ing / p)
        except Exception:  # noqa: BLE001
            continue
        if not len(df) or "label_raw" not in df:
            continue
        if fix is not None:  # corrected labels (step 6); copied captures become "Duplicate copy" and are dropped
            df = fix.apply(meta, p, df)
        rows_in += len(df)
        ip = df["seg"].map(seg_ip).astype(object).to_numpy()
        df["dir_out"] = (df["ip_src"].astype(object).to_numpy() == ip).astype(np.float32)
        df["dir_in"] = (df["ip_dst"].astype(object).to_numpy() == ip).astype(np.float32)
        lab = df["label_raw"].astype(str).map(lab_index).fillna(len(voc["labels"])).to_numpy(np.int64)
        cls = lab_cls[lab]
        sp, ep = assign(df["label_raw"], df["t"].to_numpy(), table, with_episode=True)
        keep = (cls >= 0) & (sp >= 0) & (sp <= 2)
        for f in feats:
            if f not in df:
                df[f] = np.nan
        X = df.loc[keep, feats].to_numpy(dtype=np.float32)
        K = np.column_stack([sp[keep], cls[keep], lab[keep], ep[keep]]).astype(np.int32)
        X, K, W, H = _dedupe(X, K, np.ones(len(X), dtype=np.int64), _row_hash(X, K))
        Xs.append(X), Ks.append(K), Ws.append(W), Hs.append(H)
        del df
    X = np.concatenate(Xs) if Xs else np.empty((0, len(feats)), np.float32)
    K = np.concatenate(Ks) if Ks else np.empty((0, 4), np.int32)
    W = np.concatenate(Ws) if Ws else np.empty(0, np.int64)
    H = np.concatenate(Hs) if Hs else np.empty(0, np.uint64)
    del Xs, Ks, Ws, Hs
    X, K, W, _ = _dedupe(X, K, W, H)
    order = np.lexsort((K[:, 1], K[:, 0]))
    out = pd.DataFrame(X[order], columns=feats)
    for i, k in enumerate(KEYS):
        out[k] = K[order, i].astype(np.int8 if k in ("split", "cls") else np.int16)
    out["weight"] = W[order].astype(np.int64)
    out["device_type"] = pd.Categorical([meta["device_type"]] * len(out))
    acct = out.groupby(["split", "label"])["weight"].agg(["sum", "size"]).reset_index()
    info = {"rows_in": rows_in, "rows_kept": int(W.sum()), "unique": int(len(out)),
            "accounting": acct.to_dict(orient="records")}
    return out, segs, info


# =====================================================================
# command

def run_prepare(cfg, tracer, deadline: Deadline, max_files: Optional[int] = None) -> Dict[str, Any]:
    ing, metas, table = ingest_inputs(cfg)
    if max_files:
        metas = sorted(metas, key=lambda m: m["rows"])[:max_files]
    out = prepare_dir(cfg)
    rdir = ensure_dir(out / "rows")
    voc = vocab(cfg, table)
    feats = feature_columns(cfg)
    fix = None
    if layout.corrections_enabled(cfg):
        from .relabel import Corrections

        fix = Corrections(cfg)
    atomic_write_json(out / "_spec.json", prepare_spec(cfg))
    atomic_write_json(out / "vocab.json", {**voc, "features": feats, "splits": list(SPLITS[:3])})
    ledger = Ledger(out / "_ledger.jsonl")
    done = ledger.done()

    def finished(m):
        r = done.get(m["id"])
        return bool(r) and output_ok(r["rows"]) and output_ok(r["meta"])

    todo = [m for m in metas if not finished(m)]
    total = sum(m["rows"] for m in metas)
    tracer.info(STAGE, "plan", f"{len(metas)} files ({total:,} rows) -> {out}; {len(metas) - len(todo)} already "
                f"prepared, {len(todo)} to go")
    for k, m in enumerate(todo):
        deadline.check()
        tracer.progress(STAGE, len(metas) - len(todo) + k, len(metas), "files", every_s=30,
                        current=Path(m["member"]).name)
        ledger.append(m["id"], "start")
        t0 = time.time()
        try:
            rows, segs, info = retry(lambda: prepare_member(m, ing, table, voc, feats, deadline, fix), tries=3,
                                     base_delay=10.0, on_error=lambda a, e: tracer.warn(
                                         STAGE, "io_retry", f"attempt {a}: {e}", member=m["member"]))
        except (GracefulStop, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001
            tracer.exception(STAGE, "file_failed", exc, member=m["member"])
            ledger.append(m["id"], "failed", error=f"{type(exc).__name__}: {exc}")
            continue
        fid = stable_hash(m["id"], 16)
        rpath, mpath = rdir / f"{fid}.parquet", out / f"{fid}.json"
        atomic_write_bytes(rpath, _to_parquet_bytes(rows))
        info.update({"member": m["member"], "id": m["id"], "device_type": m["device_type"],
                     "seconds": round(time.time() - t0, 1), "segments": segs.to_dict(orient="records")})
        atomic_write_json(mpath, info)
        ledger.append(m["id"], "done", rows=str(rpath), meta=str(mpath), unique=info["unique"])
        tracer.info(STAGE, "file", f"{Path(m['member']).name}: {info['rows_kept']:,} packets kept -> "
                    f"{info['unique']:,} unique rows, {len(segs)} segment(s), {info['seconds']} s")
        crash_point(STAGE)
    failed = [m["member"] for m in metas if ledger.latest().get(m["id"], {}).get("status") == "failed"]
    if failed:
        raise RuntimeError(f"{len(failed)} file(s) failed (python -m gids trace --errors); run prepare again")
    infos = [read_json(out / f"{stable_hash(m['id'], 16)}.json") for m in metas]
    return finish_prepare(cfg, tracer, out, infos, voc, feats, partial=bool(max_files))


# =====================================================================
# report

def load_rows(out: Path, infos: List[Dict[str, Any]], splits=(0, 1, 2), columns=None,
              sample: Optional[Dict[Tuple[int, int], float]] = None, seed: int = 0) -> pd.DataFrame:
    """Rows of the given splits from every file; ``sample`` maps (split, cls) -> keep probability."""
    frames = []
    for i, info in enumerate(infos):
        df = _read_parquet(out / "rows" / f"{stable_hash(info['id'], 16)}.parquet", columns=columns)
        df = df[df["split"].isin(splits)]
        if sample is not None and len(df):
            probs = pd.DataFrame([(s, c, v) for (s, c), v in sample.items()], columns=["split", "cls", "_p"])
            p = df[["split", "cls"]].astype(np.int64).merge(probs, on=["split", "cls"], how="left")["_p"]
            df = df[np.random.default_rng(seed + i).random(len(df)) < p.fillna(0.0).to_numpy()]
        df = df.assign(device_type=df["device_type"].astype(str)) if "device_type" in df else df
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _balanced_weights(w: np.ndarray, y: np.ndarray) -> np.ndarray:
    tot = pd.Series(w).groupby(y).transform("sum").to_numpy()
    return w / np.maximum(tot, 1e-12)


def _scores(y: np.ndarray, pred: np.ndarray, w: np.ndarray, n_cls: int) -> Tuple[float, float, np.ndarray]:
    """Balanced accuracy, macro F1 and per-class F1 with every class weighted equally."""
    bw = _balanced_weights(w, y)
    f1 = np.full(n_cls, np.nan)
    rec = np.full(n_cls, np.nan)
    for c in range(n_cls):
        tp = bw[(y == c) & (pred == c)].sum()
        fn = bw[(y == c) & (pred != c)].sum()
        fp = bw[(y != c) & (pred == c)].sum()
        if tp + fn == 0:
            continue
        rec[c] = tp / (tp + fn)
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1[c] = 2 * prec * rec[c] / (prec + rec[c]) if prec + rec[c] else 0.0
    return float(np.nanmean(rec)), float(np.nanmean(f1)), f1


def shortcut_check(tr: pd.DataFrame, va: pd.DataFrame, feats: List[str], classes: List[str],
                   depth: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from sklearn.tree import DecisionTreeClassifier

    n = len(classes)
    ytr, yva = np.array(tr["cls"], dtype=np.int64), np.array(va["cls"], dtype=np.int64)
    wtr = _balanced_weights(np.array(tr["weight"], dtype=np.float64), ytr)
    wva = np.array(va["weight"], dtype=np.float64)
    rows = []
    for f in feats:
        x = np.array(tr[[f]], dtype=np.float32)
        if pd.Series(x[:, 0]).nunique(dropna=False) <= 1:
            rows.append({"feature": f, "balanced_acc": np.nan, "macro_f1": np.nan, "note": "constant in train"})
            continue
        m = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(x, ytr, sample_weight=wtr)
        ba, mf1, f1 = _scores(yva, m.predict(np.array(va[[f]], dtype=np.float32)), wva, n)
        row = {"feature": f, "balanced_acc": ba, "macro_f1": mf1, "note": ""}
        row.update({f"f1 {c}": f1[i] for i, c in enumerate(classes)})
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False, na_position="last")
    m = DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=0).fit(
        np.array(tr[feats], dtype=np.float32), ytr, sample_weight=wtr)
    ba, mf1, f1 = _scores(yva, m.predict(np.array(va[feats], dtype=np.float32)), wva, n)
    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    base = {"balanced_acc": ba, "macro_f1": mf1, "f1": {c: float(f1[i]) for i, c in enumerate(classes)},
            "importance": {k: float(v) for k, v in imp.head(12).items()}}
    return res, base


def value_table(df: pd.DataFrame, f: str, classes: List[str], top: int = 8) -> pd.DataFrame:
    v = df[f].astype("float64").fillna(-99999).round(3)
    t = pd.crosstab(v, df["cls"], values=df["weight"], aggfunc="sum").fillna(0)
    t = 100 * t / t.sum(axis=0)
    t.columns = [classes[c] for c in t.columns]
    t = t.loc[t.max(axis=1).sort_values(ascending=False).index[:top]]
    t.index = [("empty" if x == -99999 else f"{x:g}") for x in t.index]
    t.index.name = f
    return t.reset_index()


def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    ok = np.isfinite(x)
    if not ok.any():
        return np.nan
    x, w = x[ok], w[ok]
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, q * cw[-1])])


def finish_prepare(cfg, tracer, out: Path, infos: List[Dict[str, Any]], voc: Dict[str, Any], feats: List[str],
                   partial: bool) -> Dict[str, Any]:
    rep = _out(cfg)
    classes = voc["classes"]
    acct = pd.DataFrame([dict(a, file=i["member"]) for i in infos for a in i["accounting"]])
    acct["label_name"] = [voc["labels"][x] for x in acct["label"]]
    acct["class"] = [classes[voc["label_class"][x]] for x in acct["label"]]
    names = list(SPLITS[:3])
    by_class = acct.pivot_table(index="class", columns="split", values="sum", aggfunc="sum").fillna(0)
    by_class.columns = [names[c] for c in by_class.columns]
    uniq = acct.pivot_table(index="class", columns="split", values="size", aggfunc="sum").fillna(0)
    uniq.columns = [f"{names[c]}_unique" for c in uniq.columns]
    by_class = by_class.join(uniq).reindex(classes).fillna(0).astype(np.int64)
    by_class.index.name = "class"
    by_label = acct.pivot_table(index="label_name", columns="split", values="sum", aggfunc="sum").fillna(0)
    by_label.columns = [names[c] for c in by_label.columns]
    by_label = by_label.astype(np.int64)

    # packets per label and split must equal step 1's split table (gaps and dropped labels excluded)
    ing_lab = pd.read_csv(layout.reports(cfg, "ingest") / "split_by_label.csv", index_col=0)
    mismatch = []
    if not partial:
        for lab_name in by_label.index:
            for s in names:
                a, b = int(by_label.loc[lab_name].get(s, 0)), int(ing_lab.loc[lab_name, s]) \
                    if lab_name in ing_lab.index else -1
                if a != b:
                    mismatch.append(f"{lab_name}/{s}: {a} vs {b}")

    segs = pd.DataFrame([dict(s, file=Path(i["member"]).name, device_type=i["device_type"])
                         for i in infos for s in i["segments"]])
    t0 = segs["t_start"].min() if len(segs) else 0.0
    if len(segs):
        segs["start_h"] = (segs["t_start"] - t0) / 3600
        segs["end_h"] = (segs["t_end"] - t0) / 3600
    single = segs.groupby("file").filter(lambda d: d["device_ip"].nunique() == 1) if len(segs) else segs
    ip_owner = single.groupby("device_ip")["file"].first().str.replace("iotsim-", "").str.replace(".csv", "") \
        if len(single) else pd.Series(dtype=str)
    multi = segs.groupby("file").filter(lambda d: d["device_ip"].nunique() > 1) if len(segs) else segs
    if len(multi):
        multi = multi.assign(ip_also_in=multi["device_ip"].map(ip_owner).fillna(""))
    segs.to_csv(rep / "segments.csv", index=False)

    # samples for the checks: up to N unique rows per class from train and val
    n_target = int(get_path(cfg, "prepare.sample_per_class", 100_000))
    uq = acct.groupby(["split", "label"])["size"].sum()
    cls_uq: Dict[Tuple[int, int], int] = {}
    for (s, l), v in uq.items():
        key = (int(s), voc["label_class"][int(l)])
        cls_uq[key] = cls_uq.get(key, 0) + int(v)
    probs = {k: min(1.0, n_target / max(v, 1)) for k, v in cls_uq.items()}
    with tracer.span(STAGE, "sample"):
        smp = load_rows(out, infos, splits=(0, 1), columns=feats + ["split", "cls", "weight"], sample=probs)
    tr, va = smp[smp["split"] == 0], smp[smp["split"] == 1]
    depth = int(get_path(cfg, "prepare.shortcut_depth", 3))
    with tracer.span(STAGE, "shortcut_check"):
        sc, base = shortcut_check(tr, va, feats, classes, depth)
    sc.to_csv(rep / "shortcut_check.csv", index=False)
    vt = {f: value_table(tr, f, classes) for f in LOWCARD if f in tr}
    for f, t in vt.items():
        t.to_csv(rep / f"values_{f}.csv", index=False)
    med = pd.DataFrame({classes[c]: {f: _wquantile(d[f].to_numpy(np.float64), d["weight"].to_numpy(np.float64),
                                                   0.5) for f in feats}
                        for c, d in tr.groupby("cls")})
    med.index.name = "feature"
    med.to_csv(rep / "medians_by_class.csv")
    by_class.to_csv(rep / "rows_by_class.csv")
    by_label.to_csv(rep / "rows_by_label.csv")

    checks: List[str] = []
    if partial:
        checks.append("Only part of the files were prepared (--max-files).")
    if mismatch:
        checks.append(f"Packets per label/split differ from step 1 for {len(mismatch)} cell(s): {mismatch[:5]}")
    low = segs[segs["share"] < 0.95] if len(segs) else segs
    if len(low):
        checks.append(f"{len(low)} capture segment(s) where the device IP is in under 95% of packets "
                      f"(see segments.csv): {', '.join((low['file'] + '#' + low['seg'].astype(str)).head(6))}")
    strong = sc[sc["macro_f1"] >= 0.9]
    for _, r in strong.iterrows():
        checks.append(f"`{r['feature']}` alone reaches macro-F1 {r['macro_f1']:.3f} on validation - check that "
                      "this is real attack behaviour and not an artefact of the testbed")
    for c in classes:
        col = f"f1 {c}"
        if col in sc:
            hit = sc[sc[col] >= 0.98]["feature"].tolist()
            if hit:
                checks.append(f"{c} is recognised with F1 >= 0.98 by a single feature: {', '.join(hit[:6])}")
    const = sc[sc["note"] == "constant in train"]["feature"].tolist()
    if const:
        checks.append(f"Constant in the training sample (no information): {', '.join(const)}")

    rows_in = sum(i["rows_in"] for i in infos)
    kept = sum(i["rows_kept"] for i in infos)
    unique = sum(i["unique"] for i in infos)
    summary = {"prepare_dir": str(out), "files": len(infos), "rows_in": rows_in, "rows_kept": kept,
               "unique_rows": unique, "classes": classes, "features": feats, "partial": partial,
               "accounting_ok": not mismatch and not partial, "baseline_tree": base, "shortcut_depth": depth,
               "rows_by_class": by_class.reset_index().to_dict(orient="records"), "checks": checks}
    atomic_write_json(rep / "summary.json", summary)
    atomic_write_text(rep / "PREPARE.md", prepare_markdown(summary, by_class, sc, base, vt, med, multi, segs,
                                                           n_target, classes))
    tracer.info(STAGE, "done", f"{kept:,} packets -> {unique:,} unique rows; report {rep / 'PREPARE.md'}")
    return summary


def prepare_markdown(s, by_class, sc, base, vt, med, multi, segs, n_target, classes) -> str:
    L = ["# Gotham dataset - step 2: prepare", ""]
    L.append(f"Files: **{s['files']}** | packets in: {s['rows_in']:,} | kept (no gap, no dropped label): "
             f"**{s['rows_kept']:,}** | unique rows after merging identical packets: **{s['unique_rows']:,}** "
             f"({s['rows_kept'] / max(s['unique_rows'], 1):.2f} packets per row) | packets per label and split "
             f"equal to step 1: **{'yes' if s['accounting_ok'] else 'NO - see section 6'}**")
    L += ["", f"Rows: `{s['prepare_dir']}/rows/` (one Parquet file per CSV; `weight` = number of identical "
          "packets merged into the row)", ""]
    L += ["## 1. Packets per class and split", "", "Packets (sum of weights) and unique rows:", "",
          md_table(by_class.reset_index(), max_rows=20), ""]
    L += ["## 2. Shortcut check", "",
          f"A depth-{s['shortcut_depth']} decision tree trained on **one feature at a time** (train sample, up to {n_target:,} unique "
          "rows per class, every class weighted equally) and scored on the validation sample. A feature that "
          "separates classes on its own is not necessarily a leak, but it is the first thing to question.", "",
          md_table(sc.round(3), max_rows=40), "",
          f"All features together, one depth-12 tree (a floor for step 3, not the model): balanced accuracy "
          f"**{base['balanced_acc']:.3f}**, macro-F1 **{base['macro_f1']:.3f}**", "",
          "| class | F1 |", "|---|---|"] + [f"| {c} | {v:.3f} |" for c, v in base["f1"].items()] + [
          "", "Most used features in that tree: " + ", ".join(f"`{k}` {v:.2f}" for k, v in
                                                             base["importance"].items()), ""]
    L += ["## 3. Value share per class (% of the class's training packets)", ""]
    for f, t in vt.items():
        L += [f"**{f}**", "", md_table(t.round(2), max_rows=10), ""]
    L += ["## 4. Median per class (training sample)", "", md_table(med.reset_index(), max_rows=40), ""]
    L += ["## 5. Capture segments", "",
          f"{len(segs)} segments in {segs['file'].nunique() if len(segs) else 0} files. Files whose segments have "
          "different device IPs (captures joined end to end); `ip_also_in` names the single-device file that has "
          "the same IP:", "",
          md_table(multi[["file", "seg", "rows", "device_ip", "share", "start_h", "end_h", "ip_also_in",
                          "labels"]] if len(multi) else multi, max_rows=60), ""]
    L += ["## 6. Things to check", ""] + ([f"- {c}" for c in s["checks"]] or ["- nothing flagged"])
    L += ["", "## 7. Next", "", "Step 3 trains imbalance-aware models (XGBoost / LightGBM / random forest) on the "
          "train rows with `weight`, tunes on validation, and keeps the test split untouched until step 4.", ""]
    return "\n".join(L)

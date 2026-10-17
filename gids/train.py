"""Step 3: train imbalance-aware models and compare feature sets on validation.

``train`` first builds the *dataset* once (``data/dataset-<hash>/``):

* rows that are identical in two files - the same capture stored twice,
  e.g. combined-cycle-10 inside combined-cycle-1 - are counted once;
* the big classes are randomly sampled down to ``dataset.max_rows_per_class``
  unique rows; ``weight_packets`` keeps how many packets every sampled row
  stands for, so metrics are still computed on the real packet counts;
* train and validation are stored; the test split is not read at all.

Then every enabled experiment in ``configs/experiments.yaml`` is trained in
order (XGBoost, LightGBM, random forest x feature sets x label schemes):

* every class gets the same total weight in training and early stopping;
* the model and its training state are saved every few rounds, so a
  disconnect only repeats the rounds since the last checkpoint;
* validation metrics (per-class precision / recall / F1 on packets,
  macro-F1, confusion matrix) and feature importance are written next to
  the model, and ``reports/train/TRAIN.md`` compares all experiments.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import layout
from .config import experiment_config, experiment_names, get_path
from .explore import md_table
from .ingest import _read_parquet, _to_parquet_bytes
from .prepare import KEYS, _row_hash, prepare_dir
from .state import (Deadline, GracefulStop, Ledger, atomic_copy, atomic_write_bytes, atomic_write_json,
                    atomic_write_text, crash_point, output_ok, read_json, stable_hash)

STAGE = "train"
# standard ports of the protocols in the testbed and of common IoT services -> service id (0 = none of these)
SERVICES = {22: 1, 23: 2, 2323: 2, 53: 3, 69: 4, 80: 5, 8080: 5, 123: 6, 443: 7, 554: 8, 1883: 9, 8883: 10,
            5683: 11, 5684: 12}
DERIVED = ("sport_class", "dport_class", "sport_service", "dport_service")


# =====================================================================
# features and labels

def _port_class(p: np.ndarray) -> np.ndarray:
    out = np.zeros(len(p), dtype=np.float32)
    ok = np.isfinite(p) & (p >= 0)
    out[ok] = np.where(p[ok] < 1024, 1, np.where(p[ok] < 49152, 2, 3))
    return out


def _service(p: np.ndarray) -> np.ndarray:
    return pd.Series(p).map(SERVICES).fillna(0).to_numpy(np.float32)


def make_matrix(df: pd.DataFrame, names: List[str]) -> np.ndarray:
    """Feature matrix (float32) for the given names, deriving the port-type features when asked."""
    cols = {}
    if any(n in DERIVED for n in names):
        sport = df["tcp_srcport"].fillna(df["udp_srcport"]).to_numpy(np.float64)
        dport = df["tcp_dstport"].fillna(df["udp_dstport"]).to_numpy(np.float64)
        cols = {"sport_class": _port_class(sport), "dport_class": _port_class(dport),
                "sport_service": _service(sport), "dport_service": _service(dport)}
    return np.column_stack([cols[n] if n in cols else df[n].to_numpy(np.float32) for n in names]).astype(np.float32)


def label_space(voc: Dict[str, Any], scheme: str) -> Tuple[List[str], np.ndarray]:
    """Class names and a lookup raw-label index -> class index for a label scheme."""
    if scheme == "fine":
        kept = [i for i, c in enumerate(voc["label_class"]) if c >= 0]
        normal = [i for i in kept if voc["classes"][voc["label_class"][i]] == voc["classes"][0]]
        order = normal + [i for i in kept if i not in normal]
        lut = np.full(len(voc["labels"]), -1, dtype=np.int64)
        lut[order] = np.arange(len(order))
        return [voc["labels"][i] for i in order], lut
    return list(voc["classes"]), np.array(voc["label_class"], dtype=np.int64)


def balanced(w: np.ndarray, y: np.ndarray, n_cls: int) -> np.ndarray:
    """Weights that give every class the same total, scaled to a mean of 1."""
    tot = np.bincount(y, weights=w, minlength=n_cls)
    present = (tot > 0).sum()
    return w / np.maximum(tot[y], 1e-12) * len(y) / max(present, 1)


# =====================================================================
# dataset (built once, shared by all experiments)

def _prepare_infos(cfg) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    pdir = prepare_dir(cfg)
    led = Ledger(pdir / "_ledger.jsonl")
    infos = sorted([read_json(r["meta"]) for r in led.done().values() if output_ok(r.get("meta", ""))],
                   key=lambda i: i["id"])
    voc = read_json(pdir / "vocab.json")
    if not infos or not voc:
        raise RuntimeError(f"No prepared files in {pdir}. Run `python -m gids prepare` first.")
    ing = Ledger(layout.ingest_dir(cfg) / "_ledger.jsonl").done()
    if len(infos) < len(ing):
        raise RuntimeError(f"Only {len(infos)} of {len(ing)} files are prepared; run `python -m gids prepare`.")
    return pdir, infos, voc


def _rows_path(pdir: Path, info: Dict[str, Any]) -> Path:
    return pdir / "rows" / f"{stable_hash(info['id'], 16)}.parquet"


def cross_file_merge(h: np.ndarray, f: np.ndarray, w: np.ndarray, infos: List[Dict[str, Any]], enabled: bool = True):
    """Rows with the same hash in several files (one capture stored twice) are kept once, in the first file.

    Returns (keep mask, weights with the dropped copies added to the kept row, dropped rows, dropped packets,
    {"file <- copy file": rows} for the ten most frequent pairs).
    """
    keep = np.ones(len(h), dtype=bool)
    wsum = w.astype(np.float64)
    if not enabled or not len(h):
        return keep, wsum, 0, 0, {}
    order = np.lexsort((f, h))
    hs_ = h[order]
    starts = np.flatnonzero(np.r_[True, hs_[1:] != hs_[:-1]])
    sizes = np.diff(np.r_[starts, len(h)])
    if not (sizes > 1).any():
        return keep, wsum, 0, 0, {}
    grp = np.repeat(np.arange(len(starts)), sizes)
    first = order[starts][grp]
    later = np.r_[False, grp[1:] == grp[:-1]]
    dropped = order[later]
    keep[dropped] = False
    np.add.at(wsum, first[later], w[dropped])
    pf = pd.Series(list(zip(f[first[later]], f[dropped]))).value_counts()
    pairs = {f"{Path(infos[a]['member']).name} <- {Path(infos[b]['member']).name}": int(v)
             for (a, b), v in pf.head(10).items()}
    return keep, wsum, int(len(dropped)), int(w[dropped].sum()), pairs


def held_out_mask(df: pd.DataFrame, labels: List[str], hold: Optional[Dict[str, Any]]) -> np.ndarray:
    """True for rows of a held-out run: raw label matching ``hold['label_regex']`` and episode ``hold['episode']``."""
    if not hold or not len(df):
        return np.zeros(len(df), dtype=bool)
    rx = re.compile(hold["label_regex"])
    held = np.array([bool(rx.search(lab)) for lab in labels])
    return held[df["label"].to_numpy(np.int64)] & (df["episode"].to_numpy(np.int64) == int(hold["episode"]))


def build_dataset(cfg, tracer, deadline: Deadline) -> Dict[str, Any]:
    ddir = layout.dataset_dir(cfg)
    meta = read_json(ddir / "dataset.json")
    if meta and output_ok(ddir / "train.parquet") and output_ok(ddir / "val.parquet"):
        return meta
    pdir, infos, voc = _prepare_infos(cfg)
    feats = voc["features"]
    spec = layout.dataset_spec(cfg)
    tracer.info(STAGE, "dataset", f"building the training/validation tables from {len(infos)} files -> {ddir}")

    # pass 1: hashes of every train/val row, to find rows stored in more than one file
    hs, fs, rs, ss, cs, ws = [], [], [], [], [], []
    with tracer.span(STAGE, "dataset.pass1"):
        for fi, info in enumerate(infos):
            deadline.check()
            df = _read_parquet(_rows_path(pdir, info), columns=feats + KEYS + ["weight"])
            idx = np.flatnonzero(df["split"].to_numpy() <= 1)
            df = df.iloc[idx]
            hs.append(_row_hash(df[feats].to_numpy(np.float32), df[KEYS].to_numpy(np.int32)))
            fs.append(np.full(len(df), fi, np.int32))
            rs.append(idx.astype(np.int64))
            ss.append(df["split"].to_numpy(np.int8))
            cs.append(df["cls"].to_numpy(np.int8))
            ws.append(df["weight"].to_numpy(np.int64))
            tracer.progress(STAGE, fi + 1, len(infos), "files (dataset pass 1)", every_s=30)
    h, f, r, s, c, w = (np.concatenate(x) for x in (hs, fs, rs, ss, cs, ws))
    del hs, fs, rs, ss, cs, ws
    keep, wsum, dup_rows, dup_packets, pairs = cross_file_merge(h, f, w, infos, spec["merge_cross_file"])
    caps = {0: spec["max_rows_per_class"], 1: spec["val_max_rows_per_class"]}
    counts = {(int(sp), int(cl)): int(n) for (sp, cl), n in
              pd.Series(keep).groupby([s, c]).sum().items()}
    prob = {k: min(1.0, caps[k[0]] / max(v, 1)) for k, v in counts.items()}
    rng = np.random.default_rng(spec["seed"])
    pk = np.zeros(2 * 128)
    for (a, b), v in prob.items():
        pk[a * 128 + b] = v
    pr = pk[s.astype(np.int64) * 128 + c.astype(np.int64)]
    take = keep & (rng.random(len(h)) < pr)
    wpk = wsum / np.where(pr > 0, pr, 1.0)      # packets each sampled row stands for
    del h

    # pass 2: read the sampled rows (rows of file fi are the contiguous block bounds[fi]:bounds[fi + 1])
    bounds = np.searchsorted(f, np.arange(len(infos) + 1))
    tables = {0: [], 1: []}
    with tracer.span(STAGE, "dataset.pass2"):
        for fi, info in enumerate(infos):
            deadline.check()
            a, b = bounds[fi], bounds[fi + 1]
            m = np.flatnonzero(take[a:b]) + a
            if not len(m):
                continue
            df = _read_parquet(_rows_path(pdir, info)).iloc[r[m]].reset_index(drop=True)
            df["weight_packets"] = wpk[m]
            df["device_type"] = df["device_type"].astype(str)
            for sp in (0, 1):
                tables[sp].append(df[df["split"] == sp])
            tracer.progress(STAGE, fi + 1, len(infos), "files (dataset pass 2)", every_s=30)
    for sp, name in ((0, "train"), (1, "val")):
        t = pd.concat(tables[sp], ignore_index=True)
        atomic_write_bytes(ddir / f"{name}.parquet", _to_parquet_bytes(t))
    rows = {f"{['train', 'val'][a]}/{voc['classes'][b]}": {"unique": v, "sampled_share": prob[(a, b)]}
            for (a, b), v in sorted(counts.items())}
    meta = {"dir": str(ddir), "spec": spec, "files": len(infos), "rows": rows,
            "train_rows": int((take & (s == 0)).sum()), "val_rows": int((take & (s == 1)).sum()),
            "cross_file_duplicate_rows": dup_rows, "cross_file_duplicate_packets": dup_packets,
            "cross_file_pairs": pairs, "classes": voc["classes"], "labels": voc["labels"],
            "label_class": voc["label_class"], "features": feats}
    atomic_write_json(ddir / "dataset.json", meta)
    tracer.info(STAGE, "dataset", f"train {meta['train_rows']:,} rows, val {meta['val_rows']:,} rows; "
                f"{dup_rows:,} rows ({dup_packets:,} packets) were stored in two files and are counted once")
    return meta


# =====================================================================
# models (each resumable from its last checkpoint)

class _Progress:
    """Shared early-stopping / checkpoint bookkeeping, saved as state.json next to the model."""

    def __init__(self, mdir: Path, patience: int, every: int, deadline: Deadline, tracer, name: str, total: int):
        self.mdir, self.patience, self.every, self.total = mdir, patience, every, total
        self.deadline, self.tracer, self.name = deadline, tracer, name
        self.state = read_json(mdir / "state.json") or {"iterations": 0, "best_iter": 0, "best_score": None,
                                                        "history": [], "seconds": 0.0, "status": "training"}
        self.t0 = time.time()

    def update(self, it: int, loss: float) -> bool:
        st = self.state
        st["history"].append(float(loss))
        st["iterations"] = it
        if st["best_score"] is None or loss < st["best_score"] - 1e-7:
            st["best_score"], st["best_iter"] = float(loss), it
        return it - st["best_iter"] >= self.patience

    def save(self, write_model) -> None:
        tmp = Path(tempfile.mkdtemp())
        try:
            write_model(tmp / "model")
            atomic_copy(tmp / "model", self.mdir / "model.bin")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.state["seconds"] += time.time() - self.t0
        self.t0 = time.time()
        atomic_write_json(self.mdir / "state.json", self.state)
        self.tracer.progress(STAGE, self.state["iterations"], self.total, f"rounds of {self.name}", every_s=30,
                             val_loss=f"{self.state['history'][-1]:.5f}", best_round=self.state["best_iter"])
        crash_point("train_ckpt")

    def check_stop(self, write_model) -> None:
        try:
            self.deadline.check()
        except GracefulStop:
            self.save(write_model)
            raise


def _gpu_available() -> bool:
    from .envinfo import _gpu

    return _gpu() is not None


def fit_xgboost(p: Dict[str, Any], X, y, w, Xv, yv, wv, n_cls: int, mdir: Path, prog: _Progress):
    import xgboost as xgb

    params = {"objective": "multi:softprob", "num_class": n_cls, "eval_metric": "mlogloss", "tree_method": "hist",
              "eta": p["learning_rate"], "max_depth": p["max_depth"], "min_child_weight": p["min_child_weight"],
              "subsample": p["subsample"], "colsample_bytree": p["colsample_bytree"], "max_bin": p["max_bin"],
              "seed": p["seed"], "nthread": os.cpu_count() or 1,
              "device": p.get("device", "auto") if p.get("device", "auto") != "auto" else
              ("cuda" if _gpu_available() else "cpu")}
    dtrain = xgb.QuantileDMatrix(X, y, weight=w, max_bin=p["max_bin"])
    dval = xgb.QuantileDMatrix(Xv, yv, weight=wv, ref=dtrain)
    init = None
    if prog.state["iterations"] and output_ok(mdir / "model.bin"):
        init = xgb.Booster(model_file=str(mdir / "model.bin"))
        prog.tracer.info(STAGE, "resume", f"{prog.name}: continuing after round {prog.state['iterations']}")
    remaining = p["n_estimators"] - prog.state["iterations"]

    class Ckpt(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            it = model.num_boosted_rounds()
            stop = prog.update(it, evals_log["val"]["mlogloss"][-1])
            def writer(path):
                model.save_model(str(path) + ".ubj")  # the extension selects the binary format
                os.replace(str(path) + ".ubj", path)

            if stop or it % p["checkpoint_every"] == 0 or it >= p["n_estimators"]:
                prog.save(writer)
            prog.check_stop(writer)
            return stop

    if remaining > 0 and not prog.state.get("stopped_early"):
        bst = xgb.train(params, dtrain, num_boost_round=remaining, evals=[(dval, "val")], callbacks=[Ckpt()],
                        xgb_model=init, verbose_eval=False)
        if bst.num_boosted_rounds() - prog.state["best_iter"] >= p["early_stopping_rounds"]:
            prog.state["stopped_early"] = True
    else:
        bst = init
    best = prog.state["best_iter"]
    return (lambda Xp: bst.predict(xgb.DMatrix(Xp), iteration_range=(0, best))), \
        bst.get_score(importance_type="total_gain")


def fit_lightgbm(p: Dict[str, Any], X, y, w, Xv, yv, wv, n_cls: int, mdir: Path, prog: _Progress):
    import lightgbm as lgb

    params = {"objective": "multiclass", "num_class": n_cls, "learning_rate": p["learning_rate"],
              "num_leaves": p["num_leaves"], "min_child_samples": p["min_child_samples"],
              "bagging_fraction": p["subsample"], "bagging_freq": 1 if p["subsample"] < 1 else 0,
              "feature_fraction": p["colsample_bytree"], "max_bin": p["max_bin"], "seed": p["seed"],
              "deterministic": True, "verbose": -1, "num_threads": os.cpu_count() or 1}
    dtrain = lgb.Dataset(X, y, weight=w, free_raw_data=False)
    dval = lgb.Dataset(Xv, yv, weight=wv, reference=dtrain)
    init = None
    if prog.state["iterations"] and output_ok(mdir / "model.bin"):
        init = str(mdir / "model.bin")
        prog.tracer.info(STAGE, "resume", f"{prog.name}: continuing after round {prog.state['iterations']}")
    remaining = p["n_estimators"] - prog.state["iterations"]

    def ckpt(env):
        it = env.iteration + 1
        stop = prog.update(it, env.evaluation_result_list[0][2])
        writer = lambda path: env.model.save_model(str(path))
        if stop or it % p["checkpoint_every"] == 0 or it >= p["n_estimators"]:
            prog.save(writer)
        prog.check_stop(writer)
        if stop:
            raise lgb.callback.EarlyStopException(prog.state["best_iter"] - 1, env.evaluation_result_list)

    if remaining > 0 and not prog.state.get("stopped_early"):
        bst = lgb.train(params, dtrain, num_boost_round=remaining, valid_sets=[dval], valid_names=["val"],
                        init_model=init, callbacks=[ckpt])
        if bst.current_iteration() - prog.state["best_iter"] >= p["early_stopping_rounds"]:
            prog.state["stopped_early"] = True
    else:
        bst = lgb.Booster(model_file=init)
    best = prog.state["best_iter"]
    imp = dict(zip(bst.feature_name(), bst.feature_importance(importance_type="gain").tolist()))
    return (lambda Xp: bst.predict(Xp, num_iteration=best)), imp


def fit_random_forest(p: Dict[str, Any], X, y, w, Xv, yv, wv, n_cls: int, mdir: Path, prog: _Progress):
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    if prog.state["iterations"] and output_ok(mdir / "model.bin"):
        rf = joblib.load(mdir / "model.bin")
        prog.tracer.info(STAGE, "resume", f"{prog.name}: continuing after {len(rf.estimators_)} trees")
    else:
        rf = RandomForestClassifier(n_estimators=0, warm_start=True, max_depth=p["max_depth"],
                                    min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                    max_samples=p["max_samples"], n_jobs=-1, random_state=p["seed"])
    Xn = np.nan_to_num(X, nan=-1.0)
    while len(getattr(rf, "estimators_", [])) < p["n_estimators"]:
        rf.n_estimators = min(p["n_estimators"], len(getattr(rf, "estimators_", [])) + p["batch"])
        rf.fit(Xn, y, sample_weight=w)
        prob = rf.predict_proba(np.nan_to_num(Xv, nan=-1.0))
        full = np.zeros((len(yv), n_cls))
        full[:, rf.classes_] = prob
        loss = float(-np.average(np.log(np.clip(full[np.arange(len(yv)), yv], 1e-15, 1)), weights=wv))
        prog.update(len(rf.estimators_), loss)
        writer = lambda path: joblib.dump(rf, path, compress=3)
        prog.save(writer)
        prog.check_stop(writer)
    prog.state["best_iter"] = len(rf.estimators_)

    def predict(Xp):
        pr = rf.predict_proba(np.nan_to_num(Xp, nan=-1.0))
        full = np.zeros((len(Xp), n_cls))
        full[:, rf.classes_] = pr
        return full

    return predict, dict(zip(range(X.shape[1]), rf.feature_importances_.tolist()))


FITTERS = {"xgboost": fit_xgboost, "lightgbm": fit_lightgbm, "random_forest": fit_random_forest}


def load_model(cfg, name: str, ds: Dict[str, Any]) -> Dict[str, Any]:
    """A finished experiment ready to predict: classes, feature names and predict(df) -> probabilities."""
    exp = experiment_config(cfg, name)
    mdir = layout.model_dir(cfg, name)
    st = read_json(mdir / "state.json") or {}
    if st.get("status") != "done" or not output_ok(mdir / "model.bin"):
        raise RuntimeError(f"Experiment {name} is not trained yet (run `python -m gids train --only {name}`)")
    model = exp.get("model", "xgboost")
    names = list(get_path(cfg, f"features.sets.{exp.get('feature_set', 'full')}"))
    classes, lut = label_space({k: ds[k] for k in ("classes", "labels", "label_class")},
                               exp.get("label_scheme", "coarse"))
    best = int(st["best_iter"])
    if model == "xgboost":
        import xgboost as xgb

        bst = xgb.Booster(model_file=str(mdir / "model.bin"))
        bst.set_param({"device": "cpu"})
        raw = lambda X: bst.inplace_predict(X, iteration_range=(0, best))
    elif model == "lightgbm":
        import lightgbm as lgb

        bst = lgb.Booster(model_file=str(mdir / "model.bin"))
        raw = lambda X: bst.predict(X, num_iteration=best)
    else:
        import joblib

        bst = joblib.load(mdir / "model.bin")

        def raw(X):
            full = np.zeros((len(X), len(classes)))
            full[:, bst.classes_] = bst.predict_proba(np.nan_to_num(X, nan=-1.0))
            return full

    def predict(df: pd.DataFrame, chunk: int = 1_000_000) -> np.ndarray:
        return np.concatenate([np.asarray(raw(make_matrix(df.iloc[i:i + chunk], names)))
                               for i in range(0, len(df), chunk)]) if len(df) else np.zeros((0, len(classes)))

    return {"name": name, "model": model, "booster": bst, "best": best, "features": names, "classes": classes,
            "lut": lut, "scheme": exp.get("label_scheme", "coarse"), "feature_set": exp.get("feature_set"),
            "holdout": exp.get("holdout"), "predict": predict, "dir": mdir}


# =====================================================================
# evaluation

def evaluate(y: np.ndarray, pred: np.ndarray, w: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    """Metrics on packets (``w`` = packets per row): per class precision / recall / F1, macro-F1, confusion."""
    k = len(classes)
    cm = np.zeros((k, k))
    np.add.at(cm, (y, pred), w)
    tp = np.diag(cm)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(support > 0, tp / support, np.nan)
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = support > 0
    per = pd.DataFrame({"class": classes, "precision": prec, "recall": rec, "f1": np.where(present, f1, np.nan),
                        "packets": support})
    return {"macro_f1": float(np.nanmean(per["f1"])), "balanced_accuracy": float(np.nanmean(rec[present])),
            "accuracy": float(tp.sum() / max(cm.sum(), 1)),
            "weighted_f1": float(np.nansum(f1 * support) / max(support.sum(), 1)),
            "per_class": per.to_dict(orient="records"), "confusion": cm.tolist(), "classes": classes}


def confusion_table(m: Dict[str, Any]) -> pd.DataFrame:
    cm = np.array(m["confusion"])
    pct = 100 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
    t = pd.DataFrame(pct, columns=[f"-> {c}" for c in m["classes"]])
    t.insert(0, "true class (% of its packets)", m["classes"])
    return t


# =====================================================================
# one experiment

def _feature_name(k, names: List[str]) -> str:
    """Map a model's internal feature id (3, "f3", "Column_3") to the feature's name."""
    if isinstance(k, (int, np.integer)):
        return names[int(k)]
    for prefix in ("Column_", "f"):
        if str(k).startswith(prefix) and str(k)[len(prefix):].isdigit():
            return names[int(str(k)[len(prefix):])]
    return str(k)


def run_experiment(cfg, name: str, ds: Dict[str, Any], tracer, deadline: Deadline,
                   cache: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    exp = experiment_config(cfg, name)
    mdir = layout.model_dir(cfg, name)
    done = read_json(mdir / "metrics.json")
    if done and (read_json(mdir / "state.json") or {}).get("status") == "done":
        return done
    model = exp.get("model", "xgboost")
    fs = exp.get("feature_set", "full")
    names = list(get_path(cfg, f"features.sets.{fs}"))
    scheme = exp.get("label_scheme", "coarse")
    params = dict(exp.get(model, {}))
    atomic_write_json(mdir / "experiment.json", {"name": name, **layout.experiment_spec(cfg, name)})
    for split in ("train", "val"):
        if split not in cache:
            cache[split] = _read_parquet(Path(ds["dir"]) / f"{split}.parquet")
    voc = {k: ds[k] for k in ("classes", "labels", "label_class")}
    classes, lut = label_space(voc, scheme)
    tr, va = cache["train"], cache["val"]
    hold = exp.get("holdout")
    if hold:  # e.g. the third Mirai run: left out of training and early stopping, tested in step 4
        tr = tr[~held_out_mask(tr, ds["labels"], hold)]
        va = va[~held_out_mask(va, ds["labels"], hold)]
    y = lut[tr["label"].to_numpy(np.int64)]
    yv = lut[va["label"].to_numpy(np.int64)]
    X, Xv = make_matrix(tr, names), make_matrix(va, names)
    w = balanced(tr["weight_packets"].to_numpy(np.float64), y, len(classes))
    wv_bal = balanced(va["weight_packets"].to_numpy(np.float64), yv, len(classes))
    tracer.info(STAGE, "experiment", f"{name}: {model}, {len(names)} features ({fs}), {len(classes)} classes "
                f"({scheme}); {len(y):,} training rows, {len(yv):,} validation rows")
    prog = _Progress(mdir, int(params.get("early_stopping_rounds", 10**9)), int(params.get("checkpoint_every", 1)),
                     deadline, tracer, name, int(params.get("n_estimators", 1)))
    predict, imp = FITTERS[model](params, X, y, w, Xv, yv, wv_bal, len(classes), mdir, prog)
    proba = predict(Xv)
    pred = np.asarray(proba).argmax(axis=1)
    wpk = va["weight_packets"].to_numpy(np.float64)
    m = evaluate(yv, pred, wpk, classes)
    if scheme == "fine":  # the same predictions seen as the six coarse classes
        to_coarse = np.array([ds["label_class"][ds["labels"].index(c)] for c in classes])
        m["as_coarse"] = evaluate(to_coarse[yv], to_coarse[pred], wpk, ds["classes"])
    imp_named = {_feature_name(k, names): float(v) for k, v in imp.items()}
    imp_s = pd.Series(imp_named, dtype=float)
    imp_s = (imp_s / imp_s.sum()).sort_values(ascending=False) if imp_s.sum() else imp_s
    st = prog.state
    m.update({"name": name, "model": model, "feature_set": fs, "label_scheme": scheme, "features": names,
              "holdout": hold,
              "iterations": st["iterations"], "best_iter": st["best_iter"], "minutes": round(st["seconds"] / 60, 1),
              "importance": imp_s.head(15).to_dict(), "train_rows": int(len(y)), "val_rows": int(len(yv)),
              "dir": str(mdir)})
    pd.DataFrame(m["per_class"]).to_csv(mdir / "per_class.csv", index=False)
    confusion_table(m).to_csv(mdir / "confusion.csv", index=False)
    imp_s.to_csv(mdir / "importance.csv", header=["share"])
    atomic_write_json(mdir / "metrics.json", m)
    st["status"] = "done"
    atomic_write_json(mdir / "state.json", st)
    tracer.info(STAGE, "result", f"{name}: validation macro-F1 {m['macro_f1']:.4f}, balanced accuracy "
                f"{m['balanced_accuracy']:.4f} ({st['best_iter']} rounds/trees, {m['minutes']} min)")
    return m


def run_train(cfg, tracer, deadline: Deadline, only: Optional[List[str]] = None) -> Dict[str, Any]:
    ds = build_dataset(cfg, tracer, deadline)
    names = [n for n in experiment_names(cfg) if not only or n in only]
    unknown = [n for n in (only or []) if n not in experiment_names(cfg)]
    if unknown:
        raise KeyError(f"Unknown experiment(s) {unknown}. Known: {experiment_names(cfg)}")
    cache: Dict[str, pd.DataFrame] = {}
    results = {}
    for n in names:
        deadline.check()
        try:
            results[n] = run_experiment(cfg, n, ds, tracer, deadline, cache)
        except (GracefulStop, KeyboardInterrupt):
            write_report(cfg, ds)
            raise
        except Exception as exc:  # noqa: BLE001
            tracer.exception(STAGE, "experiment_failed", exc, experiment=n)
        write_report(cfg, ds)
    return results


# =====================================================================
# report comparing experiments

def write_report(cfg, ds: Dict[str, Any]) -> None:
    rep = layout.reports(cfg, "train")
    rows, done = [], []
    for n in experiment_names(cfg):
        mdir = layout.model_dir(cfg, n)
        m = read_json(mdir / "metrics.json")
        st = read_json(mdir / "state.json") or {}
        exp = experiment_config(cfg, n)
        if not m:
            rows.append({"experiment": n, "model": exp.get("model"), "features": exp.get("feature_set"),
                         "labels": exp.get("label_scheme", "coarse"),
                         "status": f"{st.get('iterations', 0)} rounds so far" if st else "not started"})
            continue
        done.append(m)
        row = {"experiment": n, "model": m["model"], "features": m["feature_set"], "labels": m["label_scheme"],
               "status": "done", "rounds": m["best_iter"], "minutes": m["minutes"], "macro_f1": m["macro_f1"],
               "balanced_acc": m["balanced_accuracy"]}
        per = m["as_coarse"]["per_class"] if m["label_scheme"] == "fine" else m["per_class"]
        row.update({f"F1 {p['class']}": p["f1"] for p in per})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(rep / "experiments.csv", index=False)
    L = ["# Gotham dataset - step 3: train (validation results)", "",
         f"Dataset: `{ds['dir']}` - train {ds['train_rows']:,} rows, validation {ds['val_rows']:,} rows "
         f"(classes sampled to at most {ds['spec']['max_rows_per_class']:,} / "
         f"{ds['spec']['val_max_rows_per_class']:,} unique rows; every class has the same total weight in training; "
         "metrics below are on validation **packets**). The test split has not been read.", "",
         f"Rows stored in two files and counted once: {ds['cross_file_duplicate_rows']:,} "
         f"({ds['cross_file_duplicate_packets']:,} packets)" + (": " + "; ".join(
             f"{k} ({v:,})" for k, v in ds["cross_file_pairs"].items()) if ds["cross_file_pairs"] else ""), "",
         "## 1. Experiments", "",
         "For `fine` experiments the F1 columns are the six coarse classes, read from the fine predictions.", "",
         md_table(table.round(4), max_rows=40), ""]
    by = {(m["model"], m["feature_set"], m["label_scheme"]): m for m in done if not m.get("holdout")}
    rel = []
    for model in ("xgboost", "lightgbm", "random_forest"):
        full = by.get((model, "full", "coarse"))
        for fs, without in (("no_ttl", "TTL"), ("portable", "TTL + raw ports"),
                            ("strict", "TTL + raw ports + TCP window/options")):
            o = by.get((model, fs, "coarse"))
            if full and o:
                r = {"model": model, "without": without,
                     "macro_f1 full": full["macro_f1"], "macro_f1 without": o["macro_f1"],
                     "change": o["macro_f1"] - full["macro_f1"]}
                for a, b in zip(full["per_class"], o["per_class"]):
                    r[f"change F1 {a['class']}"] = (b["f1"] or 0) - (a["f1"] or 0)
                rel.append(r)
    L += ["## 2. How much the models rely on testbed-specific fields", "",
          "Change in validation macro-F1 when TTL (`no_ttl`) or TTL and raw port numbers (`portable`) are "
          "removed. A small change means the model recognises the attacks from their behaviour; a large one "
          "means it relied on values that identify machines or streams of this testbed.", "",
          md_table(pd.DataFrame(rel).round(4), max_rows=20) if rel else "_(needs the full model and at least one "
          "of no_ttl / portable of the same type)_", ""]
    L += ["## 3. Per experiment", ""]
    for m in done:
        L += [f"### {m['name']}", "",
              f"{m['model']}, feature set `{m['feature_set']}` ({len(m['features'])} features), "
              f"{m['label_scheme']} labels, {m['best_iter']} rounds/trees kept of {m['iterations']}, "
              f"{m['minutes']} min. Validation macro-F1 **{m['macro_f1']:.4f}**, balanced accuracy "
              f"{m['balanced_accuracy']:.4f}, accuracy {m['accuracy']:.4f}.", "",
              md_table(pd.DataFrame(m["per_class"]).round(4), max_rows=30), "",
              "Confusion (rows: true class, % of its packets):", "",
              md_table(confusion_table(m).round(3), max_rows=30), "",
              "Most important features (share of total gain): " + ", ".join(
                  f"`{k}` {v:.3f}" for k, v in list(m["importance"].items())[:10]), ""]
    checks = []
    for m in done:
        weak = [p["class"] for p in m["per_class"] if p["f1"] is not None and p["f1"] == p["f1"] and p["f1"] < 0.9]
        if weak:
            checks.append(f"{m['name']}: F1 below 0.9 for {', '.join(weak)}")
    pending = [r["experiment"] for r in rows if r["status"] != "done"]
    if pending:
        checks.append(f"Not finished yet: {', '.join(pending)} - run `python -m gids train` again to continue")
    L += ["## 4. Things to check", ""] + ([f"- {c}" for c in checks] or ["- nothing flagged"])
    L += ["", "## 5. Next", "", "Step 4 evaluates the chosen models once on the untouched test split, on the "
          "unseen third Mirai run, and per device type.", ""]
    atomic_write_text(rep / "TRAIN.md", "\n".join(L))

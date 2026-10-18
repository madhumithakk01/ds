"""Step 6: the experiments and the document for a paper.

``python -m gids paper [--parts a,b] [--max-minutes N]`` runs these parts in order.  Every part is
resumable (finished models, predictions and tables are kept on Drive and skipped), a part that fails
is logged and the next one still runs, and ``reports/paper/PAPER.md`` is rewritten after every part,
so it always shows what is finished so far.

==========  ===  ============================================================================
audit       E0   dataset audit and corrected labels (gids/relabel.py)
reference   E1   the published protocol reproduced on the same packets, P1-P9 (gids/reference.py)
score       E3   test predictions of the step 3 models, kept for the confidence intervals
corrected   E2   corrected labels: prepare, train and test the main models again
ablation    E4   this packet's header fields only vs earlier packets only
baselines   E5   logistic regression and the dataset authors' MLP layout on our features
efficiency  E6   model size and prediction speed; ingest speed
lodo        E7   each attacked device type left out of training in turn
seeds       E3   the main models trained with more seeds
==========  ===  ============================================================================

Confidence intervals: cluster bootstrap over the test packets.  A cluster is one device type x raw
label x run (episode); clusters are resampled within each raw label, so every attack keeps its
share; the same resamples are used for every model, which gives paired intervals for differences.
"""
from __future__ import annotations

import copy
import io
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import layout
from .config import experiment_config
from .explore import md_table
from .state import Deadline, GracefulStop, atomic_write_bytes, atomic_write_json, atomic_write_text, output_ok, read_json

STAGE = "paper"
PARTS = ["audit", "reference", "score", "corrected", "ablation", "baselines", "efficiency", "lodo", "seeds"]
CLASSES6 = ["Normal", "Brute Force", "C&C Communication", "DoS", "Infection", "Network Scanning"]

# test-set F1 as published (random 60/20/20 split of 35,161,611 packets; test = 7,025,321 packets)
PUBLISHED_SRC = "Alharbi & Ahmed, arXiv:2606.31594 (2026), Tables 3-7"
PUBLISHED = [
    ("random forest", 0.99, [1.00, 0.96, 1.00, 1.00, 1.00, 0.99]),
    ("XGBoost", 0.97, [1.00, 0.88, 1.00, 1.00, 0.99, 0.96]),
    ("DNN 128-128-64", 0.91, [1.00, 0.75, 0.98, 1.00, 0.77, 0.93]),
    ("logistic regression", 0.72, [1.00, 0.36, 0.91, 1.00, 0.19, 0.85]),
    ("naive Bayes", 0.43, [0.74, 0.14, 0.90, 0.47, 0.17, 0.15]),
]
# the dataset paper's federated DNN (80/20 split per device): per-class F1 and accuracy 0.89
DATASET_PAPER = ("federated DNN (dataset paper)", [0.69, 0.60, 0.97, 0.94, 0.00, 0.56])


def _p(cfg) -> Dict[str, Any]:
    return cfg.get("paper", {}) or {}


def paper_dir(cfg, *sub) -> Path:
    return layout.base_reports(cfg, "paper", *sub)


# =====================================================================
# experiments added on the fly (seeds, leave-one-device-type-out)

def _variant(cfg, base: str, seed: Optional[int] = None, holdout: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    e = copy.deepcopy(cfg["experiments"][base])
    e["enabled"] = False
    if seed is not None:
        model = experiment_config(cfg, base).get("model", "xgboost")
        e[model] = {**(e.get(model) or {}), "seed": int(seed)}
    if holdout is not None:
        e["holdout"] = holdout
    return e


def seed_experiments(cfg) -> Dict[str, Dict[str, Any]]:
    p = _p(cfg)
    return {f"{b}_seed{s}": _variant(cfg, b, seed=s) for b in p.get("seed_models", []) for s in p.get("seeds", [])}


def lodo_experiments(cfg, targets: List[str]) -> Dict[str, Dict[str, Any]]:
    base = _p(cfg).get("lodo_model", "xgb_portable")
    return {lodo_name(base, d): _variant(cfg, base, holdout={"device_type": d}) for d in targets}


def lodo_name(base: str, device_type: str) -> str:
    return f"{base}_lodo_{re.sub(r'[^0-9A-Za-z]+', '-', device_type)}"


def with_experiments(cfg, extra: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    c = copy.deepcopy(cfg)
    c.setdefault("experiments", {}).update(extra)
    return c


# =====================================================================
# shared data of one run

class Ctx:
    """Dataset, test table and bootstrap of a configuration, loaded when first needed (one test table at a time)."""

    def __init__(self, cfg, tracer, deadline: Deadline):
        self.cfg, self.tracer, self.deadline = cfg, tracer, deadline
        self._ds: Dict[str, Dict[str, Any]] = {}
        self._test: Dict[str, pd.DataFrame] = {}
        self._boot: Dict[str, "Bootstrap"] = {}

    def ds(self, cfg) -> Dict[str, Any]:
        from .train import build_dataset

        key = layout.dataset_dir(cfg).name
        if key not in self._ds:
            self._ds[key] = build_dataset(cfg, self.tracer, self.deadline)
        return self._ds[key]

    def test(self, cfg) -> pd.DataFrame:
        from .evaluate import build_test

        key = layout.dataset_dir(cfg).name
        if key not in self._test:
            self._test.clear()
            self._boot = {k: v for k, v in self._boot.items() if k == key}
            self._test[key] = build_test(cfg, self.tracer, self.deadline)
        return self._test[key]

    def boot(self, cfg) -> "Bootstrap":
        key = layout.dataset_dir(cfg).name
        if key not in self._boot:
            rounds, seed = boot_settings(cfg)
            bs = Bootstrap(self.test(cfg), self.ds(cfg), rounds, seed)
            atomic_write_json(boot_info_path(cfg), {"clusters": bs.n_clusters, "labels": bs.n_strata,
                                                    "single_cluster_labels": bs.single, "rounds": rounds,
                                                    "seed": seed, "classes": bs.classes})
            self._boot[key] = bs
        return self._boot[key]


def boot_settings(cfg) -> Tuple[int, int]:
    p = _p(cfg)
    return int(p.get("bootstrap_rounds", 1000)), int(p.get("bootstrap_seed", 0))


def boot_info_path(cfg) -> Path:
    rounds, seed = boot_settings(cfg)
    return layout.dataset_dir(cfg) / f"bootstrap-{rounds}-{seed}.json"


def _save_npy(path: Path, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    atomic_write_bytes(path, buf.getvalue())


def model_path(cfg, name: str) -> Path:
    """layout.model_dir without creating the folder (checking must not leave empty folders on Drive)."""
    from .state import stable_hash

    return layout.work(cfg) / "models" / f"{name}-{stable_hash(layout.experiment_spec(cfg, name))}"


def trained(cfg, name: str) -> bool:
    mdir = model_path(cfg, name)
    return (read_json(mdir / "state.json") or {}).get("status") == "done" and output_ok(mdir / "model.bin")


def predictions(ctx: Ctx, cfg, name: str, compute: bool = True) -> Optional[np.ndarray]:
    """Predicted class of every test row (cached as test_pred.npy next to the model)."""
    from .train import load_model

    if name not in (cfg.get("experiments") or {}):
        return None
    path = model_path(cfg, name) / "test_pred.npy"
    if output_ok(path):
        return np.load(path)
    if not compute or not trained(cfg, name):
        return None
    ds, test = ctx.ds(cfg), ctx.test(cfg)
    mdl = load_model(cfg, name, ds)
    pred = np.empty(len(test), dtype=np.int8)
    with ctx.tracer.span(STAGE, f"predict.{name}"):
        for i in range(0, len(test), 500_000):
            ctx.deadline.check()
            pred[i:i + 500_000] = np.asarray(mdl["predict"](test.iloc[i:i + 500_000])).argmax(axis=1)
    _save_npy(path, pred)
    ctx.tracer.info(STAGE, "predictions", f"{name}: {len(pred):,} test rows")
    return pred


# =====================================================================
# cluster bootstrap

def batch_metrics(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """Metrics of a batch of confusion matrices (B x k x k, rows = true class, class 0 = Normal)."""
    tp = np.einsum("bii->bi", cm)
    sup, pre = cm.sum(axis=2), cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(sup > 0, tp / sup, np.nan)
        prec = np.where(pre > 0, tp / pre, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    f1 = np.where(sup > 0, f1, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"macro_f1": np.nanmean(f1, axis=1), "f1": f1, "recall": rec,
                "benign_flagged_per_million": 1e6 * (sup[:, 0] - tp[:, 0]) / np.maximum(sup[:, 0], 1e-12),
                "attacks_missed_pct": 100 * cm[:, 1:, 0].sum(axis=1) / np.maximum(sup[:, 1:].sum(axis=1), 1e-12)}


class Bootstrap:
    """Resampling weights of the test clusters, shared by every model scored on one test table."""

    def __init__(self, test: pd.DataFrame, ds: Dict[str, Any], rounds: int, seed: int):
        lut = np.array(ds["label_class"], dtype=np.int64)
        self.classes = list(ds["classes"])
        self.k = len(self.classes)
        self.y = lut[test["label"].to_numpy(np.int64)]
        self.w = test["weight_packets"].to_numpy(np.float64)
        g = test.groupby(["device_type", "label", "episode"], sort=True, observed=True)
        self.cid = g.ngroup().to_numpy(np.int64)
        keys = g.size().reset_index()
        self.n_clusters = len(keys)
        strata = keys["label"].to_numpy()
        self.n_strata = len(np.unique(strata))
        self.single = int((pd.Series(strata).value_counts() == 1).sum())   # labels that add no variance
        rng = np.random.default_rng(seed)
        self.counts = np.zeros((rounds, self.n_clusters))
        for lab in np.unique(strata):
            idx = np.flatnonzero(strata == lab)
            self.counts[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=rounds)
        self.rounds = rounds

    def confusion(self, pred: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Confusion matrix of every cluster (n_clusters x k*k), packets."""
        cid, y, w, p = self.cid, self.y, self.w, pred.astype(np.int64)
        if rows is not None:
            cid, y, w, p = cid[rows], y[rows], w[rows], p[rows]
        idx = (cid * self.k + y) * self.k + p
        return np.bincount(idx, weights=w, minlength=self.n_clusters * self.k * self.k).reshape(
            self.n_clusters, self.k * self.k)

    def score(self, pred: np.ndarray) -> Dict[str, Any]:
        cc = self.confusion(pred)
        point = batch_metrics(cc.sum(axis=0).reshape(1, self.k, self.k))
        boot = batch_metrics((self.counts @ cc).reshape(-1, self.k, self.k))
        return {"point": {k: v[0] for k, v in point.items()}, "boot": boot,
                "confusion": cc.sum(axis=0).reshape(self.k, self.k)}


def ci(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def boot_result(ctx: Ctx, cfg, name: str) -> Optional[Dict[str, Any]]:
    """Point metrics and the metrics of every bootstrap round of one model (cached next to its predictions, so
    the test table is read only for models scored for the first time)."""
    if name not in (cfg.get("experiments") or {}):
        return None
    mdir = model_path(cfg, name)
    if not output_ok(mdir / "test_pred.npy"):
        return None
    rounds, seed = boot_settings(cfg)
    cache = mdir / f"test_boot-{rounds}-{seed}.npz"
    if output_ok(cache) and cache.stat().st_mtime >= (mdir / "test_pred.npy").stat().st_mtime:
        z = np.load(cache)
        return {"point": {k[6:]: z[k][0] for k in z.files if k.startswith("point_")},
                "boot": {k[5:]: z[k] for k in z.files if k.startswith("boot_")}}
    r = ctx.boot(cfg).score(np.load(mdir / "test_pred.npy"))
    buf = io.BytesIO()
    np.savez(buf, **{f"point_{k}": np.asarray([v]) for k, v in r["point"].items()},
             **{f"boot_{k}": v for k, v in r["boot"].items()})
    atomic_write_bytes(cache, buf.getvalue())
    return r


def scored(ctx: Ctx, cfg, names: List[str], main: Optional[str]) -> pd.DataFrame:
    """Test macro-F1 with 95 % interval for every model that has predictions; paired difference to ``main``."""
    res = {n: r for n in names if (r := boot_result(ctx, cfg, n)) is not None}
    if not res:
        return pd.DataFrame()
    classes = (read_json(boot_info_path(cfg)) or {}).get("classes") or ctx.ds(cfg)["classes"]
    base = res.get(main)
    rows = []
    for n, r in res.items():
        lo, hi = ci(r["boot"]["macro_f1"])
        row = {"model": n, "macro_f1": r["point"]["macro_f1"], "ci_low": lo, "ci_high": hi}
        if base is not None and n != main:
            d = r["boot"]["macro_f1"] - base["boot"]["macro_f1"]
            row.update({"diff_vs_main": r["point"]["macro_f1"] - base["point"]["macro_f1"],
                        "diff_low": ci(d)[0], "diff_high": ci(d)[1]})
        for j, c in enumerate(classes):
            row[f"F1 {c}"] = r["point"]["f1"][j]
        row["benign flagged /1M"] = r["point"]["benign_flagged_per_million"]
        row["attacks missed %"] = r["point"]["attacks_missed_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
# parts

def part_audit(ctx: Ctx) -> None:
    from .relabel import run_audit

    run_audit(ctx.cfg, ctx.tracer, ctx.deadline)


def part_reference(ctx: Ctx) -> None:
    from .reference import run_reference

    run_reference(ctx.cfg, ctx.tracer, ctx.deadline)


def _train_and_predict(ctx: Ctx, cfg, names: List[str]) -> None:
    from .train import run_train

    names = [n for n in names if n in (cfg.get("experiments") or {})]
    todo = [n for n in names if not trained(cfg, n)]
    if todo:
        run_train(cfg, ctx.tracer, ctx.deadline, only=todo)
    for n in names:
        ctx.deadline.check()
        predictions(ctx, cfg, n)


def part_score(ctx: Ctx) -> None:
    p = _p(ctx.cfg)
    _train_and_predict(ctx, ctx.cfg, [p.get("main", "xgb_portable")] + list(p.get("compare", [])))


def part_corrected(ctx: Ctx) -> None:
    from .prepare import run_prepare
    from .relabel import corrected_cfg

    ccfg = corrected_cfg(ctx.cfg)
    run_prepare(ccfg, ctx.tracer, ctx.deadline)
    _train_and_predict(ctx, ccfg, list(_p(ctx.cfg).get("corrected_models", [])))


def part_ablation(ctx: Ctx) -> None:
    _train_and_predict(ctx, ctx.cfg, list(_p(ctx.cfg).get("ablation", [])))


def part_baselines(ctx: Ctx) -> None:
    _train_and_predict(ctx, ctx.cfg, list(_p(ctx.cfg).get("baselines", [])))


def part_seeds(ctx: Ctx) -> None:
    cfg2 = with_experiments(ctx.cfg, seed_experiments(ctx.cfg))
    _train_and_predict(ctx, cfg2, list(seed_experiments(ctx.cfg)))


def lodo_targets(ctx: Ctx) -> List[str]:
    """Attacked device types, most attack test packets first (saved once, so later runs agree)."""
    path = paper_dir(ctx.cfg) / "lodo_targets.json"
    saved = read_json(path)
    limit = int(_p(ctx.cfg).get("lodo_limit", 8))
    if not saved:
        test, ds = ctx.test(ctx.cfg), ctx.ds(ctx.cfg)
        y = np.array(ds["label_class"])[test["label"].to_numpy(np.int64)]
        att = test.loc[y != 0].groupby("device_type", observed=True)["weight_packets"].sum()
        att = att[att > 0].sort_values(ascending=False)
        saved = {"device_types": [str(d) for d in att.index], "attack_packets": [float(v) for v in att.values]}
        atomic_write_json(path, saved)
    return saved["device_types"][:limit] if limit > 0 else saved["device_types"]


def part_lodo(ctx: Ctx) -> None:
    from .train import load_model, run_train

    cfg = ctx.cfg
    targets = lodo_targets(ctx)
    exps = lodo_experiments(cfg, targets)
    cfg2 = with_experiments(cfg, exps)
    for d, name in zip(targets, exps):
        ctx.deadline.check()
        mdir = model_path(cfg2, name)
        if output_ok(mdir / "heldout_pred.npy"):
            continue
        if not trained(cfg2, name):
            run_train(cfg2, ctx.tracer, ctx.deadline, only=[name])
        if not trained(cfg2, name):
            continue
        test = ctx.test(cfg)
        rows = test[test["device_type"].astype(str).to_numpy() == d]
        mdl = load_model(cfg2, name, ctx.ds(cfg))
        pred = np.concatenate([np.asarray(mdl["predict"](rows.iloc[i:i + 500_000])).argmax(axis=1)
                               for i in range(0, len(rows), 500_000)]).astype(np.int8)
        _save_npy(mdir / "heldout_pred.npy", pred)
        ctx.tracer.info(STAGE, "lodo", f"{d}: {len(rows):,} test rows scored by {name}")


def part_efficiency(ctx: Ctx) -> None:
    from .envinfo import _gpu
    from .train import load_model, make_matrix, raw_predictor

    cfg = ctx.cfg
    p = _p(cfg)
    names = [p.get("main", "xgb_portable")] + list(p.get("compare", [])) + list(p.get("baselines", [])) + \
        list(p.get("ablation", []))
    names = [n for n in dict.fromkeys(names) if n in cfg["experiments"] and trained(cfg, n)]
    if not names:
        return
    test, ds = ctx.test(cfg), ctx.ds(cfg)
    n = min(int(p.get("efficiency_rows", 200_000)), len(test))
    sub = test.sample(n=n, random_state=0) if n < len(test) else test
    rows = []
    for name in names:
        ctx.deadline.check()
        mdl = load_model(cfg, name, ds)
        mdir = mdl["dir"]
        size = sum(f.stat().st_size for f in mdir.glob("*.bin") if f.name in ("model.bin", "best.bin"))
        if mdl["model"] == "mlp":
            size = (mdir / "best.bin").stat().st_size
        _, raw = raw_predictor(mdl["model"], mdir, mdl["best"], len(mdl["classes"]))
        raw(make_matrix(sub.iloc[:1000], mdl["features"]))  # warm-up
        t0 = time.perf_counter()
        X = make_matrix(sub, mdl["features"])
        t1 = time.perf_counter()
        raw(X)
        t2 = time.perf_counter()
        rows.append({"model": name, "model_type": mdl["model"], "features": len(mdl["features"]),
                     "rounds_or_trees": mdl["best"], "size_mb": size / 2**20,
                     "feature_rows_per_s": n / max(t1 - t0, 1e-9), "predict_rows_per_s": n / max(t2 - t1, 1e-9),
                     "end_to_end_rows_per_s": n / max(t2 - t0, 1e-9)})
    from .relabel import _ingest_metas

    _, metas = _ingest_metas(cfg)
    ing_rows = sum(int(m.get("rows", 0)) for m in metas)
    ing_s = sum(float(m.get("seconds", 0)) for m in metas)
    atomic_write_json(paper_dir(cfg) / "efficiency.json", {
        "rows_timed": n, "cpu_count": os.cpu_count(), "gpu": _gpu(), "models": rows,
        "ingest": {"packets": ing_rows, "seconds": ing_s, "packets_per_s": ing_rows / ing_s if ing_s else None}})


PART_FUNCS = {"audit": part_audit, "reference": part_reference, "score": part_score, "corrected": part_corrected,
              "ablation": part_ablation, "baselines": part_baselines, "efficiency": part_efficiency,
              "lodo": part_lodo, "seeds": part_seeds}


def run_paper(cfg, tracer, deadline: Deadline, parts: Optional[List[str]] = None) -> Dict[str, Any]:
    parts = parts or list(PARTS)
    unknown = [x for x in parts if x not in PART_FUNCS]
    if unknown:
        raise KeyError(f"Unknown part(s) {unknown}. Parts: {', '.join(PARTS)}")
    ctx = Ctx(cfg, tracer, deadline)
    prog_path = paper_dir(cfg) / "progress.json"
    progress = read_json(prog_path) or {}
    for part in [x for x in PARTS if x in parts]:
        deadline.check()
        t0 = time.time()
        tracer.info(STAGE, "part", f"--- {part} ---")
        try:
            PART_FUNCS[part](ctx)
            progress[part] = {"status": "done", "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "minutes": round((time.time() - t0) / 60, 1)}
        except (GracefulStop, KeyboardInterrupt):
            progress[part] = {"status": "stopped (run paper again to continue)"}
            atomic_write_json(prog_path, progress)
            _write_safely(ctx, tracer)
            raise
        except Exception as exc:  # noqa: BLE001 - one failed part must not block the others
            tracer.exception(STAGE, "part_failed", exc, part=part)
            progress[part] = {"status": f"failed: {type(exc).__name__}: {exc}"[:300]}
        atomic_write_json(prog_path, progress)
        _write_safely(ctx, tracer)
    return progress


def _write_safely(ctx: Ctx, tracer) -> None:
    try:
        write_paper(ctx)
    except (GracefulStop, KeyboardInterrupt):
        raise
    except Exception as exc:  # noqa: BLE001
        tracer.exception(STAGE, "report_failed", exc)


# =====================================================================
# figures

def fig_ci(path: Path, title: str, labels: List[str], est: List[float], lo: List[float], hi: List[float],
           xlabel: str) -> None:
    """Dots with 95 % interval whiskers, one row per model; the axis starts just below the lowest interval."""
    from .results import INK, INK2, SERIES, SURFACE, _fig, _save, _style

    fig, ax, plt = _fig(7.2, 0.42 * len(labels) + 1.4)
    _style(ax)
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, lo, hi, color=SERIES[0], linewidth=2.2, zorder=2)
    ax.scatter(est, y, s=60, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    for e, h, yy in zip(est, hi, y):
        ax.annotate(f"{e:.4f}", (h, yy), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK2)
    left = max(0.0, np.floor((min(lo) - 0.005) * 100) / 100)
    right = min(1.0, max(hi)) + (min(1.0, max(hi)) - left) * 0.18
    ax.set_xlim(left, right)
    ax.set_yticks(y, labels, color=INK)
    ax.set_xlabel(f"{xlabel} (axis starts at {left:g})" if left > 0 else xlabel, color=INK2, fontsize=9)
    _save(fig, plt, path, title, ax)


# =====================================================================
# PAPER.md

def _f(x, nd: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _ci_txt(r) -> str:
    return f"{_f(r['macro_f1'])} [{_f(r['ci_low'])}, {_f(r['ci_high'])}]"


def write_paper(ctx: Ctx) -> None:
    from .envinfo import _code_version
    from .reference import PROTOCOLS, reference_dir, reference_table
    from .results import fig_dots

    cfg = ctx.cfg
    p = _p(cfg)
    out = paper_dir(cfg)
    figs = paper_dir(cfg, "figures")
    main = p.get("main", "xgb_portable")
    progress = read_json(out / "progress.json") or {}
    code = _code_version() or {}
    L = ["# Gotham Dataset 2025 - results for the paper", "",
         f"Written {time.strftime('%Y-%m-%d %H:%M')} from the outputs on Drive (code "
         f"{code.get('commit', '?')}{' + local changes' if code.get('dirty') else ''}). Parts: " + ", ".join(
             f"{k} {v.get('status', '?').split(':')[0]}" for k, v in progress.items()) + ".", ""]
    summary: List[str] = []
    body: List[str] = []

    # ---------------- E0 audit
    audit = read_json(layout.base_reports(cfg, "audit") / "audit.json")
    body += ["## E0. Dataset audit", ""]
    if audit:
        u = audit["unknown"]
        body += ["Full report: `reports/audit/AUDIT.md`.", "",
                 f"- **Copied captures:** {audit['duplicate_packets']:,} packets in "
                 f"{len(audit['duplicate_files'])} file(s) are copies of other devices' captures "
                 "(`glob(f\"{device}*.csv\")` in `data_preparation.py` also matches devices -10 ... -15 for device -1).",
                 f"- **Mirai infection labels:** of {audit['tcp_scan_packets']:,} `TCP Scan` packets, "
                 f"{audit['tcp_scan_bot_telnet_sessions']:,} ({100 * (audit['tcp_scan_bot_session_share'] or 0):.1f}%) "
                 "belong to telnet connections that carry data between internal hosts other than the attacker "
                 "(bots' login sessions) - the labelling rules give every such packet the scan label.",
                 f"- **Unknown packets:** {u['packets']:,}, of which {u['masscan_packets']:,} involve the masscan "
                 "host; the published pipelines drop them.", ""]
        if audit["duplicate_packets"] or audit["tcp_scan_bot_telnet_sessions"]:
            summary.append(f"Audit: {audit['duplicate_packets']:,} copied packets; "
                           f"{100 * (audit['tcp_scan_bot_session_share'] or 0):.1f}% of `TCP Scan` packets are bots' "
                           f"telnet login sessions; {u['masscan_packets']:,} masscan packets left `Unknown`.")
    else:
        body += ["_Not run yet (`--parts audit`)._", ""]

    # ---------------- E1 reference
    ref = reference_table(cfg)
    body += ["## E1. The published protocol, reproduced", ""]
    pub = pd.DataFrame([{"source": PUBLISHED_SRC, "model": m, "split": "random", "macro_f1": f,
                         **{f"F1 {c}": v for c, v in zip(CLASSES6, per)}} for m, f, per in PUBLISHED] +
                       [{"source": "Belarbi et al. (dataset paper)", "model": DATASET_PAPER[0], "split": "random",
                         "macro_f1": float(np.mean(DATASET_PAPER[1])),
                         **{f"F1 {c}": v for c, v in zip(CLASSES6, DATASET_PAPER[1])}}])
    body += ["Published test scores (macro-F1 of the dataset paper's DNN = mean of its reported per-class F1):", "",
             md_table(pub, max_rows=10), ""]
    if len(ref):
        rmeta = read_json(reference_dir(cfg) / "rows.json") or {}
        body += ["Reproduction on the same packets (all protocols: the authors' six classes, class-balanced training, "
                 "early stopping on validation, test metrics on packets). Protocols:", ""]
        body += [f"- **{k}**: {v[2]}" for k, v in PROTOCOLS.items()] + [""]
        body += [md_table(_round(ref), max_rows=30), ""]
        if rmeta:
            body += [f"Reference rows: {rmeta['rows']:,} packets sampled from {rmeta['packets']:,} "
                     f"(at most {rmeta['spec']['max_rows_per_label']:,} per raw label, each row weighted by "
                     "1/p); random split stratified per raw label with seed "
                     f"{rmeta['spec']['seed']}; numeric inputs standardised on the training rows. The preprocessing "
                     "is a vectorised re-implementation of the authors' functions; `tools/check_reference.py` "
                     "compares the two input by input.", ""]
            other = [f"{d['label']} ({d['grouping']})" for d in rmeta.get("labels", [])
                     if str(d.get("grouping", "")).startswith("not in")]
            if other:
                body += ["Raw labels outside the authors' grouping: " + "; ".join(other) + ".", ""]
        get = {(row["protocol"], row["model"]): row["macro_f1"] for _, row in ref.iterrows()}
        sets = [("reference inputs (70)", "P1", "P4"), ("reference without timestamp", "P2", "P5"),
                ("timestamp only", "P3", None), ("our portable features", "P6", "P8"),
                ("our full features", "P7", "P9")]
        series = {"random split": [get.get((a, "XGBoost")) for _, a, _ in sets],
                  "time-blocked split": [get.get((b, "XGBoost")) if b else None for _, _, b in sets]}
        if any(v is not None for v in series["random split"] + series["time-blocked split"]):
            fig_dots(figs / "e1_protocols.png", "XGBoost test macro-F1 by inputs and split", [s for s, _, _ in sets],
                     series, "test macro-F1")
            body += ["![protocols](figures/e1_protocols.png)", ""]
        p1, p4 = get.get(("P1", "XGBoost")), get.get(("P4", "XGBoost"))
        p1rf, p3 = get.get(("P1", "random forest")), get.get(("P3", "XGBoost"))
        p6, p8 = get.get(("P6", "XGBoost")), get.get(("P8", "XGBoost"))
        if p1 is not None:
            s = f"Published protocol reproduced: P1 XGBoost {p1:.3f} (published 0.97)"
            s += f", random forest {p1rf:.3f} (published 0.99)" if p1rf is not None else ""
            s += f"; the arrival time alone gives {p3:.3f}" if p3 is not None else ""
            s += f"; the same inputs with a time-blocked split give {p4:.3f}" if p4 is not None else ""
            summary.append(s + ".")
        if p6 is not None and p8 is not None:
            summary.append(f"Our portable features: {p6:.3f} with a random packet split vs {p8:.3f} time-blocked "
                           f"(difference {p6 - p8:+.3f}).")
    else:
        body += ["_Not run yet (`--parts reference`)._", ""]

    # ---------------- E2 corrected labels
    body += ["## E2. Corrected labels", ""]
    e2 = _corrected_table(ctx, cfg)
    if len(e2):
        body += ["Each pipeline on its own time-blocked test split (the corrected split is recomputed from the "
                 "corrected labels and leaves the copied captures out). Intervals as in E3.", "",
                 md_table(_round(e2), max_rows=10), ""]
        e2.to_csv(out / "e2_corrected.csv", index=False)
        for n in p.get("corrected_models", []):
            a = e2[(e2["model"] == n) & (e2["labels"] == "original")]
            b = e2[(e2["model"] == n) & (e2["labels"] == "corrected")]
            if len(a) and len(b):
                a, b = a.iloc[0], b.iloc[0]
                summary.append(f"Corrected labels, `{n}`: macro-F1 {_ci_txt(a)} -> {_ci_txt(b)}; Brute Force F1 "
                               f"{_f(a['F1 Brute Force'], 3)} -> {_f(b['F1 Brute Force'], 3)}, Network Scanning F1 "
                               f"{_f(a['F1 Network Scanning'], 3)} -> {_f(b['F1 Network Scanning'], 3)}.")
                break
    else:
        body += ["_Not run yet (`--parts corrected`)._", ""]

    # ---------------- E3 main results with intervals
    names = list(dict.fromkeys([main] + list(p.get("compare", []))))
    tab = _safe_scored(ctx, cfg, names, main)
    body += ["## E3. Test results with 95% confidence intervals", ""]
    if len(tab):
        bi = read_json(boot_info_path(cfg)) or {}
        body += [f"Cluster bootstrap: {bi.get('clusters')} clusters (device type x raw label x run) in "
                 f"{bi.get('labels')} raw labels, resampled within each label, {bi.get('rounds')} rounds; `diff` = "
                 f"macro-F1 minus `{main}` on the same resamples. {bi.get('single_cluster_labels')} raw label(s) have "
                 "a single cluster and add no resampling variance, so the intervals are a lower bound for those "
                 "classes.", "", md_table(_round(tab), max_rows=30), ""]
        tab.to_csv(out / "e3_test_ci.csv", index=False)
        fig_ci(figs / "e3_test_ci.png", "Test macro-F1 with 95% intervals", list(tab["model"]),
               list(tab["macro_f1"]), list(tab["ci_low"]), list(tab["ci_high"]), "test macro-F1")
        body += ["![intervals](figures/e3_test_ci.png)", ""]
        m = tab[tab["model"] == main]
        if len(m):
            summary.append(f"Main model `{main}`: test macro-F1 {_ci_txt(m.iloc[0])} (95% cluster-bootstrap "
                           "interval).")
    else:
        body += ["_No test predictions yet (`--parts score`)._", ""]
    seeds = _seed_table(ctx, cfg)
    if len(seeds):
        body += ["Training seeds (seed 0 = the step 3 model):", "", md_table(_round(seeds), max_rows=10), ""]
        for _, r in seeds.iterrows():
            if r["seeds"] > 1:
                summary.append(f"`{r['model']}` over {int(r['seeds'])} training seeds: macro-F1 "
                               f"{r['mean']:.4f} ± {r['sd']:.4f} (min {r['min']:.4f}).")

    # ---------------- E4 ablation, E5 baselines
    for key, title, part, lead in (
            ("ablation", "E4. Which features matter", "ablation",
             "`header`: fields of the packet itself (sizes, flags, TCP window and options, port classes, direction). "
             "`context`: direction plus features computed from earlier packets only. `portable` = both."),
            ("baselines", "E5. Model families on the same features", "baselines",
             "Logistic regression and a 128-128-64 ReLU network (the dataset authors' layout) on the portable "
             "features, next to the tree ensembles.")):
        extra = list(p.get(key, []))
        if key == "baselines":
            extra = [x for x in p.get("compare", []) if x.endswith("_portable")] + extra
        t = _safe_scored(ctx, cfg, [main] + extra, main)
        body += [f"## {title}", "", lead, ""]
        if len(t) > 1:
            body += [md_table(_round(t), max_rows=20), ""]
            t.to_csv(out / f"{key}.csv", index=False)
            parts_txt = [f"`{r['model']}` {_f(r['macro_f1'])} (diff {_f(r.get('diff_vs_main'))} "
                         f"[{_f(r.get('diff_low'))}, {_f(r.get('diff_high'))}])"
                         for _, r in t.iterrows() if r["model"] != main]
            summary.append(f"{title.split('. ')[1]}: " + "; ".join(parts_txt) + f" vs `{main}`.")
        else:
            body += [f"_Not run yet (`--parts {part}`)._", ""]

    # ---------------- E6 efficiency
    eff = read_json(out / "efficiency.json")
    body += ["## E6. Efficiency", ""]
    if eff:
        et = pd.DataFrame(eff["models"])
        body += [f"Prediction timed on {eff['rows_timed']:,} test rows in one batch, {eff['cpu_count']} CPU cores"
                 f"{' (GPU present, not used for prediction)' if eff.get('gpu') else ''}. "
                 "Feature rows/s = building the inputs from the stored fields.", "", md_table(_round(et), max_rows=20),
                 ""]
        ing = eff.get("ingest") or {}
        if ing.get("packets_per_s"):
            body += [f"Step 1 (CSV parsing + context features, one process): {ing['packets']:,} packets in "
                     f"{ing['seconds'] / 3600:.1f} h = {ing['packets_per_s']:,.0f} packets/s.", ""]
        mm = et[et["model"] == main]
        if len(mm):
            summary.append(f"Efficiency: `{main}` {mm.iloc[0]['size_mb']:.1f} MB, "
                           f"{mm.iloc[0]['end_to_end_rows_per_s']:,.0f} packets/s on {eff['cpu_count']} CPU cores.")
    else:
        body += ["_Not run yet (`--parts efficiency`)._", ""]

    # ---------------- E7 LODO
    body += ["## E7. Device types never seen in training", ""]
    lt = _lodo_table(ctx, cfg)
    if len(lt):
        body += [f"`{p.get('lodo_model', 'xgb_portable')}` trained without one device type (its training and "
                 "validation packets removed), scored on that device type's test packets, next to the model that saw "
                 "it. `detected` = attack packets called any attack; `right class` = attack packets given their "
                 "class.", "", md_table(_round(lt), max_rows=40), ""]
        lt.to_csv(out / "e7_lodo.csv", index=False)
        fig_dots(figs / "e7_lodo.png", "Attack packets detected on each device type", list(lt["device_type"]),
                 {"seen in training": list(lt["detected seen"]), "left out of training": list(lt["detected unseen"])},
                 "share of attack packets called an attack")
        body += ["![lodo](figures/e7_lodo.png)", ""]
        v = lt["detected unseen"].dropna()
        if len(v):
            summary.append(f"Unseen device types ({len(v)}): attack packets detected {v.min():.3f}-{v.max():.3f} "
                           f"(median {v.median():.3f}); right class median {lt['right class unseen'].median():.3f}; "
                           f"benign flagged per million median {lt['benign /1M unseen'].median():.1f}.")
    else:
        body += ["_Not run yet (`--parts lodo`)._", ""]

    L += ["## Summary", ""] + ([f"- {s}" for s in summary] or ["- nothing finished yet"]) + [""] + body
    L += ["## Reproduce", "", "```", "python -m gids paper            # every part, resumable",
          "python -m gids paper --parts reference,score", "```", "",
          "Settings: `paper:` in `configs/experiments.yaml`. Models: `models/<experiment>-<hash>/`; test predictions "
          "`test_pred.npy` next to each model.", ""]
    atomic_write_text(out / "PAPER.md", "\n".join(L) + "\n")


def _round(df: pd.DataFrame) -> pd.DataFrame:
    """4 decimals for scores; rates and counts as whole numbers with thousands separators."""
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind != "f":
            continue
        if "/1M" in c or "rows_per_s" in c:
            d[c] = [f"{v:,.0f}" if np.isfinite(v) else "" for v in d[c]]
        else:
            d[c] = d[c].round(1 if c.endswith("_mb") or c.endswith("%") else 4)
    return d


def _safe_scored(ctx: Ctx, cfg, names: List[str], main: Optional[str]) -> pd.DataFrame:
    try:
        return scored(ctx, cfg, names, main)
    except (GracefulStop, KeyboardInterrupt):
        raise
    except Exception as exc:  # noqa: BLE001
        ctx.tracer.exception(STAGE, "scoring_failed", exc, models=names)
        return pd.DataFrame()


def _seed_table(ctx: Ctx, cfg) -> pd.DataFrame:
    cfg2 = with_experiments(cfg, seed_experiments(cfg))
    rows = []
    for base in _p(cfg).get("seed_models", []):
        names = [base] + [n for n in seed_experiments(cfg) if n.startswith(f"{base}_seed")]
        t = _safe_scored(ctx, cfg2, names, None)
        if len(t):
            v = t["macro_f1"].to_numpy()
            rows.append({"model": base, "seeds": len(v), "mean": float(v.mean()),
                         "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan, "min": float(v.min()),
                         "max": float(v.max())})
    return pd.DataFrame(rows)


def _corrected_table(ctx: Ctx, cfg) -> pd.DataFrame:
    from .relabel import corrected_cfg

    names = list(_p(cfg).get("corrected_models", []))
    ccfg = corrected_cfg(cfg)
    if not any(predictions(ctx, ccfg, n, compute=False) is not None for n in names):
        return pd.DataFrame()
    frames = []
    for tag, c in (("original", cfg), ("corrected", ccfg)):
        t = _safe_scored(ctx, c, names, None)
        if len(t):
            t.insert(1, "labels", tag)
            frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _lodo_table(ctx: Ctx, cfg) -> pd.DataFrame:
    from .train import evaluate

    path = paper_dir(cfg) / "lodo_targets.json"
    if not read_json(path):
        return pd.DataFrame()
    targets = lodo_targets(ctx)
    cfg2 = with_experiments(cfg, lodo_experiments(cfg, targets))
    main = _p(cfg).get("lodo_model", "xgb_portable")
    seen_all = predictions(ctx, cfg, main, compute=False)
    rows = []
    for d in targets:
        mdir = model_path(cfg2, lodo_name(main, d))
        if not output_ok(mdir / "heldout_pred.npy") or seen_all is None:
            continue
        cached = read_json(mdir / "heldout_metrics.json")
        if cached and cached.get("main_pred_mtime") == (model_path(cfg, main) / "test_pred.npy").stat().st_mtime:
            rows.append(cached["row"])
            continue
        test, ds = ctx.test(cfg), ctx.ds(cfg)
        k = test["device_type"].astype(str).to_numpy() == d
        y = np.array(ds["label_class"])[test.loc[k, "label"].to_numpy(np.int64)]
        w = test.loc[k, "weight_packets"].to_numpy(np.float64)
        row = {"device_type": d, "attack packets": float(w[y != 0].sum())}
        for tag, pred in (("seen", seen_all[k].astype(np.int64)), ("unseen", np.load(mdir / "heldout_pred.npy"))):
            att, ben = y != 0, y == 0
            row[f"detected {tag}"] = float(w[att & (pred != 0)].sum() / max(w[att].sum(), 1e-12))
            row[f"right class {tag}"] = float(w[att & (pred == y)].sum() / max(w[att].sum(), 1e-12))
            row[f"benign /1M {tag}"] = float(1e6 * w[ben & (pred != 0)].sum() / max(w[ben].sum(), 1e-12))
            row[f"macro_f1 {tag}"] = evaluate(y, pred, w, ds["classes"])["macro_f1"]
        atomic_write_json(mdir / "heldout_metrics.json",
                          {"row": row, "main_pred_mtime": (model_path(cfg, main) / "test_pred.npy").stat().st_mtime})
        rows.append(row)
    return pd.DataFrame(rows)

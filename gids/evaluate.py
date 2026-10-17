"""Step 4: one evaluation on the untouched test split, a harder test, and explanations.

``evaluate`` (resumable: every finished part is saved and skipped on a re-run)

1. trains what is still missing (``xgb_strict`` and the Mirai hold-out model)
   through the normal, checkpointed training;
2. builds the test table once (``data/dataset-<hash>/test.parquet``): every
   test row of every file, with rows stored in two files counted once, exactly
   as for training - nothing is sampled, metrics are on all test packets;
3. scores every model in ``evaluate.models``: per-class precision / recall /
   F1, macro-F1, confusion, benign packets flagged as attacks per million,
   attack packets let through as benign, and the same per device type;
4. unseen-run test: the model trained without the third Mirai run is scored
   on that run's test packets next to the model that saw its first part;
5. explanations for the main model with XGBoost's built-in TreeSHAP
   (``pred_contribs``): which features drive each class, and what pushes the
   scanning packets that are called brute force across the boundary.

``reports/evaluate/EVALUATE.md`` collects everything.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import layout
from .config import get_path
from .explore import md_table
from .ingest import _read_parquet, _to_parquet_bytes
from .prepare import KEYS, _row_hash
from .state import (Deadline, atomic_write_bytes, atomic_write_json, atomic_write_text, crash_point, output_ok,
                    read_json)
from .train import (_prepare_infos, _rows_path, build_dataset, confusion_table, cross_file_merge, evaluate,
                    held_out_mask, load_model, run_train)

STAGE = "evaluate"


def _out(cfg) -> Path:
    return layout.reports(cfg, "evaluate")


# =====================================================================
# test table

def build_test(cfg, tracer, deadline: Deadline) -> pd.DataFrame:
    path = layout.dataset_dir(cfg) / "test.parquet"
    meta_path = layout.dataset_dir(cfg) / "test.json"
    if output_ok(path) and read_json(meta_path):
        return _read_parquet(path)
    pdir, infos, voc = _prepare_infos(cfg)
    feats = voc["features"]
    frames, hs, fs = [], [], []
    with tracer.span(STAGE, "test_table"):
        for fi, info in enumerate(infos):
            deadline.check()
            df = _read_parquet(_rows_path(pdir, info))
            df = df[df["split"] == 2].reset_index(drop=True)
            df["device_type"] = df["device_type"].astype(str)
            hs.append(_row_hash(df[feats].to_numpy(np.float32), df[KEYS].to_numpy(np.int32)))
            fs.append(np.full(len(df), fi, np.int32))
            frames.append(df)
            tracer.progress(STAGE, fi + 1, len(infos), "files (test table)", every_s=30)
    test = pd.concat(frames, ignore_index=True)
    del frames
    keep, wsum, dup_rows, dup_packets, pairs = cross_file_merge(
        np.concatenate(hs), np.concatenate(fs), test["weight"].to_numpy(np.int64), infos,
        bool(get_path(cfg, "dataset.merge_cross_file", True)))
    test["weight_packets"] = wsum
    test = test[keep].reset_index(drop=True)
    atomic_write_bytes(path, _to_parquet_bytes(test))
    atomic_write_json(meta_path, {"rows": int(len(test)), "packets": float(test["weight_packets"].sum()),
                                  "cross_file_duplicate_rows": dup_rows, "cross_file_duplicate_packets": dup_packets,
                                  "cross_file_pairs": pairs})
    tracer.info(STAGE, "test_table", f"{len(test):,} test rows ({test['weight_packets'].sum():,.0f} packets); "
                f"{dup_rows:,} rows stored in two files counted once")
    return test


# =====================================================================
# scoring

def _alarm_stats(y: np.ndarray, pred: np.ndarray, w: np.ndarray, normal: int = 0) -> Dict[str, float]:
    ben = y == normal
    att = ~ben
    return {"benign_flagged_per_million": float(1e6 * w[ben & (pred != normal)].sum() / max(w[ben].sum(), 1)),
            "attacks_missed_pct": float(100 * w[att & (pred == normal)].sum() / max(w[att].sum(), 1))}


def score_model(mdl: Dict[str, Any], test: pd.DataFrame, ds: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    y = mdl["lut"][test["label"].to_numpy(np.int64)]
    w = test["weight_packets"].to_numpy(np.float64)
    pred = mdl["predict"](test).argmax(axis=1)
    m = evaluate(y, pred, w, mdl["classes"])
    m.update(_alarm_stats(y, pred, w))
    if mdl["scheme"] == "fine":
        to_coarse = np.array([ds["label_class"][ds["labels"].index(c)] for c in mdl["classes"]])
        yc, pc = to_coarse[y], to_coarse[pred]
        m["as_coarse"] = evaluate(yc, pc, w, ds["classes"])
        m["as_coarse"].update(_alarm_stats(yc, pc, w))
    else:
        yc, pc = y, pred
    dev = []
    dt = test["device_type"].to_numpy()
    for d in sorted(set(dt)):
        k = dt == d
        md = evaluate(yc[k], pc[k], w[k], ds["classes"])
        a = _alarm_stats(yc[k], pc[k], w[k])
        dev.append({"device_type": d, "packets": float(w[k].sum()),
                    "attack_packets": float(w[k & (yc != 0)].sum()),
                    "macro_f1": md["macro_f1"], "attack_recall": float(
                        w[k & (yc != 0) & (pc == yc)].sum() / max(w[k & (yc != 0)].sum(), 1))
                    if (k & (yc != 0)).any() else np.nan, **a})
    m["per_device_type"] = dev
    m.update({"name": mdl["name"], "model": mdl["model"], "feature_set": mdl["feature_set"],
              "label_scheme": mdl["scheme"], "seconds": round(time.time() - t0, 1)})
    return m


def unseen_run(cfg, test: pd.DataFrame, ds: Dict[str, Any], holdout_name: str, seen_name: str) -> Dict[str, Any]:
    """Recall on the held-out Mirai run's test packets: model that never saw the run vs the one that saw its start."""
    ho = load_model(cfg, holdout_name, ds)
    seen = load_model(cfg, seen_name, ds)
    mask = held_out_mask(test, ds["labels"], ho["holdout"])
    sub = test[mask].reset_index(drop=True)
    rows = []
    preds = {}
    for tag, mdl in (("unseen (trained without this run)", ho), ("seen (trained on its first 60%)", seen)):
        y = mdl["lut"][sub["label"].to_numpy(np.int64)]
        p = mdl["predict"](sub).argmax(axis=1)
        preds[tag] = p
        w = sub["weight_packets"].to_numpy(np.float64)
        for li in sorted(set(sub["label"])):
            k = sub["label"].to_numpy() == li
            rows.append({"model": tag, "raw label": ds["labels"][li], "packets": float(w[k].sum()),
                         "recall (right class)": float(w[k & (p == y)].sum() / max(w[k].sum(), 1)),
                         "called benign %": float(100 * w[k & (p == 0)].sum() / max(w[k].sum(), 1))})
    full = score_model(ho, test, ds)
    return {"holdout": ho["holdout"], "packets": float(sub["weight_packets"].sum()), "rows": rows,
            "holdout_model_test": {k: full[k] for k in ("macro_f1", "balanced_accuracy",
                                                         "benign_flagged_per_million", "attacks_missed_pct")},
            "holdout_model_per_class": full["per_class"]}


# =====================================================================
# explanations (XGBoost TreeSHAP)

def explain(mdl: Dict[str, Any], test: pd.DataFrame, labels: List[str], n_per_class: int,
            seed: int = 0) -> Dict[str, Any]:
    import xgboost as xgb

    if mdl["model"] != "xgboost":
        raise ValueError("explanations need an XGBoost model")
    rng = np.random.default_rng(seed)
    y_all = mdl["lut"][test["label"].to_numpy(np.int64)]
    classes, names = mdl["classes"], mdl["features"]
    # stratified sample: every class equally represented, rows drawn in proportion to the packets they stand for
    idx = []
    w_all = test["weight_packets"].to_numpy(np.float64)
    for c in range(len(classes)):
        k = np.flatnonzero(y_all == c)
        if len(k) <= n_per_class:
            idx.append(k)
        else:
            idx.append(rng.choice(k, size=n_per_class, replace=False, p=w_all[k] / w_all[k].sum()))
    idx = np.concatenate(idx)
    sub = test.iloc[idx].reset_index(drop=True)
    y = y_all[idx]
    from .train import make_matrix

    X = make_matrix(sub, names)
    contrib = mdl["booster"].predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=(0, mdl["best"]))
    contrib = contrib.reshape(len(X), len(classes), len(names) + 1)
    pred = mdl["booster"].inplace_predict(X, iteration_range=(0, mdl["best"])).argmax(axis=1)
    # mean |SHAP| of each feature for the score of the packet's own class
    own = np.abs(contrib[np.arange(len(X)), y, :-1])
    by_class = pd.DataFrame({classes[c]: own[y == c].mean(axis=0) for c in range(len(classes)) if (y == c).any()},
                            index=names)
    by_class["all classes"] = by_class.mean(axis=1)
    by_class = by_class.sort_values("all classes", ascending=False)
    out = {"rows": int(len(X)), "mean_abs_shap": by_class.round(4).reset_index().rename(
        columns={"index": "feature"}).to_dict(orient="records")}
    # the brute-force / scanning boundary
    if "Brute Force" in classes and "Network Scanning" in classes:
        bf, sc = classes.index("Brute Force"), classes.index("Network Scanning")
        groups = {"scanning -> scanning": (y == sc) & (pred == sc), "scanning -> brute force": (y == sc) & (pred == bf),
                  "brute force -> brute force": (y == bf) & (pred == bf),
                  "brute force -> scanning": (y == bf) & (pred == sc)}
        med = pd.DataFrame({g: pd.DataFrame(X[k], columns=names).median() for g, k in groups.items() if k.any()})
        push = contrib[:, bf, :-1] - contrib[:, sc, :-1]  # > 0 pushes towards brute force
        a, b = groups["scanning -> brute force"], groups["scanning -> scanning"]
        diff = pd.Series(push[a].mean(axis=0) - push[b].mean(axis=0), index=names) if a.any() and b.any() \
            else pd.Series(dtype=float)
        raw = sub["label"].to_numpy()
        out["boundary"] = {
            "counts": {g: int(k.sum()) for g, k in groups.items()},
            "raw_labels_of_confused": {lab: int(v) for lab, v in pd.Series(
                [labels[int(x)] for x in raw[a]]).value_counts().items()},
            "medians": med.round(4).reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
            "push_towards_brute_force": diff.sort_values(ascending=False).round(4).to_dict()}
    return out


# =====================================================================
# command

def run_evaluate(cfg, tracer, deadline: Deadline) -> Dict[str, Any]:
    ev = cfg.get("evaluate", {})
    models: List[str] = list(ev.get("models", []))
    main, holdout, seen = ev.get("main"), ev.get("holdout_experiment"), ev.get("holdout_compare")
    needed = list(dict.fromkeys(models + [x for x in (holdout, seen) if x]))
    run_train(cfg, tracer, deadline, only=needed)
    ds = build_dataset(cfg, tracer, deadline)
    test = build_test(cfg, tracer, deadline)
    rep = _out(cfg)
    results: Dict[str, Any] = {}
    for n in models:
        deadline.check()
        f = rep / f"test_{n}.json"
        res = read_json(f)
        if res is None:
            try:
                mdl = load_model(cfg, n, ds)
            except RuntimeError as exc:
                tracer.warn(STAGE, "model_missing", str(exc))
                continue
            with tracer.span(STAGE, f"score.{n}"):
                res = score_model(mdl, test, ds)
            res["model_dir"] = str(mdl["dir"])
            atomic_write_json(f, res)
            crash_point(STAGE)
            tracer.info(STAGE, "test", f"{n}: test macro-F1 {res['macro_f1']:.4f}, benign flagged "
                        f"{res['benign_flagged_per_million']:.1f} per million, attacks missed "
                        f"{res['attacks_missed_pct']:.3f}%")
        results[n] = res
    unseen = read_json(rep / "unseen_run.json")
    if unseen is None and holdout and seen:
        deadline.check()
        try:
            with tracer.span(STAGE, "unseen_run"):
                unseen = unseen_run(cfg, test, ds, holdout, seen)
            atomic_write_json(rep / "unseen_run.json", unseen)
        except RuntimeError as exc:
            tracer.warn(STAGE, "model_missing", str(exc))
    expl = read_json(rep / "explain.json")
    if expl is None and main:
        deadline.check()
        try:
            with tracer.span(STAGE, "explain"):
                expl = explain(load_model(cfg, main, ds), test, ds["labels"], int(ev.get("shap_rows_per_class", 2000)))
            atomic_write_json(rep / "explain.json", expl)
        except (RuntimeError, ValueError) as exc:
            tracer.warn(STAGE, "explain_skipped", str(exc))
    tmeta = read_json(layout.dataset_dir(cfg) / "test.json") or {}
    write_report(cfg, ds, tmeta, results, unseen, expl, main)
    return results


def _val_metrics(cfg, name: str) -> Optional[Dict[str, Any]]:
    return read_json(layout.model_dir(cfg, name) / "metrics.json")


def write_report(cfg, ds, tmeta, results, unseen, expl, main) -> None:
    rep = _out(cfg)
    L = ["# Gotham dataset - step 4: test results and explanations", "",
         f"Test split: **{tmeta.get('rows', 0):,} rows = {tmeta.get('packets', 0):,.0f} packets**, every test "
         f"packet of every file (nothing sampled); {tmeta.get('cross_file_duplicate_rows', 0):,} rows "
         f"({tmeta.get('cross_file_duplicate_packets', 0):,} packets) stored in two files are counted once. "
         "All numbers are on packets. The test split was not used for training, early stopping or any choice.", ""]
    rows = []
    for n, m in results.items():
        mc = m.get("as_coarse", m)
        v = _val_metrics(cfg, n) or {}
        r = {"model": n, "features": m["feature_set"], "labels": m["label_scheme"],
             "test macro_f1": m["macro_f1"], "val macro_f1": v.get("macro_f1"),
             "test balanced_acc": m["balanced_accuracy"],
             "benign flagged /1M": mc["benign_flagged_per_million"], "attacks missed %": mc["attacks_missed_pct"]}
        r.update({f"F1 {p['class']}": p["f1"] for p in mc["per_class"]})
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(rep / "test_results.csv", index=False)
    L += ["## 1. Test results", "",
          "`benign flagged /1M`: benign packets called an attack per million benign packets. `attacks missed %`: "
          "attack packets called benign. For the fine model the F1 columns are the six classes read from its "
          "17-label predictions; its own 17-label macro-F1 is in `test macro_f1`.", "",
          md_table(table.round(4), max_rows=20), ""]
    if main in results:
        m = results[main]
        L += [f"## 2. Main model `{main}` in detail", "", md_table(pd.DataFrame(m["per_class"]).round(4)), "",
              "Confusion (rows: true class, % of its test packets):", "",
              md_table(confusion_table(m).round(3), max_rows=20), "",
              "Per device type (classes present on that device type; attacks hit only some device types):", "",
              md_table(pd.DataFrame(m["per_device_type"]).round(4), max_rows=20), ""]
        pd.DataFrame(m["per_device_type"]).to_csv(rep / f"per_device_type_{main}.csv", index=False)
    fine = [m for m in results.values() if m["label_scheme"] == "fine"]
    for m in fine:
        L += [f"## 3. 17 raw labels (`{m['name']}`)", "", md_table(pd.DataFrame(m["per_class"]).round(4),
                                                                   max_rows=30), ""]
    if unseen:
        L += ["## 4. Unseen attack run", "",
              f"The run `{unseen['holdout']['label_regex']}` episode {unseen['holdout']['episode']} (episodes count "
              "from 0, so episode 2 of Mirai is its third run) was left out of training and early stopping. Its "
              f"{unseen['packets']:,.0f} test packets are scored by that model and by the model that was trained on "
              "the first 60% of the same run. Recall = share of the run's packets given the right class.", "",
              md_table(pd.DataFrame(unseen["rows"]).round(4), max_rows=20), "",
              "The hold-out model on the whole test split: " + ", ".join(
                  f"{k} {v:.4f}" for k, v in unseen["holdout_model_test"].items()), ""]
    if expl:
        L += ["## 5. What the main model looks at (TreeSHAP)", "",
              f"Mean |SHAP| of each feature for the score of the packet's own class, on {expl['rows']:,} test packets "
              "sampled equally per class.", "", md_table(pd.DataFrame(expl["mean_abs_shap"]), max_rows=30), ""]
        b = expl.get("boundary")
        if b:
            L += ["### Brute force vs scanning", "",
                  "Sampled packets per outcome: " + ", ".join(f"{k}: {v}" for k, v in b["counts"].items()), "",
                  "Raw labels of the scanning packets called brute force: " + (", ".join(
                      f"{k} ({v})" for k, v in b["raw_labels_of_confused"].items()) or "none"), "",
                  "Median feature values per outcome:", "", md_table(pd.DataFrame(b["medians"]), max_rows=30), "",
                  "Features that push the misclassified scanning packets towards brute force more than the "
                  "correctly classified ones (difference in mean SHAP of brute force minus scanning):", "",
                  "| feature | push |", "|---|---|"] + [f"| {k} | {v:.4f} |" for k, v in
                                                       list(b["push_towards_brute_force"].items())[:12]] + [""]
    checks = []
    for n, m in results.items():
        v = _val_metrics(cfg, n)
        if v and m["macro_f1"] < v["macro_f1"] - 0.02:
            checks.append(f"{n}: test macro-F1 {m['macro_f1']:.4f} is clearly below validation "
                          f"{v['macro_f1']:.4f} - the later part of each attack differs from the earlier one")
        mc = m.get("as_coarse", m)
        if mc["benign_flagged_per_million"] > 100:
            checks.append(f"{n}: {mc['benign_flagged_per_million']:.0f} benign packets per million flagged")
    if unseen:
        low = [r for r in unseen["rows"] if r["model"].startswith("unseen") and r["recall (right class)"] < 0.9]
        for r in low:
            checks.append(f"unseen Mirai run: {r['raw label']} recall {r['recall (right class)']:.3f}")
    L += ["## 6. Things to check", ""] + ([f"- {c}" for c in checks] or ["- nothing flagged"])
    atomic_write_text(rep / "EVALUATE.md", "\n".join(L) + "\n")

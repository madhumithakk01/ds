"""Step 5: one results document and figures from everything the earlier steps saved.

``results`` re-runs nothing: it reads the summaries, metrics and evaluation
files on Drive and writes ``reports/results/RESULTS.md`` plus PNG figures in
``reports/results/figures/``.  It also computes one extra view exactly from
the saved test confusion matrices (which are in packets): Brute Force and
Network Scanning merged into one *telnet intrusion* class, because step 4
showed that the scanning packets called brute force are telnet login traffic.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import layout
from .config import experiment_config, get_path
from .explore import md_table
from .state import atomic_write_json, atomic_write_text, ensure_dir, read_json
from .trace import list_manifests

STAGE = "results"
MERGE = {"Telnet intrusion (scan + brute force)": ["Brute Force", "Network Scanning"]}

# reference palette (light surface): categorical slots 1-2, text tokens, sequential blue ramp
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]


# =====================================================================
# metrics from a confusion matrix

def metrics_from_confusion(cm: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    tp = np.diag(cm)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(support > 0, tp / support, np.nan)
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    f1 = np.where(support > 0, f1, np.nan)
    per = pd.DataFrame({"class": classes, "precision": prec, "recall": rec, "f1": f1, "packets": support})
    ben = classes.index("Normal") if "Normal" in classes else 0
    att = [i for i in range(len(classes)) if i != ben]
    return {"macro_f1": float(np.nanmean(f1)), "balanced_accuracy": float(np.nanmean(rec[support > 0])),
            "benign_flagged_per_million": float(1e6 * (cm[ben].sum() - cm[ben, ben]) / max(cm[ben].sum(), 1)),
            "attacks_missed_pct": float(100 * cm[att, ben].sum() / max(cm[att].sum(), 1)),
            "per_class": per.to_dict(orient="records")}


def merge_classes(cm: np.ndarray, classes: List[str], groups: Dict[str, List[str]]):
    """Confusion matrix with each group of classes merged into one class (rows and columns summed)."""
    target = []
    names: List[str] = []
    for c in classes:
        g = next((k for k, v in groups.items() if c in v), c)
        if g not in names:
            names.append(g)
        target.append(names.index(g))
    a = np.zeros((len(names), len(classes)))
    a[target, np.arange(len(classes))] = 1.0
    return a @ cm @ a.T, names


# =====================================================================
# figures

def _style(ax, grid_axis="x") -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w, h):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax, plt


def _save(fig, plt, path: Path, title: str, ax) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_dots(path: Path, title: str, labels: List[str], series: Dict[str, List[Optional[float]]],
             xlabel: str, top: float = 1.0) -> None:
    """Horizontal dot plot: one row per label, one marker shape + colour per series, values written beside.

    The axis starts just below the lowest value (dots need no zero baseline); the label says where it starts.
    """
    keep = [i for i in range(len(labels)) if any(v[i] is not None and v[i] == v[i] for v in series.values())]
    labels = [labels[i] for i in keep]
    series = {k: [v[i] for i in keep] for k, v in series.items()}
    vals_all = [x for v in series.values() for x in v if x is not None and x == x]
    lo = max(0.0, np.floor((min(vals_all) - 0.01) * 50) / 50) if vals_all else 0.0
    xlim = (lo, top + (top - lo) * 0.14)
    xlabel = f"{xlabel} (axis starts at {lo:g})" if lo > 0 else xlabel
    fig, ax, plt = _fig(7.2, 0.5 * len(labels) + 1.4)
    _style(ax)
    y = np.arange(len(labels))[::-1]
    markers = ["o", "D"]
    for i, (name, vals) in enumerate(series.items()):
        v = np.array([np.nan if x is None else x for x in vals], dtype=float)
        ax.scatter(v, y + (0.12 if len(series) > 1 and i == 0 else -0.12 if len(series) > 1 else 0), s=64,
                   marker=markers[i], color=SERIES[i], edgecolor=SURFACE, linewidth=2, zorder=3, label=name)
        for xv, yv in zip(v, y):
            if np.isfinite(xv):
                ax.annotate(f"{xv:.4f}", (xv, yv + (0.12 if len(series) > 1 and i == 0 else
                                                   -0.12 if len(series) > 1 else 0)),
                            xytext=(7, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, labels, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_xticks([t for t in ax.get_xticks() if lo <= t <= top + 1e-9])
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                  handletextpad=0.3, borderaxespad=0.1)
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=30)
        fig.tight_layout()
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        return
    _save(fig, plt, path, title, ax)


def fig_bars(path: Path, title: str, labels: List[str], values: List[float], xlabel: str) -> None:
    """Horizontal bars from zero, one colour, value labels."""
    fig, ax, plt = _fig(7.2, 0.36 * len(labels) + 1.2)
    _style(ax)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for v, yy in zip(values, y):
        ax.annotate(f"{v:.3f}", (v, yy), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK2)
    ax.set_yticks(y, labels, color=INK)
    ax.set_xlim(0, max(values) * 1.15 if values else 1)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    _save(fig, plt, path, title, ax)


def fig_confusion(path: Path, title: str, cm: np.ndarray, classes: List[str]) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    pct = 100 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
    fig, ax, plt = _fig(7.2, 6.0)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("blues", [SURFACE] + BLUES)
    # colour by sqrt(%) so that the small off-diagonal shares stay visible next to ~100 % diagonals
    ax.imshow(np.sqrt(pct), cmap=cmap, vmin=0, vmax=10, aspect="auto")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = pct[i, j]
            txt = "0" if v == 0 else ("<0.01" if v < 0.01 else f"{v:.2f}" if v < 10 else f"{v:.1f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color=SURFACE if np.sqrt(v) > 6 else INK)
    short = [c.replace(" Communication", "").replace("Network ", "") for c in classes]
    ax.set_xticks(range(len(classes)), short, rotation=30, ha="right", color=INK, fontsize=9)
    ax.set_yticks(range(len(classes)), short, color=INK, fontsize=9)
    ax.set_xlabel("predicted class", color=INK2, fontsize=9)
    ax.set_ylabel("true class (% of its test packets)", color=INK2, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(classes)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(classes)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    _save(fig, plt, path, title, ax)


# =====================================================================
# collecting what the earlier steps saved

def _latest_runs(work: Path) -> pd.DataFrame:
    rows = {}
    for m in list_manifests(work):
        if m.get("status") == "ok" and m.get("command") in ("explore", "ingest", "prepare", "train", "evaluate"):
            code = (m.get("env") or {}).get("code") or {}
            rows[m["command"]] = {"stage": m["command"], "run": m["run_id"], "finished": m.get("finished", ""),
                                  "code commit": (code.get("commit") or "?") + (" (+ local changes)"
                                                                               if code.get("dirty") else ""),
                                  "minutes": round(m.get("elapsed_s", 0) / 60, 1)}
    order = ["explore", "ingest", "prepare", "train", "evaluate"]
    return pd.DataFrame([rows[k] for k in order if k in rows])


def run_results(cfg, tracer) -> Dict[str, Any]:
    work = layout.work(cfg)
    rep = layout.reports(cfg, "results")
    figs = ensure_dir(rep / "figures")
    ex = read_json(layout.reports(cfg, "explore") / "summary.json") or {}
    ing = read_json(layout.reports(cfg, "ingest") / "summary.json") or {}
    prep = read_json(layout.reports(cfg, "prepare") / "summary.json") or {}
    ds = read_json(layout.dataset_dir(cfg) / "dataset.json") or {}
    tmeta = read_json(layout.dataset_dir(cfg) / "test.json") or {}
    evdir = layout.reports(cfg, "evaluate")
    ev = cfg.get("evaluate", {})
    models = [n for n in ev.get("models", []) if (evdir / f"test_{n}.json").exists()]
    if not models:
        raise RuntimeError("No test results yet. Run `python -m gids evaluate` first.")
    tests = {n: read_json(evdir / f"test_{n}.json") for n in models}
    main = ev.get("main") if ev.get("main") in tests else models[0]
    unseen = read_json(evdir / "unseen_run.json")
    expl = read_json(evdir / "explain.json")

    # ---- tables
    rows, merged_rows = [], []
    for n, m in tests.items():
        mc = m.get("as_coarse", m)
        val = read_json(layout.model_dir(cfg, n) / "metrics.json") or {}
        cm = np.array(mc["confusion"])
        mm = metrics_from_confusion(*merge_classes(cm, mc["classes"], MERGE))
        rows.append({"model": n, "features": m["feature_set"], "labels": m["label_scheme"],
                     "val macro-F1": val.get("macro_f1"), "test macro-F1": m["macro_f1"],
                     "test macro-F1, 6 classes": mc["macro_f1"], "telnet merged macro-F1": mm["macro_f1"],
                     "benign flagged /1M": mc["benign_flagged_per_million"],
                     "attacks missed %": mc["attacks_missed_pct"]})
        merged_rows.append({"model": n, **{p["class"]: p["f1"] for p in mm["per_class"]}})
    results = pd.DataFrame(rows)
    merged = pd.DataFrame(merged_rows)
    results.to_csv(rep / "test_results.csv", index=False)
    merged.to_csv(rep / "test_telnet_merged_f1.csv", index=False)

    rel = []
    for fs in ("full", "no_ttl", "portable", "strict"):
        name = next((n for n in (get_path(cfg, "experiments") or {}) if
                     experiment_config(cfg, n).get("model") == "xgboost" and
                     experiment_config(cfg, n).get("feature_set") == fs and
                     experiment_config(cfg, n).get("label_scheme", "coarse") == "coarse" and
                     not experiment_config(cfg, n).get("holdout")), None)
        if not name:
            continue
        val = read_json(layout.model_dir(cfg, name) / "metrics.json") or {}
        t = tests.get(name)
        rel.append({"feature set": fs, "experiment": name, "features": len(get_path(cfg, f"features.sets.{fs}")),
                    "val macro-F1": val.get("macro_f1"), "test macro-F1": t["macro_f1"] if t else None,
                    "test F1 Brute Force": next((p["f1"] for p in t["per_class"] if p["class"] == "Brute Force"),
                                                None) if t else None,
                    "test F1 Network Scanning": next((p["f1"] for p in t["per_class"]
                                                      if p["class"] == "Network Scanning"), None) if t else None})
    rel = pd.DataFrame(rel)

    # ---- figures
    names = list(tests)
    fig_dots(figs / "model_comparison.png", "Test macro-F1 per model", names,
             {"6 classes": [tests[n].get("as_coarse", tests[n])["macro_f1"] for n in names],
              "scan + brute force merged": [r["telnet merged macro-F1"] for r in rows]},
             "macro-F1 on all test packets")
    mt = tests[main]
    fig_confusion(figs / f"confusion_{main}.png", f"Confusion on the test split - {main}",
                  np.array(mt["confusion"]), mt["classes"])
    fig_dots(figs / f"per_class_f1_{main}.png", f"Test F1 per class - {main}",
             [p["class"] for p in mt["per_class"]], {"F1": [p["f1"] for p in mt["per_class"]]},
             "F1 on test packets")
    if len(rel):
        fig_dots(figs / "feature_set_reliance.png", "Macro-F1 as testbed-specific fields are removed (XGBoost)",
                 [f"{r['feature set']} ({r['features']})" for r in rel.to_dict(orient="records")],
                 {"validation": list(rel["val macro-F1"]), "test": [None if pd.isna(x) else x
                                                                    for x in rel["test macro-F1"]]},
                 "macro-F1, 6 classes")
    shap = pd.DataFrame(expl["mean_abs_shap"]) if expl else pd.DataFrame()
    if len(shap):
        top = shap.sort_values("all classes", ascending=False).head(12)
        fig_bars(figs / f"shap_{main}.png", f"Mean |SHAP| per feature, averaged over classes - {main}",
                 list(top["feature"]), list(top["all classes"]), "mean |SHAP| (log-odds of the packet's own class)")

    # ---- document
    doc = results_markdown(cfg, ex, ing, prep, ds, tmeta, tests, main, results, merged, rel, unseen, expl,
                           _latest_runs(work))
    atomic_write_text(rep / "RESULTS.md", doc)
    summary = {"main": main, "models": names, "results": rows, "figures": sorted(p.name for p in figs.glob("*.png"))}
    atomic_write_json(rep / "results.json", summary)
    tracer.info(STAGE, "done", f"{rep / 'RESULTS.md'} and {len(summary['figures'])} figures")
    return summary


def _pct(x) -> str:
    return f"{100 * x:.2f}%"


def results_markdown(cfg, ex, ing, prep, ds, tmeta, tests, main, results, merged, rel, unseen, expl, runs) -> str:
    mt = tests[main]
    mc = mt.get("as_coarse", mt)
    pc = {p["class"]: p for p in mt["per_class"]}
    mm = metrics_from_confusion(*merge_classes(np.array(mc["confusion"]), mc["classes"], MERGE))
    cm = np.array(mt["confusion"])
    k = mt["classes"]
    best = results.sort_values("test macro-F1, 6 classes", ascending=False).iloc[0]
    L = ["# Gotham IoT intrusion detection - results", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `python -m gids results` from the "
         "outputs saved by steps 0-4; nothing was re-run. Figures are in `figures/`._", ""]

    L += ["## 1. Summary", ""]
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(np.argmax(off), off.shape)
    err_share = 100 * off[i, j] / max(off.sum(), 1)
    strong = [c for c, p in pc.items() if p["f1"] is not None and p["f1"] == p["f1"] and p["f1"] >= 0.99]
    L += [f"- **Data:** {ex.get('rows', 0):,} labelled packets from {ex.get('files_scanned', 0)} device captures "
          f"({ex.get('devices', '?')} emulated IoT devices, {ex.get('device_types', '?')} device types) of the Gotham "
          f"Dataset 2025; {len(mc['classes'])} classes after mapping {len(ex.get('labels', {}))} raw labels.",
          f"- **Main model** (`{main}`, {mt['model']} on {len(get_path(cfg, 'features.sets.' + mt['feature_set']))} "
          f"features, set `{mt['feature_set']}`): test macro-F1 **{mc['macro_f1']:.4f}**; benign packets flagged as "
          f"attacks: **{mc['benign_flagged_per_million']:.1f} per million**; attack packets let through as benign: "
          f"**{mc['attacks_missed_pct']:.4f}%**.",
          f"- **Where the errors are:** classes with test F1 >= 0.99: {', '.join(strong) or 'none'}. The largest "
          f"error is {k[i]} predicted as {k[j]}: {100 * off[i, j] / max(cm[i].sum(), 1):.2f}% of {k[i]} packets, "
          f"{err_share:.1f}% of all misclassified test packets."]
    if {"Brute Force", "Network Scanning"} <= set(mc["classes"]):
        L += [f"- With Brute Force and Network Scanning merged into one telnet-intrusion class the main model's "
              f"macro-F1 is **{mm['macro_f1']:.4f}** (Section 4.3)."]
    L += [f"- **Highest test macro-F1 (six classes):** `{best['model']}` ({best['features']} features), "
          f"{best['test macro-F1, 6 classes']:.4f}."]
    tested = rel.dropna(subset=["test macro-F1"]) if len(rel) else rel
    if len(tested) >= 2:
        first, last = tested.iloc[0], tested.iloc[-1]
        t0, t1 = tests[first["experiment"]], tests[last["experiment"]]
        drops = sorted(((p0["f1"] - p1["f1"], p0["class"]) for p0, p1 in zip(t0["per_class"], t1["per_class"])
                        if p0["f1"] == p0["f1"] and p1["f1"] == p1["f1"]), reverse=True)
        L += [f"- **Reliance on testbed-specific fields:** from feature set `{first['feature set']}` to "
              f"`{last['feature set']}` the test macro-F1 changes from {first['test macro-F1']:.4f} to "
              f"{last['test macro-F1']:.4f}; the largest per-class F1 drops are "
              + ", ".join(f"{c} ({-d:+.4f})" for d, c in drops[:2]) + "."]
    if unseen and unseen["rows"]:
        rr = [r for r in unseen["rows"] if r["model"].startswith("unseen")]
        h = unseen["holdout"]
        L += [f"- **Unseen attack run:** a model trained without the packets of `{h['label_regex']}` episode "
              f"{h['episode']} gives them the right class with recall between "
              f"{min(r['recall (right class)'] for r in rr):.4f} and {max(r['recall (right class)'] for r in rr):.4f}."]
    L += [""]

    L += ["## 2. Data", "",
          f"- {ex.get('files_scanned', 0)} processed CSV files ({ex.get('csv_gb', 0):.2f} GB of text) read straight "
          f"from the dataset zip; {ex.get('bad_lines', 0):,} malformed lines; every timestamp parsed.",
          f"- Step 1 turned {ing.get('rows', 0):,} rows into typed features; row counts equal step 0: "
          f"{'yes' if ing.get('rows_match_explore') else 'no'}.",
          "- The device IP is found per capture segment, because a file can hold several devices' captures "
          "joined end to end; rows stored in two files are counted once: "
          f"{ds.get('cross_file_duplicate_rows', 0):,} training/validation rows and "
          f"{tmeta.get('cross_file_duplicate_rows', 0):,} test rows"
          + (" (largest overlaps: " + "; ".join(f"{k} ({v:,})" for k, v in list(ds["cross_file_pairs"].items())[:4])
             + ")" if ds.get("cross_file_pairs") else "") + ".",
          f"- The raw label `Unknown` is left out; packets kept: {prep.get('rows_kept', 0):,}.", "",
          "Packets per class and split (after the split gaps; before merging the copies stored in two files):", "",
          md_table(pd.DataFrame(prep.get("rows_by_class", [])), max_rows=10), "",
          "**Split.** Attacks occur in short windows (episodes) while benign traffic runs throughout, so the "
          "split is made per raw label in time order: each episode is cut into blocks of at most "
          f"{get_path(cfg, 'split.block_h', 1)} h; in every block the first 60% of packets are training, the next 10% "
          "validation, the last 30% test, with a short gap left out between them. A random split would put "
          "near-identical packets of one flood into both training and test.", ""]

    sets = get_path(cfg, "features.sets") or {}
    L += ["## 3. Method", "",
          "- **Unit:** one prediction per packet.",
          "- **Features:** header fields (sizes, flags, protocol, TCP window / options), direction relative to the "
          "device, and context features computed only from *earlier* packets of the same capture segment: packets "
          "and bytes per source, destination, IP pair and 5-tuple flow in the last 1 s / 10 s, inter-arrival times, "
          "and new destination ports per source. Addresses, device and time are not features.",
          "- **Feature sets:** " + "; ".join(f"`{k}` ({len(v)} features)" for k, v in sets.items()) +
          ". `portable` drops TTL and replaces port numbers by their range and standard service; `strict` also "
          "drops the TCP window and options fields, which partly fingerprint the sender's operating system.",
          "- **Training:** XGBoost, LightGBM and random forest; DoS and Normal sampled to "
          f"{ds.get('spec', {}).get('max_rows_per_class', 0):,} unique rows; every class has the same total weight; "
          "early stopping on class-balanced validation loss.",
          "- **Evaluation:** all metrics are on packets; the test split was used once, after every choice was made "
          "on validation.", ""]

    L += ["## 4. Results", "", "### 4.1 Test results", "", "![model comparison](figures/model_comparison.png)", "",
          md_table(results.round(4), max_rows=20), "",
          "`test macro-F1` is over each model's own classes (17 for the fine model); `6 classes` reads the fine "
          "model's predictions as the six classes.", ""]
    L += [f"### 4.2 Main model `{main}`", "", f"![confusion](figures/confusion_{main}.png)", "",
          f"![per-class F1](figures/per_class_f1_{main}.png)", "",
          md_table(pd.DataFrame(mt["per_class"]).round(4)), ""]
    L += ["### 4.3 Brute force vs scanning", ""]
    if expl and expl.get("boundary"):
        b = expl["boundary"]
        med = pd.DataFrame(b["medians"]).set_index("feature")
        cols = [c for c in ("scanning -> scanning", "scanning -> brute force", "brute force -> brute force")
                if c in med]
        closer = []
        if len(cols) == 3:
            for f in med.index:
                a, x, c = med.loc[f, cols[0]], med.loc[f, cols[1]], med.loc[f, cols[2]]
                if pd.notna(a) and pd.notna(x) and pd.notna(c) and a != c and abs(x - c) < abs(x - a):
                    closer.append(f)
        show = [f for f in ("tcp_flags", "tcp_window_size_value", "ip_flags", "frame_len", "sport_service",
                            "dport_service", "dir_out", "ctx_pair_iat", "ctx_flow_iat") if f in med.index]
        L += [f"In a sample of test packets ({', '.join(f'{k}: {v}' for k, v in b['counts'].items())}), the scanning "
              "packets called brute force have raw label(s) " + ", ".join(b["raw_labels_of_confused"]) + ". Their "
              f"median values are closer to brute force than to the other scanning packets for {len(closer)} of "
              f"{len(med)} features ({', '.join(closer[:10])}{'...' if len(closer) > 10 else ''}).", "",
              md_table(med.loc[show, cols].reset_index(), max_rows=12), ""]
        conf = med[cols[1]] if len(cols) == 3 else pd.Series(dtype=float)
        if len(conf) and conf.get("tcp_flags") == 24 and 2 in (conf.get("sport_service"), conf.get("dport_service")):
            L += ["Reading: these are telnet session packets (data-carrying PSH/ACK on port 23 with the operating "
              "system's normal TCP behaviour) recorded while the scan phase was running. Mirai's scanner itself "
              "tries telnet logins on the hosts that answer, so per packet the two labels overlap; telling them "
              "apart needs the attack phase, not the packet.", ""]
    L += ["F1 with the two classes merged (computed exactly from the test confusion matrices):", "",
          md_table(merged.round(4), max_rows=10), ""]
    if len(rel):
        L += ["### 4.4 Reliance on testbed-specific fields", "", "![reliance](figures/feature_set_reliance.png)", "",
              md_table(rel.round(4), max_rows=10), ""]
    if unseen:
        L += ["### 4.5 Unseen attack run", "",
              f"The model `{cfg.get('evaluate', {}).get('holdout_experiment')}` was trained without the packets of "
              f"`{unseen['holdout']['label_regex']}` episode {unseen['holdout']['episode']} (episodes count from 0); "
              "the table compares it on that episode's test packets with the main model, which saw the episode's "
              "first 60%.", "", md_table(pd.DataFrame(unseen["rows"]).round(4), max_rows=12), "",
              "A held-out episode is another run of an attack the model knows, on the same testbed: the result "
              "shows robustness to a new run, not to new attack tools or another network.", ""]
    dev = pd.DataFrame(mt.get("per_device_type", []))
    if len(dev):
        L += ["### 4.6 Per device type", "", md_table(dev.round(4), max_rows=20), ""]
    fine = [m for m in tests.values() if m["label_scheme"] == "fine"]
    for m in fine:
        L += [f"### 4.7 Seventeen raw labels (`{m['name']}`)", "",
              md_table(pd.DataFrame(m["per_class"]).round(4), max_rows=30), ""]
    if expl:
        L += ["### 4.8 What the main model uses (TreeSHAP)", "", f"![shap](figures/shap_{main}.png)", "",
              md_table(pd.DataFrame(expl["mean_abs_shap"]).head(15), max_rows=15), ""]

    L += ["## 5. Limitations", "",
          "- One emulated testbed: rates, timings and TCP behaviour come from its machines and links; scores on "
          "another network are not measured here.",
          "- Validation and test come from later parts of the same attack episodes, so they measure recognition "
          "of an ongoing attack; only attacks with several episodes allow a test on an unseen run.",
          "- Per-packet labels follow the attack phase, which makes scanning and brute force overlap (Section 4.3).",
          "- Small classes move with a handful of packets: " + ", ".join(
              f"{c} ({n:,.0f} test packets)" for c, n in sorted(
                  {p["class"]: p["packets"] for m in tests.values() for p in m["per_class"]
                   if p["packets"] and p["packets"] < 1000}.items(), key=lambda kv: kv[1])[:4]) + ".",
          "- Context features need the packets of the last 10 s of the same capture; a live system must keep that "
          "state per device.", ""]
    L += ["## 6. Reproducibility", "",
          "Every stage runs from the repository with `python -m gids <stage>` (see README). Latest successful run "
          "of each stage, with the code commit it ran:", "", md_table(runs, max_rows=10), "",
          "Settings: `configs/default.yaml` and `configs/experiments.yaml`; every run's full configuration and "
          "environment is in `runs/<run>/manifest.json` on Drive.", ""]
    return "\n".join(L)

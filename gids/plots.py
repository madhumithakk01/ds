"""Static charts written as PNG next to the CSV tables they visualise."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def bar_log(counts: pd.Series, title: str, path, xlabel: str = "rows (log scale)") -> None:
    counts = counts.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.38 * len(counts) + 1)))
    ax.barh([str(i) for i in counts.index], counts.values, color="#4C78A8")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {int(v):,}", va="center", fontsize=8)
    _save(fig, path)


def heatmap(df: pd.DataFrame, title: str, path, log: bool = True, fmt: str = "{:,.0f}",
            cmap: str = "viridis", vmax: Optional[float] = None) -> None:
    data = df.to_numpy(dtype=float)
    shown = np.log10(data + 1) if log else data
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * df.shape[1], 1.2 + 0.4 * df.shape[0]))
    im = ax.imshow(shown, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels([str(i) for i in df.index], fontsize=8)
    if df.size <= 400:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = data[i, j]
                if v:
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6,
                            color="white" if shown[i, j] < (np.nanmax(shown) * 0.6) else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, label="log10(count+1)" if log else "")
    _save(fig, path)


def confusion(cm: np.ndarray, classes: Sequence[str], title: str, path, normalize: bool = True) -> None:
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12) if normalize else cm
    fig, ax = plt.subplots(figsize=(1.6 + 0.8 * len(classes), 1.4 + 0.7 * len(classes)))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j]):,}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if shown[i, j] > 0.6 * np.nanmax(shown) else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    _save(fig, path)


def hbar(values: pd.Series, title: str, path, top: int = 25, xlabel: str = "") -> None:
    v = values.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.3 * len(v) + 1)))
    ax.barh([str(i) for i in v.index], v.values, color="#F58518")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    _save(fig, path)


def grouped_hbar(df: pd.DataFrame, title: str, path, top: int = 20) -> None:
    """Rows = features, columns = classes (e.g. mean |SHAP| per class)."""
    order = df.sum(axis=1).sort_values(ascending=False).head(top).index[::-1]
    d = df.loc[order]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(d) + 1)))
    left = np.zeros(len(d))
    cmap = plt.get_cmap("tab10")
    for k, c in enumerate(d.columns):
        ax.barh([str(i) for i in d.index], d[c].values, left=left, label=str(c), color=cmap(k % 10))
        left += d[c].values
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title)
    _save(fig, path)


def curves(hist: pd.DataFrame, path, title: str) -> None:
    metrics = [c for c in hist.columns if c != "round"]
    groups: Dict[str, List[str]] = {}
    for m in metrics:
        groups.setdefault(m.split("/", 1)[-1], []).append(m)
    n = max(1, len(groups))
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.2))
    axes = np.atleast_1d(axes)
    for ax, (name, cols) in zip(axes, groups.items()):
        for c in cols:
            ax.plot(hist["round"], hist[c], label=c.split("/", 1)[0])
        ax.set_title(name)
        ax.set_xlabel("boosting round")
        ax.legend(fontsize=7)
    fig.suptitle(title)
    _save(fig, path)


def timeline(df: pd.DataFrame, path, title: str) -> None:
    """df columns: row_label, start, end (hours), color_key."""
    if df.empty:
        return
    keys = sorted(df["color_key"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {k: cmap(i % 10) for i, k in enumerate(keys)}
    rows = list(dict.fromkeys(df["row_label"]))
    fig, ax = plt.subplots(figsize=(10, max(3, 0.28 * len(rows) + 1.2)))
    for _, r in df.iterrows():
        y = rows.index(r["row_label"])
        ax.barh(y, max(r["end"] - r["start"], 0.01), left=r["start"], color=colors[r["color_key"]],
                height=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_xlabel("hours since first packet")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in keys]
    ax.legend(handles, keys, fontsize=7, loc="upper right")
    ax.set_title(title)
    _save(fig, path)


def boxplots(sample: pd.DataFrame, features: List[str], label_col: str, path, title: str) -> None:
    feats = [f for f in features if f in sample.columns]
    if not feats:
        return
    classes = sorted(sample[label_col].dropna().unique().tolist())
    n = len(feats)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        data = [sample.loc[sample[label_col] == c, f].dropna().to_numpy() for c in classes]
        data = [d if len(d) else np.array([np.nan]) for d in data]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=7)
        ax.set_title(f)
        vals = np.concatenate([d[np.isfinite(d)] for d in data]) if data else np.array([])
        if len(vals) and np.nanmin(vals) > 0 and np.nanmax(vals) / max(np.nanmin(vals), 1e-9) > 1e3:
            ax.set_yscale("log")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title)
    _save(fig, path)


def corr(sample: pd.DataFrame, path, title: str) -> None:
    num = sample.select_dtypes(include=[np.number])
    num = num.loc[:, num.std(numeric_only=True) > 0]
    if num.shape[1] < 2:
        return
    c = num.corr().fillna(0)
    fig, ax = plt.subplots(figsize=(0.32 * c.shape[1] + 3, 0.32 * c.shape[1] + 2))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(c.shape[1]))
    ax.set_xticklabels(c.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(c.shape[1]))
    ax.set_yticklabels(c.columns, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title)
    _save(fig, path)


def pr_curves(curves_by_class: Dict[str, tuple], path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cls, (prec, rec, ap) in curves_by_class.items():
        ax.plot(rec, prec, label=f"{cls} (AP={ap:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax.set_title(title)
    _save(fig, path)

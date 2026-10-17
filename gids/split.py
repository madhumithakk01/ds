"""Leakage-aware train / validation / test split.

Every attack in the dataset happens once, in a short time window, while
benign traffic runs for the whole capture.  A random split would put packets
a few milliseconds apart into both train and test, and a plain "earlier vs
later" split would leave whole attack types out of train or out of test.

So the split is made per raw label and in time order:

1. the packets of a label are grouped into *episodes* (a pause longer than
   ``split.episode_gap_s`` starts a new one);
2. each episode is cut into *blocks* of at most ``split.block_h`` hours
   (blocks with too few packets are merged into a neighbour);
3. inside each block the earliest packets go to train, the next ones to
   validation and the latest ones to test (``split.fractions``), with a short
   *gap* between the parts whose packets are left out, so that the last
   training packets and the first test packets are not near-copies.  A gap
   lasts ``gap_frac`` of the block (at least ``gap_min_s``) but never takes
   more than ``gap_max_share`` of the block's packets, so a short burst at
   the end of a block cannot swallow the whole test part.

The result is a small table of time thresholds per block, saved as JSON; any
later stage applies it with :func:`assign` to (raw label, time) columns.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import get_path
from .state import stable_hash

SPLITS = ("train", "val", "test", "gap")
TRAIN, VAL, TEST, GAP, NONE = 0, 1, 2, 3, -1


def split_spec(cfg) -> Dict[str, Any]:
    return {
        "fractions": [float(x) for x in get_path(cfg, "split.fractions", [0.6, 0.1, 0.3])],
        "block_h": float(get_path(cfg, "split.block_h", 1.0)),
        "episode_gap_s": float(get_path(cfg, "split.episode_gap_s", 600)),
        "gap_frac": float(get_path(cfg, "split.gap_frac", 0.01)),
        "gap_min_s": float(get_path(cfg, "split.gap_min_s", 1.0)),
        "gap_max_share": float(get_path(cfg, "split.gap_max_share", 0.05)),
        "min_block_packets": int(get_path(cfg, "split.min_block_packets", 200)),
    }


def split_name(cfg) -> str:
    return f"split-{stable_hash(split_spec(cfg))}"


def _blocks_of(t: np.ndarray, spec: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    """(episode, start index, end index) of every block of one label's sorted times."""
    out: List[Tuple[int, int, int]] = []
    if len(t) == 0:
        return out
    breaks = np.flatnonzero(np.diff(t) > spec["episode_gap_s"]) + 1
    bounds = np.r_[0, breaks, len(t)]
    block_s = spec["block_h"] * 3600.0
    for ep, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        idx = np.floor((t[a:b] - t[a]) / block_s).astype(np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1] + a
        edges = list(np.r_[starts, b])
        # merge blocks that are too small to be split three ways into a neighbour
        i = 0
        while len(edges) > 2 and i < len(edges) - 1:
            if edges[i + 1] - edges[i] < spec["min_block_packets"]:
                del edges[i + 1 if i + 1 < len(edges) - 1 else i]
                i = max(i - 1, 0)
            else:
                i += 1
        out.extend((ep, int(s), int(e)) for s, e in zip(edges[:-1], edges[1:]))
    return out


def make_split(labels: np.ndarray, t: np.ndarray, spec: Dict[str, Any]) -> pd.DataFrame:
    """Compute the block table from every packet's raw label and time (epoch seconds)."""
    f_tr, f_va = spec["fractions"][0], spec["fractions"][1]
    rows = []
    lab = pd.Series(labels).astype("category")
    codes = lab.cat.codes.to_numpy()
    order = np.lexsort((t, codes))
    codes_s, t_s = codes[order], t[order]
    starts = np.r_[0, np.flatnonzero(np.diff(codes_s)) + 1]
    ends = np.r_[starts[1:], len(codes_s)]
    for a, b in zip(starts, ends):
        name = str(lab.cat.categories[codes_s[a]])
        tl = t_s[a:b]
        tl = tl[np.isfinite(tl)]
        for blk, (ep, s, e) in enumerate(_blocks_of(tl, spec)):
            tb = tl[s:e]
            n = len(tb)
            gap = max(spec["gap_frac"] * float(tb[-1] - tb[0]), spec["gap_min_s"])
            g_max = int(np.ceil(spec["gap_max_share"] * n))  # a gap never takes more than this many packets
            i_tr, i_va = min(int(f_tr * n), n - 1), min(int((f_tr + f_va) * n), n - 1)
            train_end, val_end = float(tb[i_tr]), float(tb[i_va])
            val_start = min(train_end + gap, float(tb[min(i_tr + g_max, n - 1)]))
            test_start = max(min(val_end + gap, float(tb[min(i_va + g_max, n - 1)])), val_start)
            rows.append({"label": name, "episode": ep, "block": blk, "start": float(tb[0]), "end": float(tb[-1]),
                         "train_end": train_end, "val_start": val_start, "val_end": val_end,
                         "test_start": test_start, "packets": n})
    return pd.DataFrame(rows)


def assign(labels, t: np.ndarray, table: pd.DataFrame, with_episode: bool = False):
    """Split code per packet: 0 train, 1 val, 2 test, 3 gap, -1 label not in the table.

    With ``with_episode`` also returns the episode number of every packet
    (the n-th separate run of its label; -1 when not in the table).
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.full(len(t), NONE, dtype=np.int8)
    episode = np.full(len(t), -1, dtype=np.int16)
    lab = pd.Series(labels).astype("category")
    codes = lab.cat.codes.to_numpy()
    for ci, name in enumerate(lab.cat.categories):
        tab = table[table["label"] == str(name)].sort_values("start")
        if tab.empty:
            continue
        m = np.flatnonzero(codes == ci)
        tt = t[m]
        b = np.clip(np.searchsorted(tab["start"].to_numpy(), tt, side="right") - 1, 0, len(tab) - 1)
        col = {k: tab[k].to_numpy()[b] for k in ("train_end", "val_start", "val_end", "test_start")}
        s = np.full(len(m), TEST, dtype=np.int8)
        s[tt < col["test_start"]] = GAP
        s[tt < col["val_end"]] = VAL
        s[tt < col["val_start"]] = GAP
        s[tt < col["train_end"]] = TRAIN
        s[~np.isfinite(tt)] = NONE
        out[m] = s
        episode[m] = tab["episode"].to_numpy()[b]
    return (out, episode) if with_episode else out

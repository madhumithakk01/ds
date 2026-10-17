"""Turning tshark text fields into typed columns.

tshark writes some fields as hex (``0x0012``), repeats a field when a packet
carries it twice (``64,63`` for an ICMP error that quotes an inner IP header),
and formats timestamps in several ways depending on version and options.
Every parser here reports how many non-empty values it could not interpret,
so data-quality problems end up in the trace instead of silently becoming NaN.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def normalize_name(name: str) -> str:
    s = _NON_ALNUM.sub("_", str(name).strip().lower()).strip("_")
    return s or "col"


def normalize_names(names: Iterable[str]) -> List[str]:
    out, seen = [], {}
    for n in names:
        b = normalize_name(n)
        if b in seen:
            seen[b] += 1
            b = f"{b}_{seen[b]}"
        else:
            seen[b] = 0
        out.append(b)
    return out


LABEL_CANDIDATES = ("label", "labels", "class", "attack", "attack_type", "attack_cat",
                    "category", "target", "y", "traffic_type")


def detect_label_column(cols: List[str], configured: str = "auto") -> Optional[str]:
    if configured and configured != "auto":
        c = normalize_name(configured)
        return c if c in cols else None
    for cand in LABEL_CANDIDATES:
        if cand in cols:
            return cand
    return cols[-1] if cols else None


def _as_str_series(s: pd.Series) -> pd.Series:
    if s.dtype == object or str(s.dtype) in ("string", "str") or isinstance(s.dtype, pd.StringDtype):
        return s
    return s.astype(str)


def _parse_scalar(v) -> float:
    if v is None:
        return np.nan
    v = str(v).strip()
    if not v or v.lower() in ("nan", "none", "null", "-"):
        return np.nan
    if "," in v:
        v = v.split(",", 1)[0].strip()
        if not v:
            return np.nan
    low = v.lower()
    if low.startswith("0x"):
        try:
            return float(int(low, 16))
        except ValueError:
            return np.nan
    try:
        return float(v)
    except ValueError:
        pass
    if low in ("true", "false"):
        return 1.0 if low == "true" else 0.0
    return np.nan


def _is_blank(v) -> bool:
    if v is None:
        return True
    try:
        if v != v:  # NaN
            return True
    except Exception:
        pass
    s = str(v).strip().lower()
    return s in ("", "nan", "none", "null", "-")


def parse_numeric(s: pd.Series) -> Tuple[np.ndarray, int]:
    """Parse a text column into float64.  Returns (values, n_failed)."""
    if len(s) == 0:
        return np.empty(0, dtype=np.float64), 0
    if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
        return s.to_numpy(dtype=np.float64, na_value=np.nan), 0
    s = _as_str_series(s)
    try:
        cleaned = s.where(s.str.strip() != "", None)
        vals = pd.to_numeric(cleaned, errors="raise")
        if isinstance(vals, pd.Series):
            return vals.to_numpy(dtype=np.float64, na_value=np.nan), 0
        return np.asarray(vals, dtype=np.float64), 0
    except (ValueError, TypeError):
        pass
    codes, uniques = pd.factorize(s, use_na_sentinel=True)
    parsed = np.array([_parse_scalar(u) for u in uniques], dtype=np.float64)
    blank = np.array([_is_blank(u) for u in uniques], dtype=bool)
    out = np.full(len(s), np.nan, dtype=np.float64)
    ok = codes >= 0
    out[ok] = parsed[codes[ok]]
    bad_unique = np.isnan(parsed) & ~blank
    n_failed = int(bad_unique[codes[ok]].sum()) if ok.any() else 0
    return out, n_failed


def first_value(s: pd.Series) -> pd.Series:
    """Keep the first of comma separated repeated values (outer header)."""
    s = _as_str_series(s)
    if bool(s.str.contains(",", regex=False).fillna(False).any()):
        s = s.str.split(",", n=1).str[0]
    return s.str.strip()


_TZ_TAIL = re.compile(r"\s+[A-Za-z][A-Za-z ]*$")
_FRAC = re.compile(r"(\.\d{6})\d+")
_TIME_FORMATS = (
    "%b %d, %Y %H:%M:%S.%f",
    "%b %d, %Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S.%f",
)


def parse_time(s: pd.Series, hint: Optional[str] = None) -> Tuple[np.ndarray, int, str]:
    """Parse arrival times into epoch seconds (float64).

    Returns (values, n_failed, method).  ``hint`` is the method that worked on
    the previous block, tried first to keep things fast and consistent.
    """
    n = len(s)
    if n == 0:
        return np.empty(0), 0, hint or "none"
    s = _as_str_series(s)
    nonblank = (s.str.strip() != "").to_numpy(dtype=bool, na_value=False)
    n_nonblank = int(nonblank.sum())
    if n_nonblank == 0:
        return np.full(n, np.nan), 0, hint or "none"

    def _score(vals: np.ndarray) -> float:
        return float(np.isfinite(vals[nonblank]).mean())

    methods: List[str] = ["epoch"] + [f"fmt:{f}" for f in _TIME_FORMATS] + ["infer"]
    if hint in methods:
        methods.remove(hint)
        methods.insert(0, hint)

    cleaned = None
    best: Tuple[float, np.ndarray, str] = (-1.0, np.full(n, np.nan), "none")
    for m in methods:
        if m == "epoch":
            vals, _ = parse_numeric(s)
            if np.isfinite(vals).any():
                med = np.nanmedian(vals)
                if not (1e8 < med < 5e9):
                    vals = np.full(n, np.nan)
        else:
            if cleaned is None:
                cleaned = s.str.strip().str.replace(r"\s+", " ", regex=True)
                cleaned = cleaned.str.replace(_TZ_TAIL, "", regex=True)
                cleaned = cleaned.str.replace(_FRAC, r"\1", regex=True)
            try:
                if m == "infer":
                    dt = pd.to_datetime(cleaned, errors="coerce", utc=True, format="mixed")
                else:
                    dt = pd.to_datetime(cleaned, errors="coerce", utc=True, format=m[4:])
                epoch = pd.Timestamp("1970-01-01", tz="UTC")
                vals = ((dt - epoch) / pd.Timedelta(seconds=1)).to_numpy(dtype=np.float64, na_value=np.nan)
            except (ValueError, TypeError, OverflowError):
                continue
        score = _score(vals)
        if score > best[0]:
            best = (score, vals, m)
        if score >= 0.995:
            break
    score, vals, method = best
    n_failed = int((~np.isfinite(vals) & nonblank).sum())
    return vals, n_failed, method


def _parse_tcp_options_one(u) -> Tuple[float, float]:
    if _is_blank(u):
        return np.nan, np.nan
    h = str(u).strip()
    if "," in h:
        h = h.split(",", 1)[0]
    h = h.replace(":", "").replace(" ", "").lower()
    if h.startswith("0x"):
        h = h[2:]
    if len(h) % 2:
        return np.nan, np.nan
    try:
        b = bytes.fromhex(h)
    except ValueError:
        return np.nan, np.nan
    mask = 0
    i = 0
    n = len(b)
    while i < n:
        kind = b[i]
        mask |= 1 << min(kind, 23)
        if kind == 0:
            break
        if kind == 1:
            i += 1
            continue
        if i + 1 >= n:
            break
        ln = b[i + 1]
        if ln < 2:
            break
        i += ln
    return float(n), float(mask)


def parse_tcp_options(s: pd.Series) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (options_length_bytes, option_kind_bitmask, n_failed)."""
    if len(s) == 0:
        return np.empty(0), np.empty(0), 0
    s = _as_str_series(s)
    codes, uniques = pd.factorize(s, use_na_sentinel=True)
    lens = np.empty(len(uniques), dtype=np.float64)
    masks = np.empty(len(uniques), dtype=np.float64)
    blanks = np.empty(len(uniques), dtype=bool)
    for i, u in enumerate(uniques):
        lens[i], masks[i] = _parse_tcp_options_one(u)
        blanks[i] = _is_blank(u)
    out_len = np.full(len(s), np.nan)
    out_mask = np.full(len(s), np.nan)
    ok = codes >= 0
    out_len[ok] = lens[codes[ok]]
    out_mask[ok] = masks[codes[ok]]
    bad = np.isnan(lens) & ~blanks
    n_failed = int(bad[codes[ok]].sum()) if ok.any() else 0
    return out_len, out_mask, n_failed


def clean_label(s: pd.Series) -> pd.Series:
    s = _as_str_series(s).str.strip().str.replace(r"\s+", " ", regex=True)
    return s.where(s != "", "UNKNOWN")


def mode_ip(src: pd.Series, dst: pd.Series) -> Optional[str]:
    both = pd.concat([src, dst], ignore_index=True)
    both = both[(both != "") & both.notna()]
    if both.empty:
        return None
    vc = both.value_counts()
    return str(vc.index[0])


def device_from_member(member_path: str) -> Tuple[str, str]:
    """``processed/iotsim-cooler-motor-3.csv`` -> (``cooler-motor-3``, ``cooler-motor``)."""
    stem = member_path.replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    if stem.lower().startswith("iotsim-"):
        stem = stem[7:]
    m = re.match(r"^(?P<type>.+?)-(?P<idx>\d+)$", stem)
    if m:
        return stem, m.group("type")
    return stem, stem


def summarize_values(s: pd.Series, top: int = 8) -> Dict[str, int]:
    vc = s.value_counts(dropna=False).head(top)
    return {str(k): int(v) for k, v in vc.items()}

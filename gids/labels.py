"""Mapping raw dataset labels to the classes a model is trained on.

``scheme: coarse`` groups the ten packet labels of the dataset into the six
families used in the dataset paper's own evaluation (Normal, DoS, Network
Scanning, Brute Force, C&C Communication, Infection).  ``scheme: fine`` keeps
every raw label as its own class.  The mapping actually applied is written to
the reports so it can be checked against the label inventory.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DUPLICATE = "Duplicate copy"  # rows of another device's capture copied into a file (step 6 corrections)


class LabelMapper:
    def __init__(self, cfg_labels: Dict):
        self.scheme = cfg_labels.get("scheme", "coarse")
        self.normal = cfg_labels.get("normal_class", "Normal")
        self.explicit = {self._norm(k): v for k, v in (cfg_labels.get("explicit_map") or {}).items()}
        self.rules: List[Tuple[re.Pattern, str]] = [
            (re.compile(p, re.IGNORECASE), c) for p, c in (cfg_labels.get("rules") or [])
        ]
        self.unmapped = cfg_labels.get("unmapped", "keep")
        self.drop = {self._norm(x) for x in (cfg_labels.get("drop_raw") or [])}

    @staticmethod
    def _norm(s) -> str:
        return re.sub(r"\s+", " ", str(s).strip())

    def explain(self, raw: str) -> Tuple[Optional[str], str]:
        r = self._norm(raw)
        if r == DUPLICATE:
            return None, "copy of another device's capture (corrected labels)"
        if r in self.drop:
            return None, "dropped by labels.drop_raw"
        if self.scheme == "fine":
            if re.search(r"normal|benign", r, re.IGNORECASE):
                return self.normal, "fine scheme (normal)"
            return r, "fine scheme"
        if r in self.explicit:
            return self.explicit[r], "explicit_map"
        for pat, cls in self.rules:
            if pat.search(r):
                return cls, f"rule /{pat.pattern}/"
        if self.unmapped == "drop":
            return None, "unmapped (dropped)"
        return r, "unmapped (kept as own class)"

    def map_one(self, raw: str) -> Optional[str]:
        return self.explain(raw)[0]

    def mapping(self, raws: Iterable[str]) -> Dict[str, Optional[str]]:
        return {str(r): self.map_one(r) for r in raws}

    def classes(self, raws: Iterable[str]) -> List[str]:
        cls = {c for c in self.mapping(raws).values() if c is not None}
        rest = sorted(c for c in cls if c != self.normal)
        return ([self.normal] if self.normal in cls else []) + rest

    def table(self, raw_counts: Dict[str, int]) -> pd.DataFrame:
        rows = []
        for raw, n in sorted(raw_counts.items(), key=lambda kv: -kv[1]):
            cls, why = self.explain(raw)
            rows.append({"raw_label": raw, "class": cls, "rule": why, "rows": int(n)})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["share"] = df["rows"] / df["rows"].sum()
        return df

    def unmapped_labels(self, raws: Iterable[str]) -> List[str]:
        return [r for r in raws if self.explain(r)[1].startswith("unmapped")]

    def encode(self, raw_series: pd.Series, classes: List[str]):
        """Raw label column -> int class index (-1 when dropped/unknown)."""
        import numpy as np

        cats = pd.Series(raw_series).astype("category")
        mp = self.mapping(cats.cat.categories)
        index = {c: i for i, c in enumerate(classes)}
        lut = np.array([index.get(mp[str(c)], -1) if mp[str(c)] is not None else -1
                        for c in cats.cat.categories] + [-1], dtype=np.int32)
        codes = cats.cat.codes.to_numpy()
        return lut[np.where(codes < 0, len(lut) - 1, codes)]

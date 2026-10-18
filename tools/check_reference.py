"""Check gids/reference.py against the dataset authors' own preprocessing functions, input by input.

    git clone https://github.com/othmbela/gotham-network-packet-labeller /content/labeller
    python tools/check_reference.py --zip /content/drive/MyDrive/Ds/GothamDataset2025.zip \\
        --labeller /content/labeller --rows 200000

Runs the helper functions of the authors' notebook ("GothamDataset2025 -- Data Preparation.ipynb":
extract_protocols_and_ports, convert_time, convert_ports, convert_checksums, fill_missing_values,
unpack_flags and the one-hot encoder) and gids.reference on the same packets (the first --rows rows of one
CSV in the archive) and reports every model input that differs.  Rows the notebook's own code cannot
convert are left out of the comparison and counted.  This executes code from the labeller repository.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gids import reference as R  # noqa: E402
from gids.explore import BlockParser  # noqa: E402
from gids.parse import normalize_names  # noqa: E402

NOTEBOOK = "notebooks/GothamDataset2025 -- Data Preparation.ipynb"


def notebook_functions(labeller: Path) -> dict:
    nb = json.loads((labeller / NOTEBOOK).read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    helpers = next(c for c in cells if "def extract_protocols_and_ports" in c)
    consts = next(c for c in cells if "PORT_HIERARCHY_MAP_IOT" in c and "GLOBAL_CATEGORICAL_VALUES" in c)
    helpers = helpers.replace("        axis=1,\n        inplace=True,", "        inplace=True,")  # pandas >= 3
    ns: dict = {}
    exec("import pandas as pd\nimport numpy as np\n" + helpers + "\n" + consts, ns)  # noqa: S102
    return ns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--labeller", required=True, help="clone of github.com/othmbela/gotham-network-packet-labeller")
    ap.add_argument("--member", help="CSV inside the zip (default: the first processed CSV)")
    ap.add_argument("--rows", type=int, default=200_000)
    a = ap.parse_args()
    from sklearn.preprocessing import OneHotEncoder

    ns = notebook_functions(Path(a.labeller))
    with zipfile.ZipFile(a.zip) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv") and "processed" in n.lower()]
        member = a.member or names[0]
        with zf.open(member) as fh:
            lines = [fh.readline() for _ in range(a.rows + 1)]
    text = b"".join(x for x in lines if x)
    print(f"{member}: first {len(lines) - 1:,} rows")
    df = pd.read_csv(io.BytesIO(text), low_memory=False, on_bad_lines="skip")
    hdr = text.split(b"\n", 1)[0].decode()
    raw, _ = BlockParser(normalize_names(next(iter([hdr.split(",")]))), ",", None, prefer="pandas").parse(
        text.split(b"\n", 1)[1])
    if len(df) != len(raw):
        print(f"row counts differ ({len(df)} vs {len(raw)}); malformed lines - use another --member")
        return 2
    keep = (df["label"] != "Unknown").to_numpy().copy()
    fp = df["frame.protocols"].astype(str)
    crash = ((fp.str.contains(":tcp") & df["tcp.srcport"].isna())
             | (~fp.str.contains(":tcp") & fp.str.contains(":udp") & df["udp.srcport"].isna())
             | df[["ip.ttl", "ip.flags", "frame.len"]].astype(str).apply(lambda c: c.str.contains(",")).any(axis=1)
             ).to_numpy()
    print(f"rows the notebook cannot convert (left out): {int((keep & crash).sum()):,}")
    keep &= ~crash
    df, raw = df[keep].reset_index(drop=True), raw[keep].reset_index(drop=True)
    # the notebook expects "... .ffffff000 BST"
    df["frame.time"] = df["frame.time"].astype(str).str.replace(r"\s+[A-Za-z]{2,5}$", " BST", regex=True)
    df["frame.time"] = df["frame.time"].str.replace(r"(\.\d{6})\d*", r"\g<1>000", regex=True)
    ref = ns["extract_protocols_and_ports"](df.copy())
    ref = ns["convert_time"](ref)
    ref["timestamp"] = ref["frame.time"].values.astype("datetime64[ns]").astype(np.int64) // 10**9  # pandas 2 units
    ref = ns["convert_ports"](ref, ns["PORT_HIERARCHY_MAP_IOT"])
    ref = ns["convert_checksums"](ref)
    ref = ns["fill_missing_values"](ref)
    enc = OneHotEncoder(drop="first", sparse_output=False).fit(ns["GLOBAL_CATEGORICAL_VALUES"])
    nums = [nm for nm, _ in R.REF_NUMERIC]
    A = np.hstack([ns["unpack_flags"](ref[["ip.flags", "tcp.flags"]]),
                   enc.transform(ref[["ip.protocol", "src.port", "dst.port"]]),
                   ref[nums].to_numpy(np.float64)])
    mine, _ = R.reference_features(raw, "frame_time", None)
    unit = {nm: (0.0, 1.0) for nm in nums}
    B = np.hstack([R.reference_inputs(mine, "reference", unit)[:, :61].astype(np.float64),
                   mine[[c for _, c in R.REF_NUMERIC]].to_numpy(np.float64)])
    names = R.reference_input_names("reference")
    same_names = [n.split("=", 1)[1] for n in names[11:61]] == [f.split("_", 1)[1] for f in enc.get_feature_names_out()]
    bad = [(names[j], int((~np.isclose(A[:, j], B[:, j], rtol=1e-12, atol=0)).sum()))
           for j in range(A.shape[1]) if not np.allclose(A[:, j], B[:, j], rtol=1e-12, atol=0)]
    print(f"{len(A):,} packets compared, {A.shape[1]} inputs; one-hot order identical: {same_names}")
    print("inputs that differ: " + (", ".join(f"{n} ({k:,} rows)" for n, k in bad) if bad else "none"))
    return 0 if same_names and not bad else 1


if __name__ == "__main__":
    sys.exit(main())

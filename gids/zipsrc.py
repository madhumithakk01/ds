"""Reading the dataset archive without unpacking it.

The Zenodo archive holds raw PCAPs (~25 GB) next to the processed CSVs
(~9 GB).  Only the CSVs are needed, so members are streamed straight out of
the zip on Drive.  Reading is done in line-aligned byte blocks whose offsets
are recorded, which is what makes resuming in the middle of a 2 GB file cheap.
"""
from __future__ import annotations

import fnmatch
import io
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .state import ensure_dir, retry


@dataclass
class Member:
    id: str              # stable identifier used in ledgers
    source: str          # path of the zip file or directory
    name: str            # path inside the zip (or relative path in the directory)
    size: int            # uncompressed bytes
    csize: int           # compressed bytes
    kind: str            # processed_csv | raw_pcap | metadata | nested_zip | other
    inner: Optional[str] = None  # member name of a nested zip that contains this file

    def to_dict(self):
        return asdict(self)


def _classify(name: str, cfg_inv: dict) -> str:
    low = name.replace("\\", "/").lower()
    base = low.rsplit("/", 1)[-1]
    if base.startswith(".") or "__macosx" in low or low.endswith("/"):
        return "other"
    excl = [e.lower() for e in cfg_inv.get("exclude", [])]
    if any(fnmatch.fnmatch(low, e) or e in low for e in excl):
        if low.endswith((".pcap", ".pcapng")):
            return "raw_pcap"
        return "other"
    if low.endswith(".csv"):
        incl = [i.lower() for i in cfg_inv.get("include", [])]
        if not incl or any(fnmatch.fnmatch(low, i) or i in low for i in incl):
            return "processed_csv"
        return "other_csv"
    if low.endswith((".pcap", ".pcapng")):
        return "raw_pcap"
    if low.endswith(".json"):
        return "metadata"
    if low.endswith(".zip"):
        return "nested_zip"
    return "other"


def find_zip(cfg_paths: dict) -> Optional[str]:
    """Return the configured zip path, or look for a likely candidate on Drive."""
    zp = cfg_paths.get("zip_path")
    if zp and Path(zp).exists():
        return zp
    roots = [r for r in cfg_paths.get("search_roots", []) if Path(r).exists()]
    hits: List[Tuple[int, str]] = []
    for root in roots:
        root_depth = str(root).rstrip("/").count("/")
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count("/") - root_depth >= 3:
                dirnames[:] = []
            for f in filenames:
                if f.lower().endswith(".zip") and "gotham" in f.lower():
                    p = os.path.join(dirpath, f)
                    try:
                        hits.append((os.path.getsize(p), p))
                    except OSError:
                        pass
    if hits:
        hits.sort(reverse=True)
        return hits[0][1]
    return None


def build_inventory(cfg: dict) -> Dict:
    paths = cfg["paths"]
    inv_cfg = cfg.get("inventory", {})
    members: List[Member] = []
    data_dir = paths.get("data_dir")
    if data_dir:
        root = Path(data_dir)
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(root)).replace("\\", "/")
                st = p.stat().st_size
                members.append(Member(id=rel, source=str(root), name=rel, size=st, csize=st,
                                      kind=_classify(rel, inv_cfg)))
        source = {"type": "dir", "path": str(root)}
    else:
        zp = find_zip(paths)
        if not zp:
            raise FileNotFoundError(
                f"Dataset zip not found at {paths.get('zip_path')} and no candidate found under "
                f"{paths.get('search_roots')}. Set paths.zip_path (or GIDS_ZIP_PATH).")

        def _list():
            with zipfile.ZipFile(zp) as zf:
                return [(i.filename, i.file_size, i.compress_size, i.compress_type, i.CRC)
                        for i in zf.infolist() if not i.is_dir()]

        infos = retry(_list, tries=4, base_delay=3.0)
        for name, size, csize, ctype, crc in infos:
            members.append(Member(id=name, source=zp, name=name, size=size, csize=csize,
                                  kind=_classify(name, inv_cfg)))
        source = {"type": "zip", "path": zp, "bytes": os.path.getsize(zp)}

    # CSVs packed inside a nested zip
    if not any(m.kind == "processed_csv" for m in members):
        for nz in [m for m in members if m.kind == "nested_zip"]:
            local = extract_nested(nz, cfg)
            with zipfile.ZipFile(local) as zf:
                for i in zf.infolist():
                    if i.is_dir():
                        continue
                    kind = _classify(i.filename, inv_cfg)
                    members.append(Member(id=f"{nz.name}!{i.filename}", source=nz.source,
                                          name=i.filename, size=i.file_size, csize=i.compress_size,
                                          kind=kind, inner=nz.name))

    summary: Dict[str, Dict[str, float]] = {}
    for m in members:
        top = m.name.replace("\\", "/").split("/")
        group = "/".join(top[:-1][-2:]) or "."
        key = f"{m.kind}:{group}"
        s = summary.setdefault(key, {"files": 0, "gb": 0.0, "compressed_gb": 0.0})
        s["files"] += 1
        s["gb"] += m.size / 2**30
        s["compressed_gb"] += m.csize / 2**30
    for s in summary.values():
        s["gb"] = round(s["gb"], 3)
        s["compressed_gb"] = round(s["compressed_gb"], 3)
    return {"source": source, "members": [m.to_dict() for m in members], "summary": summary}


def extract_nested(nz: Member, cfg: dict) -> str:
    tmp = ensure_dir(Path(cfg["paths"]["local_tmp"]) / "nested")
    out = tmp / Path(nz.name).name
    if out.exists() and out.stat().st_size == nz.size:
        return str(out)

    def _extract():
        with zipfile.ZipFile(nz.source) as zf, zf.open(nz.name) as src, open(out, "wb") as dst:
            while True:
                buf = src.read(64 * 2**20)
                if not buf:
                    break
                dst.write(buf)

    retry(_extract, tries=3, base_delay=5.0)
    return str(out)


class MemberStream:
    """Context manager returning a binary stream for a member."""

    def __init__(self, member: dict, cfg: dict):
        self.m = member
        self.cfg = cfg
        self._zf = None
        self._fh = None

    def __enter__(self):
        m = self.m
        if m.get("inner"):
            local = extract_nested(Member(**{**m, "name": m["inner"], "inner": None,
                                             "kind": "nested_zip"}), self.cfg)
            self._zf = zipfile.ZipFile(local)
            self._fh = self._zf.open(m["name"])
        elif Path(m["source"]).is_dir():
            self._fh = open(Path(m["source"]) / m["name"], "rb")
        else:
            self._zf = zipfile.ZipFile(m["source"])
            self._fh = self._zf.open(m["name"])
        return self._fh

    def __exit__(self, *exc):
        for h in (self._fh, self._zf):
            try:
                if h is not None:
                    h.close()
            except Exception:
                pass
        return False


def read_header(stream) -> Tuple[bytes, int]:
    line = stream.readline()
    return line, len(line)


def skip_bytes(stream, n: int, chunk: int = 64 * 2**20) -> None:
    left = n
    while left > 0:
        buf = stream.read(min(chunk, left))
        if not buf:
            raise EOFError(f"member ended while skipping ({left} bytes short)")
        left -= len(buf)


def iter_blocks(stream, start: int, block_bytes: int) -> Iterator[Tuple[int, int, bytes]]:
    """Yield (start_offset, end_offset, bytes) blocks that end on a newline.

    ``start`` is the absolute offset of the stream position on entry.
    """
    pos = start
    carry = b""
    while True:
        chunk = stream.read(block_bytes)
        if not chunk:
            if carry:
                yield pos, pos + len(carry), carry
            return
        data = carry + chunk if carry else chunk
        cut = data.rfind(b"\n")
        if cut == -1:
            carry = data
            continue
        block = data[:cut + 1]
        carry = data[cut + 1:]
        yield pos, pos + len(block), block
        pos += len(block)


def peek_member(member: dict, cfg: dict, n_lines: int = 15, max_bytes: int = 256 * 1024) -> Dict:
    with MemberStream(member, cfg) as fh:
        raw = fh.read(max_bytes)
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()[: n_lines + 1]
    header = lines[0] if lines else ""
    delim = "\t" if header.count("\t") > header.count(",") else ","
    return {"member": member["name"], "delimiter": "tab" if delim == "\t" else "comma",
            "header": header, "lines": lines[1:]}


def bytes_to_text_io(block: bytes) -> io.StringIO:
    return io.StringIO(block.decode("utf-8", errors="replace"))

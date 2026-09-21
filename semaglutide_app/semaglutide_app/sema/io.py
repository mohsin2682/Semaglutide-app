"""Data loading: mzML / mzXML (via pyteomics) and CSV/TSV peak lists.

Vendor files (.d, .wiff, .raw) must first be converted to mzML with ProteoWizard MSConvert
(see README and convert_raw.py).
"""
from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import looks_like_profile, pick_peaks


@dataclass
class Scan:
    idx: int
    ms_level: int
    rt: float  # minutes
    mz: np.ndarray
    inten: np.ndarray
    prec_mz: float | None = None
    prec_z: int | None = None
    label: str = ""


VENDOR_EXT = (".d", ".wiff", ".wiff2", ".raw", ".lcd")


def _clean(mz, inten, centroid: bool):
    mz = np.asarray(mz, float)
    inten = np.asarray(inten, float)
    o = np.argsort(mz)
    mz, inten = mz[o], inten[o]
    ok = inten > 0
    mz, inten = mz[ok], inten[ok]
    if centroid and looks_like_profile(mz):
        mz, inten = pick_peaks(mz, inten)
    return mz, inten


def load_run(data: bytes, filename: str, peak_pick_profile: bool = True) -> list[Scan]:
    ext = os.path.splitext(filename.lower())[1]
    if ext in VENDOR_EXT:
        raise ValueError(
            f"'{ext}' is a vendor format. Convert it to mzML first with ProteoWizard MSConvert "
            "(peak picking ON, 'Vendor' algorithm) - see README / convert_raw.py."
        )
    if ext in (".mzml", ".mzxml"):
        return _load_mz(data, ext, peak_pick_profile)
    if ext in (".csv", ".txt", ".tsv", ".xy"):
        return _load_table(data, peak_pick_profile)
    raise ValueError(f"Unsupported file type '{ext}'")


def _first(x, default=None):
    try:
        return x[0]
    except Exception:
        return default


def _load_mz(data: bytes, ext: str, pick: bool) -> list[Scan]:
    from pyteomics import mzml, mzxml

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
        fh.write(data)
        path = fh.name
    scans: list[Scan] = []
    try:
        if ext == ".mzml":
            with mzml.read(path) as rd:
                for n, sp in enumerate(rd):
                    lvl = int(sp.get("ms level", 1))
                    rt = float(_first(sp["scanList"]["scan"])["scan start time"])
                    unit = getattr(_first(sp["scanList"]["scan"])["scan start time"], "unit_info", "minute")
                    if str(unit).lower().startswith("sec"):
                        rt /= 60.0
                    pm = pz = None
                    if lvl > 1 and sp.get("precursorList"):
                        prec = _first(sp["precursorList"]["precursor"])
                        ion = _first(prec["selectedIonList"]["selectedIon"])
                        pm = float(ion["selected ion m/z"])
                        pz = int(ion["charge state"]) if "charge state" in ion else None
                    mz, it = _clean(sp["m/z array"], sp["intensity array"], pick)
                    if mz.size:
                        scans.append(Scan(n, lvl, rt, mz, it, pm, pz, sp.get("id", str(n))))
        else:
            with mzxml.read(path) as rd:
                for n, sp in enumerate(rd):
                    lvl = int(sp.get("msLevel", 1))
                    rt = float(sp.get("retentionTime", 0.0))
                    pm = pz = None
                    if lvl > 1 and sp.get("precursorMz"):
                        p0 = sp["precursorMz"][0]
                        pm = float(p0["precursorMz"])
                        pz = int(p0["precursorCharge"]) if "precursorCharge" in p0 else None
                    mz, it = _clean(sp["m/z array"], sp["intensity array"], pick)
                    if mz.size:
                        scans.append(Scan(n, lvl, rt, mz, it, pm, pz, str(sp.get("num", n))))
    finally:
        os.unlink(path)
    if not scans:
        raise ValueError("No spectra found in file.")
    return scans


def _load_table(data: bytes, pick: bool) -> list[Scan]:
    """CSV/TSV/whitespace table. Minimum: two columns m/z and intensity (header optional).
    Optional columns: scan / rt / ms_level / precursor_mz / precursor_z -> multiple scans."""
    text = data.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python", comment="#")
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    def find(*names):
        for n in names:
            for c in df.columns:
                if n in c:
                    return c
        return None

    if not any(k in df.columns for k in ("mz", "m/z")) and find("mz", "m/z", "mass") is None:
        # headerless two-column file
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", header=None, comment="#")
        df = df.iloc[:, :2]
        df.columns = ["mz", "intensity"]
    mzc = find("m/z", "mz", "mass")
    itc = find("intens", "abund", "height", "area", "counts")
    if mzc is None or itc is None:
        raise ValueError("Could not find m/z and intensity columns.")
    scan_c = find("scan", "spectrum")
    rt_c = find("rt", "retention", "time")
    lvl_c = find("ms_level", "mslevel", "level")
    pm_c = find("precursor_mz", "precursor", "prec")
    pz_c = find("precursor_z", "prec_z", "charge")
    group = scan_c or (rt_c if rt_c and lvl_c is not None else None)
    scans: list[Scan] = []
    if group is None:
        mz, it = _clean(df[mzc], df[itc], pick)
        pm = float(df[pm_c].dropna().iloc[0]) if pm_c and df[pm_c].notna().any() else None
        pz = int(df[pz_c].dropna().iloc[0]) if pz_c and df[pz_c].notna().any() else None
        return [Scan(0, 2 if pm else 1, 0.0, mz, it, pm, pz, "single spectrum")]
    for n, (key, g) in enumerate(df.groupby(group, sort=True)):
        lvl = int(g[lvl_c].iloc[0]) if lvl_c else 1
        rt = float(g[rt_c].iloc[0]) if rt_c else float(n)
        pm = float(g[pm_c].iloc[0]) if pm_c and pd.notna(g[pm_c].iloc[0]) else None
        pz = int(g[pz_c].iloc[0]) if pz_c and pd.notna(g[pz_c].iloc[0]) else None
        mz, it = _clean(g[mzc], g[itc], pick)
        scans.append(Scan(n, lvl, rt, mz, it, pm, pz, str(key)))
    return scans


def scans_to_table(scans: list[Scan]) -> pd.DataFrame:
    rows = []
    for s in scans:
        for m, i in zip(s.mz, s.inten):
            rows.append((s.idx, s.ms_level, s.rt, s.prec_mz, s.prec_z, m, i))
    return pd.DataFrame(rows, columns=["scan", "ms_level", "rt", "precursor_mz", "precursor_z", "mz", "intensity"])

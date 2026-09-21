"""Synthetic QTOF-like LC-MS/MS run of semaglutide with spiked impurities (for testing the app)."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .analysis import merge_peaks
from .chem import C13_SPACING, PROTON, Peptide, fadd, fmass, fragment_table, isotope_pattern, mz_of, parse_formula, semaglutide
from .io import Scan

# name, formula delta, RT (min), relative abundance vs main, MS2-able
IMPURITIES = [
    ("Trp oxidation", "O", 11.60, 0.030),
    ("Gln deamidation", "N-1H-1O", 12.05, 0.040),
    ("des-Gly31", "C-2H-3N-1O-1", 12.55, 0.008),
    ("Loss of OEG", "C-6H-11N-1O-3", 13.10, 0.006),
    ("des-His1", "C-6H-7N-3O-1", 10.90, 0.005),
]
CS_DIST = {4: 0.55, 5: 1.0, 6: 0.45, 7: 0.08}
MAIN_RT = 12.30


def _envelope(formula: Counter, z: int, apex: float, n_iso: int = 9):
    M = fmass(formula)
    pat = isotope_pattern(formula, n_iso)
    return np.array([mz_of(M, z, k) for k in range(n_iso)]), pat * apex


def _fragments_spectrum(pep: Peptide, rng, shifted_from: int | None = None, delta: float = 0.0,
                        coverage: float = 0.75):
    """Simulated CID spectrum. If shifted_from is given, fragments that contain residues >= that
    position carry the +delta mass (modelling a localised modification)."""
    n = pep.n
    frag = fragment_table(pep, ("b", "y"), 3)
    mzs, its = [], []
    for r in frag.itertuples():
        if rng.random() > coverage * (1.0 if r.z == 1 else 0.7) or (r.index < 2):
            continue
        m = r.mz
        if shifted_from is not None:
            contains = (r.index >= shifted_from) if r.type == "b" else (r.index >= n - shifted_from + 1)
            if contains:
                m += delta / r.z
        m *= 1 + rng.normal(0, 3e-6)
        base = rng.lognormal(mean=np.log(5e3), sigma=0.9) * (1.5 if r.type == "y" else 1.0)
        mzs.append(m)
        its.append(base)
    # noise
    k = 250
    nm = rng.uniform(200, 1700, k)
    ni = rng.lognormal(np.log(150), 0.7, k)
    mz = np.concatenate([mzs, nm])
    it = np.concatenate([its, ni])
    o = np.argsort(mz)
    return mz[o], it[o]


def generate_demo_run(seed: int = 7) -> list[Scan]:
    rng = np.random.default_rng(seed)
    pep = semaglutide()
    main_f = pep.formula()
    species = [("main", main_f, MAIN_RT, 1.0)]
    for name, d, rt, rel in IMPURITIES:
        species.append((name, fadd(main_f, parse_formula(d)), rt, rel))

    scans: list[Scan] = []
    idx = 0
    rts = np.arange(9.5, 14.5, 0.04)
    for rt in rts:
        peaks_mz, peaks_it = [], []
        for name, f, rt0, rel in species:
            prof = np.exp(-0.5 * ((rt - rt0) / 0.09) ** 2)
            if prof < 1e-3:
                continue
            for z, w in CS_DIST.items():
                mzs, its = _envelope(f, z, 8e5 * rel * w * prof)
                peaks_mz.append(mzs)
                peaks_it.append(its)
        mz = np.concatenate(peaks_mz) if peaks_mz else np.array([])
        it = np.concatenate(peaks_it) if peaks_it else np.array([])
        if mz.size:
            mz, it = merge_peaks(mz, it, 8.0)  # unresolved overlaps (e.g. deamidation vs 13C)
            mz = mz * (1 + rng.normal(0, 1.2e-6, mz.size))
            it = it * rng.lognormal(0, 0.01, it.size)
        # chemical noise
        k = 250
        nm = rng.uniform(300, 2000, k)
        ni = rng.lognormal(np.log(400), 0.8, k)
        mz = np.concatenate([mz, nm])
        it = np.concatenate([it, ni])
        keep = it > 250
        mz, it = mz[keep], it[keep]
        o = np.argsort(mz)
        scans.append(Scan(idx, 1, float(rt), mz[o], it[o], label=f"MS1 {idx}"))
        idx += 1

    # MS/MS: main species 5+
    mzm, itm = _fragments_spectrum(pep, rng)
    scans.append(Scan(idx, 2, MAIN_RT + 0.01, mzm, itm, mz_of(pep.mass(), 5), 5, "MS2 semaglutide [M+5H]5+"))
    idx += 1
    # MS/MS: Trp-oxidised impurity (+15.9949 on Trp25)
    ox_delta = fmass(parse_formula("O"))
    # modelled by an explicit +O delta on all fragments that contain Trp25
    mzo, ito = _fragments_spectrum(pep, rng, shifted_from=25, delta=ox_delta, coverage=0.75)
    scans.append(Scan(idx, 2, 11.61, mzo, ito, mz_of(pep.mass() + ox_delta, 5), 5, "MS2 Trp-oxidised [M+5H]5+"))
    return scans


def write_demo_csvs(out_dir: str):
    """Write demo peak lists (summed MS1, main MS/MS, impurity MS/MS) as CSV for testing the CSV loader."""
    import os
    import pandas as pd
    from .analysis import sum_spectra

    scans = generate_demo_run()
    ms1 = [s for s in scans if s.ms_level == 1 and abs(s.rt - MAIN_RT) < 1.5]
    mz, it = sum_spectra(ms1, 5.0)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"mz": mz, "intensity": it}).to_csv(f"{out_dir}/demo_ms1_summed.csv", index=False)
    for s, name in [(scans[-2], "demo_msms_semaglutide_5+.csv"), (scans[-1], "demo_msms_trp_oxidised_5+.csv")]:
        pd.DataFrame({"mz": s.mz, "intensity": s.inten, "precursor_mz": s.prec_mz}).to_csv(f"{out_dir}/{name}", index=False)

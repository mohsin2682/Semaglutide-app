"""MS1 identification, impurity search, MS/MS annotation and modification localisation."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .chem import (
    C13_SPACING, PROTON, RESIDUES, AA_NAMES, GGLU, OEG, Peptide, fadd, fmass, fstr,
    fragment_table, isotope_pattern, mz_of, parse_formula,
)


# --------------------------------------------------------------------------- spectrum utils
def merge_peaks(mz, inten, ppm: float = 5.0):
    """Merge centroids closer than `ppm` (chain clustering); intensities summed, m/z intensity-weighted."""
    mz = np.asarray(mz, float)
    inten = np.asarray(inten, float)
    if mz.size == 0:
        return mz, inten
    o = np.argsort(mz)
    mz, inten = mz[o], inten[o]
    gap = np.diff(mz) > mz[:-1] * ppm * 1e-6
    grp = np.concatenate([[0], np.cumsum(gap)])
    s = np.bincount(grp, weights=inten)
    m = np.bincount(grp, weights=inten * mz) / np.where(s > 0, s, 1)
    return m, s


def pick_peaks(mz, inten, rel_height: float = 0.001):
    """Centroid profile data: local maxima with parabolic m/z refinement."""
    mz = np.asarray(mz, float)
    inten = np.asarray(inten, float)
    if mz.size < 3:
        return mz, inten
    idx, _ = find_peaks(inten, height=inten.max() * rel_height)
    out_mz = []
    for i in idx:
        if 0 < i < len(mz) - 1:
            y0, y1, y2 = inten[i - 1], inten[i], inten[i + 1]
            d = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / d if d != 0 else 0.0
            out_mz.append(mz[i] + off * (mz[i + 1] - mz[i]))
        else:
            out_mz.append(mz[i])
    return np.array(out_mz), inten[idx]


def looks_like_profile(mz, ppm_thresh: float = 25.0) -> bool:
    """Heuristic: profile data has very small, regular point spacing."""
    mz = np.sort(np.asarray(mz, float))
    if mz.size < 50:
        return False
    d = np.median(np.diff(mz) / mz[:-1]) * 1e6
    return d < ppm_thresh


def sum_spectra(scans, ppm: float = 5.0):
    if not scans:
        return np.array([]), np.array([])
    mz = np.concatenate([s.mz for s in scans])
    it = np.concatenate([s.inten for s in scans])
    return merge_peaks(mz, it, ppm)


def apply_noise_filter(mz, inten, rel_pct: float):
    if inten.size == 0:
        return mz, inten
    keep = inten >= inten.max() * rel_pct / 100.0
    return mz[keep], inten[keep]


def _window(mz, target, ppm):
    tol = target * ppm * 1e-6
    lo = np.searchsorted(mz, target - tol, "left")
    hi = np.searchsorted(mz, target + tol, "right")
    return lo, hi


def eic(scans, target_mz: float, ppm: float = 10.0):
    rt, val = [], []
    for s in scans:
        lo, hi = _window(s.mz, target_mz, ppm)
        rt.append(s.rt)
        val.append(s.inten[lo:hi].sum() if hi > lo else 0.0)
    return np.array(rt), np.array(val)


# --------------------------------------------------------------------------- envelope matching
def match_envelope(mz, inten, formula: Counter, z: int, ppm: float, n_iso: int = 8) -> dict:
    """Match a theoretical isotope envelope for [M+zH]z+ against a centroid spectrum
    (mz sorted ascending)."""
    M = fmass(formula)
    pat = isotope_pattern(formula, n_iso)
    sig = pat >= 0.10
    obs = np.zeros(n_iso)
    obs_mz = np.full(n_iso, np.nan)
    obs_idx = np.full(n_iso, -1)
    for k in range(n_iso):
        t = mz_of(M, z, k)
        lo, hi = _window(mz, t, ppm)
        if hi > lo:
            j = lo + int(np.argmax(inten[lo:hi]))
            obs[k], obs_mz[k], obs_idx[k] = inten[j], mz[j], j
    found = (obs > 0) & sig
    n_sig = int(sig.sum())
    res = dict(z=z, theo_mz=mz_of(M, z), obs_mz=np.nan, ppm=np.nan, score=0.0, intensity=0.0,
               n_found=int(found.sum()), n_sig=n_sig, mass_obs=np.nan, scale=0.0,
               obs=obs, obs_idx=obs_idx, obs_mz_all=obs_mz, pattern=pat, matched=False)
    if found.sum() < max(2, int(np.ceil(0.6 * n_sig))):
        return res
    o, p = obs[sig], pat[sig]
    denom = np.linalg.norm(o) * np.linalg.norm(p)
    score = float(o @ p / denom) if denom > 0 else 0.0
    theo = np.array([mz_of(M, z, k) for k in range(n_iso)])
    err = (obs_mz - theo) / theo * 1e6
    w = np.where(found, obs, 0)
    ppm_w = float(np.nansum(err * w) / w.sum())
    k_ref = 0 if found[0] else int(np.argmax(np.where(found, pat, -1)))
    scale = float((obs[sig] @ pat[sig]) / (pat[sig] @ pat[sig]))
    res.update(
        obs_mz=float(obs_mz[k_ref]),
        ppm=ppm_w,
        score=score,
        intensity=float(obs[obs > 0].sum()),
        mass_obs=float((obs_mz[k_ref] - PROTON) * z - k_ref * C13_SPACING),
        scale=scale,
        matched=True,
    )
    return res


def identify_main(mz, inten, pep: Peptide, zs, ppm: float, min_score: float) -> pd.DataFrame:
    rows = []
    f = pep.formula()
    for z in zs:
        m = match_envelope(mz, inten, f, z, ppm)
        ok = m["matched"] and m["score"] >= min_score and abs(m["ppm"]) <= ppm
        rows.append({
            "z": z, "Theoretical m/z": m["theo_mz"], "Observed m/z": m["obs_mz"],
            "Error (ppm)": m["ppm"], "Isotope score": m["score"],
            "Isotope peaks found": f"{m['n_found']}/{m['n_sig']}",
            "Envelope intensity": m["intensity"], "Observed mass (Da)": m["mass_obs"],
            "Pass": bool(ok), "envelope": m,
        })
    return pd.DataFrame(rows)


def subtract_envelopes(mz, inten, formula: Counter, id_df: pd.DataFrame):
    """Remove the scaled main-species envelopes from a copy of the spectrum (residual spectrum)."""
    res = inten.copy()
    for _, r in id_df[id_df["Pass"]].iterrows():
        m = r["envelope"]
        for k, j in enumerate(m["obs_idx"]):
            if j >= 0:
                res[j] = max(0.0, res[j] - m["scale"] * m["pattern"][k])
    return res


# --------------------------------------------------------------------------- impurity library
def build_library(pep: Peptide) -> pd.DataFrame:
    """Candidate impurities as formula deltas relative to the intact peptide."""
    lib = []

    def add(name, cat, delta, note=""):
        d = parse_formula(delta) if isinstance(delta, str) else delta
        lib.append(dict(name=name, category=cat, delta=d, note=note))

    # chemical modifications
    add("Oxidation (+O)", "Oxidation", "O", "Trp25 (Trp31 GLP-1 numbering) is the usual site; also Tyr/His")
    add("Dioxidation (+2O)", "Oxidation", "O2", "Trp -> N-formylkynurenine-type / dihydroxy-Trp")
    add("Kynurenine (Trp +O -C)", "Oxidation", "C-1O", "Trp -> Kyn (+3.9949 Da)")
    add("Deamidation (Gln/Asn)", "Deamidation", "N-1H-1O", "+0.984 Da; overlaps 13C peak of the intact species")
    add("Dehydration (-H2O)", "Degradation", "H-2O-1", "Asp/Glu cyclisation, succinimide")
    add("Hydrolysis (+H2O)", "Degradation", "H2O")
    add("Formylation (+CO)", "Process", "CO")
    add("Acetylation (+C2H2O)", "Process", "C2H2O", "Capping by-product")
    add("Methylation (+CH2)", "Process", "CH2")
    add("Sodium adduct [M+Na-H]", "Adduct", "Na1H-1")
    add("tert-Butyl (+C4H8)", "Process", "C4H8", "Incomplete protecting-group removal")
    # side-chain related
    add("Loss of OEG (-1x)", "Side chain", "C-6H-11N-1O-3", "Missing 8-amino-3,6-dioxaoctanoic acid")
    add("Loss of OEG (-2x)", "Side chain", "C-12H-22N-2O-6")
    add("Extra OEG (+1x)", "Side chain", "C6H11NO3")
    add("Loss of gGlu", "Side chain", "C-5H-7N-1O-3")
    add("Extra gGlu", "Side chain", "C5H7NO3")
    add("Diacid -2C (-C2H4)", "Side chain", "C-2H-4", "Shorter fatty diacid by one CH2-CH2 unit")
    add("Diacid +2C (+C2H4)", "Side chain", "C2H4", "Longer fatty diacid by one CH2-CH2 unit")
    # derive "loss of the diacid" / "loss of the whole side chain" from the peptide's own
    # attached modification, rather than assuming semaglutide's C18 diacid (works for any
    # Lys-linked fatty-diacid peptide, e.g. tirzepatide's C20 diacid, as long as it uses the
    # same gGlu + n*OEG linker chemistry)
    if pep.mods:
        pos0, (mod_name, chain_f) = next(iter(pep.mods.items()))
        add(f"Loss of complete side chain ({mod_name}, free Lys)", "Side chain",
            Counter({e: -n for e, n in chain_f.items()}))
        diacid_f = fadd(fadd(chain_f, parse_formula(GGLU), -1), parse_formula(OEG), -2)
        if diacid_f and all(n > 0 for n in diacid_f.values()):
            n_c = diacid_f.get("C", 0)
            add(f"Loss of C{n_c} diacid", "Side chain", Counter({e: -n for e, n in diacid_f.items()}),
                "Assumes the standard gGlu + 2xOEG linker; loses only the fatty-diacid portion")
    # sequence variants: single deletions / insertions / substitutions
    res_f = {i: pep.residue_formula(i) for i in range(pep.n)}
    groups: dict = {}
    for i in range(pep.n):
        neg = Counter({e: -n for e, n in res_f[i].items()})
        groups.setdefault(fstr(neg), []).append((i, neg))
    for _, members in groups.items():
        names = "/".join(f"{AA_NAMES[pep.seq[i]]}{i + 1}" for i, _ in members)
        label = f"des-{names}" if len(members) == 1 else f"des-residue ({names})"
        add(label, "Deletion", members[0][1], "Single residue deletion (position cannot be assigned by MS1 alone)"
            if len(members) > 1 else f"Deletion of residue {members[0][0] + 1}")
    for a in "GAVLSTDEKRHFYWQNPMC":
        add(f"ins-{AA_NAMES[a]}", "Insertion", RESIDUES[a], "Double coupling / insertion")
    add("ins-Aib", "Insertion", RESIDUES["B"])
    for (s, t, d) in [("B", "A", "Aib->Ala"), ("B", "G", "Aib->Gly"), ("A", "B", "Ala->Aib"),
                      ("D", "N", "Asp->Asn"), ("E", "Q", "Glu->Gln")]:
        if s in pep.seq:
            delta = fadd(parse_formula(RESIDUES[t]), parse_formula(RESIDUES[s]), -1)
            add(d, "Substitution", delta)
    # terminal truncations (2..3 residues)
    for k in (2, 3):
        f: Counter = Counter()
        for i in range(k):
            f = fadd(f, res_f[i], -1)
        add(f"N-term truncation (des 1-{k})", "Truncation", f)
        f = Counter()
        for i in range(pep.n - k, pep.n):
            f = fadd(f, res_f[i], -1)
        add(f"C-term truncation (des {pep.n - k + 1}-{pep.n})", "Truncation", f)
    df = pd.DataFrame([r for r in lib if r])
    df["delta_mass"] = [fmass(d) for d in df["delta"]]
    df["delta_formula"] = [fstr(d) for d in df["delta"]]
    df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
    return df



def overlap_fit(mz, inten, main_f: Counter, cand_f: Counter, z: int, ppm: float, d_int: int, n_iso: int = 8):
    """Two-component NNLS for a candidate whose envelope is shifted by ~d_int x 13C spacing relative to the
    main species (unresolved by QTOF, e.g. deamidation +0.984 Da). Returns dict or None."""
    from scipy.optimize import nnls

    M = fmass(main_f)
    pm = isotope_pattern(main_f, n_iso)
    pc = isotope_pattern(cand_f, n_iso)
    K = n_iso + d_int
    obs = np.zeros(K)
    for k in range(K):
        lo, hi = _window(mz, mz_of(M, z, k), ppm)
        if hi > lo:
            obs[k] = inten[lo:hi].max()
    A = np.zeros((K, 2))
    A[:n_iso, 0] = pm
    A[d_int:, 1] = pc[:K - d_int]
    coef, _ = nnls(A, obs)
    a, b = coef
    if a <= 0:
        return None
    r_full = np.linalg.norm(A @ coef - obs)
    a1 = float(A[:, 0] @ obs / (A[:, 0] @ A[:, 0]))
    r_main = np.linalg.norm(a1 * A[:, 0] - obs)
    fit = A @ coef
    cos = float(fit @ obs / (np.linalg.norm(fit) * np.linalg.norm(obs) + 1e-12))
    return dict(rel=100 * b / a, improvement=r_full / (r_main + 1e-12), score=cos)


def search_impurities(mz, inten, pep: Peptide, id_df: pd.DataFrame, zs, ppm: float,
                      min_score: float, min_rel_pct: float, include_dimer: bool = True) -> pd.DataFrame:
    main_f = pep.formula()
    main_M = fmass(main_f)
    residual = subtract_envelopes(mz, inten, main_f, id_df)
    good_z = [int(r.z) for r in id_df.itertuples() if r.Pass]
    main_int = {int(r.z): r.envelope["obs"].max() for r in id_df.itertuples() if r.Pass}
    if not good_z:
        return pd.DataFrame()
    lib = build_library(pep)
    cands = [(r.name, r.category, fadd(main_f, r.delta), r.delta_formula, r.note) for r in lib.itertuples()]
    if include_dimer:
        d = Counter({e: 2 * n for e, n in main_f.items()})
        cands.append(("Dimer (2M)", "Aggregate", d, "2M", "Non-covalent/covalent dimer"))
    out = []
    for name, cat, f, dform, note in cands:
        M = fmass(f)
        hits = []
        zrange = zs if name != "Dimer (2M)" else [z * 2 for z in zs] + [z * 2 - 1 for z in zs]
        d_int = int(round((M - main_M) / C13_SPACING))
        overlap = 0 < d_int <= 3 and abs((M - main_M) - d_int * C13_SPACING) < 0.05
        if overlap:
            fits = []
            for z in good_z:
                r = overlap_fit(mz, inten, main_f, f, z, ppm, d_int)
                if r and r["improvement"] < 0.6 and r["rel"] >= min_rel_pct and r["score"] >= min_score:
                    fits.append((z, r))
            if fits:
                out.append({
                    "Impurity": name, "Category": cat, "Δ formula": dform,
                    "Δ mass (Da)": M - main_M, "Theoretical mass (Da)": M,
                    "Observed mass (Da)": np.nan, "Error (ppm)": np.nan,
                    "Charge states": ", ".join(str(z) for z, _ in fits), "# z": len(fits),
                    "Isotope score": float(np.mean([r["score"] for _, r in fits])),
                    "Relative abundance (%)": float(np.mean([r["rel"] for _, r in fits])),
                    "Confidence": "High" if len(fits) >= 2 else "Tentative (1 charge state)",
                    "Note": (note + "; " if note else "") + "estimated by isotope-overlap (NNLS) fit - not resolved from intact species",
                })
            continue
        for z in sorted(set(zrange)):
            m = match_envelope(mz, residual, f, z, ppm)
            if m["matched"] and m["score"] >= min_score and abs(m["ppm"]) <= ppm:
                # relative abundance vs main at the same z (fallback: mean of main)
                ref = main_int.get(z, np.mean(list(main_int.values())))
                m["rel"] = 100 * m["obs"].max() / ref
                hits.append(m)
        if not hits:
            continue
        rel = float(np.mean([h["rel"] for h in hits]))
        if rel < min_rel_pct:
            continue
        out.append({
            "Impurity": name, "Category": cat, "Δ formula": dform,
            "Δ mass (Da)": M - main_M, "Theoretical mass (Da)": M,
            "Observed mass (Da)": float(np.mean([h["mass_obs"] for h in hits])),
            "Error (ppm)": float(np.mean([h["ppm"] for h in hits])),
            "Charge states": ", ".join(str(h["z"]) for h in hits),
            "# z": len(hits), "Isotope score": float(np.mean([h["score"] for h in hits])),
            "Relative abundance (%)": rel,
            "Confidence": "High" if len(hits) >= 2 else "Tentative (1 charge state)",
            "Note": note,
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    # flag isobaric alternatives (same mass within 0.005 Da)
    alt = []
    for i, r in df.iterrows():
        same = df[(abs(df["Theoretical mass (Da)"] - r["Theoretical mass (Da)"]) < 0.005) & (df.index != i)]
        alt.append(", ".join(same["Impurity"]))
    df["Isobaric alternatives"] = alt
    return df.sort_values("Relative abundance (%)", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- MS/MS
def annotate_msms(mz, inten, pep: Peptide, ion_types, max_charge: int, ppm: float,
                  min_rel_pct: float = 0.5, frag_cache=None):
    """Match experimental peaks to theoretical fragments. Returns (matches df, coverage dict)."""
    mz = np.asarray(mz, float)
    inten = np.asarray(inten, float)
    order = np.argsort(mz)
    mz, inten = mz[order], inten[order]
    keep = inten >= inten.max() * min_rel_pct / 100.0 if inten.size else np.array([], bool)
    mz, inten = mz[keep], inten[keep]
    frag = fragment_table(pep, ion_types, max_charge) if frag_cache is None else frag_cache
    fm = frag["mz"].values
    rows = []
    for m, it in zip(mz, inten):
        lo, hi = _window(fm, m, ppm)
        if hi <= lo:
            continue
        cand = frag.iloc[lo:hi].copy()
        cand["err"] = (m - cand["mz"]) / cand["mz"] * 1e6
        prim = cand["type"].isin(["b", "y"]).astype(int)
        cand = cand.assign(_p=-prim, _e=cand["err"].abs()).sort_values(["_p", "_e"]).iloc[0]
        rows.append(dict(mz=m, intensity=it, ion=cand["label"], type=cand["type"], series=cand["series"],
                         index=int(cand["index"]), z=int(cand["z"]), theo_mz=cand["mz"], ppm=cand["err"]))
    df = pd.DataFrame(rows)
    n = pep.n
    b_sites, y_sites = set(), set()
    if not df.empty:
        b_sites = set(df[df["type"] == "b"]["index"])
        y_sites = set(n - df[df["type"] == "y"]["index"])
    cov_sites = b_sites | y_sites
    cov = dict(
        b_sites=sorted(b_sites), y_sites=sorted(y_sites), covered=sorted(cov_sites),
        coverage_pct=100 * len(cov_sites) / (n - 1),
        matched_peaks=len(df), total_peaks=int(len(mz)),
        matched_int_pct=100 * df["intensity"].sum() / inten.sum() if len(df) and inten.sum() else 0.0,
        median_abs_ppm=float(df["ppm"].abs().median()) if len(df) else np.nan,
    )
    return df, cov


def localize_shift(mz, inten, pep: Peptide, delta_mass: float, ppm: float, max_charge: int = 3,
                   min_rel_pct: float = 0.5) -> tuple[pd.DataFrame, tuple[int, int] | None]:
    """For each b/y ion check whether the unshifted or the delta-shifted form is present; infer the
    residue range carrying the modification (assumes a single, localised modification)."""
    mz = np.asarray(mz, float)
    inten = np.asarray(inten, float)
    o = np.argsort(mz)
    mz, inten = mz[o], inten[o]
    thr = inten.max() * min_rel_pct / 100.0
    frag = fragment_table(pep, ("b", "y"), max_charge)
    rows = []
    n = pep.n
    for (t, i), g in frag.groupby(["type", "index"]):
        un = sh = False
        for r in g.itertuples():
            for shifted in (False, True):
                target = r.mz + (delta_mass / r.z if shifted else 0.0)
                lo, hi = _window(mz, target, ppm)
                if hi > lo and inten[lo:hi].max() >= thr:
                    if shifted:
                        sh = True
                    else:
                        un = True
        rows.append(dict(type=t, index=int(i), unshifted=un, shifted=sh))
    df = pd.DataFrame(rows)
    lower, upper = 1, n
    for r in df.itertuples():
        if r.type == "b":
            if r.unshifted and not r.shifted:
                lower = max(lower, r.index + 1)
            if r.shifted and not r.unshifted:
                upper = min(upper, r.index)
        else:
            if r.unshifted and not r.shifted:
                upper = min(upper, n - r.index)
            if r.shifted and not r.unshifted:
                lower = max(lower, n - r.index + 1)
    if not (df["shifted"].any()):
        return df, None
    return df, (lower, upper)

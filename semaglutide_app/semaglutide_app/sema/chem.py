"""Peptide chemistry: formulas, masses, isotope patterns and fragment ions.

Semaglutide is represented with one-letter codes; ``B`` = Aib (2-aminoisobutyric acid).
The Lys26 (GLP-1 numbering; position 20 of the 31-mer) side chain is a *modification*
of the residue: C18 diacid - gamma-Glu - 2x OEG.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- constants
MONO = {
    "C": 12.0,
    "H": 1.00782503207,
    "N": 14.0030740048,
    "O": 15.99491461956,
    "S": 31.97207100,
    "Na": 22.9897692809,
}
# nominal shift -> abundance
ISOTOPES = {
    "C": [(0, 0.9893), (1, 0.0107)],
    "H": [(0, 0.999885), (1, 0.000115)],
    "N": [(0, 0.99636), (1, 0.00364)],
    "O": [(0, 0.99757), (1, 0.00038), (2, 0.00205)],
    "S": [(0, 0.9499), (1, 0.0075), (2, 0.0425), (4, 0.0001)],
    "Na": [(0, 1.0)],
}
PROTON = 1.007276466812
C13_SPACING = 1.0033548378
CO = 27.9949146196
NH3 = 17.0265491
H2O_MASS = 18.0105646863

RESIDUES = {  # residue (amino acid - H2O) formulas
    "G": "C2H3NO", "A": "C3H5NO", "S": "C3H5NO2", "P": "C5H7NO", "V": "C5H9NO",
    "T": "C4H7NO2", "C": "C3H5NOS", "L": "C6H11NO", "I": "C6H11NO", "N": "C4H6N2O2",
    "D": "C4H5NO3", "Q": "C5H8N2O2", "K": "C6H12N2O", "E": "C5H7NO3", "M": "C5H9NOS",
    "H": "C6H7N3O", "F": "C9H9NO", "R": "C6H12N4O", "Y": "C9H9NO2", "W": "C11H10N2O",
    "B": "C4H7NO",  # Aib
}
AA_NAMES = {
    "G": "Gly", "A": "Ala", "S": "Ser", "P": "Pro", "V": "Val", "T": "Thr", "C": "Cys",
    "L": "Leu", "I": "Ile", "N": "Asn", "D": "Asp", "Q": "Gln", "K": "Lys", "E": "Glu",
    "M": "Met", "H": "His", "F": "Phe", "R": "Arg", "Y": "Tyr", "W": "Trp", "B": "Aib",
}

SEMAGLUTIDE_SEQ = "HBEGTFTSDVSSYLEGQAAKEFIAWLVRGRG"  # B = Aib
SEMAGLUTIDE_LYS_POS = 20  # 1-based (Lys26 in GLP-1(7-37) numbering)
GLP1_OFFSET = 6  # position + 6 = GLP-1 numbering

# side chain building blocks (as formulas *added* to the Lys residue)
OEG = "C6H11NO3"          # 8-amino-3,6-dioxaoctanoic acid residue
GGLU = "C5H7NO3"          # gamma-Glu residue
DIACID = "C18H32O3"       # HOOC-(CH2)16-CO- replacing an H
SEMA_SIDECHAIN = "C35H61N3O12"  # = DIACID + GGLU + 2*OEG


# --------------------------------------------------------------------------- formulas
def parse_formula(text: str) -> Counter:
    """'C6H11NO3' -> Counter. Negative counts are allowed via a leading '-' per element: 'C-1H-2'."""
    out: Counter = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(-?\d*)", text.replace(" ", "")):
        if not el:
            continue
        if el not in MONO:
            raise ValueError(f"Unknown element '{el}' in formula '{text}'")
        out[el] += int(n) if n not in ("", "-") else 1
    return out


def fadd(a: Counter, b: Counter, k: int = 1) -> Counter:
    out = Counter(a)
    for el, n in b.items():
        out[el] += k * n
    return Counter({el: n for el, n in out.items() if n != 0})


def fmass(f: Counter) -> float:
    return float(sum(MONO[el] * n for el, n in f.items()))


def fstr(f: Counter) -> str:
    order = ["C", "H", "N", "O", "S", "Na"]
    return "".join(f"{el}{f[el] if f[el] != 1 else ''}" for el in order if f.get(el, 0))


H2O = parse_formula("H2O")


def isotope_pattern(formula: Counter, n: int = 12) -> np.ndarray:
    """Coarse (nominal-mass) isotope distribution, normalised to a maximum of 1."""
    pat = np.zeros(n)
    pat[0] = 1.0
    for el, cnt in formula.items():
        cnt = int(cnt)
        if cnt <= 0:
            continue
        base = np.zeros(n)
        for shift, ab in ISOTOPES[el]:
            if shift < n:
                base[shift] += ab
        res = np.zeros(n)
        res[0] = 1.0
        p = base.copy()
        while cnt:
            if cnt & 1:
                res = np.convolve(res, p)[:n]
            p = np.convolve(p, p)[:n]
            cnt >>= 1
        pat = np.convolve(pat, res)[:n]
        pat /= pat.sum()
    return pat / pat.max()


def mz_of(mass: float, z: int, k: int = 0) -> float:
    """m/z of the [M+zH]z+ ion for isotope k (k x 13C spacing)."""
    return (mass + k * C13_SPACING + z * PROTON) / z


# --------------------------------------------------------------------------- peptide
@dataclass
class Peptide:
    seq: str
    mods: dict = field(default_factory=dict)  # pos(1-based) -> (name, Counter)
    name: str = "peptide"

    def __post_init__(self):
        bad = [c for c in self.seq if c not in RESIDUES]
        if bad:
            raise ValueError(f"Unsupported residue(s): {sorted(set(bad))}")

    @property
    def n(self) -> int:
        return len(self.seq)

    def residue_formula(self, i: int) -> Counter:
        f = parse_formula(RESIDUES[self.seq[i]])
        if (i + 1) in self.mods:
            f = fadd(f, self.mods[i + 1][1])
        return f

    def formula(self) -> Counter:
        f: Counter = Counter()
        for i in range(self.n):
            f = fadd(f, self.residue_formula(i))
        return fadd(f, H2O)

    def mass(self) -> float:
        return fmass(self.formula())

    def residue_masses(self) -> np.ndarray:
        return np.array([fmass(self.residue_formula(i)) for i in range(self.n)])

    def label(self, i: int) -> str:
        """Residue label in GLP-1 numbering for semaglutide-like sequences, e.g. 'Trp31'."""
        return f"{AA_NAMES[self.seq[i]]}{i + 1}"


def semaglutide(with_sidechain: bool = True, extra_mod: str | None = None) -> Peptide:
    mods = {}
    if with_sidechain:
        mods[SEMAGLUTIDE_LYS_POS] = ("C18 diacid-gGlu-2xOEG", parse_formula(SEMA_SIDECHAIN))
    return Peptide(SEMAGLUTIDE_SEQ, mods, "semaglutide")


# --------------------------------------------------------------------------- fragments
ION_TYPES = {
    # name: (series, neutral mass offset added to the residue sum)
    "a": ("N", -CO),
    "b": ("N", 0.0),
    "b-H2O": ("N", -H2O_MASS),
    "b-NH3": ("N", -NH3),
    "y": ("C", H2O_MASS),
    "y-H2O": ("C", 0.0),
    "y-NH3": ("C", H2O_MASS - NH3),
}


def fragment_table(pep: Peptide, ion_types=("b", "y"), max_charge: int = 4) -> pd.DataFrame:
    res = pep.residue_masses()
    n = pep.n
    rows = []
    for t in ion_types:
        series, off = ION_TYPES[t]
        for i in range(1, n):
            base = res[:i].sum() if series == "N" else res[n - i:].sum()
            neutral = base + off
            for z in range(1, max_charge + 1):
                rows.append((t, series, i, z, (neutral + z * PROTON) / z))
    df = pd.DataFrame(rows, columns=["type", "series", "index", "z", "mz"])
    ch = {1: "⁺", 2: "²⁺", 3: "³⁺", 4: "⁴⁺", 5: "⁵⁺", 6: "⁶⁺"}
    df["label"] = df["type"] + df["index"].astype(str) + df["z"].map(lambda z: ch.get(z, f"({z}+)"))
    return df.sort_values("mz").reset_index(drop=True)

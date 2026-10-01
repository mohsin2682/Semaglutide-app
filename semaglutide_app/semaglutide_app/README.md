# Peptide QTOF Analyzer

Streamlit dashboard for **identification (MS1)**, **sequence confirmation (MS/MS)** and **impurity profiling**
of peptide drugs from QTOF data.

```
pip install -r requirements.txt
streamlit run app.py
```

Pick *Demo data* in the sidebar to try everything immediately (synthetic LC-MS/MS run with spiked impurities).
`demo_data/` holds the same demo as CSV peak lists to test the CSV loader.

## Inputs
| Format | Notes |
|---|---|
| mzML / mzXML | Preferred. MS1 + MS2 scans, RT and precursor m/z/charge are read automatically. |
| CSV / TSV | Columns `mz`, `intensity` (+ optional `scan`, `rt`, `ms_level`, `precursor_mz`, `precursor_z`). |
| .d / .wiff / .raw | Convert first: `python convert_raw.py sample.d` (ProteoWizard MSConvert, centroid, MS1-2). |

## What it does
1. **Run overview** - TIC + EIC of [M+5H]5+, choose the RT window that is summed (whole run for impurity work).
2. **MS1 identification** - for each charge state (default 3-8) the theoretical isotope envelope of
   C187H291N45O59 (monoisotopic 4111.1154 Da) is matched: isotope-pattern score, ppm error, observed neutral mass.
   Identity is *confirmed* when >= 2 charge states pass.
3. **Sequence confirmation** - b/y (optionally a, -H2O, -NH3) fragment annotation with charge up to z-1, sequence
   coverage map, ppm error stats. The Lys26 side chain (C18 diacid-gGlu-2xOEG, +C35H61N3O12) and Aib8 (`B`) are built in.
   The *localisation* panel tests unshifted vs shifted fragments to place a mass shift (e.g. +15.995 on Trp31).
4. **Impurities** - the main envelope is subtracted, then ~100 candidates (oxidation, deamidation, side-chain
   truncations/extensions, residue deletions/insertions, Aib->Ala, terminal truncations, adducts, dimer) are searched
   by isotope-envelope matching. Deamidation (+0.984 Da, unresolved from the 13C peak) is estimated by a two-component
   isotope-overlap fit.
5. **Report** - Excel workbook (summary, MS1, impurities, MS/MS matches, coverage, theoretical fragments).

## Numbering
Internal positions are 1-31 of the peptide (H1 = His7 in GLP-1 numbering): Aib = pos 2 (Aib8), Lys(side chain) = pos 20
(Lys26), Trp = pos 25 (Trp31).

## Limits / notes
* MS1 relative abundance is a signal ratio, not corrected for ionisation efficiency - not for reportable purity.
* Isobaric variants (Ile/Leu, Asp isomerisation, positional oxidation) need MS/MS and retention time.
* Thresholds (ppm, isotope score, coverage %) are starting values - tune to your instrument and validate on your own data.
* CID of semaglutide produces side-chain neutral losses; use the a/-H2O/-NH3 ion options to explore them.

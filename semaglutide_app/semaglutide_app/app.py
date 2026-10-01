"""Peptide drugs QTOF dashboard: MS1 identification, MS/MS sequence confirmation, impurity profiling.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from sema.analysis import (
    annotate_msms, apply_noise_filter, build_library, eic, identify_main, localize_shift,
    search_impurities, sum_spectra,
)
from sema.chem import (
    AA_NAMES, C13_SPACING, PROTON, SEMAGLUTIDE_LYS_POS, SEMAGLUTIDE_SEQ, Peptide, fmass, fstr,
    isotope_pattern, mz_of, parse_formula, semaglutide, SEMA_SIDECHAIN, fragment_table,
    TIRZEPATIDE_SEQ, TIRZEPATIDE_LYS_POS, TIRZEPATIDE_SIDECHAIN, tirzepatide,
)
from sema.demo import generate_demo_run
from sema.io import load_run

st.set_page_config(page_title="Peptide Drugs QTOF Analyzer", page_icon="🧬", layout="wide")

BLUE, ORANGE, GREY, RED, GREEN = "#2a6fbb", "#d9822b", "#8a8f98", "#c0392b", "#2e8b57"


# ============================================================================ helpers
def stem(mz, inten):
    x, y = [], []
    for m, i in zip(mz, inten):
        x += [m, m, None]
        y += [0, i, None]
    return x, y


def spectrum_fig(mz, inten, title="", xr=None, height=380):
    fig = go.Figure()
    x, y = stem(mz, inten)
    fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", line=dict(color=GREY, width=1), name="peaks", hoverinfo="skip"))
    fig.add_trace(go.Scattergl(x=mz, y=inten, mode="markers", marker=dict(size=3, color=GREY), name="peaks",
                               hovertemplate="m/z %{x:.4f}<br>int %{y:.3g}<extra></extra>", showlegend=False))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="m/z", yaxis_title="Intensity", template="plotly_white")
    if xr:
        fig.update_xaxes(range=xr)
    return fig


def fmt(df, cols_2=(), cols_4=(), cols_0=()):
    df = df.copy()
    for c in cols_2:
        if c in df:
            df[c] = df[c].astype(float).round(2)
    for c in cols_4:
        if c in df:
            df[c] = df[c].astype(float).round(4)
    for c in cols_0:
        if c in df:
            df[c] = df[c].astype(float).round(0)
    return df


@st.cache_data(show_spinner="Reading file…")
def cached_load(data: bytes, name: str, pick: bool):
    return load_run(data, name, pick)


@st.cache_data(show_spinner="Generating demo run…")
def cached_demo():
    return generate_demo_run()


# ============================================================================ sidebar
with st.sidebar:
    st.title("🧬 Semaglutide QTOF")
    st.caption("MS1 identification · MS/MS sequence confirmation · impurity profiling")

    st.subheader("1 · Data")
    source = st.radio("Source", ["Demo data (synthetic)", "Upload files"], label_visibility="collapsed")
    up_run = up_ms2 = None
    pick = st.checkbox("Peak-pick profile data", True, help="Only applied if data look like profile mode.")
    if source == "Upload files":
        up_run = st.file_uploader("LC-MS run / spectrum (mzML, mzXML, CSV)", type=["mzml", "mzxml", "csv", "txt", "tsv"])
        up_ms2 = st.file_uploader("MS/MS spectrum (optional, CSV/mzML)", type=["mzml", "mzxml", "csv", "txt", "tsv"])
        st.caption("Vendor .d / .wiff files: convert to mzML with MSConvert first (see README).")

    st.subheader("2 · Peptide")
    PRESETS = {
        "Semaglutide": dict(seq=SEMAGLUTIDE_SEQ, k=SEMAGLUTIDE_LYS_POS, chain=SEMA_SIDECHAIN,
                           chain_name="C18 diacid-γGlu-2×OEG", cterm=False),
        "Tirzepatide": dict(seq=TIRZEPATIDE_SEQ, k=TIRZEPATIDE_LYS_POS, chain=TIRZEPATIDE_SIDECHAIN,
                           chain_name="C20 diacid-γGlu-2×AEEA", cterm=True),
        "Custom / other": None,
    }
    preset = st.selectbox("Peptide preset", list(PRESETS), index=0)
    cfg = PRESETS[preset]
    if st.session_state.get("_last_preset") != preset:
        st.session_state["seq_input"] = cfg["seq"] if cfg else SEMAGLUTIDE_SEQ
        st.session_state["k_pos_input"] = cfg["k"] if cfg else 0
        st.session_state["cterm_input"] = cfg["cterm"] if cfg else False
        st.session_state["side_input"] = bool(cfg)
        st.session_state["extra_mod_input"] = ""
        st.session_state["_last_preset"] = preset

    seq = st.text_input("Sequence (B = Aib)", key="seq_input").strip().upper()
    if cfg:
        side = st.checkbox(f"Include lipidated side chain: {cfg['chain_name']}", key="side_input")
    else:
        side = False
    k_pos = st.number_input("Modified-residue position (1-based; 0 = none)", 0, 300, step=1, key="k_pos_input",
                            help="Position of the residue (usually Lys) carrying the side chain / extra modification. "
                                 "Auto-filled for the presets above; set manually for a custom sequence.")
    cterm_amide = st.checkbox("C-terminal amide (unchecked = free-acid C-terminus)", key="cterm_input")
    extra_mod = st.text_input("Extra modification formula (optional)", key="extra_mod_input",
                              help="Elemental formula added at the modified-residue position, e.g. C2H2O")
    try:
        mods = {}
        if side and cfg and k_pos > 0:
            mods[k_pos] = ("side chain", parse_formula(cfg["chain"]))
        pep = Peptide(seq, mods, "peptide", cterm_amide=cterm_amide)
        if extra_mod.strip() and k_pos > 0:
            from sema.chem import fadd
            base = mods.get(k_pos, ("", parse_formula("")))[1]
            pep.mods[k_pos] = ("side chain+extra", fadd(base, parse_formula(extra_mod)))
    except Exception as e:  # noqa
        st.error(str(e))
        st.stop()

    st.subheader("3 · Parameters")
    ppm1 = st.number_input("MS1 tolerance (ppm)", 1.0, 50.0, 10.0, 1.0)
    ppm2 = st.number_input("MS/MS tolerance (ppm)", 1.0, 100.0, 20.0, 1.0)
    zmin, zmax = st.slider("MS1 charge states", 2, 12, (3, 8))
    min_score = st.slider("Min. isotope-pattern score", 0.5, 1.0, 0.85, 0.01)
    noise_pct = st.number_input("MS1 noise cut-off (% of base peak)", 0.0, 5.0, 0.02, 0.01, format="%.2f")
    merge_ppm = st.number_input("Merge peaks when summing scans (ppm)", 0.5, 20.0, 5.0, 0.5)
    min_imp = st.number_input("Report impurities ≥ (% of main)", 0.0, 10.0, 0.1, 0.05)
    ion_types = st.multiselect("MS/MS ion types", ["b", "y", "a", "b-H2O", "y-H2O", "b-NH3", "y-NH3"], ["b", "y"])
    ms2_min_rel = st.number_input("MS/MS min. peak (% of base)", 0.0, 10.0, 0.5, 0.1)
    cov_pass = st.slider("Sequence-coverage pass criterion (%)", 30, 100, 60)

zs = list(range(zmin, zmax + 1))

# ============================================================================ load data
scans = None
try:
    if source == "Demo data (synthetic)":
        scans = cached_demo()
    elif up_run is not None:
        scans = cached_load(up_run.getvalue(), up_run.name, pick)
except Exception as e:  # noqa
    st.error(f"Could not read file: {e}")
    st.stop()

extra_ms2 = []
if source == "Upload files" and up_ms2 is not None:
    try:
        extra_ms2 = [s for s in cached_load(up_ms2.getvalue(), up_ms2.name, pick)]
    except Exception as e:  # noqa
        st.error(f"Could not read MS/MS file: {e}")

st.title("Semaglutide identification, sequence confirmation & impurity analysis")
if scans is None:
    st.info("Upload an LC-MS file in the sidebar, or switch to **Demo data** to explore the app.")
    st.stop()

ms1 = [s for s in scans if s.ms_level == 1]
ms2 = [s for s in scans if s.ms_level >= 2] + [s for s in extra_ms2]
M0 = pep.mass()
formula = pep.formula()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Formula", fstr(formula))
c2.metric("Monoisotopic mass", f"{M0:.4f} Da")
c3.metric("Average-mass check", f"{sum({'C':12.0107,'H':1.00794,'N':14.0067,'O':15.9994,'S':32.065,'Na':22.98977}[e]*n for e,n in formula.items()):.2f} Da")
c4.metric("m/z (5+ / 4+ / 6+)", " / ".join(f"{mz_of(M0, z):.3f}" for z in (5, 4, 6)))

# ============================================================================ RT window / summed MS1
tab_run, tab_id, tab_seq, tab_imp, tab_rep = st.tabs(
    ["① Run overview", "② MS1 identification", "③ Sequence confirmation (MS/MS)", "④ Impurities", "⑤ Report"])

with tab_run:
    if len(ms1) > 1:
        rt_all = np.array([s.rt for s in ms1])
        tic = np.array([s.inten.sum() for s in ms1])
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=rt_all, y=tic, name="TIC (MS1)", line=dict(color=GREY)))
        best_z = 5 if 5 in zs else zs[len(zs) // 2]
        rt_e, v_e = eic(ms1, mz_of(M0, best_z), ppm1)
        fig.add_trace(go.Scatter(x=rt_e, y=v_e, name=f"EIC [M+{best_z}H]{best_z}+ {mz_of(M0, best_z):.4f}", line=dict(color=BLUE)))
        for s in ms2:
            fig.add_vline(x=s.rt, line=dict(color=ORANGE, dash="dot", width=1))
        fig.update_layout(height=320, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="Retention time (min)", yaxis_title="Intensity",
                          title="Chromatograms (dotted = MS/MS scans)")
        st.plotly_chart(fig, use_container_width=True)
        apex = rt_e[np.argmax(v_e)] if v_e.max() > 0 else rt_all[np.argmax(tic)]
        mode = st.radio("MS1 summing window", ["Whole run (best for impurity profiling)", "Around main-peak apex", "Custom"],
                        horizontal=True)
        if mode.startswith("Whole"):
            rt_lo, rt_hi = rt_all.min(), rt_all.max()
        elif mode.startswith("Around"):
            hw = st.slider("± half-width (min)", 0.02, 1.0, 0.15, 0.01)
            rt_lo, rt_hi = apex - hw, apex + hw
        else:
            rt_lo, rt_hi = st.slider("RT window (min)", float(rt_all.min()), float(rt_all.max()),
                                     (float(apex - 0.2), float(apex + 0.2)))
        sel = [s for s in ms1 if rt_lo <= s.rt <= rt_hi]
        st.caption(f"{len(sel)} MS1 scans summed ({rt_lo:.2f}–{rt_hi:.2f} min); main peak apex ≈ {apex:.2f} min")
    else:
        sel = ms1
        rt_lo = rt_hi = 0.0
        st.caption("Single MS1 spectrum loaded.")
    if not sel:
        st.warning("No MS1 scans in the selected window.")
        st.stop()
    mz_s, it_s = sum_spectra(sel, merge_ppm)
    mz_s, it_s = apply_noise_filter(mz_s, it_s, noise_pct)
    st.write(f"Summed MS1 spectrum: **{len(mz_s):,}** centroids after noise filter · "
             f"**{len(ms2)}** MS/MS spectra available")
    if ms2:
        st.dataframe(pd.DataFrame([{"Scan": s.label or s.idx, "RT (min)": round(s.rt, 3),
                                    "Precursor m/z": None if s.prec_mz is None else round(s.prec_mz, 4),
                                    "z": s.prec_z, "Peaks": len(s.mz)} for s in ms2]),
                     hide_index=True, use_container_width=True)

# ============================================================================ MS1 identification
id_df = identify_main(mz_s, it_s, pep, zs, ppm1, min_score)
passed = id_df[id_df["Pass"]]
with tab_id:
    n_ok = len(passed)
    mean_ppm = passed["Error (ppm)"].mean() if n_ok else np.nan
    mass_obs = passed["Observed mass (Da)"].mean() if n_ok else np.nan
    ms1_ok = n_ok >= 2 and abs(mean_ppm) <= ppm1
    (st.success if ms1_ok else st.error)(
        f"MS1 identity {'CONFIRMED' if ms1_ok else 'NOT confirmed'}: {n_ok} charge state(s) matched "
        f"(≥2 required), mean error {mean_ppm:+.2f} ppm, observed monoisotopic mass "
        f"{mass_obs:.4f} Da vs theoretical {M0:.4f} Da." if n_ok else
        "No charge state matched the theoretical semaglutide isotope envelope.")
    a, b, c = st.columns(3)
    a.metric("Charge states matched", f"{n_ok} / {len(zs)}")
    b.metric("Mean mass error", "–" if not n_ok else f"{mean_ppm:+.2f} ppm")
    c.metric("Mean isotope score", "–" if not n_ok else f"{passed['Isotope score'].mean():.3f}")

    fig = spectrum_fig(mz_s, it_s, "Summed MS1 spectrum (labelled = semaglutide charge states)")
    for _, r in passed.iterrows():
        fig.add_annotation(x=r["Observed m/z"], y=float(np.max(r["envelope"]["obs"])), text=f"{int(r['z'])}+",
                           showarrow=True, arrowhead=0, ay=-25, font=dict(color=BLUE, size=13), arrowcolor=BLUE)
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    show = id_df.drop(columns=["envelope"])
    st.dataframe(fmt(show, cols_4=["Theoretical m/z", "Observed m/z", "Observed mass (Da)"],
                     cols_2=["Error (ppm)"]).style.format({"Isotope score": "{:.3f}", "Envelope intensity": "{:.3g}"}),
                 hide_index=True, use_container_width=True)

    zsel = st.selectbox("Isotope envelope check for charge state", [int(z) for z in id_df["z"]],
                        index=min(2, len(id_df) - 1))
    row = id_df[id_df["z"] == zsel].iloc[0]["envelope"]
    theo_x = [mz_of(M0, zsel, k) for k in range(len(row["pattern"]))]
    scale = row["obs"].max() if row["obs"].max() > 0 else 1
    figi = go.Figure()
    figi.add_trace(go.Bar(x=theo_x, y=row["pattern"] * scale, name="Theoretical", marker_color=ORANGE, opacity=0.6, width=0.6 / zsel))
    figi.add_trace(go.Bar(x=[m if not np.isnan(m) else t for m, t in zip(row["obs_mz_all"], theo_x)],
                          y=row["obs"], name="Observed", marker_color=BLUE, opacity=0.8, width=0.3 / zsel))
    figi.update_layout(barmode="overlay", height=300, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
                       xaxis_title="m/z", yaxis_title="Intensity",
                       title=f"[M+{zsel}H]{zsel}+ isotope envelope · score {row['score']:.3f}")
    st.plotly_chart(figi, use_container_width=True)

# ============================================================================ impurities (computed once)
imp_df = search_impurities(mz_s, it_s, pep, id_df, zs, ppm1, min_score, min_imp) if n_ok else pd.DataFrame()

# ============================================================================ MS/MS
cov_result = None
ms2_df = pd.DataFrame()
with tab_seq:
    if not ms2:
        st.info("No MS/MS spectra found. Upload an MS/MS peak list (CSV) or an mzML with MS2 scans.")
    else:
        labels = [f"{s.label or s.idx} · RT {s.rt:.2f} · prec {s.prec_mz:.4f}" if s.prec_mz else f"{s.label or s.idx} · RT {s.rt:.2f}"
                  for s in ms2]
        pick_i = st.selectbox("MS/MS spectrum", range(len(ms2)), format_func=lambda i: labels[i])
        s2 = ms2[pick_i]
        auto_z = s2.prec_z or 5
        zprec = st.number_input("Precursor charge", 2, 12, int(auto_z))
        maxfz = st.slider("Max fragment charge", 1, 6, int(min(3, max(1, zprec - 1))))
        if s2.prec_mz:
            neutral = (s2.prec_mz - PROTON) * zprec
            delta = neutral - M0
            lib = build_library(pep)
            near = lib.iloc[(lib["delta_mass"] - delta).abs().argsort()[:1]].iloc[0]
            txt = (f"Precursor neutral mass **{neutral:.4f} Da** → Δ vs. semaglutide **{delta:+.4f} Da**. ")
            if abs(delta) < 0.02:
                txt += "Precursor = intact semaglutide."
            elif abs(near["delta_mass"] - delta) < 0.02:
                txt += f"Matches library entry **{near['name']}** ({near['delta_mass']:+.4f} Da)."
            else:
                txt += "No library impurity matches this shift."
            st.markdown(txt)
        ms2_df, cov_result = annotate_msms(s2.mz, s2.inten, pep, ion_types, maxfz, ppm2, ms2_min_rel)
        cv = cov_result
        seq_ok = cv["coverage_pct"] >= cov_pass
        (st.success if seq_ok else st.warning)(
            f"Sequence coverage {cv['coverage_pct']:.0f}% of {pep.n - 1} inter-residue bonds "
            f"({'meets' if seq_ok else 'below'} the {cov_pass}% criterion).")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bonds covered", f"{len(cv['covered'])}/{pep.n - 1}")
        m2.metric("Peaks annotated", f"{cv['matched_peaks']}/{cv['total_peaks']}")
        m3.metric("Annotated intensity", f"{cv['matched_int_pct']:.0f}%")
        m4.metric("Median |error|", f"{cv['median_abs_ppm']:.1f} ppm" if len(ms2_df) else "–")

        # ---- annotated spectrum
        fig = spectrum_fig(s2.mz, s2.inten, "Annotated MS/MS spectrum", height=420)
        if len(ms2_df):
            for typ, col in (("b", BLUE), ("y", ORANGE)):
                d = ms2_df[ms2_df["type"] == typ]
                x, y = stem(d["mz"], d["intensity"])
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=col, width=2), name=f"{typ} ions", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=d["mz"], y=d["intensity"], mode="markers+text", text=d["ion"],
                                         textposition="top center", textfont=dict(size=9, color=col),
                                         marker=dict(color=col, size=5), name=f"{typ} labels", showlegend=False,
                                         hovertemplate="%{text}<br>m/z %{x:.4f}<br>%{customdata:+.1f} ppm<extra></extra>",
                                         customdata=d["ppm"]))
            o = ms2_df[~ms2_df["type"].isin(["b", "y"])]
            if len(o):
                fig.add_trace(go.Scatter(x=o["mz"], y=o["intensity"], mode="markers+text", text=o["ion"], textposition="top center",
                                         textfont=dict(size=9, color=GREEN), marker=dict(color=GREEN, size=5), name="other"))
        st.plotly_chart(fig, use_container_width=True)

        # ---- coverage map
        n = pep.n
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=list(range(1, n + 1)), y=[0] * n, mode="text",
                                  text=[c for c in pep.seq.replace("B", "Ū")], textfont=dict(size=20),
                                  hovertext=[f"{pep.label(i)} (GLP-1 #{i + 7})" for i in range(n)], hoverinfo="text",
                                  showlegend=False))
        for i in range(1, n):
            figc.add_shape(type="line", x0=i + 0.5, x1=i + 0.5, y0=-0.35, y1=0.35, line=dict(color="#d0d3d8", width=1))
        for i in cv["b_sites"]:
            figc.add_shape(type="path", path=f"M {i + 0.5} 0.6 L {i + 0.5} 0.35 L {i + 0.1} 0.35", line=dict(color=BLUE, width=2))
        for i in cv["y_sites"]:
            figc.add_shape(type="path", path=f"M {i + 0.5} -0.6 L {i + 0.5} -0.35 L {i + 0.9} -0.35", line=dict(color=ORANGE, width=2))
        for i in range(1, n + 1):
            figc.add_annotation(x=i, y=-0.95, text=str(i), showarrow=False, font=dict(size=9, color=GREY))
        for i in range(n):
            if (i + 1) in pep.mods:
                figc.add_annotation(x=i + 1, y=0.95, text="side chain", showarrow=True, ay=-14, ax=0, font=dict(size=9, color=GREEN))
        figc.update_layout(height=230, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10),
                           xaxis=dict(visible=False, range=[0.3, n + 0.7]), yaxis=dict(visible=False, range=[-1.2, 1.4]),
                           title="Sequence coverage map  (blue = b ions · orange = y ions · Ū = Aib)")
        st.plotly_chart(figc, use_container_width=True)

        with st.expander("Matched fragment ions"):
            st.dataframe(fmt(ms2_df, cols_4=["mz", "theo_mz"], cols_2=["ppm"]), hide_index=True, use_container_width=True)

        # ---- modification localisation
        with st.expander("Localise a modification / mass shift (for impurity MS/MS)"):
            choices = {"Custom Δ mass": None}
            if len(imp_df):
                choices.update({f"{r['Impurity']} ({r['Δ mass (Da)']:+.4f})": r["Δ mass (Da)"] for _, r in imp_df.iterrows()})
            ch = st.selectbox("Mass shift", list(choices))
            dm = choices[ch]
            if dm is None:
                dm = st.number_input("Δ mass (Da)", value=15.9949, format="%.4f")
            loc_df, rng_ = localize_shift(s2.mz, s2.inten, pep, dm, ppm2, maxfz, ms2_min_rel)
            if rng_:
                lo_, hi_ = rng_
                if lo_ == hi_:
                    st.success(f"Modification localised to **{pep.label(lo_ - 1)}** (GLP-1 position {lo_ + 6}).")
                elif lo_ < hi_:
                    st.info(f"Modification lies between **{pep.label(lo_ - 1)}** and **{pep.label(hi_ - 1)}** "
                            f"(residues {lo_}–{hi_}).")
                else:
                    st.warning("Fragment evidence is inconsistent with a single localised modification.")
            else:
                st.write("No shifted fragment ions found – this spectrum does not carry the selected modification.")
            st.dataframe(loc_df.sort_values(["type", "index"]), hide_index=True, use_container_width=True)

# ============================================================================ impurities tab
with tab_imp:
    if not n_ok:
        st.warning("Main species not identified – impurity search needs a confirmed semaglutide envelope.")
    elif imp_df.empty:
        st.info("No impurities above the reporting threshold matched.")
    else:
        st.write(f"**{len(imp_df)}** candidate impurities detected (main-species envelope subtracted before searching).")
        figb = go.Figure(go.Bar(y=imp_df["Impurity"][::-1], x=imp_df["Relative abundance (%)"][::-1], orientation="h",
                                marker_color=[BLUE if c == "High" else GREY for c in imp_df["Confidence"][::-1]],
                                text=[f"{v:.2f}%" for v in imp_df["Relative abundance (%)"][::-1]], textposition="outside"))
        figb.update_layout(height=max(250, 40 * len(imp_df) + 80), template="plotly_white", xaxis_title="% of main (peak-height, MS1)",
                           margin=dict(l=10, r=40, t=30, b=10), title="Impurity profile (blue = ≥2 charge states, grey = tentative)")
        st.plotly_chart(figb, use_container_width=True)
        st.dataframe(fmt(imp_df, cols_4=["Δ mass (Da)", "Theoretical mass (Da)", "Observed mass (Da)"],
                         cols_2=["Error (ppm)", "Relative abundance (%)"]).style.format({"Isotope score": "{:.3f}"}),
                     hide_index=True, use_container_width=True)
        st.caption("Relative abundance is a signal ratio at MS1 level (no response-factor correction) – use UV/LC "
                   "quantitation for reportable purity. Isobaric variants (e.g. Ile/Leu, Asp isomerisation) cannot be "
                   "distinguished by mass; use MS/MS localisation and retention time.")
    with st.expander("Candidate library searched"):
        lib = build_library(pep)
        st.dataframe(lib[["name", "category", "delta_formula", "delta_mass", "note"]].round(4), hide_index=True, use_container_width=True)

# ============================================================================ report
with tab_rep:
    st.subheader("Summary")
    summ = pd.DataFrame([
        ("Peptide sequence (B = Aib)", pep.seq), ("Formula", fstr(formula)), ("Theoretical monoisotopic mass (Da)", round(M0, 4)),
        ("MS1 charge states matched", f"{n_ok}/{len(zs)}"),
        ("Observed mass (Da)", "" if not n_ok else round(mass_obs, 4)),
        ("Mean mass error (ppm)", "" if not n_ok else round(mean_ppm, 2)),
        ("MS1 identity", "CONFIRMED" if ms1_ok else "NOT confirmed"),
        ("MS/MS sequence coverage (%)", "" if cov_result is None else round(cov_result["coverage_pct"], 1)),
        ("MS/MS criterion", "" if cov_result is None else ("PASS" if cov_result["coverage_pct"] >= cov_pass else "FAIL")),
        ("Impurities detected", len(imp_df)),
        ("MS1 tolerance / MS2 tolerance (ppm)", f"{ppm1} / {ppm2}"),
    ], columns=["Item", "Value"])
    st.dataframe(summ.astype(str), hide_index=True, use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summ.astype(str).to_excel(xw, sheet_name="Summary", index=False)
        id_df.drop(columns=["envelope"]).to_excel(xw, sheet_name="MS1_identification", index=False)
        (imp_df if len(imp_df) else pd.DataFrame({"info": ["none detected"]})).to_excel(xw, sheet_name="Impurities", index=False)
        (ms2_df if len(ms2_df) else pd.DataFrame({"info": ["no MS/MS"]})).to_excel(xw, sheet_name="MSMS_matches", index=False)
        pd.DataFrame({"position": range(1, pep.n + 1), "residue": list(pep.seq),
                      "name": [AA_NAMES[c] for c in pep.seq],
                      "b_ion_after": [i in (cov_result or {}).get("b_sites", []) for i in range(1, pep.n + 1)],
                      "y_ion_after": [i in (cov_result or {}).get("y_sites", []) for i in range(1, pep.n + 1)],
                      }).to_excel(xw, sheet_name="Coverage", index=False)
        fragment_table(pep, ("b", "y"), 3).round(5).to_excel(xw, sheet_name="Theoretical_fragments", index=False)
    st.download_button("⬇ Download Excel report", buf.getvalue(), "semaglutide_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    if len(imp_df):
        st.download_button("⬇ Impurities (CSV)", imp_df.to_csv(index=False).encode(), "semaglutide_impurities.csv", "text/csv")

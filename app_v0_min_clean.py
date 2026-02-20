# app_v0_min_clean.py
# Streamlit teaching app (V0-min): Weibull (2p/3p) + sliders + MLE fit + probability plot
# V0 intentionally WITHOUT censoring and WITHOUT top didactic block (minimal UI).
#
# CSV supports flexible time columns: Zeit, TimeFailure, Time, T, ...
# Optional Status/Failure/Event column: in V0 it is used only as a filter (Status==1).
#
# Suggested OER license: CC BY 4.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import io
import re

from scipy import stats
from scipy.special import gamma


# -----------------------------
# Utilities
# -----------------------------
def _as_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _norm_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _should_try_datetime(series: pd.Series) -> bool:
    return series.dtype == object or pd.api.types.is_string_dtype(series.dtype)


def _try_parse_datetime(series: pd.Series) -> Optional[pd.Series]:
    if not _should_try_datetime(series):
        return None
    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if parsed.notna().mean() >= 0.9:
            return parsed
    except Exception:
        pass
    return None


@dataclass
class WeibullParams:
    beta: float   # shape
    eta: float    # scale
    t0: float = 0.0  # location/threshold (0 for 2-parameter)


# -----------------------------
# Weibull model (2- or 3-parameter)
# -----------------------------
def weibull_sf(t: np.ndarray, p: WeibullParams) -> np.ndarray:
    t = _as_float_array(t)
    x = t - p.t0
    out = np.ones_like(t, dtype=float)
    mask = x > 0
    out[mask] = np.exp(- (x[mask] / p.eta) ** p.beta)
    return out


def weibull_cdf(t: np.ndarray, p: WeibullParams) -> np.ndarray:
    return 1.0 - weibull_sf(t, p)


def weibull_pdf(t: np.ndarray, p: WeibullParams) -> np.ndarray:
    t = _as_float_array(t)
    x = t - p.t0
    out = np.zeros_like(t, dtype=float)
    mask = x > 0
    xb = x[mask]
    out[mask] = (p.beta / p.eta) * (xb / p.eta) ** (p.beta - 1.0) * np.exp(- (xb / p.eta) ** p.beta)
    return out


def weibull_hazard(t: np.ndarray, p: WeibullParams) -> np.ndarray:
    t = _as_float_array(t)
    x = t - p.t0
    out = np.zeros_like(t, dtype=float)
    mask = x > 0
    xb = x[mask]
    out[mask] = (p.beta / p.eta) * (xb / p.eta) ** (p.beta - 1.0)
    return out


def weibull_mean(p: WeibullParams) -> float:
    return p.t0 + p.eta * float(gamma(1.0 + 1.0 / p.beta))


def weibull_median(p: WeibullParams) -> float:
    return p.t0 + p.eta * (math.log(2.0) ** (1.0 / p.beta))


def weibull_quantile(p: WeibullParams, q: float) -> float:
    q = float(q)
    q = min(max(q, 1e-12), 1.0 - 1e-12)
    return p.t0 + p.eta * ((-math.log(1.0 - q)) ** (1.0 / p.beta))


# -----------------------------
# Fit (MLE, no censoring)
# -----------------------------
def fit_weibull_mle(time: np.ndarray, use_3p: bool) -> WeibullParams:
    t = _as_float_array(time)
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise ValueError("Zeitwerte müssen endlich und >= 0 sein.")
    if len(t) < 3:
        raise ValueError("Für einen stabilen Fit bitte mindestens 3 Zeitwerte.")
    if use_3p:
        c, loc, scale = stats.weibull_min.fit(t)
        return WeibullParams(beta=float(c), eta=float(scale), t0=float(loc))
    c, loc, scale = stats.weibull_min.fit(t, floc=0.0)
    return WeibullParams(beta=float(c), eta=float(scale), t0=0.0)


# -----------------------------
# Probability Plot (Weibull)
# -----------------------------
def weibull_probability_plot(time: np.ndarray, use_3p: bool, t0: float) -> Tuple[np.ndarray, np.ndarray]:
    t = _as_float_array(time)
    x = t - float(t0) if use_3p else t
    x = x[x > 0]  # IMPORTANT: ignore zeros/negatives for log
    if len(x) < 3:
        raise ValueError("Zu wenige Werte für Probability Plot (mind. 3, alle > t0).")

    x_sorted = np.sort(x)
    n = len(x_sorted)
    i = np.arange(1, n + 1, dtype=float)
    F = (i - 0.3) / (n + 0.4)  # median ranks
    X = np.log(np.clip(x_sorted, 1e-300, None))
    Y = np.log(-np.log(1.0 - F))
    return X, Y


def line_fit_xy(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    X = _as_float_array(X)
    Y = _as_float_array(Y)
    A = np.vstack([np.ones_like(X), X]).T
    coeff, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return float(coeff[0]), float(coeff[1])


def _read_csv_flexible(file) -> pd.DataFrame:
    """
    Robust CSV reader for typical teaching/Excel exports:
    - Auto-detect delimiter (comma/semicolon/tab) via sep=None + python engine
    - Handle UTF-8 with BOM, and fall back to latin-1 if needed
    """
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
    else:
        text = str(raw)

    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


# -----------------------------
# Data loading (aliases)
# -----------------------------
TIME_ALIASES = {"t", "zeit", "time", "timefailure", "timetofailure", "time_to_failure", "ttf", "lifetime", "duration"}
STATUS_ALIASES = {"status", "event", "failure", "fault", "ausfall", "isfailure"}


def load_times_from_csv(file) -> Tuple[np.ndarray, Optional[np.ndarray], str, pd.DataFrame]:
    df = _read_csv_flexible(file)
    if df.empty:
        raise ValueError("CSV ist leer.")

    norm_map = {_norm_col(c): c for c in df.columns}

    time_col = None
    for key_norm, orig in norm_map.items():
        if key_norm in TIME_ALIASES:
            time_col = orig
            break
    if time_col is None:
        raise ValueError("Keine Zeit-Spalte gefunden. Unterstützt: Zeit, TimeFailure, Time, T, ...")

    s_time = df[time_col]

    dt = _try_parse_datetime(s_time)
    if dt is not None:
        dt_min = dt.min()
        time = ((dt - dt_min).dt.total_seconds() / 3600.0).to_numpy(dtype=float)
        time_note = f"Zeitstempel erkannt → umgerechnet in Stunden seit {dt_min} (UTC)."
    else:
        s_time_num = s_time.astype(str)
        # If values look like German decimal commas (e.g. "12,5"), convert to dots.
        if s_time.dtype == object or pd.api.types.is_string_dtype(s_time.dtype):
            if s_time_num.str.contains(r"\d,\d").mean() > 0.5:
                s_time_num = s_time_num.str.replace(",", ".", regex=False)
        time = pd.to_numeric(s_time_num, errors="coerce").to_numpy(dtype=float)
        time_note = "Zeit als numerische Dauer interpretiert (Einheit wie im CSV)."

    status_col = None
    for key_norm, orig in norm_map.items():
        if key_norm in STATUS_ALIASES:
            status_col = orig
            break

    status = None
    status_note = "Keine Status-Spalte gefunden (ok)."
    if status_col is not None:
        s_status = df[status_col].astype(str)
        if df[status_col].dtype == object or pd.api.types.is_string_dtype(df[status_col].dtype):
            if s_status.str.contains(r"\d,\d").mean() > 0.5:
                s_status = s_status.str.replace(",", ".", regex=False)
        status = pd.to_numeric(s_status, errors="coerce").fillna(0).astype(int).to_numpy()
        status_note = f"Status-Spalte '{status_col}' gefunden. V0 nutzt nur Status==1."

    valid = np.isfinite(time) & (time >= 0)
    dropped = int((~valid).sum())
    time = time[valid]
    if status is not None:
        status = status[valid]

    info = time_note + " " + status_note
    if dropped > 0:
        info += f" ({dropped} Zeilen mit ungültiger Zeit entfernt.)"
    return time, status, info, df


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Weibull PdM (V0 minimal)", layout="wide")
st.title("📈 Weibull-Verteilung – PdM & Instandhaltung (V0 minimal)")

# Sidebar
st.sidebar.header("Daten")
uploaded = st.sidebar.file_uploader("CSV hochladen", type=["csv"])

use_3p = st.sidebar.toggle("3-Parameter-Weibull (t0)", value=False)
mode = st.sidebar.radio("Modus", ["Slider (manuell)", "Fit auf Daten (MLE)"], index=0)

time = None
status = None
info = None
df_preview = None

if uploaded is not None:
    try:
        time, status, info, df_preview = load_times_from_csv(uploaded)
        st.sidebar.success(f"Daten geladen: n={len(time)}")
    except Exception as ex:
        st.sidebar.error(f"CSV-Fehler: {ex}")
        time = None

if info:
    st.info(info)

# If status exists: use only Status==1
if time is not None and status is not None:
    mask = (status == 1)
    used = int(mask.sum())
    total = int(len(time))
    if used == 0:
        st.error("Status-Spalte vorhanden, aber keine Zeilen mit Status==1.")
        time = None
    else:
        if used < total:
            st.warning(f"V0: Nutze nur Status==1 → {used}/{total} Zeilen.")
        time = time[mask]

# Plot grid
if time is not None and len(time) > 0:
    t_grid_max = max(float(np.nanmax(time)) * 1.25, 1.0)
else:
    t_grid_max = 200.0
t_grid = np.linspace(0.0, t_grid_max, 600)

# Slider params
st.sidebar.subheader("Parameter (Slider)")
beta = st.sidebar.slider("β (Shape)", 0.2, 10.0, 1.5, 0.05)
eta = st.sidebar.slider("η (Scale)", 0.1, float(max(t_grid_max * 2.0, 2.0)), float(max(t_grid_max / 2.0, 1.0)), 0.1)
t0 = 0.0
if use_3p:
    t0 = st.sidebar.slider("t0 (Location)", 0.0, float(t_grid_max * 0.8), 0.0, 0.1)

params = WeibullParams(beta=float(beta), eta=float(eta), t0=float(t0))

# Fit
fit_params = None
fit_error = None
if mode == "Fit auf Daten (MLE)":
    if time is None:
        st.warning("Bitte CSV hochladen oder Slider-Modus nutzen.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("🔧 Fit starten (MLE)"):
                try:
                    fit_params = fit_weibull_mle(time=time, use_3p=use_3p)
                except Exception as ex:
                    fit_error = str(ex)
        with c2:
            st.caption("Fit nutzt SciPy (MLE): scipy.stats.weibull_min.fit")

if fit_error:
    st.error(fit_error)

active = fit_params if fit_params is not None else params

# KPIs
kpi = pd.DataFrame({
    "Kennwert": ["β (Shape)", "η (Scale)", "t0 (Location)", "Mean (≈MTTF)", "Median (B50)", "B10", "B90"],
    "Wert": [
        active.beta,
        active.eta,
        active.t0,
        weibull_mean(active),
        weibull_median(active),
        weibull_quantile(active, 0.10),
        weibull_quantile(active, 0.90),
    ]
})

left, right = st.columns([2.2, 1.2], gap="large")

with right:
    st.subheader("Kennwerte")
    st.dataframe(kpi, hide_index=True, use_container_width=True)

    with st.expander("Formeln", expanded=True):
        if not use_3p:
            st.markdown("- **PDF (Dichte)**:")
            st.latex(r"f(t)=\frac{\beta}{\eta}\left(\frac{t}{\eta}\right)^{\beta-1}\exp\left[-\left(\frac{t}{\eta}\right)^\beta\right],\quad t\ge 0")
            st.markdown("- **CDF (Verteilungsfunktion)**:")
            st.latex(r"F(t)=1-\exp\left[-\left(\frac{t}{\eta}\right)^\beta\right]")
            st.markdown("- **Reliability / Survival**:")
            st.latex(r"R(t)=1-F(t)=\exp\left[-\left(\frac{t}{\eta}\right)^\beta\right]")
            st.markdown("- **Hazard (Ausfallrate)**:")
            st.latex(r"h(t)=\frac{f(t)}{R(t)}=\frac{\beta}{\eta}\left(\frac{t}{\eta}\right)^{\beta-1}")
        else:
            st.markdown("- **Reliability / Survival**:")
            st.latex(r"R(t)=\exp\left[-\left(\frac{t-t_0}{\eta}\right)^\beta\right],\quad t>t_0;\quad R(t)=1\;\text{für}\;t\le t_0")
            st.markdown("- **CDF (Verteilungsfunktion)**:")
            st.latex(r"F(t)=1-R(t)")
            st.markdown("- **PDF (Dichte)**:")
            st.latex(r"f(t)=\frac{\beta}{\eta}\left(\frac{t-t_0}{\eta}\right)^{\beta-1}\exp\left[-\left(\frac{t-t_0}{\eta}\right)^\beta\right],\quad t>t_0")
            st.markdown("- **Hazard (Ausfallrate)**:")
            st.latex(r"h(t)=\frac{f(t)}{R(t)}=\frac{\beta}{\eta}\left(\frac{t-t_0}{\eta}\right)^{\beta-1},\quad t>t_0")

    if df_preview is not None:
        with st.expander("CSV-Vorschau (erste 20 Zeilen)"):
            st.dataframe(df_preview.head(20), use_container_width=True)

# Curves
pdf = weibull_pdf(t_grid, active)
cdf = weibull_cdf(t_grid, active)
sf = weibull_sf(t_grid, active)
haz = weibull_hazard(t_grid, active)

with left:
    st.subheader("Kurven")
    tabs = st.tabs(["PDF f(t)", "CDF F(t)", "Reliability R(t)", "Hazard h(t)", "Probability Plot"])

    with tabs[0]:
        fig, ax = plt.subplots()
        ax.plot(t_grid, pdf)
        ax.set_xlabel("t")
        ax.set_ylabel("f(t)")
        ax.set_title("Weibull PDF")
        if time is not None and len(time) > 0:
            ax.hist(time, bins=30, density=True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

    with tabs[1]:
        fig, ax = plt.subplots()
        ax.plot(t_grid, cdf)
        ax.set_xlabel("t")
        ax.set_ylabel("F(t)")
        ax.set_ylim(0, 1)
        ax.set_title("Weibull CDF")
        st.pyplot(fig, clear_figure=True)

    with tabs[2]:
        fig, ax = plt.subplots()
        ax.plot(t_grid, sf)
        ax.set_xlabel("t")
        ax.set_ylabel("R(t)")
        ax.set_ylim(0, 1)
        ax.set_title("Weibull Reliability / Survival")
        st.pyplot(fig, clear_figure=True)

    with tabs[3]:
        fig, ax = plt.subplots()
        ax.plot(t_grid, haz)
        ax.set_xlabel("t")
        ax.set_ylabel("h(t)")
        ax.set_title("Weibull Hazard / Ausfallrate")
        st.pyplot(fig, clear_figure=True)

    with tabs[4]:
        if time is None or len(time) < 3:
            st.info("Für den Probability Plot bitte CSV mit mindestens 3 Zeitwerten hochladen.")
        else:
            try:
                X, Y = weibull_probability_plot(time, use_3p=use_3p, t0=active.t0)
                a, b = line_fit_xy(X, Y)
                fig, ax = plt.subplots()
                ax.scatter(X, Y, s=18)
                Xline = np.linspace(np.min(X), np.max(X), 100)
                ax.plot(Xline, a + b * Xline)
                ax.set_xlabel(r"$\ln(t-t_0)$" if use_3p else r"$\ln(t)$")
                ax.set_ylabel(r"$\ln(-\ln(1-\hat F(t)))$")
                ax.set_title("Weibull Probability Plot (Linearisierung)")
                st.pyplot(fig, clear_figure=True)
            except Exception as ex:
                st.error(f"Probability Plot fehlgeschlagen: {ex}")

st.caption("V0 minimal: Weibull-only, ohne Zensierung.")

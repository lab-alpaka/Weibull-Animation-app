# app.py
# Streamlit teaching app (V0): Weibull (2- & 3-parameter) + sliders + MLE fit + probability plot
# V0 intentionally WITHOUT censoring to keep it minimal.
#
# Suggested license for OER: CC BY 4.0 (add attribution in your repo).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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


def _try_parse_datetime(series: pd.Series) -> Optional[pd.Series]:
    """Try parsing a series as datetime. Returns parsed series or None."""
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
    """Survival function R(t) = P(T>t). For 3p Weibull, defined for t>t0; for t<=t0, R=1."""
    t = _as_float_array(t)
    x = t - p.t0
    out = np.ones_like(t, dtype=float)
    mask = x > 0
    out[mask] = np.exp(- (x[mask] / p.eta) ** p.beta)
    return out


def weibull_cdf(t: np.ndarray, p: WeibullParams) -> np.ndarray:
    """CDF F(t). For 3p Weibull, F(t)=0 for t<=t0."""
    return 1.0 - weibull_sf(t, p)


def weibull_pdf(t: np.ndarray, p: WeibullParams) -> np.ndarray:
    """PDF f(t). For 3p Weibull, f(t)=0 for t<=t0."""
    t = _as_float_array(t)
    x = t - p.t0
    out = np.zeros_like(t, dtype=float)
    mask = x > 0
    xb = x[mask]
    out[mask] = (p.beta / p.eta) * (xb / p.eta) ** (p.beta - 1.0) * np.exp(- (xb / p.eta) ** p.beta)
    return out


def weibull_hazard(t: np.ndarray, p: WeibullParams) -> np.ndarray:
    """Hazard h(t) = f(t)/R(t). For 3p Weibull, h(t)=0 for t<=t0."""
    t = _as_float_array(t)
    x = t - p.t0
    out = np.zeros_like(t, dtype=float)
    mask = x > 0
    xb = x[mask]
    out[mask] = (p.beta / p.eta) * (xb / p.eta) ** (p.beta - 1.0)
    return out


def weibull_mean(p: WeibullParams) -> float:
    """Mean time to failure (MTTF) for Weibull."""
    return p.t0 + p.eta * float(gamma(1.0 + 1.0 / p.beta))


def weibull_median(p: WeibullParams) -> float:
    """Median (B50) for Weibull."""
    return p.t0 + p.eta * (math.log(2.0) ** (1.0 / p.beta))


def weibull_quantile(p: WeibullParams, q: float) -> float:
    """Quantile t_q with F(t_q)=q, 0<q<1."""
    q = float(q)
    q = min(max(q, 1e-12), 1.0 - 1e-12)
    return p.t0 + p.eta * ((-math.log(1.0 - q)) ** (1.0 / p.beta))


# -----------------------------
# Weibull fitting (NO censoring in V0)
# -----------------------------
def fit_weibull_mle(time: np.ndarray, use_3p: bool) -> WeibullParams:
    """
    Fit Weibull parameters via SciPy MLE:
      scipy.stats.weibull_min.fit -> returns (beta, loc=t0, scale=eta)
    """
    t = _as_float_array(time)
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise ValueError("Zeitwerte müssen endlich und >= 0 sein.")
    if len(t) < 3:
        raise ValueError("Für einen stabilen Fit bitte mindestens 3 Zeitwerte.")

    if use_3p:
        c, loc, scale = stats.weibull_min.fit(t)            # loc free
        return WeibullParams(beta=float(c), eta=float(scale), t0=float(loc))
    else:
        c, loc, scale = stats.weibull_min.fit(t, floc=0.0)  # 2p: loc fixed
        return WeibullParams(beta=float(c), eta=float(scale), t0=0.0)


# -----------------------------
# Weibull Probability Plot (NO censoring in V0)
# -----------------------------
def weibull_probability_plot(time: np.ndarray, use_3p: bool, t0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weibull probability plot:
      X = ln(t - t0) , Y = ln(-ln(1-Fhat(t)))
    Uses median ranks (Bernard's approximation) for empirical Fhat.
    """
    t = _as_float_array(time)
    if use_3p:
        x = t - float(t0)
        x = x[x > 0]
    else:
        x = t

    if len(x) < 3:
        raise ValueError("Zu wenige Werte für Probability Plot (mind. 3).")

    x_sorted = np.sort(x)
    n = len(x_sorted)
    i = np.arange(1, n + 1, dtype=float)
    F = (i - 0.3) / (n + 0.4)  # median ranks

    X = np.log(np.clip(x_sorted, 1e-300, None))
    Y = np.log(-np.log(1.0 - F))
    return X, Y


def line_fit_xy(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    """Least squares line fit: Y = a + b X."""
    X = _as_float_array(X)
    Y = _as_float_array(Y)
    A = np.vstack([np.ones_like(X), X]).T
    coeff, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return float(coeff[0]), float(coeff[1])


# -----------------------------
# Data loading helpers (aliases)
# -----------------------------
TIME_ALIASES = {
    # requested + common
    "t", "zeit", "time", "timefailure", "time_to_failure", "ttf", "lifetime", "duration"
}
STATUS_ALIASES = {
    # requested + common
    "status", "failure", "event", "fault", "ausfall", "isfailure"
}


def load_times_from_csv(file) -> Tuple[np.ndarray, Optional[np.ndarray], str, pd.DataFrame]:
    """
    Load time values from CSV with flexible column names.
    Returns: (time, status_or_none, info_message, df_preview)

    - time column: supports T, Zeit, Time, TimeFailure, ...
    - status column (optional): supports Status, Failure, Event, ...

    In V0 (no censoring):
      If status exists, we FILTER to rows where status==1 and ignore status==0.
    """
    df = pd.read_csv(file)
    if df.empty:
        raise ValueError("CSV ist leer.")

    norm_map = {_norm_col(c): c for c in df.columns}

    # find time column
    time_col = None
    for key_norm, orig in norm_map.items():
        if key_norm in TIME_ALIASES:
            time_col = orig
            break
    if time_col is None:
        raise ValueError("Keine Zeit-Spalte gefunden. Unterstützt: T, Zeit, Time, TimeFailure, ...")

    s_time = df[time_col]

    # Try datetime first (user mentioned Zeitstempel)
    dt = _try_parse_datetime(s_time)
    if dt is not None:
        dt_min = dt.min()
        time = ((dt - dt_min).dt.total_seconds() / 3600.0).to_numpy(dtype=float)
        time_note = f"Zeit interpretiert als Zeitstempel → umgerechnet in Stunden seit {dt_min} (UTC)."
    else:
        time = pd.to_numeric(s_time, errors="coerce").to_numpy(dtype=float)
        time_note = "Zeit als numerische Dauer interpretiert (Einheit wie im CSV)."

    # optional status column
    status_col = None
    for key_norm, orig in norm_map.items():
        if key_norm in STATUS_ALIASES:
            status_col = orig
            break

    status = None
    status_note = "Keine Status-Spalte gefunden (ok)."
    if status_col is not None:
        status = pd.to_numeric(df[status_col], errors="coerce").fillna(0).astype(int).to_numpy()
        status_note = f"Status-Spalte '{status_col}' gefunden. In V0 werden nur Zeilen mit Status==1 für Fit/Plots genutzt."

    # drop invalid times
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
st.set_page_config(page_title="Weibull in PdM/Instanthaltung (V0)", layout="wide")
st.title("📈 Weibull-Verteilung für Instandhaltung & Predictive Maintenance (V0 – minimal)")

with st.expander("Didaktik: Was siehst du hier?", expanded=True):
    st.markdown(
        """
**Ziel:** Minimaler Lehr-Prototyp zur Weibull-Verteilung.

Du kannst:
- Parameter per **Slider** verändern (Kurven bewegen sich live).
- Optional Daten hochladen (**CSV**) und per **MLE (SciPy)** Parameter schätzen.
- **PDF, CDF, Reliability (Survival), Hazard** anzeigen.
- Einen **Weibull-Probability-Plot** (Linearisierung) betrachten.

**CSV-Spalten (flexibel):**
- Zeit-Spalte: `T` / `Zeit` / `Time` / `TimeFailure` / …
- Optional: `Status` / `Failure` / `Event` / …
  - In **V0** wird `Status==1` als „Ausfall/verwenden“ interpretiert, `Status==0` wird ignoriert.
        """
    )

st.sidebar.header("Einstellungen (minimal)")
uploaded = st.sidebar.file_uploader("CSV hochladen", type=["csv"])
use_3p = st.sidebar.toggle("3-Parameter-Weibull (t0)", value=False)
mode = st.sidebar.radio("Modus", ["Slider (manuell)", "Fit auf Daten (MLE)"], index=0)

# Load data
time = None
status = None
load_info = None
df_preview = None

if uploaded is not None:
    try:
        time, status, load_info, df_preview = load_times_from_csv(uploaded)
        st.sidebar.success(f"Daten geladen: n={len(time)}")
    except Exception as ex:
        st.sidebar.error(f"CSV-Fehler: {ex}")
        time = None
        status = None
        load_info = None
        df_preview = None

if load_info:
    st.info(load_info)

# If status exists, filter to status==1 for analysis (V0)
if time is not None and status is not None:
    mask = (status == 1)
    used = int(mask.sum())
    total = int(len(time))
    if used == 0:
        st.error("Status-Spalte vorhanden, aber keine Zeilen mit Status==1. Bitte Daten prüfen.")
        time = None
    else:
        if used < total:
            st.warning(f"V0: Nutze nur Status==1 → {used}/{total} Zeilen.")
        time = time[mask]

# Determine plotting range
if time is not None and len(time) > 0:
    t_grid_max = max(float(np.nanmax(time)) * 1.25, 1.0)
else:
    t_grid_max = 200.0

t_grid = np.linspace(0.0, t_grid_max, 600)

# Parameter sliders (defaults)
default_beta = 1.5
default_eta = max(t_grid_max / 2.0, 1.0)
default_t0 = 0.0

if time is not None and len(time) > 0:
    med0 = float(np.nanmedian(time))
    if np.isfinite(med0) and med0 > 0:
        default_eta = max(med0, 1.0)

st.sidebar.subheader("Parameter (Slider)")
beta = st.sidebar.slider("β (Form/Shape)", 0.2, 10.0, float(default_beta), 0.05)
eta = st.sidebar.slider("η (Skala/Scale)", 0.1, float(max(t_grid_max * 2.0, 2.0)), float(default_eta), 0.1)

t0 = 0.0
if use_3p:
    upper = float(t_grid_max * 0.8)
    t0 = st.sidebar.slider("t0 (Schwelle/Location)", 0.0, upper, float(default_t0), 0.1)

params = WeibullParams(beta=float(beta), eta=float(eta), t0=float(t0))

# Fit
fit_params = None
fit_error = None

if mode == "Fit auf Daten (MLE)":
    if time is None:
        st.warning("Bitte CSV hochladen oder in den Slider-Modus wechseln.")
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

active_params = fit_params if fit_params is not None else params

# KPIs
kpi = pd.DataFrame({
    "Kennwert": ["β (Shape)", "η (Scale)", "t0 (Location)", "MTTF / Mean", "Median (B50)", "B10", "B90"],
    "Wert": [
        active_params.beta,
        active_params.eta,
        active_params.t0,
        weibull_mean(active_params),
        weibull_median(active_params),
        weibull_quantile(active_params, 0.10),
        weibull_quantile(active_params, 0.90),
    ]
})

# Layout
left, right = st.columns([2.2, 1.2], gap="large")

with right:
    st.subheader("Kennwerte")
    st.dataframe(kpi, hide_index=True, use_container_width=True)

    with st.expander("Formeln (Weibull)", expanded=True):
        if not use_3p:
            st.latex(r"f(t)=\frac{\beta}{\eta}\left(\frac{t}{\eta}\right)^{\beta-1}\exp\left[-\left(\frac{t}{\eta}\right)^\beta\right],\quad t\ge 0")
            st.latex(r"F(t)=1-\exp\left[-\left(\frac{t}{\eta}\right)^\beta\right]")
            st.latex(r"R(t)=\exp\left[-\left(\frac{t}{\eta}\right)^\beta\right]")
            st.latex(r"h(t)=\frac{\beta}{\eta}\left(\frac{t}{\eta}\right)^{\beta-1}")
            st.latex(r"\mathbb{E}[T]=\eta\,\Gamma\left(1+\frac{1}{\beta}\right)")
        else:
            st.latex(r"R(t)=\exp\left[-\left(\frac{t-t_0}{\eta}\right)^\beta\right],\quad t>t_0;\quad R(t)=1 \text{ für } t\le t_0")
            st.latex(r"F(t)=1-R(t)")
            st.latex(r"f(t)=\frac{\beta}{\eta}\left(\frac{t-t_0}{\eta}\right)^{\beta-1}\exp\left[-\left(\frac{t-t_0}{\eta}\right)^\beta\right],\quad t>t_0")
            st.latex(r"h(t)=\frac{\beta}{\eta}\left(\frac{t-t_0}{\eta}\right)^{\beta-1},\quad t>t_0")
            st.latex(r"\mathbb{E}[T]=t_0+\eta\,\Gamma\left(1+\frac{1}{\beta}\right)")

    if df_preview is not None:
        with st.expander("CSV-Vorschau (erste 20 Zeilen)"):
            st.dataframe(df_preview.head(20), use_container_width=True)

# Curves
pdf = weibull_pdf(t_grid, active_params)
cdf = weibull_cdf(t_grid, active_params)
sf = weibull_sf(t_grid, active_params)
haz = weibull_hazard(t_grid, active_params)

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
                X, Y = weibull_probability_plot(time, use_3p=use_3p, t0=active_params.t0)
                a, b = line_fit_xy(X, Y)

                beta_pp = b
                eta_pp = math.exp(-a / b) if b != 0 else float("nan")

                fig, ax = plt.subplots()
                ax.scatter(X, Y, s=18)
                Xline = np.linspace(np.min(X), np.max(X), 100)
                ax.plot(Xline, a + b * Xline)
                ax.set_xlabel(r"$\ln(t-t_0)$" if use_3p else r"$\ln(t)$")
                ax.set_ylabel(r"$\ln(-\ln(1-\hat F(t)))$")
                ax.set_title("Weibull Probability Plot (Linearisierung)")
                st.pyplot(fig, clear_figure=True)

                st.caption("Interpretation: Wenn die Punkte ungefähr auf einer Geraden liegen, passt das Weibull-Modell gut.")
                st.write({"β (aus Plot)": beta_pp, "η (aus Plot)": eta_pp, "t0": active_params.t0})

            except Exception as ex:
                st.error(f"Probability Plot fehlgeschlagen: {ex}")

st.divider()
st.caption("V0: Minimal (Weibull-only, ohne Zensierung). Erweiterungen (Normal/Lognormal/Exponential/Gamma) folgen in V1.")

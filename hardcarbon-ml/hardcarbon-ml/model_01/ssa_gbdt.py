import matplotlib
matplotlib.use("Agg")

import os
import json
import math
import joblib
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


# =========================================================
# STRICT GBDT PIPELINE FOR SSA (SSA TARGET)
# Updated to match your CatBoost (Raman XGB-style) plotting format:
# - Manual axis control + auto gap control
# - Square symmetric axes (force_square_axes + same_xy_limits)
# - Transparent annotation bbox (doesn't hide points)
# - Saves PNG + SVG + PDF
#
# Modeling:
# - SSA>0 filter
# - RAW IQR outlier removal on SSA BEFORE split (k=IQR_K_RAW)
# - Temperature_01/02 -> BoxCox FEATURES (global; params saved)
# - Factorize categoricals on TRAIN only (unseen -> -1)
# - Target SSA -> BoxCox fit on TRAIN ONLY + RobustScaler
# - RobustScaler on continuous X
# - GBDT GridSearchCV
# - Inference: inverse scaler_y -> inverse BoxCox -> RAW SSA units
# =========================================================


# ---------------- CONFIG ----------------
# ---- Repository paths (relative to the repository root) ----
from pathlib import Path as _RepoPath
REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "SI_Model_01.xlsx"
DATA_SHEET = "Model_01_Surface_Area_Imputated"
OUTPUT_DIR = REPO_ROOT / "outputs" / "model_01/ssa_gbdt"

INPUT_FP = str(DATA_FILE)
SAVE_DIR = str(OUTPUT_DIR)
REPORT_JSON = os.path.join(SAVE_DIR, "SSA_gbdt_report.json")
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
target_col = "SSA"
model_name = "GBDT"

AUTO_CREATE_MISSING_COLUMNS = True

# RAW IQR outlier removal
IQR_K_RAW = 20.0

# Optional: histogram plot (raw vs boxcox) after RAW-IQR
SAVE_TARGET_SPREAD_PLOT = True
TARGET_SPREAD_PLOT_FP = os.path.join(SAVE_DIR, "SSA_spread_after_raw_IQR_raw_vs_boxcox.jpeg")

RUN_SHAP = True
SHAP_MAX_BACKGROUND = 200
SHAP_MAX_EXPLAIN = 500
SHAP_TOP_N = 10
SHAP_SEED = 42


# =========================
# PDP CONFIG
# =========================
RUN_PDP = True
PDP_TOP_N_GROUPS = 3          # top 3 grouped SHAP features
PDP_GRID_POINTS = 60
PDP_PERCENTILE_RANGE = (0.05, 0.95)
PDP_MAX_BASE_ROWS = 500       # rows used for averaging PDP
PDP_RANDOM_SEED = 42

# Main-paper PDP settings: selected from the actual SSA SHAP ranking.
# From your SSA SHAP panel, the leading drivers are:
# 1) Tem02, 2) Tem01, 3) Postwash, followed by FC%, HR01/O% type descriptors.
#
# Therefore, the main figure uses:
# - Tem01 x Tem02: continuous-continuous thermal design window
# - Tem01 x Postwash: continuous-categorical heatmap
# - Tem02 x Postwash: continuous-categorical heatmap
# - Tem01 x Fixed Carbon: continuous-continuous backup chemistry window
#
# Important: Postwash is categorical/label encoded, so it is plotted as a category heatmap,
# not a smooth continuous PDP surface.
RUN_MAIN_2D_PDP = True
MAIN_2D_PDP_GRID_POINTS = 60
MAIN_2D_PDP_MAX_CATEGORIES = 12
MAIN_2D_PDP_PERCENTILE_RANGE = (0.05, 0.95)  # data-supported range but broad enough for Tem02

# =========================
# ROUTE-AWARE SHAP / PDP CONFIG
# =========================
RUN_ROUTE_AWARE_SHAP = True
ROUTE_FEATURE = "Type"
ONE_STEP_KEYWORDS = ("one", "single", "1")
TWO_STEP_KEYWORDS = ("two", "double", "2")
MIN_ROUTE_ROWS_FOR_SHAP = 5
MIN_ROUTE_ROWS_FOR_PDP = 10
MAIN_2D_PDP_PAIRS = [
    # Valid for all samples: Tem01 is the main/first thermal treatment.
    {"x": "Temperature_01_boxcox", "y": "Fixed Carbon%", "label": "Tem01 × FC%"},
    {"x": "Temperature_01_boxcox", "y": "Oxygen%", "label": "Tem01 × O%"},
    {"x": "Temperature_01_boxcox", "y": "Postwash", "label": "Tem01 × Post"},

    # Chemically valid only for two-step samples:
    # Tem02 is not a real zero-temperature process in one-step samples;
    # it means "no second step". Therefore, this panel is filtered to two-step records.
    {"x": "Temperature_01_boxcox", "y": "Temperature_02_boxcox", "label": "Tem01 × Tem02", "subset": "two_step"},
]

PDP_PLOT_STYLE = {
    # Journal-style 2D PDP/design-window maps.
    # These are intentionally lighter than parity/SHAP poster-style figures.
    "figsize": (5.0, 4.6),
    "dpi": 600,
    "font_family": "Arial",
    "label_fs": 16,
    "tick_fs": 13,
    "spine_lw": 1.4,
    "line_lw": 1.2,
    "scatter_s": 12,
    "scatter_alpha": 0.25,
    "cmap": "viridis",
    "n_contour_levels": 16,
    "contour_lw": 0.55,
    "contour_alpha": 0.40,
    "panel_label_fs": 14,
    "panel_label_alpha": 0.62,
    "colorbar_tick_fs": 11,
    "colorbar_label_fs": 13,
}

# Empty dictionaries mean auto-scaled SHAP x-axes.
SHAP_AXES_RECT_BAR = [0.42, 0.25, 0.65, 0.80]
SHAP_AXES_RECT_STRIP = [0.42, 0.25, 0.65, 0.80]
SHAP_AXIS_SETTINGS_BAR: Dict[str, Any] = {}
SHAP_AXIS_SETTINGS_STRIP: Dict[str, Any] = {}

SHAP_LABEL_MAP = {
    "Cellulose%": "Cel%",
    "Hemi-cellulose%": "HCel%",
    "Lignin%": "Lig%",
    "Volatile Matter%": "VM%",
    "Ash%": "Ash%",
    "Fixed Carbon%": "FC%",
    "Carbon%": "C%",
    "Oxygen%": "O%",
    "Hydrogen%": "H%",
    "Time_01": "T01",
    "Heating rate_01": "HR01",
    "Time_02": "T02",
    "Heating rate_02": "HR02",
    "Temperature_01_boxcox": "Tem01",
    "Temperature_02_boxcox": "Tem02",
    "Prewash": "Pre",
    "Postwash": "Post",
    "Type": "Type",
    "Method_01": "M01",
    "Activator_01": "Act01",
    "Method_02": "M02",
    "Activator_02": "Act02",
}


def sanitize_filename(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


def make_json_serializable(o: Any):
    if o is None:
        return None
    if isinstance(o, dict):
        return {k: make_json_serializable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [make_json_serializable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return make_json_serializable(o.tolist())
    try:
        json.dumps(o)
        return o
    except Exception:
        return str(o)


# =========================================================
# ------------- AXIS + GAP CONTROL (EDIT HERE) ------------
# =========================================================
AXIS_SETTINGS_TRUEPRED: Dict[str, Dict[str, Any]] = {
    "SSA": {
        # ---------------- MANUAL AXIS (OPTIONAL) ----------------
        # Uncomment these 4 lines to force the same axis for all splits:
        "xlim": (0, 2500),
        "ylim": (0, 2500),
        "xtick_step": 500,
        "ytick_step": 500,

        # ---------------- AUTO AXIS TUNING (GAPS) ----------------
        "axis_padding_frac": 0.05,  # increase to 0.10 for more gap
        "n_ticks": 6,
        "clamp_min_zero": True,

        # ---------------- SQUARE + SYMMETRIC LOOK ----------------
        "force_square_axes": True,
        "same_xy_limits": True,
    }
}


# =========================================================
# ---------------- Raman XGB CONSISTENT PLOT STYLE --------
# =========================================================
PLOT_STYLE = {
    "font_family": "Arial",

    "label_fs": 22,
    "tick_fs": 18,
    "spine_lw": 5,

    "grid": False,

    "scatter_alpha": 0.7,
    "scatter_s": 100,
    "scatter_color": "#1f77b4",

    "diag_color": "red",
    "diag_ls": "--",
    "diag_lw": 2,

    "ann_fs": 20,
    # transparent background so it doesn't hide points
    "ann_bbox": dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0", alpha=0.0),

    "figsize": (5, 5),
    "dpi_png": 600,

    "svg_fonttype": "none",
    "pdf_fonttype": 42,
    "ps_fonttype": 42,
}

SHAP_PLOT_STYLE = {
    # Journal-style SHAP figures. Keep readable but less heavy than poster-style plots.
    "figsize_bar": (5.2, 4.8),
    "figsize_strip": (5.2, 4.8),
    "dpi_png": 600,
    "spine_lw": 1.5,
    "label_fs": 14,
    "tick_fs": 11,
    "ytick_fs": 11,
    "bar_color": "#1f77b4",
    "bar_alpha": 0.9,
    "strip_s": 24,
    "strip_alpha": 0.70,
}


def apply_plot_style():
    plt.rcParams["font.family"] = PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = PLOT_STYLE["svg_fonttype"]
    plt.rcParams["pdf.fonttype"] = PLOT_STYLE["pdf_fonttype"]
    plt.rcParams["ps.fonttype"] = PLOT_STYLE["ps_fonttype"]


def save_figure_multi(basepath_no_ext: str, save_png=True, save_svg=True, save_pdf=True):
    if save_png:
        plt.savefig(basepath_no_ext + ".png", dpi=PLOT_STYLE["dpi_png"], bbox_inches="tight")
    if save_svg:
        plt.savefig(basepath_no_ext + ".svg", bbox_inches="tight")
    if save_pdf:
        plt.savefig(basepath_no_ext + ".pdf", bbox_inches="tight")


def save_shap_figure_multi(fig, basepath_no_ext: str):
    fig.savefig(basepath_no_ext + ".png", dpi=SHAP_PLOT_STYLE["dpi_png"], bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".svg", bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".pdf", bbox_inches="tight")


def set_spines(ax):
    for s in ax.spines.values():
        s.set_linewidth(PLOT_STYLE["spine_lw"])


def set_spines_lw(ax, lw: float):
    for s in ax.spines.values():
        s.set_linewidth(lw)


def nice_step(raw_step: float) -> float:
    """Round raw_step to a 'nice' step: 1,2,3,5,10 × 10^k."""
    if raw_step <= 0 or not np.isfinite(raw_step):
        return 1.0
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10 ** exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 3:
        nice_fraction = 3
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return nice_fraction * (10 ** exponent)


def nice_limits_ticks(data, n_ticks=6, padding=0.05, clamp_min_zero=True):
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]

    if data.size == 0:
        vmin, vmax = (0.0, 1.0)
    else:
        vmin, vmax = float(data.min()), float(data.max())
        if np.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5

    span = vmax - vmin
    vmin_pad = vmin - span * padding
    vmax_pad = vmax + span * padding

    if clamp_min_zero:
        vmin_pad = max(0.0, vmin_pad)

    raw_step = (vmax_pad - vmin_pad) / max(1, (n_ticks - 1))
    step = nice_step(raw_step)

    vmin_nice = np.floor(vmin_pad / step) * step
    vmax_nice = np.ceil(vmax_pad / step) * step

    if clamp_min_zero:
        vmin_nice = max(0.0, vmin_nice)

    ticks = np.arange(vmin_nice, vmax_nice + 0.5 * step, step)
    return vmin_nice, vmax_nice, step, ticks


def auto_shap_xaxis(ax, values, start_at_zero: bool = False, n_ticks: int = 5):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        x_min, x_max = (0.0, 1.0) if start_at_zero else (-1.0, 1.0)
    else:
        vmin = float(values.min())
        vmax = float(values.max())
        if start_at_zero:
            x_min = 0.0
            x_max = vmax
        else:
            max_abs = max(abs(vmin), abs(vmax))
            x_min = -max_abs
            x_max = max_abs

        if np.isclose(x_min, x_max):
            pad = 1.0 if np.isclose(x_min, 0.0) else abs(x_min) * 0.2
            x_min -= pad
            x_max += pad
        else:
            pad = (x_max - x_min) * 0.05
            if start_at_zero:
                x_max += pad
            else:
                x_min -= pad
                x_max += pad

    step = nice_step((x_max - x_min) / max(1, n_ticks - 1))
    x0 = 0.0 if start_at_zero else np.floor(x_min / step) * step
    x1 = np.ceil(x_max / step) * step
    ticks = np.arange(x0, x1 + 0.5 * step, step)
    ax.set_xlim((float(x0), float(x1)))
    ax.set_xticks(ticks)


def apply_shap_xaxis_controls(ax, settings: Dict[str, Any], values, start_at_zero: bool = False):
    if settings.get("xlim") is not None:
        xlim = tuple(float(v) for v in settings["xlim"])
        ax.set_xlim(xlim)
        if settings.get("xtick_step") is not None:
            step = float(settings["xtick_step"])
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * step, step))
    elif settings.get("xtick_step") is not None:
        auto_shap_xaxis(ax, values, start_at_zero=start_at_zero)
        xlim = ax.get_xlim()
        step = float(settings["xtick_step"])
        ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * step, step))
    else:
        auto_shap_xaxis(ax, values, start_at_zero=start_at_zero)


# ---------------- LEGACY RENAME MAP ----------------
LEGACY_RENAME_MAP: Dict[str, str] = {
    "Cellulose %": "Cellulose%",
    "Hemi-cellulose %": "Hemi-cellulose%",
    "Lignin %": "Lignin%",
    "Volatile Matter %": "Volatile Matter%",
    "Ash %": "Ash%",
    "Fixed Carbon": "Fixed Carbon%",
    "Carbon %": "Carbon%",
    "Oxygen %": "Oxygen%",
    "Hydrogen %": "Hydrogen%",
    "Time 01": "Time_01",
    "Heating rate 01": "Heating rate_01",
    "Temperature 01": "Temperature_01",
    "Temperature 02": "Temperature_02",
    "Method 01": "Method_01",
    "Activator 01": "Activator_01",
    "Method 02": "Method_02",
    "Activatior 02": "Activator_02",
    "Time 02": "Time_02",
    "Heating rate 02": "Heating rate_02",
}

# ---------------- FEATURE GROUPS ----------------
CONTINUOUS_COLS: List[str] = [
    "Cellulose%", "Hemi-cellulose%", "Lignin%", "Volatile Matter%", "Ash%",
    "Fixed Carbon%", "Carbon%", "Oxygen%", "Hydrogen%",
    "Time_01", "Heating rate_01",
    "Time_02", "Heating rate_02",
]
BOXCOX_BASES: List[str] = ["Temperature_01", "Temperature_02"]
CATEGORICAL_COLS: List[str] = [
    "Prewash", "Postwash", "Type", "Method_01", "Activator_01", "Method_02", "Activator_02"
]
REQUIRED_COLUMNS = sorted(list(set(CONTINUOUS_COLS + BOXCOX_BASES + CATEGORICAL_COLS + [target_col])))


# ---------------- DATA HELPERS ----------------
def check_and_create_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    auto_create: bool = True
) -> Tuple[pd.DataFrame, List[str], Dict[str, bool]]:
    present = list(df.columns)
    presence_map = {c: (c in present) for c in required_cols}
    missing = [c for c, ok in presence_map.items() if not ok]
    if not missing:
        return df, [], presence_map
    if auto_create:
        for c in missing:
            df[c] = np.nan
        presence_map = {c: (c in df.columns) for c in required_cols}
        return df, missing, presence_map
    raise ValueError(f"Missing required columns: {missing}")


def dropna_align(Xd: pd.DataFrame, yd: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comb = pd.concat([Xd, yd], axis=1).dropna().reset_index(drop=True)
    return comb[Xd.columns], comb[yd.columns]


def get_metrics(y_true, y_pred) -> Tuple[float, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return float(r2), float(rmse)


def remove_outliers_iqr_raw(df: pd.DataFrame, target: str, k: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    s = pd.to_numeric(df[target], errors="coerce")
    mask_valid = s.notna() & (s > 0)
    s_valid = s[mask_valid]

    if s_valid.empty:
        return df.copy(), {"method": "iqr_raw", "note": "no valid rows"}

    q1 = float(np.quantile(s_valid, 0.25))
    q3 = float(np.quantile(s_valid, 0.75))
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    keep = (s_valid >= lower) & (s_valid <= upper)
    idx_keep = keep.index[keep]
    out = df.loc[idx_keep].copy().reset_index(drop=True)

    details = {
        "method": "iqr_raw",
        "k": float(k),
        "q1_raw": q1,
        "q3_raw": q3,
        "iqr_raw": float(iqr),
        "lower_raw": float(lower),
        "upper_raw": float(upper),
        "raw_min": float(s_valid.min()),
        "raw_max": float(s_valid.max()),
        "rows_before": int(len(df)),
        "rows_valid": int(mask_valid.sum()),
        "rows_after": int(len(out)),
        "rows_removed_total": int(len(df) - len(out)),
    }

    print(
        f"[IQR raw] {target}: k={k} "
        f"min/max=({details['raw_min']:.4f},{details['raw_max']:.4f}) "
        f"Q1={q1:.4f} Q3={q3:.4f} "
        f"low={lower:.4f} high={upper:.4f} "
        f"before={details['rows_before']} after={details['rows_after']} removed={details['rows_removed_total']}"
    )
    return out, details


def save_target_spread_plot_raw_vs_boxcox(y_raw: np.ndarray, out_fp: str):
    y_raw = np.asarray(y_raw).ravel()
    y_raw = y_raw[np.isfinite(y_raw)]
    y_raw = y_raw[y_raw > 0]

    if y_raw.size > 0:
        try:
            y_bc, lam_plot = boxcox(y_raw)
        except Exception:
            y_bc, lam_plot = y_raw, None
    else:
        y_bc, lam_plot = y_raw, None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=600)
    for ax in axes:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    axes[0].hist(y_raw, bins=30, color="#1f77b4", alpha=0.85)
    axes[0].set_title("SSA (raw) AFTER RAW-IQR")
    axes[0].set_xlabel("SSA")
    axes[0].set_ylabel("Count")

    axes[1].hist(y_bc, bins=30, color="#1f77b4", alpha=0.85)
    if lam_plot is not None:
        axes[1].set_title(f"BoxCox(SSA) (λ≈{lam_plot:.3f})")
    else:
        axes[1].set_title("BoxCox(SSA) (plot fallback)")
    axes[1].set_xlabel("BoxCox(SSA)")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(out_fp, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved target spread plot:", out_fp)


# ---------------- Target BoxCox (train-only fit) ----------------
def boxcox_transform_train_only(y_train_raw: np.ndarray, y_other_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    y_train_raw = np.asarray(y_train_raw).reshape(-1)
    y_other_raw = np.asarray(y_other_raw).reshape(-1)

    if np.any(y_train_raw <= 0):
        raise ValueError("BoxCox target transform requires y_train_raw > 0 for all rows.")
    if np.any(y_other_raw <= 0):
        raise ValueError("BoxCox target transform requires y_other_raw > 0 for all rows.")

    y_train_bc, lam = boxcox(y_train_raw)
    if abs(lam) < 1e-12:
        y_other_bc = np.log(y_other_raw)
    else:
        y_other_bc = (np.power(y_other_raw, lam) - 1.0) / lam

    params = {"lam": float(lam), "shift": 0.0}
    return y_train_bc.reshape(-1, 1), y_other_bc.reshape(-1, 1), params


def boxcox_inverse(y_bc: np.ndarray, lam: float) -> np.ndarray:
    """Safe inverse Box-Cox used for model predictions.

    Tree models can occasionally extrapolate to a Box-Cox value where
    lam * y + 1 becomes slightly negative. Without clipping, this creates
    NaN regions in PDP heatmaps. Clipping avoids white/blank PDP areas while
    keeping the response in physically meaningful positive units.
    """
    y_bc = np.asarray(y_bc, dtype=float).reshape(-1)
    if abs(lam) < 1e-12:
        y = np.exp(y_bc)
    else:
        base = lam * y_bc + 1.0
        base = np.maximum(base, 1e-12)
        y = np.power(base, 1.0 / lam)
    return y.reshape(-1, 1)


# ---------------- Raman-style plot ----------------
def plot_true_vs_pred(
    y_true, y_pred, set_name, target_name, model_name, save_dir,
    xlim=None, ylim=None, xtick_step=None, ytick_step=None,
    axis_padding_frac=0.05, n_ticks=6, clamp_min_zero=True,
    force_square_axes=True, same_xy_limits=True
):
    apply_plot_style()

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r2, rmse = get_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])

    ax.scatter(
        y_true, y_pred,
        alpha=PLOT_STYLE["scatter_alpha"],
        s=PLOT_STYLE["scatter_s"],
        color=PLOT_STYLE["scatter_color"],
    )
    ax.set_xlabel("True Value", fontsize=PLOT_STYLE["label_fs"])
    ax.set_ylabel("Predicted Value", fontsize=PLOT_STYLE["label_fs"])

    manual_limits = (xlim is not None) or (ylim is not None) or (xtick_step is not None) or (ytick_step is not None)

    if (not manual_limits) and same_xy_limits:
        both = np.concatenate([y_true[np.isfinite(y_true)], y_pred[np.isfinite(y_pred)]])
        a0, a1, astep, _ = nice_limits_ticks(both, n_ticks=n_ticks, padding=axis_padding_frac, clamp_min_zero=clamp_min_zero)
        xlim = (a0, a1) if xlim is None else xlim
        ylim = (a0, a1) if ylim is None else ylim
        xtick_step = astep if xtick_step is None else xtick_step
        ytick_step = astep if ytick_step is None else ytick_step
    else:
        if xlim is None or xtick_step is None:
            x0, x1, xstep, _ = nice_limits_ticks(y_true, n_ticks=n_ticks, padding=axis_padding_frac, clamp_min_zero=clamp_min_zero)
            if xlim is None:
                xlim = (x0, x1)
            if xtick_step is None:
                xtick_step = xstep

        if ylim is None or ytick_step is None:
            y0, y1, ystep, _ = nice_limits_ticks(y_pred, n_ticks=n_ticks, padding=axis_padding_frac, clamp_min_zero=clamp_min_zero)
            if ylim is None:
                ylim = (y0, y1)
            if ytick_step is None:
                ytick_step = ystep

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * xtick_step, xtick_step))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.5 * ytick_step, ytick_step))
    ax.tick_params(axis="both", labelsize=PLOT_STYLE["tick_fs"])

    lo = min(float(xlim[0]), float(ylim[0]))
    hi = max(float(xlim[1]), float(ylim[1]))
    ax.plot([lo, hi], [lo, hi], color=PLOT_STYLE["diag_color"], linestyle=PLOT_STYLE["diag_ls"], linewidth=PLOT_STYLE["diag_lw"])

    txt = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\n{model_name}"
    ax.text(
        0.95, 0.05, txt,
        transform=ax.transAxes,
        fontsize=PLOT_STYLE["ann_fs"],
        ha="right", va="bottom",
        bbox=PLOT_STYLE["ann_bbox"],
    )

    set_spines(ax)

    if force_square_axes:
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    base = os.path.join(save_dir, f"{sanitize_filename(target_name)}_{set_name}_{sanitize_filename(model_name)}_true_vs_pred")
    save_figure_multi(base, save_png=True, save_svg=True, save_pdf=True)
    plt.close(fig)
    print("Saved scatter (png/svg/pdf):", base + ".*")
    return base



# =========================================================
# ---------------- GROUPED SHAP + PDP HELPERS -------------
# =========================================================
def inverse_boxcox_feature(x_bc: np.ndarray, lam: Optional[float], shift: float = 0.0) -> np.ndarray:
    """Convert Box-Cox transformed feature values back to original units for PDP x-axis."""
    x_bc = np.asarray(x_bc, dtype=float)
    if lam is None:
        return x_bc - shift
    if abs(lam) < 1e-12:
        return np.exp(x_bc) - shift
    return np.power(lam * x_bc + 1.0, 1.0 / lam) - shift


def make_group_name(source_feature: str) -> str:
    """
    Group features for grouped SHAP.
    For Model 1, most features are already single physical descriptors.
    Temperature Box-Cox features are grouped back to the original temperature names.

    For Model 2, extend this function to group one-hot encoded variables, e.g.:
    EC_DEC_1_1, EC_DMC_1_1, DIGLYME -> Electrolyte type or Solvent type.
    """
    if source_feature == "Temperature_01_boxcox":
        return "Temperature_01"
    if source_feature == "Temperature_02_boxcox":
        return "Temperature_02"
    return source_feature


def make_group_display_name(group_name: str) -> str:
    display_map = dict(SHAP_LABEL_MAP)
    display_map.update({
        "Temperature_01": "Tem01",
        "Temperature_02": "Tem02",
    })
    return display_map.get(group_name, group_name)


def compute_grouped_shap_importance(
    shap_vals: np.ndarray,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Compute grouped SHAP as mean(abs(sum SHAP within each group))."""
    group_to_features: Dict[str, List[str]] = {}
    for f in feature_names:
        g = make_group_name(f)
        group_to_features.setdefault(g, []).append(f)

    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    rows = []
    for group_name, feats in group_to_features.items():
        idxs = [feature_to_idx[f] for f in feats]
        grouped_values = shap_vals[:, idxs].sum(axis=1)
        rows.append({
            "group": group_name,
            "group_display": make_group_display_name(group_name),
            "features_in_group": ";".join(feats),
            "n_features": len(feats),
            "mean_abs_grouped_shap": float(np.abs(grouped_values).mean()),
        })

    grouped_df = (
        pd.DataFrame(rows)
        .sort_values("mean_abs_grouped_shap", ascending=False)
        .reset_index(drop=True)
    )
    return grouped_df, group_to_features


def save_pdp_plot_numeric(
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    x_label: str,
    y_label: str,
    title_label: str,
    out_base: str,
):
    plt.rcParams["font.family"] = PDP_PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=PDP_PLOT_STYLE["figsize"])
    ax.plot(x_grid, y_mean, linewidth=PDP_PLOT_STYLE["line_lw"])
    ax.scatter(x_grid, y_mean, s=PDP_PLOT_STYLE["scatter_s"], alpha=PDP_PLOT_STYLE["scatter_alpha"])
    ax.set_xlabel(x_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.set_ylabel(y_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.tick_params(axis="both", labelsize=PDP_PLOT_STYLE["tick_fs"])
    ax.text(
        0.05, 0.95, title_label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=PDP_PLOT_STYLE["label_fs"],
        fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(PDP_PLOT_STYLE["spine_lw"])
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=PDP_PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def save_pdp_plot_categorical(
    labels: List[str],
    y_mean: np.ndarray,
    x_label: str,
    y_label: str,
    title_label: str,
    out_base: str,
):
    plt.rcParams["font.family"] = PDP_PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(labels))
    ax.bar(x, y_mean)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=PDP_PLOT_STYLE["tick_fs"])
    ax.set_xlabel(x_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.set_ylabel(y_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.tick_params(axis="y", labelsize=PDP_PLOT_STYLE["tick_fs"])
    ax.text(
        0.05, 0.95, title_label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=PDP_PLOT_STYLE["label_fs"],
        fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(PDP_PLOT_STYLE["spine_lw"])
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=PDP_PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def get_feature_display_label(feature: str) -> str:
    if feature == "Temperature_01_boxcox":
        return "Temperature 01 (°C)"
    if feature == "Temperature_02_boxcox":
        return "Temperature 02 (°C)"
    return SHAP_LABEL_MAP.get(feature, feature)


def convert_boxcox_feature_to_raw_for_plot(
    feature: str,
    values: np.ndarray,
    boxcox_params_X: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    if feature == "Temperature_01_boxcox":
        return inverse_boxcox_feature(
            values,
            lam=boxcox_params_X["Temperature_01"]["lam"],
            shift=boxcox_params_X["Temperature_01"]["shift"],
        )
    if feature == "Temperature_02_boxcox":
        return inverse_boxcox_feature(
            values,
            lam=boxcox_params_X["Temperature_02"]["lam"],
            shift=boxcox_params_X["Temperature_02"]["shift"],
        )
    return values


def build_pdp_axis_values(
    feature: str,
    X_base_pdp: pd.DataFrame,
    continuous_actual: List[str],
    boxcox_actual: List[str],
    categorical_actual: List[str],
    cat_maps: Dict[str, List[Any]],
    boxcox_params_X: Dict[str, Dict[str, Any]],
    grid_points: int,
    percentile_range: Tuple[float, float],
    max_categories: int,
) -> Optional[Dict[str, Any]]:
    """Build model-grid values and plot-grid values for one PDP axis."""
    if feature not in X_base_pdp.columns:
        print(f"[MAIN 2D PDP] Feature not found and skipped: {feature}")
        return None

    if feature in continuous_actual:
        vals = X_base_pdp[feature].dropna().to_numpy(dtype=float)
        if vals.size < 3:
            print(f"[MAIN 2D PDP] Too few values and skipped: {feature}")
            return None
        lo = np.nanquantile(vals, percentile_range[0])
        hi = np.nanquantile(vals, percentile_range[1])
        if np.isclose(lo, hi):
            print(f"[MAIN 2D PDP] Quantile range too small and skipped: {feature}")
            return None
        model_values = np.linspace(lo, hi, grid_points)
        plot_values = model_values.copy()
        actual_plot_values = vals.copy()
        return {
            "feature": feature,
            "kind": "continuous",
            "model_values": model_values,
            "plot_values": plot_values,
            "actual_plot_values": actual_plot_values,
            "tick_labels": None,
            "axis_label": get_feature_display_label(feature),
        }

    if feature in boxcox_actual:
        vals = X_base_pdp[feature].dropna().to_numpy(dtype=float)
        if vals.size < 3:
            print(f"[MAIN 2D PDP] Too few values and skipped: {feature}")
            return None
        lo = np.nanquantile(vals, percentile_range[0])
        hi = np.nanquantile(vals, percentile_range[1])
        if np.isclose(lo, hi):
            print(f"[MAIN 2D PDP] Quantile range too small and skipped: {feature}")
            return None
        model_values = np.linspace(lo, hi, grid_points)
        plot_values = convert_boxcox_feature_to_raw_for_plot(feature, model_values, boxcox_params_X)
        actual_plot_values = convert_boxcox_feature_to_raw_for_plot(feature, vals, boxcox_params_X)
        return {
            "feature": feature,
            "kind": "boxcox_continuous",
            "model_values": model_values,
            "plot_values": plot_values,
            "actual_plot_values": actual_plot_values,
            "tick_labels": None,
            "axis_label": get_feature_display_label(feature),
        }

    if feature in categorical_actual:
        codes = sorted([
            int(c) for c in X_base_pdp[feature].dropna().unique()
            if int(c) >= 0
        ])
        if len(codes) == 0:
            print(f"[MAIN 2D PDP] No valid category codes and skipped: {feature}")
            return None
        if len(codes) > max_categories:
            # Keep the most frequent categories for readability.
            freq = X_base_pdp[feature].value_counts().sort_values(ascending=False)
            codes = [int(c) for c in freq.index if int(c) >= 0][:max_categories]
            codes = sorted(codes)

        labels = []
        for code in codes:
            if feature in cat_maps and int(code) < len(cat_maps[feature]):
                labels.append(str(cat_maps[feature][int(code)]))
            else:
                labels.append(str(code))
        plot_values = np.arange(len(codes), dtype=float)
        actual_codes = X_base_pdp[feature].dropna().to_numpy(dtype=float)
        actual_plot_values = []
        code_to_pos = {int(code): i for i, code in enumerate(codes)}
        for c in actual_codes:
            if int(c) in code_to_pos:
                actual_plot_values.append(code_to_pos[int(c)])
        actual_plot_values = np.asarray(actual_plot_values, dtype=float)
        return {
            "feature": feature,
            "kind": "categorical",
            "model_values": np.asarray(codes, dtype=float),
            "plot_values": plot_values,
            "actual_plot_values": actual_plot_values,
            "tick_labels": labels,
            "axis_label": get_feature_display_label(feature),
        }

    print(f"[MAIN 2D PDP] Unrecognized feature type and skipped: {feature}")
    return None



def infer_route_codes(
    cat_maps: Dict[str, List[Any]],
    route_feature: str = "Type",
    one_keywords: Tuple[str, ...] = ("one", "single", "1"),
    two_keywords: Tuple[str, ...] = ("two", "double", "2"),
) -> Dict[str, List[int]]:
    """Identify one-step and two-step encoded category codes from cat_maps."""
    labels = cat_maps.get(route_feature, [])
    one_codes: List[int] = []
    two_codes: List[int] = []

    for i, label in enumerate(labels):
        s = str(label).lower()
        if any(k in s for k in one_keywords):
            one_codes.append(i)
        if any(k in s for k in two_keywords):
            two_codes.append(i)

    if not one_codes and len(labels) >= 1:
        one_codes = [0]
    if not two_codes and len(labels) >= 2:
        two_codes = [1]

    return {"one_step": sorted(set(one_codes)), "two_step": sorted(set(two_codes))}


def get_route_mask(
    X_df: pd.DataFrame,
    cat_maps: Dict[str, List[Any]],
    subset: Optional[str],
    route_feature: str = "Type",
) -> np.ndarray:
    """Return a boolean mask for all / one-step / two-step rows."""
    if subset is None or str(subset).lower() in ("all", "none"):
        return np.ones(len(X_df), dtype=bool)

    if route_feature not in X_df.columns:
        print(f"[Route-aware] Route feature '{route_feature}' not found. Using all rows.")
        return np.ones(len(X_df), dtype=bool)

    route_codes = infer_route_codes(cat_maps, route_feature=route_feature)
    subset_norm = str(subset).lower().strip()

    if subset_norm in ("one", "one_step", "single", "single_step"):
        codes = route_codes["one_step"]
    elif subset_norm in ("two", "two_step", "double", "double_step"):
        codes = route_codes["two_step"]
    else:
        print(f"[Route-aware] Unknown subset '{subset}'. Using all rows.")
        return np.ones(len(X_df), dtype=bool)

    route_values = pd.to_numeric(X_df[route_feature], errors="coerce")
    mask = route_values.isin(codes).to_numpy()

    print(
        f"[Route-aware] subset={subset_norm}, feature={route_feature}, "
        f"codes={codes}, rows={int(mask.sum())}/{len(mask)}"
    )
    return mask


def save_route_shap_bar(shap_importance_df: pd.DataFrame, out_base: str, title_label: str, top_n: int = 10):
    """Save a compact SHAP bar plot for route-specific subset analysis."""
    if shap_importance_df.empty:
        return

    plt.rcParams["font.family"] = PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    top = shap_importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("mean(|SHAP value|)", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.text(
        0.03, 0.97, title_label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0),
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def run_route_aware_shap_analysis(
    shap_vals: np.ndarray,
    X_explain: pd.DataFrame,
    cat_maps: Dict[str, List[Any]],
    feature_names: List[str],
    output_dir: str,
    target_col: str,
    model_name: str,
    route_feature: str = "Type",
    min_rows: int = 5,
) -> Dict[str, Any]:
    """
    Save route-specific SHAP results:
    - global/all rows
    - one-step rows
    - two-step rows
    - Tem02 SHAP by route
    """
    route_dir = os.path.join(output_dir, "route_aware_SHAP")
    os.makedirs(route_dir, exist_ok=True)

    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    route_outputs: Dict[str, Any] = {
        "enabled": True,
        "output_dir": route_dir,
        "route_feature": route_feature,
    }

    route_masks = {
        "all": np.ones(len(X_explain), dtype=bool),
        "one_step": get_route_mask(X_explain, cat_maps, "one_step", route_feature=route_feature),
        "two_step": get_route_mask(X_explain, cat_maps, "two_step", route_feature=route_feature),
    }

    subset_records = []

    for subset_name, mask in route_masks.items():
        n_rows = int(mask.sum())
        if n_rows < min_rows:
            print(f"[Route SHAP] Skipped {subset_name}: only {n_rows} rows.")
            continue

        subset_shap = shap_vals[mask, :]
        mean_abs = np.abs(subset_shap).mean(axis=0)

        subset_imp = (
            pd.DataFrame({
                "feature": [SHAP_LABEL_MAP.get(f, f) for f in feature_names],
                "source_feature": feature_names,
                "mean_abs_shap": mean_abs,
                "subset": subset_name,
                "n_rows": n_rows,
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        csv_fp = os.path.join(route_dir, f"{sanitize_filename(target_col)}_{model_name}_{subset_name}_mean_abs_SHAP.csv")
        subset_imp.to_csv(csv_fp, index=False)

        bar_base = os.path.join(route_dir, f"{sanitize_filename(target_col)}_{model_name}_{subset_name}_SHAP_bar")
        save_route_shap_bar(subset_imp, bar_base, title_label=f"{subset_name.replace('_', '-')} SHAP", top_n=10)

        subset_records.append({
            "subset": subset_name,
            "n_rows": n_rows,
            "csv": csv_fp,
            "bar_basepath": bar_base,
            "top_feature": subset_imp.iloc[0]["source_feature"],
            "top_mean_abs_shap": float(subset_imp.iloc[0]["mean_abs_shap"]),
        })

    # Tem02 SHAP by route/type
    tem02_feature = "Temperature_02_boxcox"
    if tem02_feature in feature_to_idx and route_feature in X_explain.columns:
        tem02_idx = feature_to_idx[tem02_feature]
        type_codes = pd.to_numeric(X_explain[route_feature], errors="coerce").to_numpy()
        type_labels_map = {i: str(label) for i, label in enumerate(cat_maps.get(route_feature, []))}

        tem02_records = []
        for shap_v, code in zip(shap_vals[:, tem02_idx], type_codes):
            if np.isfinite(code):
                code_int = int(code)
                tem02_records.append({
                    route_feature: code_int,
                    "route_label": type_labels_map.get(code_int, str(code_int)),
                    "Tem02_SHAP": float(shap_v),
                    "abs_Tem02_SHAP": float(abs(shap_v)),
                })

        tem02_df = pd.DataFrame(tem02_records)
        tem02_csv = os.path.join(route_dir, f"{sanitize_filename(target_col)}_Tem02_SHAP_by_route.csv")
        tem02_df.to_csv(tem02_csv, index=False)

        if not tem02_df.empty:
            plt.rcParams["font.family"] = PLOT_STYLE["font_family"]
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            labels = list(tem02_df["route_label"].dropna().unique())
            data = [tem02_df.loc[tem02_df["route_label"] == lab, "Tem02_SHAP"].to_numpy() for lab in labels]
            ax.boxplot(data, labels=labels, showfliers=True)
            ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
            ax.set_ylabel("Tem02 SHAP value", fontsize=12)
            ax.set_xlabel("Synthesis route", fontsize=12)
            ax.tick_params(axis="both", labelsize=10)
            for tick in ax.get_xticklabels():
                tick.set_rotation(25)
                tick.set_ha("right")
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            fig.tight_layout()
            tem02_box_base = os.path.join(route_dir, f"{sanitize_filename(target_col)}_Tem02_SHAP_by_route_boxplot")
            fig.savefig(tem02_box_base + ".png", dpi=600, bbox_inches="tight")
            fig.savefig(tem02_box_base + ".svg", bbox_inches="tight")
            fig.savefig(tem02_box_base + ".pdf", bbox_inches="tight")
            plt.close(fig)

            route_outputs["tem02_shap_by_route_csv"] = tem02_csv
            route_outputs["tem02_shap_by_route_boxplot_basepath"] = tem02_box_base

    route_summary_csv = os.path.join(route_dir, f"{sanitize_filename(target_col)}_{model_name}_route_SHAP_summary.csv")
    pd.DataFrame(subset_records).to_csv(route_summary_csv, index=False)
    route_outputs["route_summary_csv"] = route_summary_csv
    route_outputs["subsets"] = subset_records
    return route_outputs


def run_type_tem02_shap_interaction(
    model,
    X_explain: pd.DataFrame,
    feature_names: List[str],
    cat_maps: Dict[str, List[Any]],
    output_dir: str,
    target_col: str,
    route_feature: str = "Type",
) -> Dict[str, Any]:
    """Try to compute SHAP interaction values for Type × Tem02."""
    interaction_dir = os.path.join(output_dir, "route_aware_SHAP_interactions")
    os.makedirs(interaction_dir, exist_ok=True)

    out: Dict[str, Any] = {
        "enabled": True,
        "output_dir": interaction_dir,
        "pair": f"{route_feature} x Temperature_02_boxcox",
        "status": "not_run",
    }

    if route_feature not in feature_names or "Temperature_02_boxcox" not in feature_names:
        out["status"] = "skipped_missing_feature"
        return out

    try:
        import shap
        interaction_explainer = shap.TreeExplainer(model)
        interaction_vals = interaction_explainer.shap_interaction_values(X_explain)

        if isinstance(interaction_vals, list):
            interaction_vals = interaction_vals[0]

        interaction_vals = np.asarray(interaction_vals)
        if interaction_vals.ndim != 3:
            out["status"] = f"skipped_unexpected_shape_{interaction_vals.shape}"
            return out

        idx_type = feature_names.index(route_feature)
        idx_tem02 = feature_names.index("Temperature_02_boxcox")
        pair_vals = interaction_vals[:, idx_type, idx_tem02]

        type_codes = pd.to_numeric(X_explain[route_feature], errors="coerce").to_numpy()
        type_labels_map = {i: str(label) for i, label in enumerate(cat_maps.get(route_feature, []))}

        df_pair = pd.DataFrame({
            route_feature: type_codes,
            "route_label": [type_labels_map.get(int(c), str(c)) if np.isfinite(c) else "NA" for c in type_codes],
            "Type_x_Tem02_SHAP_interaction": pair_vals,
            "abs_interaction": np.abs(pair_vals),
        })

        csv_fp = os.path.join(interaction_dir, f"{sanitize_filename(target_col)}_Type_x_Tem02_SHAP_interaction.csv")
        df_pair.to_csv(csv_fp, index=False)

        plt.rcParams["font.family"] = PLOT_STYLE["font_family"]
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        labels = list(df_pair["route_label"].dropna().unique())
        data = [df_pair.loc[df_pair["route_label"] == lab, "Type_x_Tem02_SHAP_interaction"].to_numpy() for lab in labels]
        ax.boxplot(data, labels=labels, showfliers=True)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_ylabel("SHAP interaction value", fontsize=12)
        ax.set_xlabel("Synthesis route", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        for tick in ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_ha("right")
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        fig.tight_layout()

        box_base = os.path.join(interaction_dir, f"{sanitize_filename(target_col)}_Type_x_Tem02_SHAP_interaction_boxplot")
        fig.savefig(box_base + ".png", dpi=600, bbox_inches="tight")
        fig.savefig(box_base + ".svg", bbox_inches="tight")
        fig.savefig(box_base + ".pdf", bbox_inches="tight")
        plt.close(fig)

        out.update({
            "status": "success",
            "csv": csv_fp,
            "boxplot_basepath": box_base,
            "mean_abs_interaction": float(np.mean(np.abs(pair_vals))),
        })
        return out

    except Exception as exc:
        out["status"] = "failed"
        out["error"] = str(exc)
        note_fp = os.path.join(interaction_dir, f"{sanitize_filename(target_col)}_Type_x_Tem02_interaction_failed.txt")
        with open(note_fp, "w", encoding="utf-8") as f:
            f.write(
                "SHAP interaction values could not be computed for this model/explainer combination.\\n"
                f"Error: {exc}\\n"
                "Use the route-specific Tem02 SHAP boxplot as the main diagnostic.\\n"
            )
        out["note"] = note_fp
        return out


def make_2d_pdp_panel(
    x_feature: str,
    y_feature: str,
    panel_label: str,
    X_base_pdp: pd.DataFrame,
    predict_raw_from_working_df,
    continuous_actual: List[str],
    boxcox_actual: List[str],
    categorical_actual: List[str],
    cat_maps: Dict[str, List[Any]],
    boxcox_params_X: Dict[str, Dict[str, Any]],
    grid_points: int,
    percentile_range: Tuple[float, float],
    max_categories: int,
) -> Optional[Dict[str, Any]]:
    x_axis = build_pdp_axis_values(
        x_feature, X_base_pdp, continuous_actual, boxcox_actual, categorical_actual,
        cat_maps, boxcox_params_X, grid_points, percentile_range, max_categories,
    )
    y_axis = build_pdp_axis_values(
        y_feature, X_base_pdp, continuous_actual, boxcox_actual, categorical_actual,
        cat_maps, boxcox_params_X, grid_points, percentile_range, max_categories,
    )
    if x_axis is None or y_axis is None:
        return None

    Z = np.zeros((len(y_axis["model_values"]), len(x_axis["model_values"])), dtype=float)
    for j, yv in enumerate(y_axis["model_values"]):
        for i, xv in enumerate(x_axis["model_values"]):
            X_tmp = X_base_pdp.copy()
            X_tmp[x_feature] = xv
            X_tmp[y_feature] = yv
            Z[j, i] = float(np.mean(predict_raw_from_working_df(X_tmp)))

    return {
        "panel_label": panel_label,
        "x_feature": x_feature,
        "y_feature": y_feature,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "Z": Z,
    }


def _axis_extent(values: np.ndarray, is_categorical: bool) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return float(values[0] - 0.5), float(values[0] + 0.5)
    if is_categorical:
        return -0.5, float(values.size - 0.5)
    return float(np.nanmin(values)), float(np.nanmax(values))


def plot_2d_pdp_panel_on_ax(
    ax,
    panel: Dict[str, Any],
    target_col: str,
    add_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """CEJ/Nature-style 2D PDP panel.

    - Uses contourf for continuous-continuous panels.
    - Uses tight data-supported axes based on the PDP grid.
    - Adds marginal rug marks to show real data locations.
    - Avoids the large white margins seen when full raw feature limits are used.
    """
    x_axis = panel["x_axis"]
    y_axis = panel["y_axis"]
    Z = np.asarray(panel["Z"], dtype=float)

    x_is_cat = x_axis["kind"] == "categorical"
    y_is_cat = y_axis["kind"] == "categorical"

    cmap = PDP_PLOT_STYLE.get("cmap", "viridis")
    levels_n = int(PDP_PLOT_STYLE.get("n_contour_levels", 16))

    # Robust color limits. Use shared vmin/vmax when supplied by composite.
    finite_z = Z[np.isfinite(Z)]
    if finite_z.size == 0:
        finite_z = np.array([0.0, 1.0])
    if vmin is None:
        vmin = float(np.nanpercentile(finite_z, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(finite_z, 98))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    x_vals = np.asarray(x_axis["plot_values"], dtype=float)
    y_vals = np.asarray(y_axis["plot_values"], dtype=float)

    if (not x_is_cat) and (not y_is_cat):
        Xg, Yg = np.meshgrid(x_vals, y_vals)
        levels = np.linspace(vmin, vmax, levels_n)
        im = ax.contourf(Xg, Yg, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both")
        # Light contour lines give the panel the same design-window feel as high-impact PDP figures.
        ax.contour(
            Xg, Yg, Z,
            levels=levels[::2],
            colors="k",
            linewidths=PDP_PLOT_STYLE.get("contour_lw", 0.55),
            alpha=PDP_PLOT_STYLE.get("contour_alpha", 0.40),
        )
        ax.set_xlim(float(np.nanmin(x_vals)), float(np.nanmax(x_vals)))
        ax.set_ylim(float(np.nanmin(y_vals)), float(np.nanmax(y_vals)))
    else:
        x0, x1 = _axis_extent(x_vals, x_is_cat)
        y0, y1 = _axis_extent(y_vals, y_is_cat)
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[x0, x1, y0, y1],
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        cbar.ax.tick_params(labelsize=PDP_PLOT_STYLE.get("colorbar_tick_fs", 11))
        cbar.set_label(f"Mean predicted {target_col}", fontsize=PDP_PLOT_STYLE.get("colorbar_label_fs", 13))

    ax.set_xlabel(x_axis["axis_label"], fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.set_ylabel(y_axis["axis_label"], fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.tick_params(axis="both", labelsize=PDP_PLOT_STYLE["tick_fs"], width=1.1, length=3.5)

    if x_is_cat:
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_axis["tick_labels"], rotation=35, ha="right", fontsize=10)
    if y_is_cat:
        ax.set_yticks(y_vals)
        ax.set_yticklabels(y_axis["tick_labels"], fontsize=10)

    # Marginal rug marks: actual literature data positions.
    # These are essential because PDP trends outside dense regions are extrapolative.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr = np.asarray(x_axis["actual_plot_values"], dtype=float)
    yr = np.asarray(y_axis["actual_plot_values"], dtype=float)

    if xr.size > 0 and not x_is_cat:
        xr = xr[np.isfinite(xr)]
        xr = xr[(xr >= min(x0, x1)) & (xr <= max(x0, x1))]
        if xr.size > 0:
            y_rug = min(y0, y1) + 0.015 * abs(y1 - y0)
            ax.scatter(xr, np.full_like(xr, y_rug), marker="|", s=18, c="k", alpha=0.35, linewidths=0.6, zorder=4)

    if yr.size > 0 and not y_is_cat:
        yr = yr[np.isfinite(yr)]
        yr = yr[(yr >= min(y0, y1)) & (yr <= max(y0, y1))]
        if yr.size > 0:
            x_rug = min(x0, x1) + 0.015 * abs(x1 - x0)
            ax.scatter(np.full_like(yr, x_rug), yr, marker="_", s=18, c="k", alpha=0.35, linewidths=0.6, zorder=4)

    ax.text(
        0.03, 0.97, panel["panel_label"],
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=PDP_PLOT_STYLE.get("panel_label_fs", 14),
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=PDP_PLOT_STYLE.get("panel_label_alpha", 0.62), pad=1.0),
    )

    for spine in ax.spines.values():
        spine.set_linewidth(PDP_PLOT_STYLE["spine_lw"])

    return im


def save_2d_pdp_panel(panel: Dict[str, Any], target_col: str, out_base: str):
    plt.rcParams["font.family"] = PDP_PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=PDP_PLOT_STYLE["figsize"])
    plot_2d_pdp_panel_on_ax(ax, panel, target_col=target_col, add_colorbar=True)
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=PDP_PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def save_2d_pdp_composite(panels: List[Dict[str, Any]], target_col: str, out_base: str, ncols: int = 2):
    if not panels:
        return
    plt.rcParams["font.family"] = PDP_PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Shared color scale because all panels are the same output target (SSA).
    all_z = np.concatenate([np.asarray(p["Z"], dtype=float).ravel() for p in panels])
    all_z = all_z[np.isfinite(all_z)]
    if all_z.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanpercentile(all_z, 2))
        vmax = float(np.nanpercentile(all_z, 98))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    last_im = None
    for ax, panel in zip(axes_flat, panels):
        last_im = plot_2d_pdp_panel_on_ax(ax, panel, target_col=target_col, add_colorbar=False, vmin=vmin, vmax=vmax)
    for ax in axes_flat[len(panels):]:
        ax.axis("off")

    # One shared colorbar keeps the main figure cleaner and closer to journal style.
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_flat[:len(panels)], fraction=0.028, pad=0.025)
        cbar.ax.tick_params(labelsize=PDP_PLOT_STYLE.get("colorbar_tick_fs", 11))
        cbar.set_label(f"Mean predicted {target_col}", fontsize=PDP_PLOT_STYLE.get("colorbar_label_fs", 13))

    fig.savefig(out_base + ".png", dpi=PDP_PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------- MAIN ----------------
def main():
    print(f"Loading input: {INPUT_FP}")
    df = pd.read_excel(INPUT_FP, sheet_name=DATA_SHEET)
    df.columns = [c.strip() for c in df.columns]

    rename_actions = {c: LEGACY_RENAME_MAP[c] for c in df.columns if c in LEGACY_RENAME_MAP}
    if rename_actions:
        df = df.rename(columns=rename_actions)
        print(f"Applied legacy renames for {len(rename_actions)} columns.")

    df, missing_created, presence_map = check_and_create_required_columns(df, REQUIRED_COLUMNS, auto_create=AUTO_CREATE_MISSING_COLUMNS)
    if missing_created:
        print("Created missing columns as NaN:", missing_created)

    # coerce numeric
    for col in CONTINUOUS_COLS + BOXCOX_BASES + [target_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # BoxCox temperatures (FEATURES; global)
    boxcox_params_X: Dict[str, Dict[str, Any]] = {}
    for col in BOXCOX_BASES:
        bc_col = col + "_boxcox"
        data = df[col].dropna().astype(float) if col in df.columns else pd.Series(dtype=float)
        if data.empty:
            df[bc_col] = np.nan
            boxcox_params_X[col] = {"lam": None, "shift": 0.0}
            continue

        min_val = float(data.min())
        shift = 0.0 if min_val > 0 else abs(min_val) + 1.0
        shifted = data + shift
        try:
            transformed, lam = boxcox(shifted)
            df[bc_col] = np.nan
            df.loc[data.index, bc_col] = transformed
            boxcox_params_X[col] = {"lam": float(lam), "shift": float(shift)}
        except Exception as e:
            print(f"BoxCox failed for {col}, using shifted raw values. Error: {e}")
            df[bc_col] = np.nan
            df.loc[data.index, bc_col] = shifted
            boxcox_params_X[col] = {"lam": None, "shift": float(shift), "note": str(e)}

    boxcox_transformed_cols = [c + "_boxcox" for c in BOXCOX_BASES]

    input_cols = (
        [c for c in CONTINUOUS_COLS if c in df.columns]
        + [c for c in boxcox_transformed_cols if c in df.columns]
        + [c for c in CATEGORICAL_COLS if c in df.columns]
    )
    final_df = df[input_cols + [target_col]].copy().reset_index(drop=True)

    # Filter SSA > 0
    final_df = final_df[final_df[target_col].notnull()]
    final_df = final_df[final_df[target_col] > 0].reset_index(drop=True)

    print(f"Target ({target_col}) stats after filtering (pre outlier removal):")
    print(final_df[target_col].describe())

    # RAW-IQR outlier removal
    final_df, outlier_details = remove_outliers_iqr_raw(final_df, target_col, k=IQR_K_RAW)

    print(f"Target ({target_col}) stats after RAW-IQR outlier removal:")
    print(final_df[target_col].describe())

    if SAVE_TARGET_SPREAD_PLOT:
        save_target_spread_plot_raw_vs_boxcox(final_df[target_col].to_numpy(), TARGET_SPREAD_PLOT_FP)

    # Split
    y = final_df[[target_col]]
    X = final_df[input_cols]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=SEED)

    # Align NA drop
    X_train_sel, y_train_sel = dropna_align(X_train[input_cols], y_train)
    X_val_sel, y_val_sel = dropna_align(X_val[input_cols], y_val)
    X_test_sel, y_test_sel = dropna_align(X_test[input_cols], y_test)

    continuous_actual = [c for c in CONTINUOUS_COLS if c in X_train_sel.columns]
    boxcox_actual = [c for c in boxcox_transformed_cols if c in X_train_sel.columns]
    categorical_actual = [c for c in CATEGORICAL_COLS if c in X_train_sel.columns]

    # Factorize categoricals on TRAIN only (unseen -> -1)
    cat_maps: Dict[str, List[Any]] = {}
    for c in categorical_actual:
        vals = X_train_sel[c].astype(object).fillna("__NA__")
        codes, uniques = pd.factorize(vals, sort=True)
        X_train_sel[c] = codes
        cat_maps[c] = [None if u == "__NA__" else u for u in uniques.tolist()]

        mapping = {str(u): i for i, u in enumerate(uniques)}
        def map_cat(series: pd.Series) -> pd.Series:
            s = series.astype(object).fillna("__NA__").astype(str)
            return s.map(mapping).fillna(-1).astype(int)

        X_val_sel[c] = map_cat(X_val_sel[c])
        X_test_sel[c] = map_cat(X_test_sel[c])

    # Robust scale continuous X only
    scaler_X: Optional[RobustScaler] = None
    if continuous_actual:
        scaler_X = RobustScaler()
        X_train_cont_scaled = scaler_X.fit_transform(X_train_sel[continuous_actual])
        X_val_cont_scaled = scaler_X.transform(X_val_sel[continuous_actual])
        X_test_cont_scaled = scaler_X.transform(X_test_sel[continuous_actual])
    else:
        X_train_cont_scaled = np.empty((len(X_train_sel), 0))
        X_val_cont_scaled = np.empty((len(X_val_sel), 0))
        X_test_cont_scaled = np.empty((len(X_test_sel), 0))

    X_train_box = X_train_sel[boxcox_actual].to_numpy() if boxcox_actual else np.empty((len(X_train_sel), 0))
    X_val_box = X_val_sel[boxcox_actual].to_numpy() if boxcox_actual else np.empty((len(X_val_sel), 0))
    X_test_box = X_test_sel[boxcox_actual].to_numpy() if boxcox_actual else np.empty((len(X_test_sel), 0))

    X_train_cat = X_train_sel[categorical_actual].to_numpy() if categorical_actual else np.empty((len(X_train_sel), 0))
    X_val_cat = X_val_sel[categorical_actual].to_numpy() if categorical_actual else np.empty((len(X_val_sel), 0))
    X_test_cat = X_test_sel[categorical_actual].to_numpy() if categorical_actual else np.empty((len(X_test_sel), 0))

    X_train_final = np.concatenate([X_train_cont_scaled, X_train_box, X_train_cat], axis=1)
    X_val_final = np.concatenate([X_val_cont_scaled, X_val_box, X_val_cat], axis=1)
    X_test_final = np.concatenate([X_test_cont_scaled, X_test_box, X_test_cat], axis=1)

    # ---------------- y transform: BoxCox (train-only fit) + Robust scale ----------------
    y_train_raw = y_train_sel.to_numpy().reshape(-1, 1)
    y_val_raw = y_val_sel.to_numpy().reshape(-1, 1)
    y_test_raw = y_test_sel.to_numpy().reshape(-1, 1)

    if (y_train_raw <= 0).any() or (y_val_raw <= 0).any() or (y_test_raw <= 0).any():
        raise ValueError("SSA must be > 0 for BoxCox target transform. Check filtering/outliers.")

    y_train_bc, y_val_bc, boxcox_y_params = boxcox_transform_train_only(y_train_raw, y_val_raw)
    _, y_test_bc, _ = boxcox_transform_train_only(y_train_raw, y_test_raw)

    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_bc)
    y_val_scaled = scaler_y.transform(y_val_bc)
    y_test_scaled = scaler_y.transform(y_test_bc)

    print(f"[Target transform] SSA: BoxCox (train-only λ={boxcox_y_params['lam']:.6f})")

    # ---------------- GBDT GridSearch ----------------
    gbdt_param_grid = {
        "n_estimators": [600],
        "learning_rate": [0.1],
        "max_depth": [None],
        "subsample": [0.6],
        "max_features": ["sqrt"],
        "min_samples_leaf": [6],
        "min_samples_split": [3],
        "ccp_alpha": [1e-4],
    }

    base = GradientBoostingRegressor(random_state=SEED)
    grid = GridSearchCV(
        estimator=base,
        param_grid=gbdt_param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )

    print("=== GridSearchCV start (GBDT, SSA; RAW-IQR then BoxCox(y)) ===")
    grid.fit(X_train_final, y_train_scaled.ravel())
    best_params = grid.best_params_
    print("Best GBDT params:", best_params)

    model = GradientBoostingRegressor(**best_params, random_state=SEED)
    model.fit(X_train_final, y_train_scaled.ravel())

    # Predict -> inverse scaler -> inverse BoxCox -> RAW SSA
    y_train_pred_scaled = model.predict(X_train_final).reshape(-1, 1)
    y_val_pred_scaled = model.predict(X_val_final).reshape(-1, 1)
    y_test_pred_scaled = model.predict(X_test_final).reshape(-1, 1)

    y_train_pred_bc = scaler_y.inverse_transform(y_train_pred_scaled)
    y_val_pred_bc = scaler_y.inverse_transform(y_val_pred_scaled)
    y_test_pred_bc = scaler_y.inverse_transform(y_test_pred_scaled)

    lam = float(boxcox_y_params["lam"])
    y_train_pred = boxcox_inverse(y_train_pred_bc, lam=lam)
    y_val_pred = boxcox_inverse(y_val_pred_bc, lam=lam)
    y_test_pred = boxcox_inverse(y_test_pred_bc, lam=lam)

    y_train_pred = np.maximum(y_train_pred, 0)
    y_val_pred = np.maximum(y_val_pred, 0)
    y_test_pred = np.maximum(y_test_pred, 0)

    y_train_actual = y_train_raw
    y_val_actual = y_val_raw
    y_test_actual = y_test_raw

    r2_train, rmse_train = get_metrics(y_train_actual, y_train_pred)
    r2_val, rmse_val = get_metrics(y_val_actual, y_val_pred)
    r2_test, rmse_test = get_metrics(y_test_actual, y_test_pred)

    print(f"Training ({target_col}): R2={r2_train:.3f}, RMSE={rmse_train:.3f}")
    print(f"Validation ({target_col}): R2={r2_val:.3f}, RMSE={rmse_val:.3f}")
    print(f"Test ({target_col}): R2={r2_test:.3f}, RMSE={rmse_test:.3f}")

    # Plots (CatBoost style)
    settings = AXIS_SETTINGS_TRUEPRED.get(target_col, {})
    plot_files = {
        "train": plot_true_vs_pred(
            y_train_actual, y_train_pred, "Train", target_col, model_name, SAVE_DIR, **settings
        ),
        "validation": plot_true_vs_pred(
            y_val_actual, y_val_pred, "Val", target_col, model_name, SAVE_DIR, **settings
        ),
        "test": plot_true_vs_pred(
            y_test_actual, y_test_pred, "Test", target_col, model_name, SAVE_DIR, **settings
        ),
    }

    shap_outputs: Dict[str, Any] = {"enabled": bool(RUN_SHAP)}
    pdp_outputs: Dict[str, Any] = {"enabled": False}
    if RUN_SHAP:
        try:
            import shap
            print("[SHAP] shap imported successfully.")
        except Exception as exc:
            raise RuntimeError("SHAP is required. Install with: pip install shap") from exc

        shap_dir = os.path.join(SAVE_DIR, "SHAP")
        os.makedirs(shap_dir, exist_ok=True)
        print("[SHAP] Output dir:", shap_dir)

        feature_names = continuous_actual + boxcox_actual + categorical_actual
        display_names = [SHAP_LABEL_MAP.get(name, name) for name in feature_names]
        seen: Dict[str, int] = {}
        unique_display_names = []
        for name in display_names:
            if name not in seen:
                seen[name] = 0
                unique_display_names.append(name)
            else:
                seen[name] += 1
                unique_display_names.append(f"{name}_{seen[name]}")
        display_name_by_source = dict(zip(feature_names, unique_display_names))

        X_train_df_shap = pd.DataFrame(X_train_final, columns=feature_names)
        X_test_df_shap = pd.DataFrame(X_test_final, columns=feature_names)

        rng = np.random.default_rng(SHAP_SEED)
        if len(X_train_df_shap) > SHAP_MAX_BACKGROUND:
            bg_idx = rng.choice(len(X_train_df_shap), size=SHAP_MAX_BACKGROUND, replace=False)
            X_bg = X_train_df_shap.iloc[bg_idx]
        else:
            X_bg = X_train_df_shap

        if len(X_test_df_shap) > SHAP_MAX_EXPLAIN:
            ex_idx = rng.choice(len(X_test_df_shap), size=SHAP_MAX_EXPLAIN, replace=False)
            X_explain = X_test_df_shap.iloc[ex_idx]
        else:
            X_explain = X_test_df_shap

        explainer_name = "shap.TreeExplainer(model, data=X_bg)"
        try:
            explainer = shap.TreeExplainer(model, data=X_bg)
            shap_exp = explainer(X_explain)
            shap_vals = np.asarray(shap_exp.values)
        except Exception as tree_exc:
            print("[SHAP] TreeExplainer failed; using PermutationExplainer:", tree_exc)
            max_evals = 2 * X_explain.shape[1] + 1
            explainer = shap.PermutationExplainer(model.predict, X_bg)
            shap_exp = explainer(X_explain, max_evals=max_evals)
            shap_vals = np.asarray(shap_exp.values)
            explainer_name = "shap.PermutationExplainer(model.predict, X_bg)"

        if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
            shap_vals = shap_vals[:, :, 0]
        print("[SHAP] SHAP computed:", shap_vals.shape)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_imp = (
            pd.DataFrame(
                {
                    "feature": [display_name_by_source.get(name, name) for name in X_explain.columns],
                    "source_feature": X_explain.columns,
                    "mean_abs_shap": mean_abs,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        target_safe = sanitize_filename(target_col)
        shap_imp_csv = os.path.join(shap_dir, f"{target_safe}_{model_name}_mean_abs_shap.csv")
        shap_imp.to_csv(shap_imp_csv, index=False)
        print("[SHAP] Saved:", shap_imp_csv)

        # =========================================================
        # ROUTE-AWARE SHAP ANALYSIS
        # =========================================================
        route_shap_outputs: Dict[str, Any] = {"enabled": False}
        route_interaction_outputs: Dict[str, Any] = {"enabled": False}

        if RUN_ROUTE_AWARE_SHAP:
            route_shap_outputs = run_route_aware_shap_analysis(
                shap_vals=shap_vals,
                X_explain=X_explain,
                cat_maps=cat_maps,
                feature_names=list(X_explain.columns),
                output_dir=shap_dir,
                target_col=target_col,
                model_name=model_name,
                route_feature=ROUTE_FEATURE,
                min_rows=MIN_ROUTE_ROWS_FOR_SHAP,
            )

            route_interaction_outputs = run_type_tem02_shap_interaction(
                model=model,
                X_explain=X_explain,
                feature_names=list(X_explain.columns),
                cat_maps=cat_maps,
                output_dir=shap_dir,
                target_col=target_col,
                route_feature=ROUTE_FEATURE,
            )


        # =========================================================
        # GROUPED SHAP + TOP 3 PDP
        # =========================================================
        pdp_outputs = {"enabled": False}
        if RUN_PDP:
            pdp_dir = os.path.join(SAVE_DIR, "PDP_top3_grouped_SHAP")
            os.makedirs(pdp_dir, exist_ok=True)

            grouped_shap_df, group_to_features = compute_grouped_shap_importance(
                shap_vals=shap_vals,
                feature_names=list(X_explain.columns),
            )
            grouped_shap_csv = os.path.join(
                pdp_dir,
                f"{sanitize_filename(target_col)}_{model_name}_grouped_SHAP_importance.csv",
            )
            grouped_shap_df.to_csv(grouped_shap_csv, index=False)
            print("[PDP] Saved grouped SHAP:", grouped_shap_csv)

            top_groups = grouped_shap_df.head(PDP_TOP_N_GROUPS).copy()

            # Use original working dataframe for PDP averaging:
            # continuous features are raw values;
            # temperature features are already Box-Cox transformed;
            # categorical features are already encoded with train-only maps.
            X_working_all = pd.concat([X_train_sel, X_val_sel, X_test_sel], axis=0, ignore_index=True)

            rng_pdp = np.random.default_rng(PDP_RANDOM_SEED)
            if len(X_working_all) > PDP_MAX_BASE_ROWS:
                sampled_idx = rng_pdp.choice(len(X_working_all), size=PDP_MAX_BASE_ROWS, replace=False)
                X_base_pdp = X_working_all.iloc[sampled_idx].copy().reset_index(drop=True)
            else:
                X_base_pdp = X_working_all.copy().reset_index(drop=True)

            def transform_working_df_to_model_array(X_work: pd.DataFrame) -> np.ndarray:
                if continuous_actual:
                    X_cont = scaler_X.transform(X_work[continuous_actual])
                else:
                    X_cont = np.empty((len(X_work), 0))

                X_box = X_work[boxcox_actual].to_numpy() if boxcox_actual else np.empty((len(X_work), 0))
                X_cat = X_work[categorical_actual].to_numpy() if categorical_actual else np.empty((len(X_work), 0))
                return np.concatenate([X_cont, X_box, X_cat], axis=1)

            def predict_raw_from_working_df(X_work: pd.DataFrame) -> np.ndarray:
                X_arr = transform_working_df_to_model_array(X_work)
                y_scaled_pred = model.predict(X_arr).reshape(-1, 1)
                y_bc_pred = scaler_y.inverse_transform(y_scaled_pred)
                y_raw_pred = boxcox_inverse(y_bc_pred, lam=lam)
                y_raw_pred = np.maximum(y_raw_pred, 0)
                return y_raw_pred.ravel()

            pdp_records = []

            for rank_i, row in top_groups.iterrows():
                group_name = row["group"]
                group_display = row["group_display"]
                features_in_group = group_to_features[group_name]

                # For Model 1, most groups contain one feature.
                # If a group contains multiple features, use the strongest individual feature inside that group.
                if len(features_in_group) > 1:
                    local_imp = shap_imp[shap_imp["source_feature"].isin(features_in_group)].copy()
                    local_imp = local_imp.sort_values("mean_abs_shap", ascending=False)
                    pdp_feature = local_imp.iloc[0]["source_feature"]
                    note = f"group_multiple_features_used_top_individual: {pdp_feature}"
                else:
                    pdp_feature = features_in_group[0]
                    note = "single_feature_group"

                print(f"[PDP] Rank {rank_i + 1}: group={group_name}, feature={pdp_feature}")

                # ---------------- Continuous 1D PDP ----------------
                if pdp_feature in continuous_actual:
                    values = X_base_pdp[pdp_feature].dropna().to_numpy(dtype=float)
                    lo = np.nanquantile(values, PDP_PERCENTILE_RANGE[0])
                    hi = np.nanquantile(values, PDP_PERCENTILE_RANGE[1])
                    if np.isclose(lo, hi):
                        print(f"[PDP] Skipped {pdp_feature}: quantile range too small.")
                        continue

                    grid = np.linspace(lo, hi, PDP_GRID_POINTS)
                    y_means = []
                    for v in grid:
                        X_tmp = X_base_pdp.copy()
                        X_tmp[pdp_feature] = v
                        y_means.append(np.mean(predict_raw_from_working_df(X_tmp)))
                    y_means = np.asarray(y_means)

                    out_base = os.path.join(
                        pdp_dir,
                        f"rank{rank_i + 1}_{sanitize_filename(group_display)}_{sanitize_filename(pdp_feature)}_PDP",
                    )
                    save_pdp_plot_numeric(
                        x_grid=grid,
                        y_mean=y_means,
                        x_label=SHAP_LABEL_MAP.get(pdp_feature, pdp_feature),
                        y_label=f"Mean predicted {target_col}",
                        title_label=f"({rank_i + 1}) {group_display}",
                        out_base=out_base,
                    )
                    pdp_records.append({
                        "rank": int(rank_i + 1),
                        "group": group_name,
                        "group_display": group_display,
                        "pdp_feature": pdp_feature,
                        "pdp_type": "continuous_1D",
                        "note": note,
                        "output_basepath": out_base,
                    })

                # ---------------- Box-Cox temperature 1D PDP ----------------
                elif pdp_feature in boxcox_actual:
                    values_bc = X_base_pdp[pdp_feature].dropna().to_numpy(dtype=float)
                    lo = np.nanquantile(values_bc, PDP_PERCENTILE_RANGE[0])
                    hi = np.nanquantile(values_bc, PDP_PERCENTILE_RANGE[1])
                    if np.isclose(lo, hi):
                        print(f"[PDP] Skipped {pdp_feature}: quantile range too small.")
                        continue

                    grid_bc = np.linspace(lo, hi, PDP_GRID_POINTS)
                    y_means = []
                    for v_bc in grid_bc:
                        X_tmp = X_base_pdp.copy()
                        X_tmp[pdp_feature] = v_bc
                        y_means.append(np.mean(predict_raw_from_working_df(X_tmp)))
                    y_means = np.asarray(y_means)

                    if pdp_feature == "Temperature_01_boxcox":
                        raw_grid = inverse_boxcox_feature(
                            grid_bc,
                            lam=boxcox_params_X["Temperature_01"]["lam"],
                            shift=boxcox_params_X["Temperature_01"]["shift"],
                        )
                        x_label = "Temperature 01 (°C)"
                    elif pdp_feature == "Temperature_02_boxcox":
                        raw_grid = inverse_boxcox_feature(
                            grid_bc,
                            lam=boxcox_params_X["Temperature_02"]["lam"],
                            shift=boxcox_params_X["Temperature_02"]["shift"],
                        )
                        x_label = "Temperature 02 (°C)"
                    else:
                        raw_grid = grid_bc
                        x_label = SHAP_LABEL_MAP.get(pdp_feature, pdp_feature)

                    out_base = os.path.join(
                        pdp_dir,
                        f"rank{rank_i + 1}_{sanitize_filename(group_display)}_{sanitize_filename(pdp_feature)}_PDP",
                    )
                    save_pdp_plot_numeric(
                        x_grid=raw_grid,
                        y_mean=y_means,
                        x_label=x_label,
                        y_label=f"Mean predicted {target_col}",
                        title_label=f"({rank_i + 1}) {group_display}",
                        out_base=out_base,
                    )
                    pdp_records.append({
                        "rank": int(rank_i + 1),
                        "group": group_name,
                        "group_display": group_display,
                        "pdp_feature": pdp_feature,
                        "pdp_type": "boxcox_temperature_1D",
                        "note": note,
                        "output_basepath": out_base,
                    })

                # ---------------- Categorical response plot ----------------
                elif pdp_feature in categorical_actual:
                    codes = sorted([
                        int(c)
                        for c in X_base_pdp[pdp_feature].dropna().unique()
                        if int(c) >= 0
                    ])
                    y_means = []
                    labels = []
                    for code in codes:
                        X_tmp = X_base_pdp.copy()
                        X_tmp[pdp_feature] = int(code)
                        y_means.append(np.mean(predict_raw_from_working_df(X_tmp)))

                        if pdp_feature in cat_maps and int(code) < len(cat_maps[pdp_feature]):
                            labels.append(str(cat_maps[pdp_feature][int(code)]))
                        else:
                            labels.append(str(code))
                    y_means = np.asarray(y_means)

                    out_base = os.path.join(
                        pdp_dir,
                        f"rank{rank_i + 1}_{sanitize_filename(group_display)}_{sanitize_filename(pdp_feature)}_category_response",
                    )
                    save_pdp_plot_categorical(
                        labels=labels,
                        y_mean=y_means,
                        x_label=SHAP_LABEL_MAP.get(pdp_feature, pdp_feature),
                        y_label=f"Mean predicted {target_col}",
                        title_label=f"({rank_i + 1}) {group_display}",
                        out_base=out_base,
                    )
                    pdp_records.append({
                        "rank": int(rank_i + 1),
                        "group": group_name,
                        "group_display": group_display,
                        "pdp_feature": pdp_feature,
                        "pdp_type": "categorical_response",
                        "note": note + "; category_response_used_not_smooth_line",
                        "output_basepath": out_base,
                    })

                else:
                    print(f"[PDP] Skipped {pdp_feature}: feature type not recognized.")

            # =========================================================
            # MAIN-PAPER 2D PDP / DESIGN-WINDOW PANELS
            # =========================================================
            main_2d_outputs = {"enabled": False}
            if RUN_MAIN_2D_PDP:
                main_2d_dir = os.path.join(SAVE_DIR, "PDP_main_2D_design_windows")
                os.makedirs(main_2d_dir, exist_ok=True)
                main_2d_records = []
                main_2d_panels = []

                for pair_i, pair in enumerate(MAIN_2D_PDP_PAIRS, start=1):
                    x_feature = pair["x"]
                    y_feature = pair["y"]
                    pair_label = pair.get("label", f"{x_feature} x {y_feature}")
                    panel_label = f"({chr(96 + pair_i)}) {pair_label}"

                    subset = pair.get("subset", "all")
                    subset_mask = get_route_mask(
                        X_base_pdp,
                        cat_maps,
                        subset=subset,
                        route_feature=ROUTE_FEATURE,
                    )
                    X_panel_base = X_base_pdp.loc[subset_mask].copy().reset_index(drop=True)

                    if len(X_panel_base) < MIN_ROUTE_ROWS_FOR_PDP:
                        print(
                            f"[Main 2D PDP] Skipped {pair_label}: subset={subset}, "
                            f"rows={len(X_panel_base)} < {MIN_ROUTE_ROWS_FOR_PDP}"
                        )
                        main_2d_records.append({
                            "rank": pair_i,
                            "x_feature": x_feature,
                            "y_feature": y_feature,
                            "label": pair_label,
                            "subset": subset,
                            "n_rows_used": int(len(X_panel_base)),
                            "status": "skipped_too_few_rows",
                            "output_basepath": None,
                        })
                        continue

                    panel = make_2d_pdp_panel(
                        x_feature=x_feature,
                        y_feature=y_feature,
                        panel_label=panel_label,
                        X_base_pdp=X_panel_base,
                        predict_raw_from_working_df=predict_raw_from_working_df,
                        continuous_actual=continuous_actual,
                        boxcox_actual=boxcox_actual,
                        categorical_actual=categorical_actual,
                        cat_maps=cat_maps,
                        boxcox_params_X=boxcox_params_X,
                        grid_points=MAIN_2D_PDP_GRID_POINTS,
                        percentile_range=MAIN_2D_PDP_PERCENTILE_RANGE,
                        max_categories=MAIN_2D_PDP_MAX_CATEGORIES,
                    )
                    if panel is None:
                        main_2d_records.append({
                            "rank": pair_i,
                            "x_feature": x_feature,
                            "y_feature": y_feature,
                            "label": pair_label,
                            "subset": subset,
                            "n_rows_used": int(len(X_panel_base)),
                            "status": "skipped",
                            "output_basepath": None,
                        })
                        continue

                    out_base = os.path.join(
                        main_2d_dir,
                        f"panel{pair_i}_{sanitize_filename(target_col)}_{sanitize_filename(pair_label)}_2D_PDP",
                    )
                    save_2d_pdp_panel(panel, target_col=target_col, out_base=out_base)
                    main_2d_panels.append(panel)
                    main_2d_records.append({
                        "rank": pair_i,
                        "x_feature": x_feature,
                        "y_feature": y_feature,
                        "label": pair_label,
                        "subset": subset,
                        "n_rows_used": int(len(X_panel_base)),
                        "status": "saved",
                        "output_basepath": out_base,
                    })
                    print("[MAIN 2D PDP] Saved:", out_base + ".png/.svg/.pdf")

                composite_base = None
                if main_2d_panels:
                    composite_base = os.path.join(
                        main_2d_dir,
                        f"{sanitize_filename(target_col)}_{model_name}_main_2D_PDP_composite",
                    )
                    save_2d_pdp_composite(
                        main_2d_panels,
                        target_col=target_col,
                        out_base=composite_base,
                        ncols=2,
                    )
                    print("[MAIN 2D PDP] Saved composite:", composite_base + ".png/.svg/.pdf")

                main_2d_summary_csv = os.path.join(
                    main_2d_dir,
                    f"{sanitize_filename(target_col)}_{model_name}_main_2D_PDP_summary.csv",
                )
                pd.DataFrame(main_2d_records).to_csv(main_2d_summary_csv, index=False)

                main_2d_outputs = {
                    "enabled": True,
                    "output_dir": main_2d_dir,
                    "summary_csv": main_2d_summary_csv,
                    "composite_basepath": composite_base,
                    "pairs": MAIN_2D_PDP_PAIRS,
                    "grid_points": int(MAIN_2D_PDP_GRID_POINTS),
                }

            pdp_summary_csv = os.path.join(
                pdp_dir,
                f"{sanitize_filename(target_col)}_{model_name}_PDP_top3_summary.csv",
            )
            pd.DataFrame(pdp_records).to_csv(pdp_summary_csv, index=False)

            pdp_outputs = {
                "enabled": True,
                "output_dir": pdp_dir,
                "grouped_shap_csv": grouped_shap_csv,
                "pdp_summary_csv": pdp_summary_csv,
                "top_n_groups": int(PDP_TOP_N_GROUPS),
                "grid_points": int(PDP_GRID_POINTS),
                "percentile_range": PDP_PERCENTILE_RANGE,
                "max_base_rows": int(PDP_MAX_BASE_ROWS),
                "main_2d_outputs": main_2d_outputs,
            }
            print("[PDP] Done. Saved summary:", pdp_summary_csv)

        top_n = int(min(SHAP_TOP_N, len(shap_imp)))
        top = shap_imp.head(top_n).copy()
        top_rev = top.iloc[::-1]
        top_source_features = top["source_feature"].tolist()
        top_labels = top["feature"].tolist()
        feat_to_idx = {feature: i for i, feature in enumerate(X_explain.columns)}

        plt.rcParams["font.family"] = PLOT_STYLE["font_family"]

        fig = plt.figure(figsize=SHAP_PLOT_STYLE["figsize_bar"])
        ax = fig.add_axes(SHAP_AXES_RECT_BAR)
        ax.barh(
            top_rev["feature"],
            top_rev["mean_abs_shap"],
            color=SHAP_PLOT_STYLE["bar_color"],
            alpha=SHAP_PLOT_STYLE["bar_alpha"],
        )
        ax.set_xlabel("mean(|SHAP value|)", fontsize=SHAP_PLOT_STYLE["label_fs"])
        ax.tick_params(axis="x", labelsize=SHAP_PLOT_STYLE["tick_fs"])
        ax.tick_params(axis="y", labelsize=SHAP_PLOT_STYLE["ytick_fs"])
        set_spines_lw(ax, SHAP_PLOT_STYLE["spine_lw"])
        apply_shap_xaxis_controls(
            ax,
            SHAP_AXIS_SETTINGS_BAR,
            values=top["mean_abs_shap"].to_numpy(float),
            start_at_zero=True,
        )

        shap_bar_base = os.path.join(shap_dir, f"{target_safe}_{model_name}_SHAP_bar_top_clean")
        save_shap_figure_multi(fig, shap_bar_base)
        plt.close(fig)
        print("[SHAP] Saved:", shap_bar_base + ".png/.svg/.pdf")

        def robust_minmax(values, lo=1, hi=99):
            values = np.asarray(values)
            vmin = np.nanpercentile(values, lo)
            vmax = np.nanpercentile(values, hi)
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
            return float(vmin), float(vmax)

        fig = plt.figure(figsize=SHAP_PLOT_STYLE["figsize_strip"])
        ax = fig.add_axes(SHAP_AXES_RECT_STRIP)
        for row_i, feature in enumerate(top_source_features[::-1]):
            j = feat_to_idx[feature]
            shap_j = shap_vals[:, j]
            x_j = X_explain[feature].to_numpy()
            jitter = (rng.random(len(shap_j)) - 0.5) * 0.55
            yj = row_i + jitter

            vmin, vmax = robust_minmax(x_j)
            colors = np.clip((x_j - vmin) / (vmax - vmin), 0, 1)
            ax.scatter(
                shap_j,
                yj,
                c=colors,
                cmap="viridis",
                s=SHAP_PLOT_STYLE["strip_s"],
                alpha=SHAP_PLOT_STYLE["strip_alpha"],
                edgecolors="none",
            )

        ax.axvline(0, color="red", linestyle="--", linewidth=2)
        ax.set_yticks(range(len(top_source_features)))
        ax.set_yticklabels(top_labels[::-1], fontsize=SHAP_PLOT_STYLE["ytick_fs"])
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=SHAP_PLOT_STYLE["label_fs"])
        ax.tick_params(axis="x", labelsize=SHAP_PLOT_STYLE["tick_fs"])
        set_spines_lw(ax, SHAP_PLOT_STYLE["spine_lw"])
        strip_axis_values = np.concatenate([shap_vals[:, feat_to_idx[feature]] for feature in top_source_features])
        apply_shap_xaxis_controls(
            ax,
            SHAP_AXIS_SETTINGS_STRIP,
            values=strip_axis_values,
            start_at_zero=False,
        )

        shap_strip_base = os.path.join(shap_dir, f"{target_safe}_{model_name}_SHAP_strip_top_clean")
        save_shap_figure_multi(fig, shap_strip_base)
        plt.close(fig)
        print("[SHAP] Saved:", shap_strip_base + ".png/.svg/.pdf")

        shap_outputs = {
            "enabled": True,
            "output_dir": shap_dir,
            "mean_abs_shap_csv": shap_imp_csv,
            "bar_top_basepath": shap_bar_base,
            "strip_top_basepath": shap_strip_base,
            "plot_file_formats": ["png", "svg", "pdf"],
            "top_n": top_n,
            "background_rows": int(len(X_bg)),
            "explained_rows": int(len(X_explain)),
            "axis_settings_bar": SHAP_AXIS_SETTINGS_BAR,
            "axis_settings_strip": SHAP_AXIS_SETTINGS_STRIP,
            "explainer": explainer_name,
            "model_output": "scaled BoxCox target prediction before inverse RobustScaler and inverse BoxCox",
            "pdp_outputs": pdp_outputs,
            "route_shap_outputs": route_shap_outputs,
            "route_interaction_outputs": route_interaction_outputs,
        }
        print("[SHAP] Done.")

    # Save artifacts
    artifacts = {
        "model_params": model.get_params(),
        "best_params": best_params,
        "metrics_original_units": {
            "train": {"r2": float(r2_train), "rmse": float(rmse_train)},
            "val": {"r2": float(r2_val), "rmse": float(rmse_val)},
            "test": {"r2": float(r2_test), "rmse": float(rmse_test)},
        },
        "legacy_renames_applied": rename_actions,
        "missing_columns_created": missing_created,
        "required_columns_presence": presence_map,
        "input_file": INPUT_FP,
        "save_dir": SAVE_DIR,
        "plot_files_basepath": plot_files,
        "plot_file_formats": ["png", "svg", "pdf"],
        "shap_outputs": shap_outputs,
        "boxcox_params_X": boxcox_params_X,
        "boxcox_params_y": boxcox_y_params,
        "feature_order": {
            "continuous": continuous_actual,
            "boxcox_transformed": boxcox_actual,
            "categorical": categorical_actual,
        },
        "final_columns": list(final_df.columns),
        "cat_maps": cat_maps,
        "raw_iqr_outlier_removal": outlier_details,
        "target_spread_plot": TARGET_SPREAD_PLOT_FP if SAVE_TARGET_SPREAD_PLOT else None,
        "plot_style": PLOT_STYLE,
        "axis_settings_truepred": AXIS_SETTINGS_TRUEPRED,
    }

    joblib.dump(model, os.path.join(SAVE_DIR, "model.pkl"))
    if scaler_X is not None:
        joblib.dump(scaler_X, os.path.join(SAVE_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(SAVE_DIR, "scaler_y.pkl"))
    joblib.dump(artifacts, os.path.join(SAVE_DIR, "artifacts.pkl"))
    joblib.dump(boxcox_y_params, os.path.join(SAVE_DIR, "boxcox_params_y.pkl"))

    results_df = pd.DataFrame([{
        "target": target_col,
        "model": model_name,
        "best_params": json.dumps(make_json_serializable(best_params)),
        "target_transform": f"boxcox(lam={boxcox_y_params['lam']:.6f})",
        "raw_iqr_k": float(IQR_K_RAW),
        "r2_train": float(r2_train),
        "rmse_train": float(rmse_train),
        "r2_val": float(r2_val),
        "rmse_val": float(rmse_val),
        "r2_test": float(r2_test),
        "rmse_test": float(rmse_test),
        "plot_train_basepath": plot_files["train"],
        "plot_validation_basepath": plot_files["validation"],
        "plot_test_basepath": plot_files["test"],
        "plot_file_formats": "png/svg/pdf",
        "shap_mean_abs_csv": shap_outputs.get("mean_abs_shap_csv"),
        "shap_bar_basepath": shap_outputs.get("bar_top_basepath"),
        "shap_strip_basepath": shap_outputs.get("strip_top_basepath"),
        "pdp_output_dir": shap_outputs.get("pdp_outputs", {}).get("output_dir"),
        "pdp_summary_csv": shap_outputs.get("pdp_outputs", {}).get("pdp_summary_csv"),
    }])
    csv_path = os.path.join(SAVE_DIR, f"{sanitize_filename(target_col.lower())}_gbdt_summary.csv")
    results_df.to_csv(csv_path, index=False)
    print("Summary saved to:", csv_path)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(artifacts), f, indent=2, ensure_ascii=False)
    print("Report JSON saved to:", REPORT_JSON)

    print("\nArtifacts saved to:", SAVE_DIR)
    print("SSA GBDT strict pipeline complete (CatBoost-style plots).")


if __name__ == "__main__":
    main()

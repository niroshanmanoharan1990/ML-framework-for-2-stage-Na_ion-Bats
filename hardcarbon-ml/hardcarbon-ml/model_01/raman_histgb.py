import matplotlib
matplotlib.use("Agg")

import os
import json
import math
import joblib
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor


# =========================================================
# RAMAN IGID HISTGB PIPELINE
# Updated journal-style version:
# - lighter journal-style parity plots
# - lighter journal-style SHAP plots
# - grouped SHAP importance
# - top grouped-SHAP 1D PDP for Supplementary
# - route-aware 2D PDP/design-window maps for main figure
# - Tem01 × Tem02 PDP is restricted to two-step samples
#
# IMPORTANT:
# This file assumes categorical variables are already numeric codes
# in the Excel file, as in your original IG/ID pipeline.
#
# If your Type coding is different:
# - update ONE_STEP_TYPE_CODES and TWO_STEP_TYPE_CODES below.
# =========================================================


# =========================
# CONFIGURATION
# =========================
# ---- Repository paths (relative to the repository root) ----
from pathlib import Path as _RepoPath
REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "SI_Model_01.xlsx"
DATA_SHEET = "Model_01_ID_IG_Imputated"
OUTPUT_DIR = REPO_ROOT / "outputs" / "model_01/raman_histgb"

input_fp = str(DATA_FILE)
save_dir = str(OUTPUT_DIR)
report_json = str(OUTPUT_DIR / "Raman_encoding_imputation_report.json")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.dirname(report_json), exist_ok=True)

SEED = 42
target_col = "IG/ID"
model_name = "HistGB"
IQR_K = 1.5


# =========================
# SHAP CONFIG
# =========================
RUN_SHAP = True
SHAP_MAX_BACKGROUND = 200
SHAP_MAX_EXPLAIN = 500
SHAP_TOP_N = 10
SHAP_SEED = 42

# Journal-style SHAP layout.
SHAP_AXES_RECT_BAR = [0.35, 0.18, 0.58, 0.74]
SHAP_AXES_RECT_STRIP = [0.35, 0.18, 0.58, 0.74]

# Keep your original axis settings for comparability.
SHAP_AXIS_SETTINGS_BAR: Dict[str, Any] = {}
SHAP_AXIS_SETTINGS_STRIP: Dict[str, Any] = {}


# =========================
# PDP CONFIG
# =========================
RUN_PDP = True
PDP_TOP_N_GROUPS = 3
PDP_GRID_POINTS = 60
PDP_PERCENTILE_RANGE = (0.05, 0.95)
PDP_MAX_BASE_ROWS = 500
PDP_RANDOM_SEED = 42

RUN_MAIN_2D_PDP = True
MAIN_2D_PDP_GRID_POINTS = 60
MAIN_2D_PDP_PERCENTILE_RANGE = (0.05, 0.95)
MAIN_2D_PDP_MAX_CATEGORIES = 12

# For Raman IG/ID, these main PDPs are chemically defensible:
# - Temperature + oxygen content: heteroatom removal and defect evolution
# - Temperature + fixed carbon: carbon-forming ability and structural ordering
# - Temperature + volatile matter: devolatilization-driven defect formation
# - Tem01 × Tem02: only for two-step samples
MAIN_2D_PDP_PAIRS = [
    {"x": "Temperature_01_boxcox", "y": "Oxygen%", "label": "Tem01 × O%"},
    {"x": "Temperature_01_boxcox", "y": "Fixed Carbon%", "label": "Tem01 × FC%"},
    {"x": "Temperature_01_boxcox", "y": "Volatile Matter%", "label": "Tem01 × VM%"},
    {"x": "Temperature_01_boxcox", "y": "Temperature_02_boxcox", "label": "Tem01 × Tem02", "subset": "two_step"},
]

# Route coding for Type.
# Update these if your Excel uses another coding system.
ROUTE_FEATURE = "Type"
ONE_STEP_TYPE_CODES = [0]
TWO_STEP_TYPE_CODES = [1]
MIN_ROUTE_ROWS_FOR_PDP = 10


# =========================
# LABEL MAP
# =========================
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
    "Temperature_01": "Tem01",
    "Temperature_02": "Tem02",
    "Prewash": "Pre",
    "Postwash": "Post",
    "Type": "Type",
    "Method_01": "M01",
    "Activator_01": "Act01",
    "Method_02": "M02",
    "Activator_02": "Act02",
}


# =========================
# JOURNAL-STYLE PLOT SETTINGS
# =========================
PLOT_STYLE = {
    "font_family": "Arial",

    # Journal-style parity-plot text and outline
    "label_fs": 16,
    "tick_fs": 13,
    "spine_lw": 1.5,

    "scatter_alpha": 0.70,
    "scatter_s": 55,
    "scatter_color": "#1f77b4",

    "diag_color": "red",
    "diag_ls": "--",
    "diag_lw": 1.3,

    "ann_fs": 12,
    "ann_bbox": dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.15", alpha=0.65),

    "figsize": (4.4, 4.4),
    "dpi_png": 600,

    "svg_fonttype": "none",
    "pdf_fonttype": 42,
    "ps_fonttype": 42,
}

SHAP_PLOT_STYLE = {
    # Journal-style SHAP figures.
    "figsize_bar": (5.2, 4.8),
    "figsize_strip": (5.2, 4.8),
    "dpi": 600,
    "spine_lw": 1.5,
    "label_fs": 14,
    "tick_fs": 11,
    "ytick_fs": 11,
    "bar_color": "#1f77b4",
    "bar_alpha": 0.9,
    "strip_s": 24,
    "strip_alpha": 0.70,
}

PDP_PLOT_STYLE = {
    # Journal-style 2D PDP/design-window maps.
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


# =========================
# STRICT REQUIRED HEADERS
# =========================
REQUIRED_COLUMNS: List[str] = [
    "Cellulose%", "Hemi-cellulose%", "Lignin%", "Volatile Matter%", "Ash%",
    "Fixed Carbon%", "Carbon%", "Oxygen%", "Hydrogen%",
    "Prewash", "Postwash", "Type", "Method_01", "Activator_01",
    "Time_01", "Heating rate_01", "Temperature_01",
    "Method_02", "Activator_02",
    "Time_02", "Heating rate_02", "Temperature_02",
    "IG/ID",
]

CONTINUOUS_COLS: List[str] = [
    "Cellulose%", "Hemi-cellulose%", "Lignin%", "Volatile Matter%", "Ash%",
    "Fixed Carbon%", "Carbon%", "Oxygen%", "Hydrogen%",
    "Time_01", "Heating rate_01",
    "Time_02", "Heating rate_02",
]
BOXCOX_BASES: List[str] = ["Temperature_01", "Temperature_02"]
CATEGORICAL_COLS: List[str] = ["Prewash", "Postwash", "Type", "Method_01", "Activator_01", "Method_02", "Activator_02"]

# Legacy renaming support for older Excel headers.
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
    "Heating Rate 01": "Heating rate_01",
    "Temperature 01": "Temperature_01",
    "Temperature 02": "Temperature_02",
    "Method 01": "Method_01",
    "Activator 01": "Activator_01",
    "Method 02": "Method_02",
    "Activatior 02": "Activator_02",
    "Time 02": "Time_02",
    "Heating rate 02": "Heating rate_02",
    "Heating Rate 02": "Heating rate_02",
    "Heating Rate_02": "Heating rate_02",
    "Heating rate_02": "Heating rate_02",
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def sanitize_filename(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("×", "x")
        .replace("%", "pct")
    )


def heuristic_heating_rate_02(columns: List[str]) -> Dict[str, str]:
    targets = {"heatingrate02", "heating_rate_02", "heating rate 02", "heating rate_02", "heatingrate_02"}
    normalized_targets = {t.replace(" ", "").replace("-", "").replace("_", "") for t in targets}
    extra = {}
    for c in columns:
        norm = str(c).lower().replace(" ", "").replace("-", "").replace("_", "")
        if norm in normalized_targets and c != "Heating rate_02":
            extra[c] = "Heating rate_02"
    return extra


def make_json_serializable(o: Any):
    if o is None:
        return None
    if isinstance(o, dict):
        return {str(k): make_json_serializable(v) for k, v in o.items()}
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


def apply_plot_style():
    plt.rcParams["font.family"] = PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = PLOT_STYLE["svg_fonttype"]
    plt.rcParams["pdf.fonttype"] = PLOT_STYLE["pdf_fonttype"]
    plt.rcParams["ps.fonttype"] = PLOT_STYLE["ps_fonttype"]


def save_figure_multi(fig, basepath_no_ext: str, dpi: Optional[int] = None):
    if dpi is None:
        dpi = PLOT_STYLE["dpi_png"]
    fig.savefig(basepath_no_ext + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".svg", bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".pdf", bbox_inches="tight")


def set_spines(ax, lw: Optional[float] = None):
    if lw is None:
        lw = PLOT_STYLE["spine_lw"]
    for spine in ax.spines.values():
        spine.set_linewidth(lw)


def get_metrics(y_true, y_pred) -> Tuple[float, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return float(r2), float(rmse)


def nice_step(raw_step: float) -> float:
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


def nice_limits_ticks(data, n_ticks=6, padding=0.05, clamp_min_zero=False):
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        vmin, vmax = 0.0, 1.0
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


def apply_shap_xaxis_controls(ax, settings: Dict[str, Any], values=None, start_at_zero=False):
    if settings.get("xlim") is not None:
        xlim = tuple(float(v) for v in settings["xlim"])
        ax.set_xlim(xlim)
        if settings.get("xtick_step") is not None:
            step = float(settings["xtick_step"])
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * step, step))
        return

    if values is None:
        return

    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    if start_at_zero:
        xmin = 0.0
        xmax = float(np.nanmax(values))
    else:
        m = float(np.nanmax(np.abs(values)))
        xmin, xmax = -m, m

    if np.isclose(xmin, xmax):
        xmax = xmin + 1.0

    pad = 0.05 * (xmax - xmin)
    if start_at_zero:
        xmax += pad
    else:
        xmin -= pad
        xmax += pad

    step = settings.get("xtick_step")
    if step is None:
        step = nice_step((xmax - xmin) / 4)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(xmin, xmax + 0.5 * step, step))


def boxcox_forward(values: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    values = pd.to_numeric(values, errors="coerce")
    data = values.dropna().astype(float)
    if data.empty:
        return pd.Series(np.nan, index=values.index), {"lam": None, "shift": 0.0, "note": "empty"}
    min_val = float(data.min())
    shift = 0.0 if min_val > 0 else abs(min_val) + 1.0
    shifted = data + shift
    out = pd.Series(np.nan, index=values.index)
    try:
        transformed, lam = boxcox(shifted)
        out.loc[data.index] = transformed
        return out, {"lam": float(lam), "shift": float(shift)}
    except Exception as exc:
        out.loc[data.index] = shifted
        return out, {"lam": None, "shift": float(shift), "note": str(exc)}


def inverse_boxcox_feature(x_bc: np.ndarray, lam: Optional[float], shift: float = 0.0) -> np.ndarray:
    x_bc = np.asarray(x_bc, dtype=float)
    if lam is None:
        return x_bc - shift
    if abs(lam) < 1e-12:
        return np.exp(x_bc) - shift
    return np.power(lam * x_bc + 1.0, 1.0 / lam) - shift


def remove_outliers_iqr(dfin: pd.DataFrame, col: str, k: float = 1.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    q1 = float(dfin[col].quantile(0.25))
    q3 = float(dfin[col].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    cleaned = dfin[(dfin[col] >= lower) & (dfin[col] <= upper)].copy().reset_index(drop=True)
    details = {
        "method": "iqr",
        "column": col,
        "k": float(k),
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "lower": float(lower),
        "upper": float(upper),
        "rows_before": int(len(dfin)),
        "rows_after": int(len(cleaned)),
        "rows_removed": int(len(dfin) - len(cleaned)),
    }
    print(
        f"Outlier removal ({col}): lower={lower:.4f}, upper={upper:.4f}, "
        f"before={len(dfin)}, after={len(cleaned)}, removed={len(dfin)-len(cleaned)}"
    )
    return cleaned, details


def dropna_align(Xd: pd.DataFrame, yd: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_comb = pd.concat([Xd, yd], axis=1)
    df_comb = df_comb.dropna().reset_index(drop=True)
    return df_comb[Xd.columns], df_comb[yd.columns]


def plot_true_vs_pred_artifact(
    y_true,
    y_pred,
    set_name,
    target_name,
    model_name="HistGB",
    save_dir=None,
    xlim=None,
    ylim=None,
    xtick_step=None,
    ytick_step=None,
):
    apply_plot_style()
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r2, rmse = get_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.scatter(
        y_true,
        y_pred,
        alpha=PLOT_STYLE["scatter_alpha"],
        s=PLOT_STYLE["scatter_s"],
        color=PLOT_STYLE["scatter_color"],
    )

    ax.set_xlabel("True Value", fontsize=PLOT_STYLE["label_fs"])
    ax.set_ylabel("Predicted Value", fontsize=PLOT_STYLE["label_fs"])

    if xlim is None or xtick_step is None:
        x0, x1, xstep, _ = nice_limits_ticks(y_true, clamp_min_zero=False)
        xlim = (x0, x1) if xlim is None else xlim
        xtick_step = xstep if xtick_step is None else xtick_step

    if ylim is None or ytick_step is None:
        y0, y1, ystep, _ = nice_limits_ticks(y_pred, clamp_min_zero=False)
        ylim = (y0, y1) if ylim is None else ylim
        ytick_step = ystep if ytick_step is None else ytick_step

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * xtick_step, xtick_step))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.5 * ytick_step, ytick_step))
    ax.tick_params(axis="both", labelsize=PLOT_STYLE["tick_fs"], width=1.1, length=3.5)

    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], color=PLOT_STYLE["diag_color"], linestyle=PLOT_STYLE["diag_ls"], linewidth=PLOT_STYLE["diag_lw"])

    textstr = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\n{model_name}"
    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=PLOT_STYLE["ann_fs"],
        va="bottom",
        ha="right",
        bbox=PLOT_STYLE["ann_bbox"],
    )

    set_spines(ax, PLOT_STYLE["spine_lw"])
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    base = None
    if save_dir is not None:
        base = os.path.join(save_dir, f"{sanitize_filename(target_name)}_{set_name}_{sanitize_filename(model_name)}_true_vs_pred")
        save_figure_multi(fig, base)
        print(f"Plot saved to: {base}.png/.svg/.pdf")

    plt.close(fig)
    return base


def make_group_name(source_feature: str) -> str:
    if source_feature == "Temperature_01_boxcox":
        return "Temperature_01"
    if source_feature == "Temperature_02_boxcox":
        return "Temperature_02"
    return source_feature


def make_group_display_name(group_name: str) -> str:
    return SHAP_LABEL_MAP.get(group_name, group_name)


def compute_grouped_shap_importance(shap_vals: np.ndarray, feature_names: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    group_to_features: Dict[str, List[str]] = {}
    for f in feature_names:
        group_to_features.setdefault(make_group_name(f), []).append(f)

    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    rows = []
    for group_name, feats in group_to_features.items():
        idxs = [feature_to_idx[f] for f in feats]
        grouped_vals = shap_vals[:, idxs].sum(axis=1)
        rows.append({
            "group": group_name,
            "group_display": make_group_display_name(group_name),
            "features_in_group": ";".join(feats),
            "n_features": len(feats),
            "mean_abs_grouped_shap": float(np.abs(grouped_vals).mean()),
        })

    grouped = pd.DataFrame(rows).sort_values("mean_abs_grouped_shap", ascending=False).reset_index(drop=True)
    return grouped, group_to_features


def save_shap_figure_multi(fig, basepath_no_ext: str):
    fig.savefig(basepath_no_ext + ".png", dpi=SHAP_PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".svg", bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".pdf", bbox_inches="tight")


def save_pdp_plot_numeric(x_grid, y_mean, x_label, y_label, title_label, out_base):
    plt.rcParams["font.family"] = PDP_PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=PDP_PLOT_STYLE["figsize"])
    ax.plot(x_grid, y_mean, linewidth=PDP_PLOT_STYLE["line_lw"])
    ax.scatter(x_grid, y_mean, s=PDP_PLOT_STYLE["scatter_s"], alpha=PDP_PLOT_STYLE["scatter_alpha"])
    ax.set_xlabel(x_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.set_ylabel(y_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.tick_params(axis="both", labelsize=PDP_PLOT_STYLE["tick_fs"], width=1.1, length=3.5)
    ax.text(
        0.03,
        0.97,
        title_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PDP_PLOT_STYLE["panel_label_fs"],
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=PDP_PLOT_STYLE["panel_label_alpha"], pad=1.0),
    )
    set_spines(ax, PDP_PLOT_STYLE["spine_lw"])
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=PDP_PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def save_pdp_plot_categorical(labels, y_mean, x_label, y_label, title_label, out_base):
    plt.rcParams["font.family"] = PDP_PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=PDP_PLOT_STYLE["figsize"])
    x = np.arange(len(labels))
    ax.bar(x, y_mean)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_xlabel(x_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.set_ylabel(y_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.tick_params(axis="y", labelsize=PDP_PLOT_STYLE["tick_fs"], width=1.1, length=3.5)
    ax.text(
        0.03,
        0.97,
        title_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PDP_PLOT_STYLE["panel_label_fs"],
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=PDP_PLOT_STYLE["panel_label_alpha"], pad=1.0),
    )
    set_spines(ax, PDP_PLOT_STYLE["spine_lw"])
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=PDP_PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def get_route_mask(X_df: pd.DataFrame, subset: Optional[str]) -> np.ndarray:
    if subset is None or str(subset).lower() in ("all", "none"):
        return np.ones(len(X_df), dtype=bool)

    if ROUTE_FEATURE not in X_df.columns:
        print(f"[Route-aware] Route feature '{ROUTE_FEATURE}' not found. Using all rows.")
        return np.ones(len(X_df), dtype=bool)

    subset_norm = str(subset).lower().strip()
    if subset_norm in ("one", "one_step", "single", "single_step"):
        codes = ONE_STEP_TYPE_CODES
    elif subset_norm in ("two", "two_step", "double", "double_step"):
        codes = TWO_STEP_TYPE_CODES
    else:
        print(f"[Route-aware] Unknown subset '{subset}'. Using all rows.")
        return np.ones(len(X_df), dtype=bool)

    route_values = pd.to_numeric(X_df[ROUTE_FEATURE], errors="coerce")
    mask = route_values.isin(codes).to_numpy()
    print(f"[Route-aware] subset={subset_norm}, feature={ROUTE_FEATURE}, codes={codes}, rows={int(mask.sum())}/{len(mask)}")
    return mask


def build_pdp_axis_values(
    feature: str,
    X_base_pdp: pd.DataFrame,
    continuous_actual: List[str],
    boxcox_actual_transformed: List[str],
    categorical_actual: List[str],
    boxcox_params_X: Dict[str, Dict[str, Any]],
    grid_points: int,
    percentile_range: Tuple[float, float],
    max_categories: int,
) -> Optional[Dict[str, Any]]:
    if feature not in X_base_pdp.columns:
        print(f"[2D PDP] Skipped {feature}: not in dataframe.")
        return None

    if feature in continuous_actual:
        vals = pd.to_numeric(X_base_pdp[feature], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size < 2:
            return None
        lo = float(np.nanquantile(vals, percentile_range[0]))
        hi = float(np.nanquantile(vals, percentile_range[1]))
        if np.isclose(lo, hi):
            return None
        model_values = np.linspace(lo, hi, grid_points)
        return {
            "kind": "continuous",
            "feature": feature,
            "model_values": model_values,
            "plot_values": model_values,
            "actual_plot_values": vals,
            "axis_label": SHAP_LABEL_MAP.get(feature, feature),
            "tick_labels": None,
        }

    if feature in boxcox_actual_transformed:
        vals_bc = pd.to_numeric(X_base_pdp[feature], errors="coerce").dropna().to_numpy(dtype=float)
        if vals_bc.size < 2:
            return None
        lo = float(np.nanquantile(vals_bc, percentile_range[0]))
        hi = float(np.nanquantile(vals_bc, percentile_range[1]))
        if np.isclose(lo, hi):
            return None
        grid_bc = np.linspace(lo, hi, grid_points)

        if feature == "Temperature_01_boxcox":
            params = boxcox_params_X.get("Temperature_01", {"lam": None, "shift": 0.0})
            plot_values = inverse_boxcox_feature(grid_bc, params.get("lam"), params.get("shift", 0.0))
            actual_plot_values = inverse_boxcox_feature(vals_bc, params.get("lam"), params.get("shift", 0.0))
            axis_label = "Temperature 01 (°C)"
        elif feature == "Temperature_02_boxcox":
            params = boxcox_params_X.get("Temperature_02", {"lam": None, "shift": 0.0})
            plot_values = inverse_boxcox_feature(grid_bc, params.get("lam"), params.get("shift", 0.0))
            actual_plot_values = inverse_boxcox_feature(vals_bc, params.get("lam"), params.get("shift", 0.0))
            axis_label = "Temperature 02 (°C)"
        else:
            plot_values = grid_bc
            actual_plot_values = vals_bc
            axis_label = SHAP_LABEL_MAP.get(feature, feature)

        return {
            "kind": "continuous",
            "feature": feature,
            "model_values": grid_bc,
            "plot_values": plot_values,
            "actual_plot_values": actual_plot_values,
            "axis_label": axis_label,
            "tick_labels": None,
        }

    if feature in categorical_actual:
        codes = sorted([int(c) for c in pd.to_numeric(X_base_pdp[feature], errors="coerce").dropna().unique() if int(c) >= 0])
        if not codes:
            return None
        codes = codes[:max_categories]
        plot_values = np.arange(len(codes), dtype=float)
        tick_labels = [str(c) for c in codes]
        actual_codes = pd.to_numeric(X_base_pdp[feature], errors="coerce").dropna().to_numpy(dtype=float)
        actual_plot = []
        code_to_pos = {code: i for i, code in enumerate(codes)}
        for c in actual_codes:
            ci = int(c)
            if ci in code_to_pos:
                actual_plot.append(code_to_pos[ci])
        return {
            "kind": "categorical",
            "feature": feature,
            "model_values": np.asarray(codes, dtype=float),
            "plot_values": plot_values,
            "actual_plot_values": np.asarray(actual_plot, dtype=float),
            "axis_label": SHAP_LABEL_MAP.get(feature, feature),
            "tick_labels": tick_labels,
        }

    return None


def make_2d_pdp_panel(
    x_feature,
    y_feature,
    panel_label,
    X_base_pdp,
    predict_raw_from_working_df,
    continuous_actual,
    boxcox_actual_transformed,
    categorical_actual,
    boxcox_params_X,
    grid_points,
    percentile_range,
    max_categories,
):
    x_axis = build_pdp_axis_values(
        x_feature, X_base_pdp, continuous_actual, boxcox_actual_transformed, categorical_actual,
        boxcox_params_X, grid_points, percentile_range, max_categories
    )
    y_axis = build_pdp_axis_values(
        y_feature, X_base_pdp, continuous_actual, boxcox_actual_transformed, categorical_actual,
        boxcox_params_X, grid_points, percentile_range, max_categories
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


def plot_2d_pdp_panel_on_ax(ax, panel, target_col, add_colorbar=True, vmin=None, vmax=None):
    x_axis = panel["x_axis"]
    y_axis = panel["y_axis"]
    Z = np.asarray(panel["Z"], dtype=float)

    x_is_cat = x_axis["kind"] == "categorical"
    y_is_cat = y_axis["kind"] == "categorical"

    cmap = PDP_PLOT_STYLE.get("cmap", "viridis")
    levels_n = int(PDP_PLOT_STYLE.get("n_contour_levels", 16))

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

    # Marginal rug marks show data-supported regions.
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

    set_spines(ax, PDP_PLOT_STYLE["spine_lw"])
    return im


def save_2d_pdp_panel(panel, target_col, out_base):
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


def save_2d_pdp_composite(panels, target_col, out_base, ncols=2):
    if not panels:
        return

    plt.rcParams["font.family"] = PDP_PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

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

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_flat[:len(panels)], fraction=0.028, pad=0.025)
        cbar.ax.tick_params(labelsize=PDP_PLOT_STYLE.get("colorbar_tick_fs", 11))
        cbar.set_label(f"Mean predicted {target_col}", fontsize=PDP_PLOT_STYLE.get("colorbar_label_fs", 13))

    fig.savefig(out_base + ".png", dpi=PDP_PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================
def main():
    # ---- 1. LOAD DATA + LEGACY HEADER RENAME ----
    df = pd.read_excel(input_fp, sheet_name=DATA_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    legacy_actions = {c: LEGACY_RENAME_MAP[c] for c in df.columns if c in LEGACY_RENAME_MAP}
    if legacy_actions:
        df = df.rename(columns=legacy_actions)

    if "Heating rate_02" not in df.columns:
        hr2_map = heuristic_heating_rate_02(list(df.columns))
        if hr2_map:
            df = df.rename(columns=hr2_map)
            legacy_actions.update(hr2_map)

    present_cols = list(df.columns)

    missing = sorted([c for c in REQUIRED_COLUMNS if c not in present_cols])
    if missing:
        msg = [
            "ERROR: Required headers are missing. The dataset must match the strict header set exactly.",
            f"- Missing ({len(missing)}): {missing}",
            f"- Present columns: {present_cols}",
            f"- Required columns: {REQUIRED_COLUMNS}",
            f"- Legacy renames applied: {legacy_actions if legacy_actions else 'None'}",
        ]
        raise ValueError("\n".join(msg))

    print("Raw IG/ID stats:")
    print(df[target_col].describe())

    # ---- 2. ENSURE NUMERIC TYPES ----
    for col in CONTINUOUS_COLS + BOXCOX_BASES + [target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for c in CATEGORICAL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- 3. Box-Cox transform for Temperature columns ----
    boxcox_params_X: Dict[str, Dict[str, Any]] = {}
    for col in BOXCOX_BASES:
        transformed, params = boxcox_forward(df[col])
        df[col + "_boxcox"] = transformed
        boxcox_params_X[col] = params

    BOXCOX_TRANSFORMED_COLS = [c + "_boxcox" for c in BOXCOX_BASES]

    # ---- 4. BUILD FINAL DATAFRAME ----
    input_cols = CONTINUOUS_COLS + BOXCOX_TRANSFORMED_COLS + CATEGORICAL_COLS
    final_df = df[input_cols + [target_col]].copy().reset_index(drop=True)

    # ---- 5. FILTER TARGET AND REMOVE RAW-TARGET OUTLIERS ----
    final_df = final_df[final_df[target_col].notnull()]
    final_df = final_df[final_df[target_col] > 0].reset_index(drop=True)
    print(f"Target ({target_col}) stats after basic filtering:")
    print(final_df[target_col].describe())

    final_df, outlier_details = remove_outliers_iqr(final_df, target_col, k=IQR_K)
    print(f"Target ({target_col}) stats after outlier removal:")
    print(final_df[target_col].describe())

    y = final_df[[target_col]]
    X = final_df[input_cols]

    # ---- 6. SPLIT ----
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=SEED)
    selected_features = X.columns.tolist()

    # ---- 7. ALIGN & SCALING ----
    X_train_sel, y_train_sel = dropna_align(X_train[selected_features], y_train)
    X_val_sel, y_val_sel = dropna_align(X_val[selected_features], y_val)
    X_test_sel, y_test_sel = dropna_align(X_test[selected_features], y_test)

    continuous_actual = [col for col in CONTINUOUS_COLS if col in X_train_sel.columns]
    boxcox_actual_transformed = [col for col in BOXCOX_TRANSFORMED_COLS if col in X_train_sel.columns]
    categorical_actual = [col for col in CATEGORICAL_COLS if col in X_train_sel.columns]

    for c in categorical_actual:
        X_train_sel[c] = pd.to_numeric(X_train_sel[c], errors="coerce")
        X_val_sel[c] = pd.to_numeric(X_val_sel[c], errors="coerce")
        X_test_sel[c] = pd.to_numeric(X_test_sel[c], errors="coerce")

    scaler_X = RobustScaler()
    if continuous_actual:
        X_train_scaled_cont = scaler_X.fit_transform(X_train_sel[continuous_actual])
        X_val_scaled_cont = scaler_X.transform(X_val_sel[continuous_actual])
        X_test_scaled_cont = scaler_X.transform(X_test_sel[continuous_actual])
    else:
        X_train_scaled_cont = np.empty((len(X_train_sel), 0))
        X_val_scaled_cont = np.empty((len(X_val_sel), 0))
        X_test_scaled_cont = np.empty((len(X_test_sel), 0))

    X_train_box = X_train_sel[boxcox_actual_transformed].to_numpy() if boxcox_actual_transformed else np.empty((len(X_train_sel), 0))
    X_val_box = X_val_sel[boxcox_actual_transformed].to_numpy() if boxcox_actual_transformed else np.empty((len(X_val_sel), 0))
    X_test_box = X_test_sel[boxcox_actual_transformed].to_numpy() if boxcox_actual_transformed else np.empty((len(X_test_sel), 0))

    X_train_cat = X_train_sel[categorical_actual].to_numpy() if categorical_actual else np.empty((len(X_train_sel), 0))
    X_val_cat = X_val_sel[categorical_actual].to_numpy() if categorical_actual else np.empty((len(X_val_sel), 0))
    X_test_cat = X_test_sel[categorical_actual].to_numpy() if categorical_actual else np.empty((len(X_test_sel), 0))

    X_train_final = np.concatenate([X_train_scaled_cont, X_train_box, X_train_cat], axis=1)
    X_val_final = np.concatenate([X_val_scaled_cont, X_val_box, X_val_cat], axis=1)
    X_test_final = np.concatenate([X_test_scaled_cont, X_test_box, X_test_cat], axis=1)

    feature_names = continuous_actual + boxcox_actual_transformed + categorical_actual

    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_sel)
    y_val_scaled = scaler_y.transform(y_val_sel)
    y_test_scaled = scaler_y.transform(y_test_sel)

    # ---- 8. HistGradientBoosting GridSearch ----
    histgb_param_grid = {
        "max_iter": [800],
        "learning_rate": [0.05],
        "max_depth": [6],
        "l2_regularization": [1.0],
        "min_samples_leaf": [2],
        "max_bins": [128],
    }
    histgb_fixed_params = {
        "random_state": SEED,
    }

    histgb = HistGradientBoostingRegressor(**histgb_fixed_params)
    grid = GridSearchCV(
        histgb,
        histgb_param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )

    print("=== GridSearchCV start (HistGB, Raman IG/ID) ===")
    grid.fit(X_train_final, y_train_scaled.ravel())
    best_params = grid.best_params_
    print("Best HistGB params:", best_params)

    all_params = dict(histgb_fixed_params)
    all_params.update(best_params)
    histgb_final = HistGradientBoostingRegressor(**all_params)
    histgb_final.fit(X_train_final, y_train_scaled.ravel())

    def inverse_predict_from_array(X_arr: np.ndarray) -> np.ndarray:
        y_pred_scaled = histgb_final.predict(X_arr).reshape(-1, 1)
        y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
        return y_pred_actual.reshape(-1)

    y_train_pred = inverse_predict_from_array(X_train_final)
    y_val_pred = inverse_predict_from_array(X_val_final)
    y_test_pred = inverse_predict_from_array(X_test_final)

    y_train_actual = scaler_y.inverse_transform(y_train_scaled).reshape(-1)
    y_val_actual = scaler_y.inverse_transform(y_val_scaled).reshape(-1)
    y_test_actual = scaler_y.inverse_transform(y_test_scaled).reshape(-1)

    axis_settings = {
        "IG/ID": {
            "xlim": (0, 3.0),
            "ylim": (0, 3.0),
            "xtick_step": 0.5,
            "ytick_step": 0.5,
        }
    }

    r2_train, rmse_train = get_metrics(y_train_actual, y_train_pred)
    r2_val, rmse_val = get_metrics(y_val_actual, y_val_pred)
    r2_test, rmse_test = get_metrics(y_test_actual, y_test_pred)

    print(f"Training (IG/ID): R2={r2_train:.3f}, RMSE={rmse_train:.3f}")
    print(f"Validation (IG/ID): R2={r2_val:.3f}, RMSE={rmse_val:.3f}")
    print(f"Test (IG/ID): R2={r2_test:.3f}, RMSE={rmse_test:.3f}")

    plot_files = {}
    for y_true, y_pred, set_name in [
        (y_train_actual, y_train_pred, "Train"),
        (y_val_actual, y_val_pred, "Val"),
        (y_test_actual, y_test_pred, "Test"),
    ]:
        settings = axis_settings.get(target_col, {})
        plot_files[set_name.lower()] = plot_true_vs_pred_artifact(
            y_true, y_pred, set_name, target_col, model_name=model_name, save_dir=save_dir, **settings
        )

    # Working dataframe for PDP: raw continuous values, Box-Cox temperature values, numeric categorical codes.
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

        X_box = X_work[boxcox_actual_transformed].to_numpy() if boxcox_actual_transformed else np.empty((len(X_work), 0))
        X_cat = X_work[categorical_actual].to_numpy() if categorical_actual else np.empty((len(X_work), 0))
        return np.concatenate([X_cont, X_box, X_cat], axis=1)

    def predict_raw_from_working_df(X_work: pd.DataFrame) -> np.ndarray:
        X_arr = transform_working_df_to_model_array(X_work)
        return inverse_predict_from_array(X_arr).reshape(-1)

    # ---- 9. SHAP ANALYSIS ----
    shap_outputs: Dict[str, Any] = {"enabled": bool(RUN_SHAP)}
    pdp_outputs: Dict[str, Any] = {"enabled": False}

    if RUN_SHAP:
        try:
            import shap
            print("[SHAP] shap imported successfully.")
        except Exception as exc:
            raise RuntimeError("SHAP is required. Install with: pip install shap") from exc

        shap_dir = os.path.join(save_dir, "SHAP")
        os.makedirs(shap_dir, exist_ok=True)
        print("[SHAP] Output dir:", shap_dir)

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

        # HistGB is not fully supported by TreeExplainer in every SHAP version.
        # PermutationExplainer is slower but robust and matches your original Raman pipeline.
        max_evals = 2 * X_explain.shape[1] + 1
        explainer_name = "shap.PermutationExplainer(histgb_final.predict, X_bg)"
        explainer = shap.PermutationExplainer(histgb_final.predict, X_bg)
        shap_exp = explainer(X_explain, max_evals=max_evals)
        shap_vals = np.asarray(shap_exp.values)

        if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
            shap_vals = shap_vals[:, :, 0]

        print("[SHAP] SHAP computed:", shap_vals.shape)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_imp = (
            pd.DataFrame({
                "feature": [SHAP_LABEL_MAP.get(name, name) for name in feature_names],
                "source_feature": feature_names,
                "mean_abs_shap": mean_abs,
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        shap_imp_csv = os.path.join(shap_dir, "IGID_HistGB_mean_abs_shap.csv")
        shap_imp.to_csv(shap_imp_csv, index=False)
        print("[SHAP] Saved:", shap_imp_csv)

        # ---------------- SHAP BAR ----------------
        top_n = int(min(SHAP_TOP_N, len(shap_imp)))
        top = shap_imp.head(top_n).copy()
        top_rev = top.iloc[::-1]

        plt.rcParams["font.family"] = PLOT_STYLE["font_family"]

        fig = plt.figure(figsize=SHAP_PLOT_STYLE["figsize_bar"], dpi=SHAP_PLOT_STYLE["dpi"])
        ax = fig.add_axes(SHAP_AXES_RECT_BAR)
        ax.barh(
            top_rev["feature"],
            top_rev["mean_abs_shap"],
            color=SHAP_PLOT_STYLE["bar_color"],
            alpha=SHAP_PLOT_STYLE["bar_alpha"],
        )
        ax.set_xlabel("mean(|SHAP value|)", fontsize=SHAP_PLOT_STYLE["label_fs"])
        ax.tick_params(axis="x", labelsize=SHAP_PLOT_STYLE["tick_fs"], width=1.1, length=3.5)
        ax.tick_params(axis="y", labelsize=SHAP_PLOT_STYLE["ytick_fs"], width=1.1, length=3.5)
        set_spines(ax, SHAP_PLOT_STYLE["spine_lw"])
        apply_shap_xaxis_controls(
            ax,
            SHAP_AXIS_SETTINGS_BAR,
            values=top["mean_abs_shap"].to_numpy(float),
            start_at_zero=True,
        )

        shap_bar_base = os.path.join(shap_dir, "IGID_HistGB_SHAP_bar_top_clean")
        save_shap_figure_multi(fig, shap_bar_base)
        plt.close(fig)
        print("[SHAP] Saved:", shap_bar_base + ".png/.svg/.pdf")

        # ---------------- SHAP STRIP ----------------
        def robust_minmax(values, lo=1, hi=99):
            values = np.asarray(values)
            vmin = np.nanpercentile(values, lo)
            vmax = np.nanpercentile(values, hi)
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
            return float(vmin), float(vmax)

        top_source_features = top["source_feature"].tolist()
        top_labels = top["feature"].tolist()
        feat_to_idx = {feature: i for i, feature in enumerate(feature_names)}

        fig = plt.figure(figsize=SHAP_PLOT_STYLE["figsize_strip"], dpi=SHAP_PLOT_STYLE["dpi"])
        ax = fig.add_axes(SHAP_AXES_RECT_STRIP)

        for row_i, feature in enumerate(top_source_features[::-1]):
            j = feat_to_idx[feature]
            shap_j = shap_vals[:, j]
            x_j = X_explain[feature].to_numpy()
            jitter = (rng.random(len(shap_j)) - 0.5) * 0.55
            yj = row_i + jitter

            vmin_f, vmax_f = robust_minmax(x_j)
            colors = np.clip((x_j - vmin_f) / (vmax_f - vmin_f), 0, 1)
            ax.scatter(
                shap_j,
                yj,
                c=colors,
                cmap="viridis",
                s=SHAP_PLOT_STYLE["strip_s"],
                alpha=SHAP_PLOT_STYLE["strip_alpha"],
                edgecolors="none",
            )

        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_yticks(range(len(top_source_features)))
        ax.set_yticklabels(top_labels[::-1], fontsize=SHAP_PLOT_STYLE["ytick_fs"])
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=SHAP_PLOT_STYLE["label_fs"])
        ax.tick_params(axis="x", labelsize=SHAP_PLOT_STYLE["tick_fs"], width=1.1, length=3.5)
        set_spines(ax, SHAP_PLOT_STYLE["spine_lw"])

        strip_axis_values = np.concatenate([shap_vals[:, feat_to_idx[feature]] for feature in top_source_features])
        apply_shap_xaxis_controls(
            ax,
            SHAP_AXIS_SETTINGS_STRIP,
            values=strip_axis_values,
            start_at_zero=False,
        )

        shap_strip_base = os.path.join(shap_dir, "IGID_HistGB_SHAP_strip_top_clean")
        save_shap_figure_multi(fig, shap_strip_base)
        plt.close(fig)
        print("[SHAP] Saved:", shap_strip_base + ".png/.svg/.pdf")

        # ---------------- GROUPED SHAP + 1D PDP ----------------
        grouped_shap_df, group_to_features = compute_grouped_shap_importance(shap_vals=shap_vals, feature_names=feature_names)

        pdp_dir = os.path.join(save_dir, "PDP_top3_grouped_SHAP")
        os.makedirs(pdp_dir, exist_ok=True)

        grouped_shap_csv = os.path.join(pdp_dir, "IGID_HistGB_grouped_SHAP_importance.csv")
        grouped_shap_df.to_csv(grouped_shap_csv, index=False)

        pdp_records = []
        top_groups = grouped_shap_df.head(PDP_TOP_N_GROUPS).copy()

        for rank_i, row in top_groups.iterrows():
            group_name = row["group"]
            group_display = row["group_display"]
            features_in_group = group_to_features[group_name]

            if len(features_in_group) > 1:
                local_imp = shap_imp[shap_imp["source_feature"].isin(features_in_group)].copy()
                local_imp = local_imp.sort_values("mean_abs_shap", ascending=False)
                pdp_feature = local_imp.iloc[0]["source_feature"]
            else:
                pdp_feature = features_in_group[0]

            if pdp_feature in continuous_actual:
                vals = pd.to_numeric(X_base_pdp[pdp_feature], errors="coerce").dropna().to_numpy(dtype=float)
                lo = np.nanquantile(vals, PDP_PERCENTILE_RANGE[0])
                hi = np.nanquantile(vals, PDP_PERCENTILE_RANGE[1])
                if np.isclose(lo, hi):
                    continue
                grid_vals = np.linspace(lo, hi, PDP_GRID_POINTS)
                y_means = []
                for v in grid_vals:
                    X_tmp = X_base_pdp.copy()
                    X_tmp[pdp_feature] = v
                    y_means.append(np.mean(predict_raw_from_working_df(X_tmp)))
                out_base = os.path.join(pdp_dir, f"rank{rank_i+1}_{sanitize_filename(group_display)}_{sanitize_filename(pdp_feature)}_PDP")
                save_pdp_plot_numeric(
                    grid_vals,
                    np.asarray(y_means),
                    SHAP_LABEL_MAP.get(pdp_feature, pdp_feature),
                    f"Mean predicted {target_col}",
                    f"({rank_i+1}) {group_display}",
                    out_base,
                )
                pdp_records.append({"rank": int(rank_i+1), "feature": pdp_feature, "type": "continuous_1D", "output_basepath": out_base})

            elif pdp_feature in boxcox_actual_transformed:
                vals = pd.to_numeric(X_base_pdp[pdp_feature], errors="coerce").dropna().to_numpy(dtype=float)
                lo = np.nanquantile(vals, PDP_PERCENTILE_RANGE[0])
                hi = np.nanquantile(vals, PDP_PERCENTILE_RANGE[1])
                if np.isclose(lo, hi):
                    continue
                grid_bc = np.linspace(lo, hi, PDP_GRID_POINTS)
                y_means = []
                for v in grid_bc:
                    X_tmp = X_base_pdp.copy()
                    X_tmp[pdp_feature] = v
                    y_means.append(np.mean(predict_raw_from_working_df(X_tmp)))

                if pdp_feature == "Temperature_01_boxcox":
                    params = boxcox_params_X["Temperature_01"]
                    x_plot = inverse_boxcox_feature(grid_bc, params.get("lam"), params.get("shift", 0.0))
                    x_label = "Temperature 01 (°C)"
                elif pdp_feature == "Temperature_02_boxcox":
                    params = boxcox_params_X["Temperature_02"]
                    x_plot = inverse_boxcox_feature(grid_bc, params.get("lam"), params.get("shift", 0.0))
                    x_label = "Temperature 02 (°C)"
                else:
                    x_plot = grid_bc
                    x_label = SHAP_LABEL_MAP.get(pdp_feature, pdp_feature)

                out_base = os.path.join(pdp_dir, f"rank{rank_i+1}_{sanitize_filename(group_display)}_{sanitize_filename(pdp_feature)}_PDP")
                save_pdp_plot_numeric(
                    x_plot,
                    np.asarray(y_means),
                    x_label,
                    f"Mean predicted {target_col}",
                    f"({rank_i+1}) {group_display}",
                    out_base,
                )
                pdp_records.append({"rank": int(rank_i+1), "feature": pdp_feature, "type": "boxcox_temperature_1D", "output_basepath": out_base})

            elif pdp_feature in categorical_actual:
                codes = sorted([int(c) for c in pd.to_numeric(X_base_pdp[pdp_feature], errors="coerce").dropna().unique() if int(c) >= 0])
                labels = [str(c) for c in codes]
                y_means = []
                for code in codes:
                    X_tmp = X_base_pdp.copy()
                    X_tmp[pdp_feature] = code
                    y_means.append(np.mean(predict_raw_from_working_df(X_tmp)))
                out_base = os.path.join(pdp_dir, f"rank{rank_i+1}_{sanitize_filename(group_display)}_{sanitize_filename(pdp_feature)}_category_response")
                save_pdp_plot_categorical(
                    labels,
                    np.asarray(y_means),
                    SHAP_LABEL_MAP.get(pdp_feature, pdp_feature),
                    f"Mean predicted {target_col}",
                    f"({rank_i+1}) {group_display}",
                    out_base,
                )
                pdp_records.append({"rank": int(rank_i+1), "feature": pdp_feature, "type": "categorical_response", "output_basepath": out_base})

        pdp_summary_csv = os.path.join(pdp_dir, "IGID_HistGB_PDP_top3_summary.csv")
        pd.DataFrame(pdp_records).to_csv(pdp_summary_csv, index=False)

        # ---------------- MAIN 2D ROUTE-AWARE PDP ----------------
        main_2d_outputs: Dict[str, Any] = {"enabled": False}
        if RUN_MAIN_2D_PDP:
            main_2d_dir = os.path.join(save_dir, "PDP_main_2D_design_windows")
            os.makedirs(main_2d_dir, exist_ok=True)

            main_2d_records = []
            main_2d_panels = []

            for pair_i, pair in enumerate(MAIN_2D_PDP_PAIRS, start=1):
                x_feature = pair["x"]
                y_feature = pair["y"]
                pair_label = pair.get("label", f"{x_feature} × {y_feature}")
                panel_label = f"({chr(96 + pair_i)}) {pair_label}"
                subset = pair.get("subset", "all")

                subset_mask = get_route_mask(X_base_pdp, subset=subset)
                X_panel_base = X_base_pdp.loc[subset_mask].copy().reset_index(drop=True)

                if len(X_panel_base) < MIN_ROUTE_ROWS_FOR_PDP:
                    print(f"[Main 2D PDP] Skipped {pair_label}: subset={subset}, rows={len(X_panel_base)}")
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
                    boxcox_actual_transformed=boxcox_actual_transformed,
                    categorical_actual=categorical_actual,
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
                composite_base = os.path.join(main_2d_dir, f"{sanitize_filename(target_col)}_{model_name}_main_2D_PDP_composite")
                save_2d_pdp_composite(main_2d_panels, target_col=target_col, out_base=composite_base, ncols=2)
                print("[MAIN 2D PDP] Saved composite:", composite_base + ".png/.svg/.pdf")

            main_2d_summary_csv = os.path.join(main_2d_dir, f"{sanitize_filename(target_col)}_{model_name}_main_2D_PDP_summary.csv")
            pd.DataFrame(main_2d_records).to_csv(main_2d_summary_csv, index=False)

            main_2d_outputs = {
                "enabled": True,
                "output_dir": main_2d_dir,
                "summary_csv": main_2d_summary_csv,
                "composite_basepath": composite_base,
                "records": main_2d_records,
            }

        pdp_outputs = {
            "enabled": True,
            "output_dir": pdp_dir,
            "grouped_shap_csv": grouped_shap_csv,
            "pdp_summary_csv": pdp_summary_csv,
            "main_2d_outputs": main_2d_outputs,
        }

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
            "model_output": "scaled target prediction before inverse RobustScaler",
            "pdp_outputs": pdp_outputs,
        }

        print("[SHAP] Done.")

    # ---- 10. SAVE RESULTS & ARTIFACTS ----
    artifacts = {
        "target": target_col,
        "model": model_name,
        "params": histgb_final.get_params(),
        "best_params": best_params,
        "metrics_original_units": {
            "train": {"r2": float(r2_train), "rmse": float(rmse_train)},
            "val": {"r2": float(r2_val), "rmse": float(rmse_val)},
            "test": {"r2": float(r2_test), "rmse": float(rmse_test)},
        },
        "legacy_renames_applied": legacy_actions,
        "plot_files_basepath": plot_files,
        "plot_file_formats": ["png", "svg", "pdf"],
        "shap_outputs": shap_outputs,
        "input_file": input_fp,
        "save_dir": save_dir,
        "required_columns": REQUIRED_COLUMNS,
        "feature_order": {
            "continuous": continuous_actual,
            "boxcox_transformed": boxcox_actual_transformed,
            "categorical": categorical_actual,
            "all_model_features": feature_names,
        },
        "boxcox_params_X": boxcox_params_X,
        "outlier_removal": outlier_details,
        "legacy_renames_applied": legacy_actions,
        "route_feature": ROUTE_FEATURE,
        "one_step_type_codes": ONE_STEP_TYPE_CODES,
        "two_step_type_codes": TWO_STEP_TYPE_CODES,
        "plot_style": PLOT_STYLE,
        "shap_plot_style": SHAP_PLOT_STYLE,
        "pdp_plot_style": PDP_PLOT_STYLE,
    }

    results = [{
        "target": target_col,
        "model": model_name,
        "params": histgb_final.get_params(),
        "best_params": best_params,
        "r2_train": float(r2_train),
        "rmse_train": float(rmse_train),
        "r2_val": float(r2_val),
        "rmse_val": float(rmse_val),
        "r2_test": float(r2_test),
        "rmse_test": float(rmse_test),
        "plot_files_basepath": plot_files,
        "plot_file_formats": ["png", "svg", "pdf"],
        "shap_outputs": shap_outputs,
    }]

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "igid_histgb_grid_strict_journal_style.csv")
    results_df.to_csv(csv_path, index=False)
    print("\nSummary (HistGB GridSearch):")
    print(results_df)
    print(f"Results table saved to {csv_path}")

    summary_json = os.path.join(save_dir, "igid_histgb_grid_strict_summary_journal_style.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(results[0]), f, indent=2, ensure_ascii=False)
    print(f"Summary JSON saved to {summary_json}")

    report = {
        "input_file": input_fp,
        "output_dir": save_dir,
        "report_file": report_json,
        "legacy_renames_applied": legacy_actions,
        "required_columns": REQUIRED_COLUMNS,
        "boxcox_params": boxcox_params_X,
        "grid_best_params": best_params,
        "plot_files_basepath": plot_files,
        "plot_file_formats": ["png", "svg", "pdf"],
        "shap_outputs": shap_outputs,
        "metrics": {
            "train": {"r2": float(r2_train), "rmse": float(rmse_train)},
            "val": {"r2": float(r2_val), "rmse": float(rmse_val)},
            "test": {"r2": float(r2_test), "rmse": float(rmse_test)},
        },
        "final_columns": list(final_df.columns),
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(report), f, indent=2, ensure_ascii=False)
    print(f"Report JSON saved to {report_json}")

    joblib.dump(histgb_final, os.path.join(save_dir, "model.pkl"))
    joblib.dump(scaler_X, os.path.join(save_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(save_dir, "scaler_y.pkl"))
    joblib.dump(artifacts, os.path.join(save_dir, "artifacts.pkl"))
    joblib.dump(boxcox_params_X, os.path.join(save_dir, "boxcox_params.pkl"))

    feature_order = {
        "continuous": CONTINUOUS_COLS,
        "boxcox_bases": BOXCOX_BASES,
        "boxcox_transformed_cols": boxcox_actual_transformed,
        "categorical": CATEGORICAL_COLS,
    }
    joblib.dump(feature_order, os.path.join(save_dir, "feature_order.pkl"))

    feature_names_full = (
        [f"SCALER:{c}" for c in CONTINUOUS_COLS]
        + boxcox_actual_transformed
        + [f"CAT:{c}" for c in CATEGORICAL_COLS]
    )
    joblib.dump(feature_names_full, os.path.join(save_dir, "feature_names.pkl"))

    print("\nArtifacts saved to:", save_dir)
    print("feature_order =", json.dumps(feature_order, indent=2))
    print("boxcox_params =", boxcox_params_X)
    print("HistGB Raman IG/ID pipeline complete with journal-style parity, SHAP, grouped SHAP, and route-aware PDP.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
HDR Capacity - Stacking (RF + XGB + CAT) with graph artifacts.

Includes:
- Carbon-style true-vs-pred PNG plots (Train/Val/Test)
- Manual axis override (x/y min/max + tick step)
- Train-only IQR filtering
- Train-only pairwise augmentation
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise SystemExit("catboost is required. Install with: pip install catboost") from exc

# =========================
# CONFIG
# =========================
# ---- Repository paths (relative to the repository root) ----
from pathlib import Path as _RepoPath
REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "SI_Model_02.xlsx"
DATA_SHEET = "HDR_Processed"
OUTPUT_DIR = REPO_ROOT / "outputs" / "model_02/hdr_stacking"

INPUT_FP = str(DATA_FILE)
INPUT_SHEET = DATA_SHEET
SAVE_DIR = str(OUTPUT_DIR)

TARGET_COL = "Capacity"
CURRENT_COL = "Current (A)"
CURRENT_FEATURE_SET = "log"  # raw | log | both

SEED = 42
IQR_ENABLED = True
IQR_K = 1.1
AUGMENTATION_ENABLED = True
PAIRWISE_FRACTION = 0.10
NOISE_FRACTION = 0.05

# Base estimator params
RF_N_ESTIMATORS = 400
RF_MAX_DEPTH = 8
RF_MAX_FEATURES = 0.75
RF_MIN_SAMPLES_SPLIT = 3
RF_MIN_SAMPLES_LEAF = 2
RF_BOOTSTRAP = True

XGB_N_ESTIMATORS = 750
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 5
XGB_SUBSAMPLE = 0.6
XGB_COLSAMPLE_BYTREE = 0.3
XGB_REG_ALPHA = 1e-4
XGB_REG_LAMBDA = 0.1
XGB_MIN_CHILD_WEIGHT = 1

CAT_ITERATIONS = 850
CAT_LEARNING_RATE = 0.08
CAT_DEPTH = 6
CAT_L2_LEAF_REG = 1.0
CAT_RANDOM_STRENGTH = 0.1
CAT_BAGGING_TEMPERATURE = 0.5

# Meta learner
RIDGE_ALPHA = 1.0
STACK_CV = 5
STACK_PASSTHROUGH = False

# =========================
# Carbon-style plot settings
# =========================
MODEL_NAME = "Stacking"
AXIS_SETTINGS_TRUEPRED: Dict[str, Dict[str, Any]] = {
    # Example:
   "Capacity": {"xlim": (0, 600), "ylim": (0, 600), "xtick_step": 150, "ytick_step": 150}
}

# Global manual axis override (applies to Train/Val/Test when set).
# Keep as None to use auto-scaling / AXIS_SETTINGS_TRUEPRED values.
AXIS_X_MIN = None
AXIS_X_MAX = None
AXIS_Y_MIN = None
AXIS_Y_MAX = None
AXIS_X_TICK_STEP = None
AXIS_Y_TICK_STEP = None

PLOT_STYLE = {
    # Carbon-style parity plots: same graph artifacts as your carbon training plots.
    "font_family": "Arial",
    "label_fs": 24,
    "tick_fs": 20,
    "spine_lw": 5,
    "scatter_alpha": 0.70,
    "scatter_s": None,  # None = matplotlib default marker size, matching the carbon-style plots
    "scatter_color": None,  # None = matplotlib default blue, matching the carbon-style plots
    "diag_color": "red",
    "diag_ls": "--",
    "diag_lw": 2.0,
    "ann_fs": 20,
    "ann_bbox": dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0"),
    "figsize": (5, 5),
    "dpi_png": 600,
    "svg_fonttype": "none",
    "pdf_fonttype": 42,
    "ps_fonttype": 42,
}

# =========================
# SHAP + PDP CONFIG
# =========================
RUN_SHAP = True
SHAP_MAX_BACKGROUND = 120
SHAP_MAX_EXPLAIN = 200
SHAP_TOP_N = 10
SHAP_SEED = 42

# Empty dictionaries mean auto-scaled SHAP x-axes.
SHAP_AXES_RECT_BAR = [0.35, 0.18, 0.58, 0.74]
SHAP_AXES_RECT_STRIP = [0.35, 0.18, 0.58, 0.74]
SHAP_AXIS_SETTINGS_BAR: Dict[str, Any] = {}
SHAP_AXIS_SETTINGS_STRIP: Dict[str, Any] = {}

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

# Fallback Model 02 HDR/Capacity PDP design-window panels.
# Main 2D panels are generated dynamically from the top 3 grouped SHAP groups.
# These fallback pairs are used only if fewer than two grouped SHAP groups are available.
MAIN_2D_PDP_PAIRS = [
    {"x_group": "Current density", "y_group": "D-S", "label": "Curr × D-S"},
    {"x_group": "Current density", "y_group": "Surface area", "label": "Curr × SSA"},
    {"x_group": "D-S", "y_group": "Surface area", "label": "D-S × SSA"},
    {"x_group": "Carbon ratio", "y_group": "Binder ratio", "label": "Carb × Binder ratio"},
]

SHAP_PLOT_STYLE = {
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

PDP_PLOT_STYLE = {
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

SHAP_LABEL_MAP = {
    "Active material ratio": "Active material ratio",
    "Active_Material_Ratio": "Active material ratio",
    "Carbon ratio": "Carbon ratio",
    "Carbon_Ratio": "Carbon ratio",
    "Binder ratio": "Binder ratio",
    "Binder_Ratio": "Binder ratio",
    "Salt concentration": "Salt concentration",
    "Salt_Concentration": "Salt concentration",
    "Additive concentration": "Additive concentration",
    "Additive_Concentration": "Additive concentration",
    "Electrolyte type": "Electrolyte type",
    "Electrolyte_Type": "Electrolyte type",
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
    "Temperature_01": "Tem01",
    "Temperature_02": "Tem02",
    "Temperature_01_boxcox": "Tem01",
    "Temperature_02_boxcox": "Tem02",
    "Prewash": "Pre",
    "Postwash": "Post",
    "Type": "Type",
    "Method_01": "M01",
    "Activator_01": "Act01",
    "Method_02": "M02",
    "Activator_02": "Act02",
    "Current (mA)": "Curr",
    "log10 current density": "log Curr",
    "Current density": "Curr",
    "Current_Density": "Curr",
    "CurrentDensity": "Curr",
    "Current": "Curr",
}

SHAP_GROUP_DEFINITIONS = {
    "Active material ratio": ["Active material ratio", "Active_Material_Ratio", "ActiveMaterialRatio", "Active ratio", "AM ratio"],
    "Carbon ratio": ["Carbon ratio", "Carbon_Ratio", "CarbonRatio", "Conductive carbon ratio"],
    "Binder ratio": ["Binder ratio", "Binder_Ratio", "BinderRatio"],
    "Separator type": ["GF", "PE", "PP"],
    "Binder type": ["CMC", "CMC+SBR", "CMC_SBR", "CMC SBR", "No Binder", "No_Binder", "PAA", "PANA", "PVDF", "SBR", "Sodium alginate", "Sodium_Alginate"],
    "Solvent type": ["NMP", "No Solvent", "No_Solvent", "Water"],
    "Salt type": ["NaClO4", "NaCLO4", "NaOTf", "NaOTF", "NaPF6", "NaTFSI"],
    "Salt concentration": ["Salt concentration", "Salt_Concentration", "SaltConcentration", "Salt conc", "Salt_conc"],
    "Additive concentration": ["Additive concentration", "Additive_Concentration", "AdditiveConcentration", "Additive conc", "Additive_conc"],
    "Electrolyte type": [
        "Electrolyte type", "Electrolyte_Type", "ElectrolyteType", "Aqueous", "Organic", "Carbonate", "Ether",
        "Ionic liquid", "Ionic_Liquid", "PC", "PC fraction", "PC_Fraction", "PCFraction",
        "EC", "EC fraction", "EC_Fraction", "ECFraction", "DMC", "DMC fraction", "DMC_Fraction",
        "DMCFraction", "DEC", "DEC fraction", "DEC_Fraction", "DECFraction", "EMC", "EMC fraction",
        "EMC_Fraction", "EMCFraction", "FEC", "FEC fraction", "FEC_Fraction", "FECFraction",
        "DME", "DME fraction", "DME_Fraction", "DMEFraction", "DOL", "DOL fraction", "DOL_Fraction",
        "DOLFraction", "TETRAGLYME", "TETRAGLYME fraction", "TETRAGLYME_Fraction", "TETRAGLYMEFraction"
    ],
    "La": ["La", "L_a", "La_nm", "La (nm)", "Lateral crystallite size", "Lateral_Crystallite_Size"],
    "D-S": ["D-S", "D_S", "D spacing", "D-spacing", "D_Spacing", "D Spacing", "d002", "d_002"],
    "Current density": ["Current density", "Current_Density", "CurrentDensity", "Current", "Curr", "mA/g", "Current (mA)", "log10 current density"],
    "Lc": ["Lc", "L_c", "Lc_nm", "Lc (nm)", "Crystallite height", "Crystallite_Height"],
    "Surface area": ["SSA", "Surface Area", "Surface area", "Specific surface area", "BET", "BET surface area", "BET_Surface_Area"],
    "Pore volume": ["PV", "Pore Volume", "Pore volume", "Pore_Volume"],
}

SHAP_GROUP_DISPLAY_MAP = {
    "Solvent type": "Solv",
    "Electrolyte type": "Elec",
    "Binder type": "Binder",
    "Salt type": "Salt",
    "D-S": "D-S",
    "Current density": "Curr",
    "Carbon ratio": "Carb",
    "La": "La",
    "Lc": "Lc",
    "Surface area": "SSA",
    "Pore volume": "PV",
    "Binder ratio": "Bi_rat",
}


def sanitize_filename(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_figure_multi(basepath_no_ext: Path) -> None:
    plt.savefig(str(basepath_no_ext) + ".png", dpi=PLOT_STYLE["dpi_png"], bbox_inches="tight")
    plt.savefig(str(basepath_no_ext) + ".svg", bbox_inches="tight")
    plt.savefig(str(basepath_no_ext) + ".pdf", bbox_inches="tight")


def save_shap_figure_multi(fig, basepath_no_ext: Path) -> None:
    fig.savefig(str(basepath_no_ext) + ".png", dpi=SHAP_PLOT_STYLE["dpi_png"], bbox_inches="tight")
    fig.savefig(str(basepath_no_ext) + ".svg", bbox_inches="tight")
    fig.savefig(str(basepath_no_ext) + ".pdf", bbox_inches="tight")


def set_spines_lw(ax, lw: float) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(lw)


def resolve_axis_settings(target_name: str) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    if target_name in AXIS_SETTINGS_TRUEPRED and isinstance(AXIS_SETTINGS_TRUEPRED[target_name], dict):
        settings.update(AXIS_SETTINGS_TRUEPRED[target_name])

    if AXIS_X_MIN is not None or AXIS_X_MAX is not None:
        if AXIS_X_MIN is None or AXIS_X_MAX is None:
            raise SystemExit("Set both AXIS_X_MIN and AXIS_X_MAX together.")
        settings["xlim"] = (float(AXIS_X_MIN), float(AXIS_X_MAX))

    if AXIS_Y_MIN is not None or AXIS_Y_MAX is not None:
        if AXIS_Y_MIN is None or AXIS_Y_MAX is None:
            raise SystemExit("Set both AXIS_Y_MIN and AXIS_Y_MAX together.")
        settings["ylim"] = (float(AXIS_Y_MIN), float(AXIS_Y_MAX))

    if AXIS_X_TICK_STEP is not None:
        settings["xtick_step"] = float(AXIS_X_TICK_STEP)
    if AXIS_Y_TICK_STEP is not None:
        settings["ytick_step"] = float(AXIS_Y_TICK_STEP)

    if "xlim" in settings:
        x0, x1 = settings["xlim"]
        if x1 <= x0:
            raise SystemExit("Invalid X axis range: AXIS_X_MAX must be greater than AXIS_X_MIN.")
    if "ylim" in settings:
        y0, y1 = settings["ylim"]
        if y1 <= y0:
            raise SystemExit("Invalid Y axis range: AXIS_Y_MAX must be greater than AXIS_Y_MIN.")
    if "xtick_step" in settings and float(settings["xtick_step"]) <= 0:
        raise SystemExit("Invalid AXIS_X_TICK_STEP: must be > 0.")
    if "ytick_step" in settings and float(settings["ytick_step"]) <= 0:
        raise SystemExit("Invalid AXIS_Y_TICK_STEP: must be > 0.")

    return settings


def get_metrics(y_true, y_pred) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    r2 = r2_score(y_true, y_pred)
    rmse_v = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_v = mean_absolute_error(y_true, y_pred)
    return float(r2), float(rmse_v), float(mae_v)


def get_data_range(series, padding=0.05, round_to=5):
    series = np.asarray(series).ravel()
    min_val = np.nanmin(series)
    max_val = np.nanmax(series)
    range_span = max_val - min_val

    if not np.isfinite(range_span) or range_span == 0:
        min_val = float(min_val) if np.isfinite(min_val) else 0.0
        max_val = float(max_val) if np.isfinite(max_val) else 1.0
        range_span = max_val - min_val if max_val != min_val else 1.0

    min_pad = min_val - range_span * padding
    max_pad = max_val + range_span * padding

    min_tick = round_to * int(min_pad // round_to)
    max_tick = round_to * int(np.ceil(max_pad / round_to))

    n_ticks = 6
    step = max(round_to, int((max_tick - min_tick) / n_ticks // round_to * round_to))
    if step == 0:
        step = round_to
    return min_tick, max_tick, step


def plot_true_vs_pred(
    y_true,
    y_pred,
    set_name,
    target_name,
    model_name="Stacking",
    save_dir=None,
    xlim=None,
    ylim=None,
    xtick_step=None,
    ytick_step=None,
):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r2, rmse_v, _ = get_metrics(y_true, y_pred)

    plt.figure(figsize=PLOT_STYLE["figsize"], dpi=PLOT_STYLE["dpi_png"])
    plt.rcParams["font.family"] = PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = PLOT_STYLE["svg_fonttype"]
    plt.rcParams["pdf.fonttype"] = PLOT_STYLE["pdf_fonttype"]
    plt.rcParams["ps.fonttype"] = PLOT_STYLE["ps_fonttype"]

    scatter_kwargs = dict(alpha=PLOT_STYLE["scatter_alpha"])
    if PLOT_STYLE.get("scatter_s") is not None:
        scatter_kwargs["s"] = PLOT_STYLE["scatter_s"]
    if PLOT_STYLE["scatter_color"] is not None:
        scatter_kwargs["color"] = PLOT_STYLE["scatter_color"]
    plt.scatter(y_true, y_pred, **scatter_kwargs)

    plt.xlabel("True Value", fontsize=PLOT_STYLE["label_fs"], fontname=PLOT_STYLE["font_family"])
    plt.ylabel("Predicted Value", fontsize=PLOT_STYLE["label_fs"], fontname=PLOT_STYLE["font_family"])

    if xlim is None or xtick_step is None:
        xmins, xmaxs, xstep = get_data_range(y_true, round_to=5)
        xlim = (xmins, xmaxs) if xlim is None else xlim
        xtick_step = xstep if xtick_step is None else xtick_step
    if ylim is None or ytick_step is None:
        ymins, ymaxs, ystep = get_data_range(y_pred, round_to=5)
        ylim = (ymins, ymaxs) if ylim is None else ylim
        ytick_step = ystep if ytick_step is None else ytick_step

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(
        np.arange(xlim[0], xlim[1] + xtick_step, xtick_step),
        fontsize=PLOT_STYLE["tick_fs"],
        fontname=PLOT_STYLE["font_family"],
    )
    plt.yticks(
        np.arange(ylim[0], ylim[1] + ytick_step, ytick_step),
        fontsize=PLOT_STYLE["tick_fs"],
        fontname=PLOT_STYLE["font_family"],
    )

    min_axis = min(xlim[0], ylim[0])
    max_axis = max(xlim[1], ylim[1])
    plt.plot(
        [min_axis, max_axis],
        [min_axis, max_axis],
        color=PLOT_STYLE["diag_color"],
        linestyle=PLOT_STYLE["diag_ls"],
        linewidth=PLOT_STYLE["diag_lw"],
    )

    textstr = f"R² = {r2:.3f}\nRMSE = {rmse_v:.3f}\n{model_name}"
    plt.gca().text(
        0.95,
        0.05,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=PLOT_STYLE["ann_fs"],
        fontname=PLOT_STYLE["font_family"],
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=PLOT_STYLE["ann_bbox"],
    )

    for spine in plt.gca().spines.values():
        spine.set_linewidth(PLOT_STYLE["spine_lw"])

    plt.tight_layout()
    base = None
    if save_dir is not None:
        fname = f"{sanitize_filename(target_name)}_{set_name}_{sanitize_filename(model_name)}_actual"
        base = Path(save_dir) / fname
        save_figure_multi(base)
        print(f"Plot saved to: {base}.png/.svg/.pdf")
    plt.close()
    return None if base is None else str(base)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"r2": float(r2_score(y_true, y_pred)), "rmse": rmse(y_true, y_pred)}


def iqr_filter_mask(y: pd.Series, k: float):
    q1 = float(y.quantile(0.25))
    q3 = float(y.quantile(0.75))
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    mask = (y >= low) & (y <= high)
    info = {
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "low_fence": float(low),
        "high_fence": float(high),
        "removed_count": int((~mask).sum()),
        "removed_below": int((y < low).sum()),
        "removed_above": int((y > high).sum()),
    }
    return mask.to_numpy(), info


def pairwise_augmentation(
    x_in: pd.DataFrame,
    y_in: pd.Series,
    fraction: float,
    noise_fraction: float,
    random_state: int,
):
    n = len(x_in)
    n_aug = int(n * fraction)
    if n_aug <= 0:
        return None, None

    rng = np.random.default_rng(random_state)
    idx1 = rng.choice(n, n_aug, replace=False)
    idx2 = rng.choice(n, n_aug, replace=False)

    x1 = x_in.iloc[idx1].to_numpy(float)
    x2 = x_in.iloc[idx2].to_numpy(float)
    x_syn = (x1 + x2) / 2.0

    binary_cols = []
    for col in x_in.columns:
        vals = pd.to_numeric(x_in[col], errors="coerce").dropna().to_numpy(float)
        if vals.size == 0:
            continue
        u = np.unique(vals)
        if np.all(np.isin(np.round(u, 12), [0.0, 1.0])):
            binary_cols.append(col)
    binary_idx = [x_in.columns.get_loc(c) for c in binary_cols]
    continuous_cols = [c for c in x_in.columns if c not in binary_cols]
    continuous_idx = [x_in.columns.get_loc(c) for c in continuous_cols]

    if binary_idx:
        choose_first = rng.integers(0, 2, size=(n_aug, len(binary_idx))).astype(bool)
        x_syn[:, binary_idx] = np.where(choose_first, x1[:, binary_idx], x2[:, binary_idx])

    if continuous_idx:
        col_stds = x_in[continuous_cols].std(axis=0, ddof=0).to_numpy(float)
        noise = rng.normal(0.0, noise_fraction * col_stds, size=(n_aug, len(continuous_idx)))
        x_syn[:, continuous_idx] = x_syn[:, continuous_idx] + noise

    y_arr = y_in.to_numpy(float)
    y_syn = (y_arr[idx1] + y_arr[idx2]) / 2.0
    return pd.DataFrame(x_syn, columns=x_in.columns), pd.Series(y_syn, name=y_in.name)



def nice_step(raw_step: float) -> float:
    """Return a clean tick step for automatically scaled axes."""
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

    return float(nice_fraction * (10 ** exponent))


def auto_shap_xaxis(ax: plt.Axes, values, start_at_zero: bool = False, n_ticks: int = 5) -> None:
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


def apply_shap_xaxis_controls(ax: plt.Axes, settings: Dict[str, Any], values, start_at_zero: bool = False) -> None:
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


def normalize_group_token(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def normalized_word_chunks(value: Any) -> List[str]:
    chunks = re.split(r"[^A-Za-z0-9]+", str(value).lower())
    return [chunk for chunk in chunks if chunk]


def feature_matches_group_token(feature_name: str, token: str) -> bool:
    feature_key = normalize_group_token(feature_name)
    token_key = normalize_group_token(token)
    if not feature_key or not token_key:
        return False
    if feature_key == token_key:
        return True

    feature_chunks = normalized_word_chunks(feature_name)
    token_chunks = normalized_word_chunks(token)
    if token_chunks and token_chunks == feature_chunks:
        return True
    if len(token_chunks) == 1 and token_chunks[0] in feature_chunks:
        return True
    if len(token_chunks) > 1:
        joined_chunks = normalize_group_token("".join(feature_chunks))
        joined_token = normalize_group_token("".join(token_chunks))
        if joined_chunks == joined_token:
            return True

    if feature_key in {f"{token_key}fraction", f"{token_key}percent", f"{token_key}percentage"}:
        return True
    if len(token_key) >= 4 and token_key in feature_key:
        return True
    return False


def build_shap_groups(feature_names: List[str], display_name_by_source: Dict[str, str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    assigned = set()
    for group_name, tokens in SHAP_GROUP_DEFINITIONS.items():
        members = []
        for feature_name in feature_names:
            if feature_name in assigned:
                continue
            if any(feature_matches_group_token(feature_name, token) for token in tokens):
                members.append(feature_name)
        if members:
            grouped[SHAP_GROUP_DISPLAY_MAP.get(group_name, group_name)] = members
            assigned.update(members)

    for feature_name in feature_names:
        if feature_name not in assigned:
            grouped[display_name_by_source.get(feature_name, feature_name)] = [feature_name]

    return grouped


def robust_minmax(values, lo: float = 1, hi: float = 99) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = np.nanpercentile(finite, lo)
    vmax = np.nanpercentile(finite, hi)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def find_group_feature_candidates(feature_names: List[str], group_name: str) -> List[str]:
    # group_name can be an original group key, e.g. "Surface area",
    # or a display label from grouped SHAP, e.g. "SSA".
    display_to_group_key = {v: k for k, v in SHAP_GROUP_DISPLAY_MAP.items()}
    canonical_group = display_to_group_key.get(group_name, group_name)

    tokens = SHAP_GROUP_DEFINITIONS.get(canonical_group, SHAP_GROUP_DEFINITIONS.get(group_name, [group_name]))
    matches = []
    for f in feature_names:
        if any(feature_matches_group_token(f, t) for t in tokens):
            matches.append(f)
    return matches

def choose_representative_feature(
    feature_names: List[str],
    group_name: str,
    X_data: pd.DataFrame,
    shap_imp: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    candidates = find_group_feature_candidates(feature_names, group_name)
    if not candidates:
        return None

    if shap_imp is not None and not shap_imp.empty:
        ranked = shap_imp[shap_imp["source_feature"].isin(candidates)].copy()
        if not ranked.empty:
            return str(ranked.sort_values("mean_abs_shap", ascending=False).iloc[0]["source_feature"])

    # Prefer a non-binary continuous column when possible.
    for c in candidates:
        vals = pd.to_numeric(X_data[c], errors="coerce").dropna().to_numpy(float)
        if vals.size > 0:
            unique = np.unique(np.round(vals, 12))
            if len(unique) > 2:
                return c
    return candidates[0]


def is_binary_or_categorical_feature(values: np.ndarray, max_categories: int = 12) -> bool:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return True
    unique_vals = np.unique(np.round(values, 12))
    return len(unique_vals) <= max_categories


def build_pdp_axis_values(
    feature: str,
    X_base_pdp: pd.DataFrame,
    grid_points: int,
    percentile_range: Tuple[float, float],
    max_categories: int,
) -> Optional[Dict[str, Any]]:
    if feature not in X_base_pdp.columns:
        return None

    vals = pd.to_numeric(X_base_pdp[feature], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size < 2:
        return None

    if is_binary_or_categorical_feature(vals, max_categories=max_categories):
        codes = sorted([float(c) for c in np.unique(vals) if np.isfinite(c)])[:max_categories]
        if not codes:
            return None
        plot_values = np.arange(len(codes), dtype=float)
        code_to_pos = {code: i for i, code in enumerate(codes)}
        actual_plot = [code_to_pos.get(float(c), np.nan) for c in vals]
        tick_labels = [str(int(c)) if float(c).is_integer() else f"{c:g}" for c in codes]
        return {
            "kind": "categorical",
            "feature": feature,
            "model_values": np.asarray(codes, dtype=float),
            "plot_values": plot_values,
            "actual_plot_values": np.asarray(actual_plot, dtype=float),
            "axis_label": SHAP_LABEL_MAP.get(feature, feature),
            "tick_labels": tick_labels,
        }

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


def make_2d_pdp_panel(
    x_feature: str,
    y_feature: str,
    panel_label: str,
    X_base_pdp: pd.DataFrame,
    predict_fn,
    grid_points: int,
    percentile_range: Tuple[float, float],
    max_categories: int,
):
    x_axis = build_pdp_axis_values(x_feature, X_base_pdp, grid_points, percentile_range, max_categories)
    y_axis = build_pdp_axis_values(y_feature, X_base_pdp, grid_points, percentile_range, max_categories)
    if x_axis is None or y_axis is None:
        return None

    Z = np.zeros((len(y_axis["model_values"]), len(x_axis["model_values"])), dtype=float)
    for j, yv in enumerate(y_axis["model_values"]):
        for i, xv in enumerate(x_axis["model_values"]):
            X_tmp = X_base_pdp.copy()
            X_tmp[x_feature] = xv
            X_tmp[y_feature] = yv
            Z[j, i] = float(np.mean(predict_fn(X_tmp)))

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


def plot_2d_pdp_panel_on_ax(ax, panel, target_col: str, add_colorbar: bool = True, vmin=None, vmax=None):
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
    set_spines_lw(ax, PDP_PLOT_STYLE["spine_lw"])
    return im


def save_2d_pdp_panel(panel, target_col: str, out_base: Path):
    plt.rcParams["font.family"] = PDP_PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=PDP_PLOT_STYLE["figsize"])
    plot_2d_pdp_panel_on_ax(ax, panel, target_col=target_col, add_colorbar=True)
    fig.tight_layout()
    fig.savefig(str(out_base) + ".png", dpi=PDP_PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def save_2d_pdp_composite(panels, target_col: str, out_base: Path, ncols: int = 2):
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

    fig.savefig(str(out_base) + ".png", dpi=PDP_PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def save_pdp_plot_numeric(x_grid, y_mean, x_label, y_label, title_label, out_base: Path):
    plt.rcParams["font.family"] = PDP_PLOT_STYLE["font_family"]
    fig, ax = plt.subplots(figsize=PDP_PLOT_STYLE["figsize"])
    ax.plot(x_grid, y_mean, linewidth=PDP_PLOT_STYLE["line_lw"])
    ax.scatter(x_grid, y_mean, s=PDP_PLOT_STYLE["scatter_s"], alpha=PDP_PLOT_STYLE["scatter_alpha"])
    ax.set_xlabel(x_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.set_ylabel(y_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.tick_params(axis="both", labelsize=PDP_PLOT_STYLE["tick_fs"], width=1.1, length=3.5)
    ax.text(
        0.03, 0.97, title_label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=PDP_PLOT_STYLE.get("panel_label_fs", 14),
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=PDP_PLOT_STYLE.get("panel_label_alpha", 0.62), pad=1.0),
    )
    set_spines_lw(ax, PDP_PLOT_STYLE["spine_lw"])
    fig.tight_layout()
    fig.savefig(str(out_base) + ".png", dpi=PDP_PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def load_prepare():
    input_fp = Path(INPUT_FP).expanduser().resolve()
    if not input_fp.exists():
        raise SystemExit(f"Input file not found: {input_fp}")

    if input_fp.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_fp, sheet_name=INPUT_SHEET)
    else:
        df = pd.read_csv(input_fp)

    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    if TARGET_COL not in df.columns:
        raise SystemExit(f"Target column not found: {TARGET_COL}")
    if CURRENT_COL not in df.columns:
        raise SystemExit(f"Current column not found: {CURRENT_COL}")

    mode = str(CURRENT_FEATURE_SET).strip().lower()
    if mode not in {"raw", "log", "both"}:
        raise SystemExit("CURRENT_FEATURE_SET must be one of: raw, log, both.")

    cur_a = pd.to_numeric(df[CURRENT_COL], errors="coerce")
    df["Current (mA)"] = cur_a * 1000.0
    cur_ma = pd.to_numeric(df["Current (mA)"], errors="coerce")
    valid_log = np.isfinite(cur_ma.to_numpy(float)) & (cur_ma.to_numpy(float) > 0)
    df["log10 current density"] = np.where(valid_log, np.log10(cur_ma.to_numpy(float)), np.nan)

    leak_cols = {
        "actual_Capacity",
        "predicted_Capacity",
        "abs_error",
        "split",
        "row_in_processed_table",
    }
    dynamic_leak_cols = [c for c in df.columns if c.lower().startswith("actual_") or c.lower().startswith("predicted_")]

    drop_cols = {
        TARGET_COL,
        "Electrolyte",
        "ICE",
        "Ref",
        CURRENT_COL,
    } | leak_cols | set(dynamic_leak_cols)

    if mode == "raw":
        drop_cols.add("log10 current density")
    elif mode == "log":
        drop_cols.add("Current (mA)")

    drop_cols = [c for c in drop_cols if c in df.columns]
    x_df = df.drop(columns=drop_cols, errors="ignore").copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    x_df = x_df.apply(pd.to_numeric, errors="coerce")

    keep = y.notna()
    dropped_due_to_current = 0
    if mode == "raw":
        bad_mask = ~x_df["Current (mA)"].notna()
        dropped_due_to_current = int((y.notna() & bad_mask).sum())
        keep = keep & x_df["Current (mA)"].notna()
    elif mode == "log":
        bad_mask = ~x_df["log10 current density"].notna()
        dropped_due_to_current = int((y.notna() & bad_mask).sum())
        keep = keep & x_df["log10 current density"].notna()
    else:
        bad_mask = ~(x_df["Current (mA)"].notna() & x_df["log10 current density"].notna())
        dropped_due_to_current = int((y.notna() & bad_mask).sum())
        keep = keep & x_df["Current (mA)"].notna() & x_df["log10 current density"].notna()

    x_df = x_df.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    x_df = x_df.fillna(x_df.median(numeric_only=True))
    all_nan_cols = x_df.columns[x_df.isna().all()].tolist()
    if all_nan_cols:
        x_df = x_df.drop(columns=all_nan_cols).copy()
    if x_df.shape[1] == 0:
        raise SystemExit("No usable features after preprocessing.")

    return x_df, y, drop_cols, all_nan_cols, dropped_due_to_current, mode


def main():
    save_dir = Path(SAVE_DIR).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    x_df, y, dropped_cols, all_nan_cols, dropped_due_to_current, mode = load_prepare()

    row_idx = np.arange(len(x_df))
    x_train, x_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        x_df, y, row_idx, test_size=0.25, random_state=SEED
    )
    x_val, x_test, y_val, y_test, idx_val, idx_test = train_test_split(
        x_temp, y_temp, idx_temp, test_size=0.4, random_state=SEED
    )

    x_train_eval = x_train.copy()
    y_train_eval = y_train.copy()

    x_fit = x_train.copy()
    y_fit = y_train.copy()

    iqr_info = {"enabled": bool(IQR_ENABLED), "k": float(IQR_K), "removed_count": 0}
    if IQR_ENABLED:
        keep_mask, info = iqr_filter_mask(y_fit, IQR_K)
        x_fit = x_fit.loc[keep_mask].reset_index(drop=True)
        y_fit = y_fit.loc[keep_mask].reset_index(drop=True)
        iqr_info = {"enabled": True, "k": float(IQR_K), **info}

    aug_fraction = PAIRWISE_FRACTION if AUGMENTATION_ENABLED else 0.0
    x_aug, y_aug = pairwise_augmentation(x_fit, y_fit, aug_fraction, NOISE_FRACTION, SEED)
    if x_aug is not None:
        x_fit = pd.concat([x_fit, x_aug], ignore_index=True)
        y_fit = pd.concat([y_fit, y_aug], ignore_index=True)

    rf = RandomForestRegressor(
        n_estimators=int(RF_N_ESTIMATORS),
        max_depth=None if RF_MAX_DEPTH is None else int(RF_MAX_DEPTH),
        max_features=RF_MAX_FEATURES,
        min_samples_split=int(RF_MIN_SAMPLES_SPLIT),
        min_samples_leaf=int(RF_MIN_SAMPLES_LEAF),
        bootstrap=bool(RF_BOOTSTRAP),
        random_state=SEED,
        n_jobs=-1,
    )
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=SEED,
        n_estimators=int(XGB_N_ESTIMATORS),
        learning_rate=float(XGB_LEARNING_RATE),
        max_depth=int(XGB_MAX_DEPTH),
        subsample=float(XGB_SUBSAMPLE),
        colsample_bytree=float(XGB_COLSAMPLE_BYTREE),
        reg_alpha=float(XGB_REG_ALPHA),
        reg_lambda=float(XGB_REG_LAMBDA),
        min_child_weight=float(XGB_MIN_CHILD_WEIGHT),
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )
    cat = CatBoostRegressor(
        loss_function="RMSE",
        iterations=int(CAT_ITERATIONS),
        learning_rate=float(CAT_LEARNING_RATE),
        depth=int(CAT_DEPTH),
        l2_leaf_reg=float(CAT_L2_LEAF_REG),
        random_strength=float(CAT_RANDOM_STRENGTH),
        bagging_temperature=float(CAT_BAGGING_TEMPERATURE),
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )

    stack = StackingRegressor(
        estimators=[("rf", rf), ("xgb", xgb), ("cat", cat)],
        final_estimator=Ridge(alpha=float(RIDGE_ALPHA)),
        cv=int(STACK_CV),
        passthrough=bool(STACK_PASSTHROUGH),
        n_jobs=-1,
    )
    stack.fit(x_fit, y_fit)

    pred_tr = stack.predict(x_train_eval)
    pred_va = stack.predict(x_val)
    pred_te = stack.predict(x_test)

    r2_tr, rmse_tr, mae_tr = get_metrics(y_train_eval.to_numpy(float), pred_tr)
    r2_va, rmse_va, mae_va = get_metrics(y_val.to_numpy(float), pred_va)
    r2_te, rmse_te, mae_te = get_metrics(y_test.to_numpy(float), pred_te)

    m_train = {"r2": r2_tr, "rmse": rmse_tr}
    m_val = {"r2": r2_va, "rmse": rmse_va}
    m_test = {"r2": r2_te, "rmse": rmse_te}

    pred_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "row_in_processed_table": idx_train,
                    "split": "train",
                    "actual_Capacity": y_train_eval.to_numpy(float),
                    "predicted_Capacity": pred_tr,
                }
            ),
            pd.DataFrame(
                {
                    "row_in_processed_table": idx_val,
                    "split": "validation",
                    "actual_Capacity": y_val.to_numpy(float),
                    "predicted_Capacity": pred_va,
                }
            ),
            pd.DataFrame(
                {
                    "row_in_processed_table": idx_test,
                    "split": "test",
                    "actual_Capacity": y_test.to_numpy(float),
                    "predicted_Capacity": pred_te,
                }
            ),
        ],
        ignore_index=True,
    ).sort_values("row_in_processed_table", ignore_index=True)
    pred_df["abs_error"] = (pred_df["actual_Capacity"] - pred_df["predicted_Capacity"]).abs()

    tag = f"{mode}current"
    pred_xlsx = save_dir / f"stacking_hdr_capacity_{tag}_actual_vs_pred.xlsx"
    pred_csv = save_dir / f"stacking_hdr_capacity_{tag}_actual_vs_pred.csv"
    model_pkl = save_dir / f"stacking_hdr_capacity_{tag}.pkl"
    metrics_json = save_dir / f"stacking_hdr_capacity_{tag}_metrics.json"

    pred_df.to_excel(pred_xlsx, index=False)
    pred_df.to_csv(pred_csv, index=False)
    joblib.dump(stack, model_pkl)
    joblib.dump(list(x_df.columns), save_dir / "feature_names.pkl")

    axis_cfg = resolve_axis_settings(TARGET_COL)
    plot_files = {
        "train": plot_true_vs_pred(
            y_train_eval.to_numpy(float), pred_tr, "Train", TARGET_COL, MODEL_NAME, save_dir, **axis_cfg
        ),
        "validation": plot_true_vs_pred(
            y_val.to_numpy(float), pred_va, "Val", TARGET_COL, MODEL_NAME, save_dir, **axis_cfg
        ),
        "test": plot_true_vs_pred(
            y_test.to_numpy(float), pred_te, "Test", TARGET_COL, MODEL_NAME, save_dir, **axis_cfg
        ),
    }

    # =========================
    # SHAP + GROUPED SHAP + PDP
    # =========================
    feature_names = list(x_df.columns)
    shap_outputs: Dict[str, Any] = {"enabled": bool(RUN_SHAP)}
    pdp_outputs: Dict[str, Any] = {"enabled": False}

    if RUN_SHAP:
        try:
            import shap
            print("[SHAP] shap imported successfully.")
        except Exception as exc:
            raise RuntimeError("SHAP is required. Install with: pip install shap") from exc

        shap_dir = save_dir / "SHAP"
        shap_dir.mkdir(parents=True, exist_ok=True)
        print("[SHAP] Output dir:", shap_dir)

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

        X_train_df_shap = pd.DataFrame(x_fit, columns=feature_names)
        X_test_df_shap = pd.DataFrame(x_test, columns=feature_names)

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

        # StackingRegressor is model-agnostic for SHAP, so use PermutationExplainer.
        max_evals = 2 * X_explain.shape[1] + 1
        explainer_name = "shap.PermutationExplainer(stack.predict, X_bg)"
        explainer = shap.PermutationExplainer(stack.predict, X_bg)
        shap_exp = explainer(X_explain, max_evals=max_evals)
        shap_vals = np.asarray(shap_exp.values)

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

        target_safe = sanitize_filename(TARGET_COL)
        shap_imp_csv = shap_dir / f"{target_safe}_{MODEL_NAME}_mean_abs_shap.csv"
        shap_imp.to_csv(shap_imp_csv, index=False)
        print("[SHAP] Saved:", shap_imp_csv)

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
        ax.tick_params(axis="x", labelsize=SHAP_PLOT_STYLE["tick_fs"], width=1.1, length=3.5)
        ax.tick_params(axis="y", labelsize=SHAP_PLOT_STYLE["ytick_fs"], width=1.1, length=3.5)
        set_spines_lw(ax, SHAP_PLOT_STYLE["spine_lw"])
        apply_shap_xaxis_controls(
            ax,
            SHAP_AXIS_SETTINGS_BAR,
            values=top["mean_abs_shap"].to_numpy(float),
            start_at_zero=True,
        )

        shap_bar_base = shap_dir / f"{target_safe}_{MODEL_NAME}_SHAP_bar_top_clean"
        save_shap_figure_multi(fig, shap_bar_base)
        plt.close(fig)
        print("[SHAP] Saved:", str(shap_bar_base) + ".png/.svg/.pdf")

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

        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_yticks(range(len(top_source_features)))
        ax.set_yticklabels(top_labels[::-1], fontsize=SHAP_PLOT_STYLE["ytick_fs"])
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=SHAP_PLOT_STYLE["label_fs"])
        ax.tick_params(axis="x", labelsize=SHAP_PLOT_STYLE["tick_fs"], width=1.1, length=3.5)
        set_spines_lw(ax, SHAP_PLOT_STYLE["spine_lw"])
        strip_axis_values = np.concatenate([shap_vals[:, feat_to_idx[feature]] for feature in top_source_features])
        apply_shap_xaxis_controls(
            ax,
            SHAP_AXIS_SETTINGS_STRIP,
            values=strip_axis_values,
            start_at_zero=False,
        )

        shap_strip_base = shap_dir / f"{target_safe}_{MODEL_NAME}_SHAP_strip_top_clean"
        save_shap_figure_multi(fig, shap_strip_base)
        plt.close(fig)
        print("[SHAP] Saved:", str(shap_strip_base) + ".png/.svg/.pdf")

        grouped_features = build_shap_groups(list(X_explain.columns), display_name_by_source)
        group_rows = []
        group_signed_values = {}
        group_color_values = {}
        for group_order, (group_name, members) in enumerate(grouped_features.items(), start=1):
            member_idx = [feat_to_idx[member] for member in members if member in feat_to_idx]
            if not member_idx:
                continue

            group_signed = shap_vals[:, member_idx].sum(axis=1)
            group_abs_signed = np.abs(group_signed)
            group_sum_abs_members = np.abs(shap_vals[:, member_idx]).sum(axis=1)
            group_data = X_explain[members].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            group_color = np.nanmean(group_data, axis=1)

            group_signed_values[group_name] = group_signed
            group_color_values[group_name] = group_color
            group_rows.append(
                {
                    "configured_order": int(group_order),
                    "group": group_name,
                    "n_features": int(len(members)),
                    "source_features": "; ".join(members),
                    "display_features": "; ".join(display_name_by_source.get(member, member) for member in members),
                    "mean_abs_group_shap": float(np.mean(group_abs_signed)),
                    "mean_sum_abs_member_shap_audit": float(np.mean(group_sum_abs_members)),
                    "mean_signed_group_shap": float(np.mean(group_signed)),
                }
            )

        grouped_shap = (
            pd.DataFrame(group_rows)
            .sort_values("mean_abs_group_shap", ascending=False)
            .reset_index(drop=True)
        )
        grouped_shap_csv = shap_dir / f"{target_safe}_{MODEL_NAME}_grouped_mean_abs_shap.csv"
        grouped_shap.to_csv(grouped_shap_csv, index=False)
        print("[SHAP] Saved grouped:", grouped_shap_csv)

        grouped_top_n = int(min(SHAP_TOP_N, len(grouped_shap)))
        grouped_top = grouped_shap.head(grouped_top_n).copy()
        grouped_top_rev = grouped_top.iloc[::-1]
        grouped_top_names = grouped_top["group"].tolist()

        fig = plt.figure(figsize=SHAP_PLOT_STYLE["figsize_bar"])
        ax = fig.add_axes(SHAP_AXES_RECT_BAR)
        ax.barh(
            grouped_top_rev["group"],
            grouped_top_rev["mean_abs_group_shap"],
            color=SHAP_PLOT_STYLE["bar_color"],
            alpha=SHAP_PLOT_STYLE["bar_alpha"],
        )
        ax.set_xlabel("mean(|sum SHAP value|)", fontsize=SHAP_PLOT_STYLE["label_fs"])
        ax.tick_params(axis="x", labelsize=SHAP_PLOT_STYLE["tick_fs"], width=1.1, length=3.5)
        ax.tick_params(axis="y", labelsize=SHAP_PLOT_STYLE["ytick_fs"], width=1.1, length=3.5)
        set_spines_lw(ax, SHAP_PLOT_STYLE["spine_lw"])
        apply_shap_xaxis_controls(
            ax,
            SHAP_AXIS_SETTINGS_BAR,
            values=grouped_top["mean_abs_group_shap"].to_numpy(float),
            start_at_zero=True,
        )

        grouped_bar_base = shap_dir / f"{target_safe}_{MODEL_NAME}_SHAP_grouped_bar_top_clean"
        save_shap_figure_multi(fig, grouped_bar_base)
        plt.close(fig)
        print("[SHAP] Saved grouped:", str(grouped_bar_base) + ".png/.svg/.pdf")

        fig = plt.figure(figsize=SHAP_PLOT_STYLE["figsize_strip"])
        ax = fig.add_axes(SHAP_AXES_RECT_STRIP)
        for row_i, group_name in enumerate(grouped_top_names[::-1]):
            group_signed = group_signed_values[group_name]
            group_color = group_color_values[group_name]
            jitter = (rng.random(len(group_signed)) - 0.5) * 0.55
            yj = row_i + jitter

            vmin, vmax = robust_minmax(group_color)
            colors = np.clip((group_color - vmin) / (vmax - vmin), 0, 1)
            ax.scatter(
                group_signed,
                yj,
                c=colors,
                cmap="viridis",
                s=SHAP_PLOT_STYLE["strip_s"],
                alpha=SHAP_PLOT_STYLE["strip_alpha"],
                edgecolors="none",
            )

        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_yticks(range(len(grouped_top_names)))
        ax.set_yticklabels(grouped_top_names[::-1], fontsize=SHAP_PLOT_STYLE["ytick_fs"])
        ax.set_xlabel("Grouped SHAP value (sum within category)", fontsize=SHAP_PLOT_STYLE["label_fs"])
        ax.tick_params(axis="x", labelsize=SHAP_PLOT_STYLE["tick_fs"], width=1.1, length=3.5)
        set_spines_lw(ax, SHAP_PLOT_STYLE["spine_lw"])
        grouped_strip_axis_values = np.concatenate([group_signed_values[group_name] for group_name in grouped_top_names])
        apply_shap_xaxis_controls(
            ax,
            SHAP_AXIS_SETTINGS_STRIP,
            values=grouped_strip_axis_values,
            start_at_zero=False,
        )

        grouped_strip_base = shap_dir / f"{target_safe}_{MODEL_NAME}_SHAP_grouped_strip_top_clean"
        save_shap_figure_multi(fig, grouped_strip_base)
        plt.close(fig)
        print("[SHAP] Saved grouped:", str(grouped_strip_base) + ".png/.svg/.pdf")

        # ---------------- Top grouped-SHAP 1D PDP ----------------
        pdp_dir = save_dir / "PDP_top3_grouped_SHAP"
        pdp_dir.mkdir(parents=True, exist_ok=True)
        pdp_records = []

        X_base_pdp_all = pd.concat([x_train_eval, x_val, x_test], axis=0, ignore_index=True)
        rng_pdp = np.random.default_rng(PDP_RANDOM_SEED)
        if len(X_base_pdp_all) > PDP_MAX_BASE_ROWS:
            sampled_idx = rng_pdp.choice(len(X_base_pdp_all), size=PDP_MAX_BASE_ROWS, replace=False)
            X_base_pdp = X_base_pdp_all.iloc[sampled_idx].copy().reset_index(drop=True)
        else:
            X_base_pdp = X_base_pdp_all.copy().reset_index(drop=True)

        def predict_fn(X_work: pd.DataFrame) -> np.ndarray:
            return np.asarray(stack.predict(X_work[feature_names]), dtype=float).reshape(-1)

        for rank_i, row in grouped_shap.head(PDP_TOP_N_GROUPS).iterrows():
            group_name = str(row["group"])
            members = [m.strip() for m in str(row["source_features"]).split(";") if m.strip()]
            ranked = shap_imp[shap_imp["source_feature"].isin(members)].copy()
            if ranked.empty:
                continue
            pdp_feature = str(ranked.sort_values("mean_abs_shap", ascending=False).iloc[0]["source_feature"])
            vals = pd.to_numeric(X_base_pdp[pdp_feature], errors="coerce").dropna().to_numpy(float)
            if vals.size < 2 or is_binary_or_categorical_feature(vals, max_categories=MAIN_2D_PDP_MAX_CATEGORIES):
                continue
            lo = np.nanquantile(vals, PDP_PERCENTILE_RANGE[0])
            hi = np.nanquantile(vals, PDP_PERCENTILE_RANGE[1])
            if np.isclose(lo, hi):
                continue
            grid_vals = np.linspace(lo, hi, PDP_GRID_POINTS)
            y_means = []
            for v in grid_vals:
                X_tmp = X_base_pdp.copy()
                X_tmp[pdp_feature] = v
                y_means.append(np.mean(predict_fn(X_tmp)))

            out_base = pdp_dir / f"rank{rank_i+1}_{sanitize_filename(group_name)}_{sanitize_filename(pdp_feature)}_PDP"
            save_pdp_plot_numeric(
                grid_vals,
                np.asarray(y_means),
                SHAP_LABEL_MAP.get(pdp_feature, pdp_feature),
                f"Mean predicted {TARGET_COL}",
                f"({rank_i+1}) {group_name}",
                out_base,
            )
            pdp_records.append({
                "rank": int(rank_i + 1),
                "group": group_name,
                "feature": pdp_feature,
                "type": "continuous_1D",
                "output_basepath": str(out_base),
            })

        pdp_summary_csv = pdp_dir / f"{target_safe}_{MODEL_NAME}_PDP_top3_summary.csv"
        pd.DataFrame(pdp_records).to_csv(pdp_summary_csv, index=False)

        # ---------------- Main 2D PDP ----------------
        main_2d_outputs: Dict[str, Any] = {"enabled": False}
        if RUN_MAIN_2D_PDP:
            main_2d_dir = save_dir / "PDP_main_2D_design_windows"
            main_2d_dir.mkdir(parents=True, exist_ok=True)

            main_2d_records = []
            main_2d_panels = []

            # IMPORTANT:
            # Main 2D PDP panels are now generated from the TOP 3 GROUPED SHAP groups.
            # Pairwise combinations are:
            #   rank 1 × rank 2
            #   rank 1 × rank 3
            #   rank 2 × rank 3
            # The manual MAIN_2D_PDP_PAIRS list is used only as a fallback.
            top3_main_groups = grouped_shap.head(3)["group"].astype(str).tolist()
            main_2d_pairs = []
            if len(top3_main_groups) >= 2:
                main_2d_pairs.append({
                    "x_group": top3_main_groups[0],
                    "y_group": top3_main_groups[1],
                    "label": f"{top3_main_groups[0]} × {top3_main_groups[1]}",
                    "source": "top3_grouped_SHAP_pair_1_2",
                })
            if len(top3_main_groups) >= 3:
                main_2d_pairs.append({
                    "x_group": top3_main_groups[0],
                    "y_group": top3_main_groups[2],
                    "label": f"{top3_main_groups[0]} × {top3_main_groups[2]}",
                    "source": "top3_grouped_SHAP_pair_1_3",
                })
                main_2d_pairs.append({
                    "x_group": top3_main_groups[1],
                    "y_group": top3_main_groups[2],
                    "label": f"{top3_main_groups[1]} × {top3_main_groups[2]}",
                    "source": "top3_grouped_SHAP_pair_2_3",
                })

            if not main_2d_pairs:
                main_2d_pairs = MAIN_2D_PDP_PAIRS

            for pair_i, pair in enumerate(main_2d_pairs, start=1):
                x_feature = choose_representative_feature(feature_names, pair["x_group"], X_base_pdp, shap_imp=shap_imp)
                y_feature = choose_representative_feature(feature_names, pair["y_group"], X_base_pdp, shap_imp=shap_imp)
                pair_label = pair.get("label", f"{pair['x_group']} × {pair['y_group']}")
                panel_label = f"({chr(96 + pair_i)}) {pair_label}"

                if x_feature is None or y_feature is None:
                    main_2d_records.append({
                        "rank": pair_i,
                        "x_group": pair["x_group"],
                        "y_group": pair["y_group"],
                        "x_feature": x_feature,
                        "y_feature": y_feature,
                        "label": pair_label,
                        "source": pair.get("source", "fallback_manual_pair"),
                        "top3_grouped_SHAP_groups": "; ".join(top3_main_groups),
                        "status": "skipped_missing_group_feature",
                        "output_basepath": None,
                    })
                    continue

                panel = make_2d_pdp_panel(
                    x_feature=x_feature,
                    y_feature=y_feature,
                    panel_label=panel_label,
                    X_base_pdp=X_base_pdp,
                    predict_fn=predict_fn,
                    grid_points=MAIN_2D_PDP_GRID_POINTS,
                    percentile_range=MAIN_2D_PDP_PERCENTILE_RANGE,
                    max_categories=MAIN_2D_PDP_MAX_CATEGORIES,
                )

                if panel is None:
                    main_2d_records.append({
                        "rank": pair_i,
                        "x_group": pair["x_group"],
                        "y_group": pair["y_group"],
                        "x_feature": x_feature,
                        "y_feature": y_feature,
                        "label": pair_label,
                        "source": pair.get("source", "fallback_manual_pair"),
                        "top3_grouped_SHAP_groups": "; ".join(top3_main_groups),
                        "status": "skipped_invalid_axis",
                        "output_basepath": None,
                    })
                    continue

                out_base = main_2d_dir / f"panel{pair_i}_{sanitize_filename(TARGET_COL)}_{sanitize_filename(pair_label)}_2D_PDP"
                save_2d_pdp_panel(panel, target_col=TARGET_COL, out_base=out_base)
                main_2d_panels.append(panel)

                main_2d_records.append({
                    "rank": pair_i,
                    "x_group": pair["x_group"],
                    "y_group": pair["y_group"],
                    "x_feature": x_feature,
                    "y_feature": y_feature,
                    "label": pair_label,
                    "source": pair.get("source", "fallback_manual_pair"),
                    "top3_grouped_SHAP_groups": "; ".join(top3_main_groups),
                    "status": "saved",
                    "output_basepath": str(out_base),
                })
                print("[MAIN 2D PDP] Saved:", str(out_base) + ".png/.svg/.pdf")

            composite_base = None
            if main_2d_panels:
                composite_base = main_2d_dir / f"{sanitize_filename(TARGET_COL)}_{MODEL_NAME}_main_2D_PDP_composite"
                save_2d_pdp_composite(main_2d_panels, target_col=TARGET_COL, out_base=composite_base, ncols=2)
                print("[MAIN 2D PDP] Saved composite:", str(composite_base) + ".png/.svg/.pdf")

            main_2d_summary_csv = main_2d_dir / f"{sanitize_filename(TARGET_COL)}_{MODEL_NAME}_main_2D_PDP_summary.csv"
            pd.DataFrame(main_2d_records).to_csv(main_2d_summary_csv, index=False)

            main_2d_outputs = {
                "enabled": True,
                "output_dir": str(main_2d_dir),
                "summary_csv": str(main_2d_summary_csv),
                "composite_basepath": None if composite_base is None else str(composite_base),
                "records": main_2d_records,
            }

        pdp_outputs = {
            "enabled": True,
            "output_dir": str(pdp_dir),
            "pdp_summary_csv": str(pdp_summary_csv),
            "main_2d_outputs": main_2d_outputs,
        }

        shap_outputs = {
            "enabled": True,
            "output_dir": str(shap_dir),
            "mean_abs_shap_csv": str(shap_imp_csv),
            "bar_top_basepath": str(shap_bar_base),
            "strip_top_basepath": str(shap_strip_base),
            "plot_file_formats": ["png", "svg", "pdf"],
            "top_n": top_n,
            "background_rows": int(len(X_bg)),
            "explained_rows": int(len(X_explain)),
            "axis_settings_bar": SHAP_AXIS_SETTINGS_BAR,
            "axis_settings_strip": SHAP_AXIS_SETTINGS_STRIP,
            "explainer": explainer_name,
            "model_output": "Stacking Capacity prediction",
            "grouped": {
                "mean_abs_shap_csv": str(grouped_shap_csv),
                "bar_top_basepath": str(grouped_bar_base),
                "strip_top_basepath": str(grouped_strip_base),
                "plot_file_formats": ["png", "svg", "pdf"],
                "top_n": grouped_top_n,
                "group_definitions": SHAP_GROUP_DEFINITIONS,
                "group_display_map": SHAP_GROUP_DISPLAY_MAP,
                "groups_resolved": grouped_features,
                "importance_definition": "mean over samples of abs(sum(signed SHAP values)) within each group; avoids over-weighting one-hot groups",
                "audit_column": "mean_sum_abs_member_shap_audit keeps sum(abs(member SHAP)) for comparison only",
            },
            "pdp_outputs": pdp_outputs,
        }
        print("[SHAP/PDP] Done.")

    metrics_row = {
        "target": TARGET_COL,
        "model": MODEL_NAME,
        "r2_train": r2_tr,
        "rmse_train": rmse_tr,
        "mae_train": mae_tr,
        "r2_val": r2_va,
        "rmse_val": rmse_va,
        "mae_val": mae_va,
        "r2_test": r2_te,
        "rmse_test": rmse_te,
        "mae_test": mae_te,
        "plot_train_basepath": plot_files["train"],
        "plot_validation_basepath": plot_files["validation"],
        "plot_test_basepath": plot_files["test"],
        "plot_file_formats": "png/svg/pdf",
    }
    pd.DataFrame([metrics_row]).to_csv(save_dir / f"stacking_hdr_capacity_{tag}_metrics.csv", index=False)

    payload = {
        "input_file": str(Path(INPUT_FP).expanduser().resolve()),
        "input_sheet": INPUT_SHEET,
        "target": TARGET_COL,
        "seed": int(SEED),
        "current_feature_set": mode,
        "dropped_rows_due_to_invalid_current_feature": int(dropped_due_to_current),
        "iqr": iqr_info,
        "augmentation": {
            "enabled": bool(AUGMENTATION_ENABLED),
            "pairwise_fraction": float(aug_fraction),
            "noise_fraction": float(NOISE_FRACTION),
        },
        "stacking": {
            "meta_model": {"type": "Ridge", "alpha": float(RIDGE_ALPHA)},
            "cv": int(STACK_CV),
            "passthrough": bool(STACK_PASSTHROUGH),
            "base_models": {
                "rf": {
                    "n_estimators": int(RF_N_ESTIMATORS),
                    "max_depth": None if RF_MAX_DEPTH is None else int(RF_MAX_DEPTH),
                    "max_features": RF_MAX_FEATURES,
                    "min_samples_split": int(RF_MIN_SAMPLES_SPLIT),
                    "min_samples_leaf": int(RF_MIN_SAMPLES_LEAF),
                    "bootstrap": bool(RF_BOOTSTRAP),
                },
                "xgb": {
                    "n_estimators": int(XGB_N_ESTIMATORS),
                    "learning_rate": float(XGB_LEARNING_RATE),
                    "max_depth": int(XGB_MAX_DEPTH),
                    "subsample": float(XGB_SUBSAMPLE),
                    "colsample_bytree": float(XGB_COLSAMPLE_BYTREE),
                    "reg_alpha": float(XGB_REG_ALPHA),
                    "reg_lambda": float(XGB_REG_LAMBDA),
                    "min_child_weight": float(XGB_MIN_CHILD_WEIGHT),
                },
                "cat": {
                    "iterations": int(CAT_ITERATIONS),
                    "learning_rate": float(CAT_LEARNING_RATE),
                    "depth": int(CAT_DEPTH),
                    "l2_leaf_reg": float(CAT_L2_LEAF_REG),
                    "random_strength": float(CAT_RANDOM_STRENGTH),
                    "bagging_temperature": float(CAT_BAGGING_TEMPERATURE),
                },
            },
        },
        "rows": {
            "n_total_after_target_filter": int(len(x_df)),
            "train_before_iqr_aug": int(len(y_train_eval)),
            "train_fit_after_iqr_aug": int(len(y_fit)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "metrics": {
            "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
            "validation": {"r2": r2_va, "rmse": rmse_va, "mae": mae_va},
            "test": {"r2": r2_te, "rmse": rmse_te, "mae": mae_te},
        },
        "plot_style_used": PLOT_STYLE,
        "axis_settings_used": axis_cfg,
        "plot_files_basepath": plot_files,
        "plot_file_formats": ["png", "svg", "pdf"],
        "shap_outputs": shap_outputs,
        "pdp_outputs": pdp_outputs,
        "features_used": list(x_df.columns),
        "dropped_columns": [str(c) for c in dropped_cols],
        "all_nan_feature_columns_dropped": [str(c) for c in all_nan_cols],
        "files": {
            "pred_xlsx": str(pred_xlsx),
            "pred_csv": str(pred_csv),
            "model_pkl": str(model_pkl),
            "metrics_csv": str(save_dir / f"stacking_hdr_capacity_{tag}_metrics.csv"),
            "plot_train_basepath": plot_files["train"],
            "plot_validation_basepath": plot_files["validation"],
            "plot_test_basepath": plot_files["test"],
        },
    }
    metrics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== HDR Capacity Stacking ({mode} current) ===")
    print(f"Train: R2={r2_tr:.3f}, RMSE={rmse_tr:.3f}, MAE={mae_tr:.3f}")
    print(f"Val:   R2={r2_va:.3f}, RMSE={rmse_va:.3f}, MAE={mae_va:.3f}")
    print(f"Test:  R2={r2_te:.3f}, RMSE={rmse_te:.3f}, MAE={mae_te:.3f}")
    print(f"IQR removed rows (train fit only): {iqr_info.get('removed_count', 0)}")
    print(f"Saved predictions: {pred_xlsx}")
    print(f"Saved metrics: {metrics_json}")


if __name__ == "__main__":
    main()


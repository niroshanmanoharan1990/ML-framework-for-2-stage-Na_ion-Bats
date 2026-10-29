#!/usr/bin/env python3
"""
Standalone GBDT trainer for Slop / Slop_Frac with SHAP.

This version is based on your ICE CatBoost SHAP plotting style, but adapted to:
- GradientBoostingRegressor / GBDT
- Slop or Slop_Frac target
- Current (A) -> Current (mA)
- Optional current feature mode: raw / log / both
- Train-only IQR filtering
- Train-only pairwise augmentation
- 75/15/10 split
- Manual grid search using validation RMSE
- Train/Val/Test actual-vs-predicted plots
- Individual SHAP bar + strip plots
- Grouped SHAP bar + grouped strip plots

Before running:
    pip install pandas numpy scikit-learn matplotlib joblib openpyxl shap
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, train_test_split


# =========================
# CONFIG
# =========================
# ---- Repository paths (relative to the repository root) ----
from pathlib import Path as _RepoPath
REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "SI_Model_02.xlsx"
DATA_SHEET = "Slope_Processed"
OUTPUT_DIR = REPO_ROOT / "outputs" / "model_02/slope_gbdt"

INPUT_FP = str(DATA_FILE)
INPUT_SHEET = DATA_SHEET
SAVE_DIR = str(OUTPUT_DIR)

# Choose "Slop" or "Slop_Frac"
TARGET = "Slop"

CURRENT_COL = "Current (A)"
CURRENT_FEATURE_SET = "log"  # options: "raw", "log", "both"

SEED = 42

IQR_ENABLED = True
IQR_K = 2

AUGMENTATION_ENABLED = True
PAIRWISE_FRACTION = 0.10
NOISE_FRACTION = 0.05

# Grid can contain one value each or multiple values.
param_grid = {
    "n_estimators": [150],
    "learning_rate": [0.0255],
    "max_depth": [7],       # None is also valid
    "subsample": [0.8],
    "max_features": [0.8],  # can be None, "sqrt", "log2", float, int
    "min_samples_leaf": [4],
    "min_samples_split": [3],
    "ccp_alpha": [1e-3],
    "loss": ["squared_error"],
}

# Fallback defaults if not present in param_grid
GBDT_SUBSAMPLE_DEFAULT = 0.8
GBDT_MAX_FEATURES_DEFAULT = 0.8
GBDT_LOSS_DEFAULT = "squared_error"

SLOP_ALIASES = ["Slop", "SLOP", "Slope", "slope"]
FRAC_ALIASES = [
    "Slop_Frac",
    "Slop_fraction",
    "Slop Fraction",
    "slop_frac",
    "slop_fraction",
    "slop fraction",
]

MODEL_NAME = "GBDT"

# =========================
# True-vs-pred graph settings
# =========================
AXIS_SETTINGS_TRUEPRED: Dict[str, Dict[str, Any]] = {
    "Slop": {
        "xlim": (0, 400),
        "ylim": (0, 400),
        "xtick_step": 100,
        "ytick_step": 100,
    },
    "Slop_Frac": {
        "xlim": (0, 1.0),
        "ylim": (0, 1.0),
        "xtick_step": 0.25,
        "ytick_step": 0.25,
    },
    "Slop_fraction": {
        "xlim": (0, 1.0),
        "ylim": (0, 1.0),
        "xtick_step": 0.25,
        "ytick_step": 0.25,
    },
}

# Optional global manual axis override.
# Leave as None unless you want to force the same axes for all Train/Val/Test plots.
AXIS_X_MIN = None
AXIS_X_MAX = None
AXIS_Y_MIN = None
AXIS_Y_MAX = None
AXIS_X_TICK_STEP = None
AXIS_Y_TICK_STEP = None

PLOT_STYLE = {
    "font_family": "Arial",
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

# =========================
# SHAP settings
# =========================
RUN_SHAP = True
SHAP_MAX_BACKGROUND = 200
SHAP_MAX_EXPLAIN = 500
SHAP_TOP_N = 10
SHAP_SEED = 42

# Empty dictionaries mean auto-scaled SHAP x-axes.
SHAP_AXES_RECT_BAR = [0.35, 0.18, 0.58, 0.74]
SHAP_AXES_RECT_STRIP = [0.35, 0.18, 0.58, 0.74]
SHAP_AXIS_SETTINGS_BAR: Dict[str, Any] = {}
SHAP_AXIS_SETTINGS_STRIP: Dict[str, Any] = {}

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

# =========================
# PDP settings
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

# Main Model 02 slope PDP design-window panels.
# These are group-aware: the code finds matching columns automatically.
MAIN_2D_PDP_PAIRS = [
    {"x_group": "Current density", "y_group": "D-S", "label": "Curr × D-S"},
    {"x_group": "Current density", "y_group": "Surface area", "label": "Curr × SSA"},
    {"x_group": "D-S", "y_group": "Surface area", "label": "D-S × SSA"},
    {"x_group": "Carbon ratio", "y_group": "Binder ratio", "label": "Carb × Binder ratio"},
]

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

# Individual feature display names.
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
    "Surface Area": "Surface area",
    "Surface area": "Surface area",
    "BET": "Surface area",
    "SSA": "Surface area",
    "Pore Volume": "Pore volume",
    "Pore volume": "Pore volume",
    "Pore_Volume": "Pore volume",
    "D-spacing": "D-Spacing",
    "D Spacing": "D-Spacing",
    "D_Spacing": "D-Spacing",
    "d002": "D-Spacing",
    "d_002": "D-Spacing",
    "Current (mA)": "Current (mA)",
    "log10 current density": "Log current",
    "Current density": "Current density",
    "Current_Density": "Current density",
    "Carbon%": "C%",
    "C%": "C%",
    "IG/ID": "IG/ID",
    "ID/IG": "ID/IG",
}

# Grouped SHAP definitions.
# These are matched against your processed feature column names.
SHAP_GROUP_DEFINITIONS = {
    "Active material ratio": [
        "Active material ratio",
        "Active_Material_Ratio",
        "ActiveMaterialRatio",
        "Active ratio",
        "AM ratio",
    ],
    "Carbon ratio": [
        "Carbon ratio",
        "Carbon_Ratio",
        "CarbonRatio",
        "Conductive carbon ratio",
    ],
    "Binder ratio": [
        "Binder ratio",
        "Binder_Ratio",
        "BinderRatio",
    ],
    "Separator type": [
        "GF",
        "PE",
        "PP",
    ],
    "Binder type": [
        "CMC",
        "CMC+SBR",
        "CMC_SBR",
        "CMC SBR",
        "No Binder",
        "No_Binder",
        "PAA",
        "PANA",
        "PVDF",
        "SBR",
        "Sodium alginate",
        "Sodium_Alginate",
    ],
    "Solvent type": [
        "NMP",
        "No Solvent",
        "No_Solvent",
        "Water",
    ],
    "Salt type": [
        "NaClO4",
        "NaCLO4",
        "NaOTf",
        "NaOTF",
        "NaPF6",
        "NaTFSI",
    ],
    "Salt concentration": [
        "Salt concentration",
        "Salt_Concentration",
        "SaltConcentration",
        "Salt conc",
        "Salt_conc",
    ],
    "Additive concentration": [
        "Additive concentration",
        "Additive_Concentration",
        "AdditiveConcentration",
        "Additive conc",
        "Additive_conc",
    ],
    "Electrolyte type": [
        "Electrolyte type",
        "Electrolyte_Type",
        "ElectrolyteType",
        "Aqueous",
        "Organic",
        "Carbonate",
        "Ether",
        "Ionic liquid",
        "Ionic_Liquid",
        "PC",
        "PC fraction",
        "PC_Fraction",
        "PCFraction",
        "EC",
        "EC fraction",
        "EC_Fraction",
        "ECFraction",
        "DMC",
        "DMC fraction",
        "DMC_Fraction",
        "DMCFraction",
        "DEC",
        "DEC fraction",
        "DEC_Fraction",
        "DECFraction",
        "EMC",
        "EMC fraction",
        "EMC_Fraction",
        "EMCFraction",
        "FEC",
        "FEC fraction",
        "FEC_Fraction",
        "FECFraction",
        "DME",
        "DME fraction",
        "DME_Fraction",
        "DMEFraction",
        "DOL",
        "DOL fraction",
        "DOL_Fraction",
        "DOLFraction",
        "TETRAGLYME",
        "TETRAGLYME fraction",
        "TETRAGLYME_Fraction",
        "TETRAGLYMEFraction",
    ],
    "La": [
        "La",
        "L_a",
        "La_nm",
        "La (nm)",
        "Lateral crystallite size",
        "Lateral_Crystallite_Size",
    ],
    "D-S": [
        "D-S",
        "D_S",
        "D spacing",
        "D-spacing",
        "D_Spacing",
        "D Spacing",
        "d002",
        "d_002",
    ],
    "Current density": [
        "Current density",
        "Current_Density",
        "CurrentDensity",
        "Current",
        "Curr",
        "Current (A)",
        "Current (mA)",
        "mA/g",
        "log10 current density",
    ],
    "Lc": [
        "Lc",
        "L_c",
        "Lc_nm",
        "Lc (nm)",
        "Crystallite height",
        "Crystallite_Height",
    ],
    "Surface area": [
        "SSA",
        "Surface Area",
        "Surface area",
        "Specific surface area",
        "BET",
        "BET surface area",
        "BET_Surface_Area",
    ],
    "Pore volume": [
        "PV",
        "Pore Volume",
        "Pore volume",
        "Pore_Volume",
    ],
    "IG/ID": [
        "IG/ID",
        "Ig/Id",
        "ID/IG",
        "Id/Ig",
        "Raman",
    ],
    "C": [
        "C",
        "Carbon%",
        "Carbon percent",
        "C%",
    ],
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
    "Active material ratio": "Act_rat",
    "Separator type": "Sep",
    "Salt concentration": "Salt_C",
    "Additive concentration": "Add_C",
    "IG/ID": "IG/ID",
    "C": "C",
}


# =========================
# Utility functions
# =========================
def apply_plot_style() -> None:
    plt.rcParams["font.family"] = PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = PLOT_STYLE["svg_fonttype"]
    plt.rcParams["pdf.fonttype"] = PLOT_STYLE["pdf_fonttype"]
    plt.rcParams["ps.fonttype"] = PLOT_STYLE["ps_fonttype"]


def sanitize_filename(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_figure_multi(fig, basepath_no_ext: str) -> None:
    fig.savefig(basepath_no_ext + ".png", dpi=PLOT_STYLE["dpi_png"], bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".svg", bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".pdf", bbox_inches="tight")


def save_shap_figure_multi(fig, basepath_no_ext: str) -> None:
    fig.savefig(basepath_no_ext + ".png", dpi=SHAP_PLOT_STYLE["dpi_png"], bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".svg", bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".pdf", bbox_inches="tight")


def set_spines(ax) -> None:
    for s in ax.spines.values():
        s.set_linewidth(PLOT_STYLE["spine_lw"])


def set_spines_lw(ax, lw: float) -> None:
    for s in ax.spines.values():
        s.set_linewidth(lw)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
    }


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

    return float(nice_fraction * (10 ** exponent))


def get_data_range(
    series: np.ndarray,
    padding: float = 0.05,
    round_to: float = 5.0,
) -> Tuple[float, float, float]:
    arr = np.asarray(series).ravel()
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return 0.0, 1.0, 0.2

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    span = max_val - min_val

    if span <= 0:
        base = abs(min_val) if min_val != 0 else 1.0
        min_val -= 0.05 * base
        max_val += 0.05 * base
        span = max_val - min_val

    min_pad = min_val - span * padding
    max_pad = max_val + span * padding

    if round_to > 0:
        min_tick = round_to * np.floor(min_pad / round_to)
        max_tick = round_to * np.ceil(max_pad / round_to)
    else:
        min_tick, max_tick = min_pad, max_pad

    n_ticks = 6
    step = (max_tick - min_tick) / n_ticks if n_ticks > 0 else (max_tick - min_tick)

    if step <= 0:
        step = 1.0

    if round_to > 0:
        step = max(round_to, round(step / round_to) * round_to)
        if step <= 0:
            step = round_to

    return float(min_tick), float(max_tick), float(step)


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

    if "xtick_step" in settings and settings["xtick_step"] <= 0:
        raise SystemExit("Invalid AXIS_X_TICK_STEP: must be > 0.")

    if "ytick_step" in settings and settings["ytick_step"] <= 0:
        raise SystemExit("Invalid AXIS_Y_TICK_STEP: must be > 0.")

    return settings


def plot_true_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    set_name: str,
    target_name: str,
    model_name: str,
    save_dir: Path,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    xtick_step: float | None = None,
    ytick_step: float | None = None,
) -> str:
    apply_plot_style()

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    r2_v = float(r2_score(y_true, y_pred))
    rmse_v = float(np.sqrt(mean_squared_error(y_true, y_pred)))

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
        x_min_auto, x_max_auto, x_step_auto = get_data_range(y_true, round_to=5.0)
        if xlim is None:
            xlim = (x_min_auto, x_max_auto)
        if xtick_step is None:
            xtick_step = x_step_auto

    if ylim is None or ytick_step is None:
        y_min_auto, y_max_auto, y_step_auto = get_data_range(y_pred, round_to=5.0)
        if ylim is None:
            ylim = (y_min_auto, y_max_auto)
        if ytick_step is None:
            ytick_step = y_step_auto

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * xtick_step, xtick_step))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.5 * ytick_step, ytick_step))
    ax.tick_params(axis="both", labelsize=PLOT_STYLE["tick_fs"], width=1.1, length=3.5)

    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=PLOT_STYLE["diag_color"],
        linestyle=PLOT_STYLE["diag_ls"],
        linewidth=PLOT_STYLE["diag_lw"],
    )

    textstr = f"R2 = {r2_v:.3f}\nRMSE = {rmse_v:.3f}\n{model_name}"
    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=PLOT_STYLE["ann_fs"],
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=PLOT_STYLE["ann_bbox"],
    )

    set_spines(ax)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()

    base = save_dir / f"{sanitize_filename(target_name)}_{set_name}_{sanitize_filename(model_name)}_true_vs_pred"
    save_figure_multi(fig, str(base))
    plt.close(fig)

    return str(base)


# =========================
# SHAP helper functions
# =========================
def auto_shap_xaxis(ax, values, start_at_zero: bool = False, n_ticks: int = 5) -> None:
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


def apply_shap_xaxis_controls(ax, settings: Dict[str, Any], values, start_at_zero: bool = False) -> None:
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


def normalize_group_token(value) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def normalized_word_chunks(value):
    chunks = re.split(r"[^A-Za-z0-9]+", str(value).lower())
    return [chunk for chunk in chunks if chunk]


def feature_matches_group_token(feature_name, token) -> bool:
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

    if feature_key in {
        f"{token_key}fraction",
        f"{token_key}percent",
        f"{token_key}percentage",
    }:
        return True

    # Long exact concepts such as SaltConcentration may appear with prefixes/suffixes.
    # Short codes such as PE/PP/GF only match word chunks, avoiding accidental substring matches.
    if len(token_key) >= 4 and token_key in feature_key:
        return True

    return False


def build_shap_groups(feature_names, display_name_by_source):
    grouped = {}
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

    # Anything not captured by your defined groups is kept as its own row.
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



# =========================
# PDP helper functions
# =========================
def find_group_feature_candidates(feature_names: List[str], group_name: str) -> List[str]:
    tokens = SHAP_GROUP_DEFINITIONS.get(group_name, [group_name])
    matches = []
    for f in feature_names:
        if any(feature_matches_group_token(f, t) for t in tokens):
            matches.append(f)
    return matches


def choose_representative_feature(
    feature_names: List[str],
    group_name: str,
    x_data: pd.DataFrame,
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
        vals = pd.to_numeric(x_data[c], errors="coerce").dropna().to_numpy(float)
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
    x_base_pdp: pd.DataFrame,
    grid_points: int,
    percentile_range: Tuple[float, float],
    max_categories: int,
) -> Optional[Dict[str, Any]]:
    if feature not in x_base_pdp.columns:
        return None

    vals = pd.to_numeric(x_base_pdp[feature], errors="coerce").dropna().to_numpy(dtype=float)
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
    x_base_pdp: pd.DataFrame,
    predict_fn,
    grid_points: int,
    percentile_range: Tuple[float, float],
    max_categories: int,
):
    x_axis = build_pdp_axis_values(x_feature, x_base_pdp, grid_points, percentile_range, max_categories)
    y_axis = build_pdp_axis_values(y_feature, x_base_pdp, grid_points, percentile_range, max_categories)
    if x_axis is None or y_axis is None:
        return None

    z = np.zeros((len(y_axis["model_values"]), len(x_axis["model_values"])), dtype=float)
    for j, yv in enumerate(y_axis["model_values"]):
        for i, xv in enumerate(x_axis["model_values"]):
            x_tmp = x_base_pdp.copy()
            x_tmp[x_feature] = xv
            x_tmp[y_feature] = yv
            z[j, i] = float(np.mean(predict_fn(x_tmp)))

    return {
        "panel_label": panel_label,
        "x_feature": x_feature,
        "y_feature": y_feature,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "Z": z,
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
    z = np.asarray(panel["Z"], dtype=float)

    x_is_cat = x_axis["kind"] == "categorical"
    y_is_cat = y_axis["kind"] == "categorical"

    cmap = PDP_PLOT_STYLE.get("cmap", "viridis")
    levels_n = int(PDP_PLOT_STYLE.get("n_contour_levels", 16))

    finite_z = z[np.isfinite(z)]
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
        xg, yg = np.meshgrid(x_vals, y_vals)
        levels = np.linspace(vmin, vmax, levels_n)
        im = ax.contourf(xg, yg, z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend="both")
        ax.contour(
            xg, yg, z,
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
            z,
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


def run_shap_analysis(
    best_model,
    x_background_df: pd.DataFrame,
    x_explain_df: pd.DataFrame,
    feature_names,
    target_col: str,
    model_name: str,
    save_dir: Path,
    x_pdp_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    shap_outputs: Dict[str, Any] = {"enabled": bool(RUN_SHAP)}

    if not RUN_SHAP:
        return shap_outputs

    try:
        import shap
        print("[SHAP] shap imported successfully.")
    except Exception as exc:
        raise RuntimeError("SHAP is required. Install with: pip install shap") from exc

    shap_dir = save_dir / "SHAP"
    shap_dir.mkdir(parents=True, exist_ok=True)
    print("[SHAP] Output dir:", shap_dir)

    display_names = [SHAP_LABEL_MAP.get(name, name) for name in feature_names]

    # Avoid duplicate display names.
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

    X_bg_full = pd.DataFrame(x_background_df, columns=feature_names)
    X_explain_full = pd.DataFrame(x_explain_df, columns=feature_names)

    rng = np.random.default_rng(SHAP_SEED)

    if len(X_bg_full) > SHAP_MAX_BACKGROUND:
        bg_idx = rng.choice(len(X_bg_full), size=SHAP_MAX_BACKGROUND, replace=False)
        X_bg = X_bg_full.iloc[bg_idx]
    else:
        X_bg = X_bg_full

    if len(X_explain_full) > SHAP_MAX_EXPLAIN:
        ex_idx = rng.choice(len(X_explain_full), size=SHAP_MAX_EXPLAIN, replace=False)
        X_explain = X_explain_full.iloc[ex_idx]
    else:
        X_explain = X_explain_full

    explainer_name = "shap.TreeExplainer(best_model, data=X_bg)"

    try:
        explainer = shap.TreeExplainer(best_model, data=X_bg)
        shap_exp = explainer(X_explain)
        shap_vals = np.asarray(shap_exp.values)
    except Exception as tree_exc:
        print("[SHAP] TreeExplainer failed; using PermutationExplainer:", tree_exc)
        max_evals = 2 * X_explain.shape[1] + 1
        explainer = shap.PermutationExplainer(best_model.predict, X_bg)
        shap_exp = explainer(X_explain, max_evals=max_evals)
        shap_vals = np.asarray(shap_exp.values)
        explainer_name = "shap.PermutationExplainer(best_model.predict, X_bg)"

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
    shap_imp_csv = shap_dir / f"{target_safe}_{model_name}_mean_abs_shap.csv"
    shap_imp.to_csv(shap_imp_csv, index=False)
    print("[SHAP] Saved:", shap_imp_csv)

    top_n = int(min(SHAP_TOP_N, len(shap_imp)))
    top = shap_imp.head(top_n).copy()
    top_rev = top.iloc[::-1]

    top_source_features = top["source_feature"].tolist()
    top_labels = top["feature"].tolist()
    feat_to_idx = {feature: i for i, feature in enumerate(X_explain.columns)}

    plt.rcParams["font.family"] = PLOT_STYLE["font_family"]

    # -------------------------
    # Individual SHAP bar
    # -------------------------
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

    shap_bar_base = shap_dir / f"{target_safe}_{model_name}_SHAP_bar_top_clean"
    save_shap_figure_multi(fig, str(shap_bar_base))
    plt.close(fig)
    print("[SHAP] Saved:", str(shap_bar_base) + ".png/.svg/.pdf")

    # -------------------------
    # Individual SHAP strip
    # -------------------------
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

    strip_axis_values = np.concatenate(
        [shap_vals[:, feat_to_idx[feature]] for feature in top_source_features]
    )

    apply_shap_xaxis_controls(
        ax,
        SHAP_AXIS_SETTINGS_STRIP,
        values=strip_axis_values,
        start_at_zero=False,
    )

    shap_strip_base = shap_dir / f"{target_safe}_{model_name}_SHAP_strip_top_clean"
    save_shap_figure_multi(fig, str(shap_strip_base))
    plt.close(fig)
    print("[SHAP] Saved:", str(shap_strip_base) + ".png/.svg/.pdf")

    # -------------------------
    # Grouped SHAP
    # -------------------------
    grouped_features = build_shap_groups(list(X_explain.columns), display_name_by_source)

    group_rows = []
    group_signed_values = {}
    group_color_values = {}

    for group_order, (group_name, members) in enumerate(grouped_features.items(), start=1):
        member_idx = [feat_to_idx[member] for member in members if member in feat_to_idx]

        if not member_idx:
            continue

        # Signed grouped SHAP for each sample.
        group_signed = shap_vals[:, member_idx].sum(axis=1)

        # Importance definition used for ranking.
        group_abs_signed = np.abs(group_signed)

        # Audit value only, not used for ranking.
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
                "display_features": "; ".join(
                    display_name_by_source.get(member, member) for member in members
                ),
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

    grouped_shap_csv = shap_dir / f"{target_safe}_{model_name}_grouped_mean_abs_shap.csv"
    grouped_shap.to_csv(grouped_shap_csv, index=False)
    print("[SHAP] Saved grouped:", grouped_shap_csv)

    grouped_top_n = int(min(SHAP_TOP_N, len(grouped_shap)))
    grouped_top = grouped_shap.head(grouped_top_n).copy()
    grouped_top_rev = grouped_top.iloc[::-1]
    grouped_top_names = grouped_top["group"].tolist()

    # Grouped bar
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

    grouped_bar_base = shap_dir / f"{target_safe}_{model_name}_SHAP_grouped_bar_top_clean"
    save_shap_figure_multi(fig, str(grouped_bar_base))
    plt.close(fig)
    print("[SHAP] Saved grouped:", str(grouped_bar_base) + ".png/.svg/.pdf")

    # Grouped strip
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

    grouped_strip_axis_values = np.concatenate(
        [group_signed_values[group_name] for group_name in grouped_top_names]
    )

    apply_shap_xaxis_controls(
        ax,
        SHAP_AXIS_SETTINGS_STRIP,
        values=grouped_strip_axis_values,
        start_at_zero=False,
    )

    grouped_strip_base = shap_dir / f"{target_safe}_{model_name}_SHAP_grouped_strip_top_clean"
    save_shap_figure_multi(fig, str(grouped_strip_base))
    plt.close(fig)
    print("[SHAP] Saved grouped:", str(grouped_strip_base) + ".png/.svg/.pdf")

    # =========================
    # PDP analysis
    # =========================
    pdp_outputs: Dict[str, Any] = {"enabled": False}

    if RUN_PDP:
        if x_pdp_df is None:
            x_pdp_df = pd.concat([x_background_df, x_explain_df], axis=0, ignore_index=True)

        x_pdp_df = pd.DataFrame(x_pdp_df, columns=feature_names).copy().reset_index(drop=True)
        rng_pdp = np.random.default_rng(PDP_RANDOM_SEED)
        if len(x_pdp_df) > PDP_MAX_BASE_ROWS:
            sampled_idx = rng_pdp.choice(len(x_pdp_df), size=PDP_MAX_BASE_ROWS, replace=False)
            x_base_pdp = x_pdp_df.iloc[sampled_idx].copy().reset_index(drop=True)
        else:
            x_base_pdp = x_pdp_df.copy().reset_index(drop=True)

        def predict_fn(x_work: pd.DataFrame) -> np.ndarray:
            return np.asarray(best_model.predict(x_work[feature_names]), dtype=float).reshape(-1)

        # ---------- Top grouped-SHAP 1D PDP ----------
        pdp_dir = save_dir / "PDP_top3_grouped_SHAP"
        pdp_dir.mkdir(parents=True, exist_ok=True)
        pdp_records = []

        for rank_i, row in grouped_shap.head(PDP_TOP_N_GROUPS).iterrows():
            group_name = str(row["group"])
            members = [m.strip() for m in str(row["source_features"]).split(";") if m.strip()]
            ranked = shap_imp[shap_imp["source_feature"].isin(members)].copy()
            if ranked.empty:
                continue

            pdp_feature = str(ranked.sort_values("mean_abs_shap", ascending=False).iloc[0]["source_feature"])
            vals = pd.to_numeric(x_base_pdp[pdp_feature], errors="coerce").dropna().to_numpy(float)
            if vals.size < 2 or is_binary_or_categorical_feature(vals, max_categories=MAIN_2D_PDP_MAX_CATEGORIES):
                continue

            lo = np.nanquantile(vals, PDP_PERCENTILE_RANGE[0])
            hi = np.nanquantile(vals, PDP_PERCENTILE_RANGE[1])
            if np.isclose(lo, hi):
                continue

            grid_vals = np.linspace(lo, hi, PDP_GRID_POINTS)
            y_means = []
            for v in grid_vals:
                x_tmp = x_base_pdp.copy()
                x_tmp[pdp_feature] = v
                y_means.append(np.mean(predict_fn(x_tmp)))

            out_base = pdp_dir / f"rank{rank_i+1}_{sanitize_filename(group_name)}_{sanitize_filename(pdp_feature)}_PDP"
            save_pdp_plot_numeric(
                grid_vals,
                np.asarray(y_means),
                SHAP_LABEL_MAP.get(pdp_feature, pdp_feature),
                f"Mean predicted {target_col}",
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

        pdp_summary_csv = pdp_dir / f"{target_safe}_{model_name}_PDP_top3_summary.csv"
        pd.DataFrame(pdp_records).to_csv(pdp_summary_csv, index=False)

        # ---------- Main 2D PDP ----------
        main_2d_outputs: Dict[str, Any] = {"enabled": False}
        if RUN_MAIN_2D_PDP:
            main_2d_dir = save_dir / "PDP_main_2D_design_windows"
            main_2d_dir.mkdir(parents=True, exist_ok=True)

            main_2d_records = []
            main_2d_panels = []

            for pair_i, pair in enumerate(MAIN_2D_PDP_PAIRS, start=1):
                x_feature = choose_representative_feature(feature_names, pair["x_group"], x_base_pdp, shap_imp=shap_imp)
                y_feature = choose_representative_feature(feature_names, pair["y_group"], x_base_pdp, shap_imp=shap_imp)
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
                        "status": "skipped_missing_group_feature",
                        "output_basepath": None,
                    })
                    continue

                panel = make_2d_pdp_panel(
                    x_feature=x_feature,
                    y_feature=y_feature,
                    panel_label=panel_label,
                    x_base_pdp=x_base_pdp,
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
                        "status": "skipped_invalid_axis",
                        "output_basepath": None,
                    })
                    continue

                out_base = main_2d_dir / f"panel{pair_i}_{sanitize_filename(target_col)}_{sanitize_filename(pair_label)}_2D_PDP"
                save_2d_pdp_panel(panel, target_col=target_col, out_base=out_base)
                main_2d_panels.append(panel)

                main_2d_records.append({
                    "rank": pair_i,
                    "x_group": pair["x_group"],
                    "y_group": pair["y_group"],
                    "x_feature": x_feature,
                    "y_feature": y_feature,
                    "label": pair_label,
                    "status": "saved",
                    "output_basepath": str(out_base),
                })
                print("[MAIN 2D PDP] Saved:", str(out_base) + ".png/.svg/.pdf")

            composite_base = None
            if main_2d_panels:
                composite_base = main_2d_dir / f"{sanitize_filename(target_col)}_{model_name}_main_2D_PDP_composite"
                save_2d_pdp_composite(main_2d_panels, target_col=target_col, out_base=composite_base, ncols=2)
                print("[MAIN 2D PDP] Saved composite:", str(composite_base) + ".png/.svg/.pdf")

            main_2d_summary_csv = main_2d_dir / f"{sanitize_filename(target_col)}_{model_name}_main_2D_PDP_summary.csv"
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
        "model_output": f"{target_col} prediction",
        "grouped": {
            "mean_abs_shap_csv": str(grouped_shap_csv),
            "bar_top_basepath": str(grouped_bar_base),
            "strip_top_basepath": str(grouped_strip_base),
            "plot_file_formats": ["png", "svg", "pdf"],
            "top_n": grouped_top_n,
            "group_definitions": SHAP_GROUP_DEFINITIONS,
            "group_display_map": SHAP_GROUP_DISPLAY_MAP,
            "groups_resolved": grouped_features,
            "importance_definition": (
                "mean over samples of abs(sum(signed SHAP values)) within each group; "
                "this avoids over-weighting one-hot groups with many dummy columns"
            ),
            "audit_column": (
                "mean_sum_abs_member_shap_audit keeps the sum(abs(member SHAP)) value "
                "for comparison only"
            ),
            "strip_definition": "sum of signed SHAP values within each group for each explained sample",
        },
        "pdp_outputs": pdp_outputs,
    }

    print("[SHAP] Done.")
    return shap_outputs


# =========================
# Data preparation functions
# =========================
def resolve_target(df: pd.DataFrame, target_mode: str) -> str:
    aliases = SLOP_ALIASES if target_mode == "Slop" else FRAC_ALIASES

    for c in aliases:
        if c in df.columns:
            return c

    raise SystemExit(f"Target aliases not found for TARGET={target_mode}")


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

    x1 = x_in.iloc[idx1].to_numpy()
    x2 = x_in.iloc[idx2].to_numpy()

    x_syn = (x1 + x2) / 2.0

    col_stds = x_in.std(axis=0, ddof=0).to_numpy()
    noise = rng.normal(0.0, noise_fraction * col_stds, size=x_syn.shape)
    x_syn = x_syn + noise

    y_arr = y_in.to_numpy(float)
    y_syn = (y_arr[idx1] + y_arr[idx2]) / 2.0

    return pd.DataFrame(x_syn, columns=x_in.columns), pd.Series(y_syn, name=y_in.name)


def prepare_xy(
    df_raw: pd.DataFrame,
    target_col: str,
    current_col: str,
    current_feature_set: str,
):
    df = df_raw.copy()

    if current_col not in df.columns:
        raise SystemExit(f"Missing current column: {current_col}")

    # Current conversion: A -> mA
    cur_a = pd.to_numeric(df[current_col], errors="coerce")
    df["Current (mA)"] = cur_a * 1000.0
    cur_ma = pd.to_numeric(df["Current (mA)"], errors="coerce")

    need_log = current_feature_set in {"log", "both"}

    if need_log:
        if (cur_ma <= 0).any():
            raise SystemExit("Current (mA) has non-positive values; cannot compute log10 current.")
        df["log10 current density"] = np.log10(cur_ma)

    output_drop = [
        "Plateau",
        "PLATEAU",
        "Platuea",
        "PLATUEA",
        "Capacity",
        "CAPACITY",
        "Pla/Cap",
        "PLA_CAP",
        "PLA/CAP",
        "Slop",
        "SLOP",
        "Slope",
        "slope",
        "Slop_Frac",
        "Slop_fraction",
        "Slop Fraction",
        target_col,
    ]

    helper_drop = [
        "actual_Slop",
        "predicted_Slop",
        "actual_Slop_Frac",
        "predicted_Slop_Frac",
        "actual_Slop_fraction",
        "predicted_Slop_fraction",
        "actual_Capacity",
        "predicted_Capacity",
        "abs_error",
        "split",
        "row_in_processed_table",
    ]

    drop_cols = [c for c in (output_drop + helper_drop) if c in df.columns]

    if current_feature_set == "raw":
        if "log10 current density" in df.columns:
            drop_cols.append("log10 current density")

    elif current_feature_set == "log":
        if "Current (mA)" in df.columns:
            drop_cols.append("Current (mA)")

    elif current_feature_set == "both":
        pass

    else:
        raise SystemExit('CURRENT_FEATURE_SET must be one of: "raw", "log", "both"')

    # Drop original current A column so that current is represented only by selected transformed feature(s).
    if current_col in df.columns:
        drop_cols.append(current_col)

    drop_cols = list(dict.fromkeys(drop_cols))

    x_df = df.drop(columns=drop_cols, errors="ignore").copy()
    y = pd.to_numeric(df[target_col], errors="coerce")

    # Force all remaining features numeric.
    # This assumes your categorical variables are already one-hot encoded.
    x_df = x_df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing target.
    keep = y.notna()
    x_df = x_df.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    # Drop all-NaN columns.
    all_nan_cols = x_df.columns[x_df.isna().all()].tolist()
    if all_nan_cols:
        x_df = x_df.drop(columns=all_nan_cols).copy()

    # Fill feature NaNs with feature medians.
    x_df = x_df.fillna(x_df.median(numeric_only=True))

    if x_df.shape[1] == 0:
        raise SystemExit("No usable features remain after preprocessing.")

    return x_df, y, drop_cols, all_nan_cols


def build_gbdt_from_params(p: dict, seed: int):
    max_depth = p.get("max_depth", 3)

    if max_depth is not None:
        max_depth = int(max_depth)

    max_features = p.get("max_features", GBDT_MAX_FEATURES_DEFAULT)

    if isinstance(max_features, str) and max_features.strip().lower() == "none":
        max_features = None

    return GradientBoostingRegressor(
        n_estimators=int(p.get("n_estimators", 100)),
        learning_rate=float(p.get("learning_rate", 0.1)),
        max_depth=max_depth,
        subsample=float(p.get("subsample", GBDT_SUBSAMPLE_DEFAULT)),
        max_features=max_features,
        min_samples_leaf=int(p.get("min_samples_leaf", 1)),
        min_samples_split=int(p.get("min_samples_split", 2)),
        ccp_alpha=float(p.get("ccp_alpha", 0.0)),
        loss=str(p.get("loss", GBDT_LOSS_DEFAULT)),
        random_state=seed,
    )


# =========================
# Main
# =========================
def main():
    save_dir = Path(SAVE_DIR).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    input_fp = Path(INPUT_FP).expanduser().resolve()

    if not input_fp.exists():
        raise SystemExit(f"Input file not found: {input_fp}")

    if input_fp.suffix.lower() in {".xlsx", ".xls"}:
        df_raw = pd.read_excel(input_fp, sheet_name=INPUT_SHEET)
    else:
        df_raw = pd.read_csv(input_fp)

    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    for c in df_raw.select_dtypes(include=["object", "string"]).columns:
        df_raw[c] = df_raw[c].astype(str).str.strip()

    target_col = resolve_target(df_raw, TARGET)
    x_df, y, dropped_cols, all_nan_cols = prepare_xy(df_raw, target_col, CURRENT_COL, CURRENT_FEATURE_SET)

    idx = np.arange(len(x_df))

    # 75/25 first split.
    x_train, x_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        x_df,
        y,
        idx,
        test_size=0.25,
        random_state=SEED,
    )

    # 15/10 validation/test split from the 25%.
    # 0.4 of 25% = 10% total test.
    x_val, x_test, y_val, y_test, idx_val, idx_test = train_test_split(
        x_temp,
        y_temp,
        idx_temp,
        test_size=0.4,
        random_state=SEED,
    )

    # Keep original train set for honest train evaluation.
    x_train_eval = x_train.copy()
    y_train_eval = y_train.copy()

    # x_fit/y_fit are used only for model fitting.
    x_fit = x_train.copy()
    y_fit = y_train.copy()

    # Train-only IQR filtering.
    iqr_info = {
        "enabled": bool(IQR_ENABLED),
        "k": float(IQR_K),
        "removed_count": 0,
    }

    if IQR_ENABLED:
        keep_mask, info = iqr_filter_mask(y_fit, IQR_K)
        x_fit = x_fit.loc[keep_mask].reset_index(drop=True)
        y_fit = y_fit.loc[keep_mask].reset_index(drop=True)
        iqr_info = {"enabled": True, "k": float(IQR_K), **info}

    # Train-only augmentation.
    aug_fraction = PAIRWISE_FRACTION if AUGMENTATION_ENABLED else 0.0
    x_aug, y_aug = pairwise_augmentation(
        x_fit,
        y_fit,
        aug_fraction,
        NOISE_FRACTION,
        SEED,
    )

    if x_aug is not None:
        x_fit = pd.concat([x_fit, x_aug], ignore_index=True)
        y_fit = pd.concat([y_fit, y_aug], ignore_index=True)

    combos = list(ParameterGrid(param_grid))
    print(f"[GBDT] trials: {len(combos)}")

    best_model = None
    best_params = None
    best_val_rmse = float("inf")
    trial_rows = []

    y_val_np = y_val.to_numpy(float)

    for i, p in enumerate(combos, start=1):
        model = build_gbdt_from_params(p, SEED)
        model.fit(x_fit, y_fit)

        pred_val = model.predict(x_val)

        val_rmse = rmse(y_val_np, pred_val)
        val_r2 = float(r2_score(y_val_np, pred_val))

        trial_rows.append(
            {
                "trial": i,
                **p,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
            }
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model
            best_params = dict(p)
            print(
                f"[GBDT] best trial {i}: "
                f"val_rmse={val_rmse:.4f}, "
                f"val_r2={val_r2:.4f}, "
                f"params={best_params}"
            )

        elif i % 25 == 0:
            print(f"[GBDT] progress: {i}/{len(combos)}")

    if best_model is None:
        raise SystemExit("No model was trained. Check param_grid and data.")

    pred_tr = best_model.predict(x_train_eval)
    pred_va = best_model.predict(x_val)
    pred_te = best_model.predict(x_test)

    m_tr = metrics(y_train_eval.to_numpy(float), pred_tr)
    m_va = metrics(y_val.to_numpy(float), pred_va)
    m_te = metrics(y_test.to_numpy(float), pred_te)

    target_slug = target_col.replace(" ", "_")

    pred_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "row_in_processed_table": idx_train,
                    "split": "train",
                    f"actual_{target_slug}": y_train_eval.to_numpy(float),
                    f"predicted_{target_slug}": pred_tr,
                }
            ),
            pd.DataFrame(
                {
                    "row_in_processed_table": idx_val,
                    "split": "validation",
                    f"actual_{target_slug}": y_val.to_numpy(float),
                    f"predicted_{target_slug}": pred_va,
                }
            ),
            pd.DataFrame(
                {
                    "row_in_processed_table": idx_test,
                    "split": "test",
                    f"actual_{target_slug}": y_test.to_numpy(float),
                    f"predicted_{target_slug}": pred_te,
                }
            ),
        ],
        ignore_index=True,
    ).sort_values("row_in_processed_table", ignore_index=True)

    pred_df["abs_error"] = (
        pred_df[f"actual_{target_slug}"] - pred_df[f"predicted_{target_slug}"]
    ).abs()

    trials_df = pd.DataFrame(trial_rows).sort_values(
        ["val_rmse", "val_r2"],
        ascending=[True, False],
        ignore_index=True,
    )

    pred_xlsx = save_dir / f"gbdt_{target_slug.lower()}_train_val_test_actual_vs_pred.xlsx"
    pred_csv = save_dir / f"gbdt_{target_slug.lower()}_train_val_test_actual_vs_pred.csv"
    trials_csv = save_dir / f"gbdt_{target_slug.lower()}_grid_trials.csv"
    model_pkl = save_dir / f"gbdt_{target_slug.lower()}_best.pkl"
    metrics_json = save_dir / f"gbdt_{target_slug.lower()}_metrics.json"

    pred_df.to_excel(pred_xlsx, index=False)
    pred_df.to_csv(pred_csv, index=False)
    trials_df.to_csv(trials_csv, index=False)

    joblib.dump(best_model, model_pkl)
    joblib.dump(list(x_df.columns), save_dir / "feature_names.pkl")

    axis_cfg = resolve_axis_settings(target_col)

    plot_train = plot_true_vs_pred(
        y_train_eval.to_numpy(float),
        pred_tr,
        "Train",
        target_col,
        MODEL_NAME,
        save_dir,
        **axis_cfg,
    )

    plot_val = plot_true_vs_pred(
        y_val.to_numpy(float),
        pred_va,
        "Val",
        target_col,
        MODEL_NAME,
        save_dir,
        **axis_cfg,
    )

    plot_test = plot_true_vs_pred(
        y_test.to_numpy(float),
        pred_te,
        "Test",
        target_col,
        MODEL_NAME,
        save_dir,
        **axis_cfg,
    )

    # =========================
    # SHAP analysis
    # =========================
    # Background = training data used for model fit.
    # Explanation = real test data only.
    shap_outputs = run_shap_analysis(
        best_model=best_model,
        x_background_df=x_fit,
        x_explain_df=x_test,
        feature_names=list(x_df.columns),
        target_col=target_col,
        model_name=MODEL_NAME,
        save_dir=save_dir,
        x_pdp_df=pd.concat([x_train_eval, x_val, x_test], axis=0, ignore_index=True),
    )

    payload = {
        "input_file": str(input_fp),
        "input_sheet": INPUT_SHEET,
        "target": target_col,
        "model": MODEL_NAME,
        "seed": int(SEED),
        "current_feature_set": CURRENT_FEATURE_SET,
        "iqr": iqr_info,
        "augmentation": {
            "enabled": bool(AUGMENTATION_ENABLED),
            "pairwise_fraction": float(aug_fraction),
            "noise_fraction": float(NOISE_FRACTION),
        },
        "best_params": best_params,
        "rows": {
            "n_total_after_target_filter": int(len(x_df)),
            "train_before_iqr_aug": int(len(y_train_eval)),
            "train_fit_after_iqr_aug": int(len(y_fit)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "metrics": {
            "train": m_tr,
            "validation": m_va,
            "test": m_te,
        },
        "axis_settings_used": axis_cfg,
        "plot_style_used": PLOT_STYLE,
        "plot_files_basepath": {
            "train": plot_train,
            "validation": plot_val,
            "test": plot_test,
        },
        "plot_file_formats": ["png", "svg", "pdf"],
        "shap_outputs": shap_outputs,
        "pdp_outputs": shap_outputs.get("pdp_outputs", {"enabled": False}),
        "dropped_columns": [str(c) for c in dropped_cols],
        "all_nan_feature_columns_dropped": [str(c) for c in all_nan_cols],
        "features_used": list(x_df.columns),
        "files": {
            "pred_xlsx": str(pred_xlsx),
            "pred_csv": str(pred_csv),
            "trials_csv": str(trials_csv),
            "model_pkl": str(model_pkl),
        },
    }

    metrics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== {MODEL_NAME} on {target_col} ===")
    print(f"Train: R2={m_tr['r2']:.3f}, RMSE={m_tr['rmse']:.3f}")
    print(f"Val:   R2={m_va['r2']:.3f}, RMSE={m_va['rmse']:.3f}")
    print(f"Test:  R2={m_te['r2']:.3f}, RMSE={m_te['rmse']:.3f}")
    print(f"Best params: {best_params}")
    print(f"Saved predictions: {pred_xlsx}")
    print(f"Saved metrics: {metrics_json}")
    print(f"Saved SHAP outputs: {save_dir / 'SHAP'}")


if __name__ == "__main__":
    main()

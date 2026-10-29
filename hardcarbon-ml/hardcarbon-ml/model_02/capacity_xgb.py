#!/usr/bin/env python3
"""
PyCharm-ready XGB trainer for Capacity (Sheet1 style).

What it does:
- Reads Excel/CSV
- Builds Train/Val/Test split = 75/15/10
- Optional IQR filter on train only
- Optional pairwise augmentation on train only
- Trains XGBoost (fixed params or small grid)
- Exports one Excel with actual vs predicted for train/val/test
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor


# =========================
# CONFIG (edit in PyCharm)
# =========================
# ---- Repository paths (relative to the repository root) ----
from pathlib import Path as _RepoPath
REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "SI_Model_02.xlsx"
DATA_SHEET = "Capacity_Processed"
OUTPUT_DIR = REPO_ROOT / "outputs" / "model_02/capacity_xgb"

INPUT_FP = str(DATA_FILE)
INPUT_SHEET = DATA_SHEET
SAVE_DIR = str(OUTPUT_DIR)
TARGET_COL = "Capacity"
SEED = 42

# Train-only augmentation
PAIRWISE_FRACTION = 0.1
NOISE_FRACTION = 0.05

# Train-only IQR filter
IQR_ENABLED = True
IQR_K = 1.9

# Fixed model config (best from 1024-trial optimization)
N_ESTIMATORS = 600
LEARNING_RATE = 0.0575
MAX_DEPTH = 5
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.35
REG_LAMBDA = 2.5

# Optional feature transform
LOG_SSA = False

# Optional small grid search (set True/False)
USE_GRID = False
GRID_PARAM = {
    "n_estimators": [250, 300, 350],
    "learning_rate": [0.10, 0.12, 0.15],
    "max_depth": [5, 6, 7],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.4, 0.5, 0.6],
    "reg_lambda": [0.5, 1.0, 2.0],
}

MODEL_NAME = "XGB"

RUN_SHAP = True
SHAP_MAX_BACKGROUND = 200
SHAP_MAX_EXPLAIN = 500
SHAP_TOP_N = 10
SHAP_SEED = 42

# Empty dictionaries mean auto-scaled SHAP x-axes.
SHAP_AXES_RECT_BAR = [0.42, 0.25, 0.65, 0.80]
SHAP_AXES_RECT_STRIP = [0.42, 0.25, 0.65, 0.80]
SHAP_AXIS_SETTINGS_BAR: Dict[str, Any] = {}
SHAP_AXIS_SETTINGS_STRIP: Dict[str, Any] = {}

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
}

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
        "mA/g",
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
}

# Default fixed axis for capacity plots (editable).
AXIS_SETTINGS_TRUEPRED = {
    "Capacity": {
        "xlim": (0.0, 600.0),
        "ylim": (0.0, 600.0),
        "xtick_step": 150.0,
        "ytick_step": 150.0,
    }
}

PLOT_STYLE = {
    "font_family": "Arial",

    # Journal-style parity plots
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
    # Journal-style SHAP figures
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

# Main-paper PDP/design-window pairs for Model 02 Capacity.
# These are resolved dynamically against your available columns.
# Continuous-carbon descriptors and cell-chemistry groups are handled automatically.
MAIN_2D_PDP_PAIRS = [
    {"x_group": "Current density", "y_group": "D-S", "label": "Curr × D-S"},
    {"x_group": "D-S", "y_group": "Surface area", "label": "D-S × SSA"},
    {"x_group": "Surface area", "y_group": "Electrolyte type", "label": "SSA × Elec"},
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


def metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


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
    }
    return mask.to_numpy(), info


def pairwise_augmentation(X_in, y_in, fraction, noise_fraction, random_state):
    rng = np.random.default_rng(random_state)
    n = len(X_in)
    n_aug = int(n * fraction)
    if n_aug <= 0:
        return None, None

    idx1 = rng.choice(n, n_aug, replace=False)
    idx2 = rng.choice(n, n_aug, replace=False)
    X1 = X_in.iloc[idx1].to_numpy(float)
    X2 = X_in.iloc[idx2].to_numpy(float)
    X_syn = (X1 + X2) / 2.0

    # Treat one-hot/binary columns as categorical: keep valid 0/1 and do not add noise.
    binary_cols = []
    for col in X_in.columns:
        vals = pd.to_numeric(X_in[col], errors="coerce").dropna().to_numpy(float)
        if vals.size == 0:
            continue
        u = np.unique(vals)
        if np.all(np.isin(np.round(u, 12), [0.0, 1.0])):
            binary_cols.append(col)
    binary_idx = [X_in.columns.get_loc(c) for c in binary_cols]
    continuous_cols = [c for c in X_in.columns if c not in binary_cols]
    continuous_idx = [X_in.columns.get_loc(c) for c in continuous_cols]

    if binary_idx:
        take_first = rng.integers(0, 2, size=(n_aug, len(binary_idx))).astype(bool)
        X_syn[:, binary_idx] = np.where(take_first, X1[:, binary_idx], X2[:, binary_idx])

    if continuous_idx:
        col_stds = X_in[continuous_cols].std(axis=0, ddof=0).to_numpy(float)
        noise = rng.normal(0.0, noise_fraction * col_stds, size=(n_aug, len(continuous_idx)))
        X_syn[:, continuous_idx] = X_syn[:, continuous_idx] + noise

    y_arr = y_in.to_numpy(float)
    y_syn = (y_arr[idx1] + y_arr[idx2]) / 2.0
    return pd.DataFrame(X_syn, columns=X_in.columns), pd.Series(y_syn, name=y_in.name)


def build_xgb(n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_lambda, seed):
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_lambda=float(reg_lambda),
        n_jobs=-1,
        verbosity=0,
    )


def sanitize_filename(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


def apply_plot_style() -> None:
    plt.rcParams["font.family"] = PLOT_STYLE["font_family"]
    plt.rcParams["svg.fonttype"] = PLOT_STYLE["svg_fonttype"]
    plt.rcParams["pdf.fonttype"] = PLOT_STYLE["pdf_fonttype"]
    plt.rcParams["ps.fonttype"] = PLOT_STYLE["ps_fonttype"]


def save_figure_multi(basepath_no_ext: str) -> None:
    plt.savefig(basepath_no_ext + ".png", dpi=PLOT_STYLE["dpi_png"], bbox_inches="tight")
    plt.savefig(basepath_no_ext + ".svg", bbox_inches="tight")
    plt.savefig(basepath_no_ext + ".pdf", bbox_inches="tight")


def save_shap_figure_multi(fig, basepath_no_ext: str) -> None:
    fig.savefig(basepath_no_ext + ".png", dpi=SHAP_PLOT_STYLE["dpi_png"], bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".svg", bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".pdf", bbox_inches="tight")


def set_spines(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_STYLE["spine_lw"])


def set_spines_lw(ax: plt.Axes, lw: float) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(lw)


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


def nice_limits_ticks(data: np.ndarray, n_ticks: int = 6, padding: float = 0.05, clamp_min_zero: bool = True):
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
    return float(vmin_nice), float(vmax_nice), float(step), ticks


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



def is_binary_feature(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if vals.size == 0:
        return False
    u = np.unique(np.round(vals, 12))
    return bool(np.all(np.isin(u, [0.0, 1.0])))


def group_display_name(group_name: str) -> str:
    return SHAP_GROUP_DISPLAY_MAP.get(group_name, SHAP_LABEL_MAP.get(group_name, group_name))


def resolve_group_members(group_name: str, feature_names: List[str]) -> List[str]:
    """Resolve a configured group name to actual dataframe columns."""
    if group_name in SHAP_GROUP_DEFINITIONS:
        tokens = SHAP_GROUP_DEFINITIONS[group_name]
        members = [
            f for f in feature_names
            if any(feature_matches_group_token(f, token) for token in tokens)
        ]
        if members:
            return members

    # Exact/source/display fallback
    exact = [f for f in feature_names if f == group_name]
    if exact:
        return exact

    display_matches = [
        f for f in feature_names
        if SHAP_LABEL_MAP.get(f, f) == group_name or group_display_name(group_name) == SHAP_LABEL_MAP.get(f, f)
    ]
    if display_matches:
        return display_matches

    fuzzy = [
        f for f in feature_names
        if feature_matches_group_token(f, group_name)
    ]
    return fuzzy


def infer_pdp_axis_kind(X_base: pd.DataFrame, members: List[str]) -> str:
    """single continuous feature -> continuous; one-hot group -> categorical_group."""
    if not members:
        return "missing"

    if len(members) == 1:
        col = members[0]
        vals = pd.to_numeric(X_base[col], errors="coerce").dropna()
        if vals.nunique() <= MAIN_2D_PDP_MAX_CATEGORIES and is_binary_feature(vals):
            return "binary_single"
        return "continuous"

    binary_count = sum(is_binary_feature(X_base[m]) for m in members if m in X_base.columns)
    if binary_count >= max(2, int(0.6 * len(members))):
        return "categorical_group"

    # Multiple non-binary members: choose the first continuous-like member for PDP.
    return "multi_numeric_use_first"


def make_axis_from_group(
    group_name: str,
    feature_names: List[str],
    X_base: pd.DataFrame,
    grid_points: int,
    percentile_range: Tuple[float, float],
    max_categories: int,
) -> Optional[Dict[str, Any]]:
    members = resolve_group_members(group_name, feature_names)
    members = [m for m in members if m in X_base.columns]

    if not members:
        print(f"[PDP] Skipped group '{group_name}': no matching feature columns.")
        return None

    kind = infer_pdp_axis_kind(X_base, members)
    display = group_display_name(group_name)

    if kind in ("continuous", "multi_numeric_use_first"):
        col = members[0]
        vals = pd.to_numeric(X_base[col], errors="coerce").dropna().to_numpy(float)
        if vals.size < 2:
            return None
        lo = float(np.nanquantile(vals, percentile_range[0]))
        hi = float(np.nanquantile(vals, percentile_range[1]))
        if np.isclose(lo, hi):
            return None
        grid = np.linspace(lo, hi, grid_points)
        return {
            "kind": "continuous",
            "group": group_name,
            "members": [col],
            "display": display,
            "axis_label": display,
            "model_values": grid,
            "plot_values": grid,
            "actual_plot_values": vals,
            "tick_labels": None,
        }

    if kind == "binary_single":
        col = members[0]
        codes = [0.0, 1.0]
        labels = [f"{display}=0", f"{display}=1"]
        vals = pd.to_numeric(X_base[col], errors="coerce").dropna().to_numpy(float)
        return {
            "kind": "categorical_single",
            "group": group_name,
            "members": [col],
            "display": display,
            "axis_label": display,
            "model_values": np.asarray(codes, dtype=float),
            "plot_values": np.arange(len(codes), dtype=float),
            "actual_plot_values": vals,
            "tick_labels": labels,
        }

    if kind == "categorical_group":
        # Limit to categories actually present and not too many.
        present = []
        for m in members:
            vals = pd.to_numeric(X_base[m], errors="coerce").fillna(0).to_numpy(float)
            if np.nanmax(vals) > 0:
                present.append(m)
        if not present:
            present = members
        present = present[:max_categories]
        tick_labels = [SHAP_LABEL_MAP.get(m, m) for m in present]

        # Actual category position from argmax over present group columns.
        group_values = X_base[present].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        actual_pos = np.argmax(group_values, axis=1) if group_values.size else np.array([], dtype=float)

        return {
            "kind": "categorical_group",
            "group": group_name,
            "members": present,
            "display": display,
            "axis_label": display,
            "model_values": np.arange(len(present), dtype=float),
            "plot_values": np.arange(len(present), dtype=float),
            "actual_plot_values": actual_pos.astype(float),
            "tick_labels": tick_labels,
        }

    return None


def apply_pdp_axis_value(X_tmp: pd.DataFrame, axis: Dict[str, Any], value: float) -> pd.DataFrame:
    """Apply one axis value to a PDP dataframe."""
    kind = axis["kind"]

    if kind == "continuous":
        col = axis["members"][0]
        X_tmp[col] = float(value)

    elif kind == "categorical_single":
        col = axis["members"][0]
        X_tmp[col] = float(value)

    elif kind == "categorical_group":
        members = axis["members"]
        category_index = int(round(float(value)))
        category_index = max(0, min(category_index, len(members) - 1))
        for m in members:
            X_tmp[m] = 0.0
        X_tmp[members[category_index]] = 1.0

    return X_tmp


def make_2d_pdp_panel(
    x_group: str,
    y_group: str,
    panel_label: str,
    feature_names: List[str],
    X_base: pd.DataFrame,
    predict_fn,
    grid_points: int,
    percentile_range: Tuple[float, float],
    max_categories: int,
) -> Optional[Dict[str, Any]]:
    x_axis = make_axis_from_group(x_group, feature_names, X_base, grid_points, percentile_range, max_categories)
    y_axis = make_axis_from_group(y_group, feature_names, X_base, grid_points, percentile_range, max_categories)
    if x_axis is None or y_axis is None:
        return None

    Z = np.zeros((len(y_axis["model_values"]), len(x_axis["model_values"])), dtype=float)

    for j, yv in enumerate(y_axis["model_values"]):
        for i, xv in enumerate(x_axis["model_values"]):
            X_tmp = X_base.copy()
            X_tmp = apply_pdp_axis_value(X_tmp, x_axis, xv)
            X_tmp = apply_pdp_axis_value(X_tmp, y_axis, yv)
            Z[j, i] = float(np.mean(predict_fn(X_tmp)))

    return {
        "panel_label": panel_label,
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


def plot_2d_pdp_panel_on_ax(ax, panel: Dict[str, Any], target_col: str, add_colorbar: bool = True, vmin=None, vmax=None):
    x_axis = panel["x_axis"]
    y_axis = panel["y_axis"]
    Z = np.asarray(panel["Z"], dtype=float)

    x_is_cat = x_axis["kind"] != "continuous"
    y_is_cat = y_axis["kind"] != "continuous"

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
        ax.set_xticklabels(x_axis["tick_labels"], rotation=35, ha="right", fontsize=9)
    if y_is_cat:
        ax.set_yticks(y_vals)
        ax.set_yticklabels(y_axis["tick_labels"], fontsize=9)

    # Marginal rug marks show data-supported regions for continuous axes.
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


def save_2d_pdp_panel(panel: Dict[str, Any], target_col: str, out_base: Path):
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


def save_2d_pdp_composite(panels: List[Dict[str, Any]], target_col: str, out_base: Path, ncols: int = 2):
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


def save_pdp_plot_categorical(labels, y_mean, x_label, y_label, title_label, out_base: Path):
    plt.rcParams["font.family"] = PDP_PLOT_STYLE["font_family"]
    fig, ax = plt.subplots(figsize=PDP_PLOT_STYLE["figsize"])
    x = np.arange(len(labels))
    ax.bar(x, y_mean)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_xlabel(x_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.set_ylabel(y_label, fontsize=PDP_PLOT_STYLE["label_fs"])
    ax.tick_params(axis="y", labelsize=PDP_PLOT_STYLE["tick_fs"], width=1.1, length=3.5)
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


def plot_true_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    set_name: str,
    target_name: str,
    model_name: str,
    save_dir: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xtick_step: float | None = None,
    ytick_step: float | None = None,
) -> str:
    apply_plot_style()

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
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
        x0, x1, xstep, _ = nice_limits_ticks(y_true, n_ticks=6, padding=0.05, clamp_min_zero=True)
        if xlim is None:
            xlim = (x0, x1)
        if xtick_step is None:
            xtick_step = xstep

    if ylim is None or ytick_step is None:
        y0, y1, ystep, _ = nice_limits_ticks(y_pred, n_ticks=6, padding=0.05, clamp_min_zero=True)
        if ylim is None:
            ylim = (y0, y1)
        if ytick_step is None:
            ytick_step = ystep

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * xtick_step, xtick_step))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.5 * ytick_step, ytick_step))
    ax.tick_params(axis="both", labelsize=PLOT_STYLE["tick_fs"])

    lo = 0.0
    hi = max(xlim[1], ylim[1])
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=PLOT_STYLE["diag_color"],
        linestyle=PLOT_STYLE["diag_ls"],
        linewidth=PLOT_STYLE["diag_lw"],
    )

    txt = f"R2 = {r2_v:.3f}\nRMSE = {rmse_v:.3f}\n{model_name}"
    ax.text(
        0.95,
        0.05,
        txt,
        transform=ax.transAxes,
        fontsize=PLOT_STYLE["ann_fs"],
        ha="right",
        va="bottom",
        bbox=PLOT_STYLE["ann_bbox"],
    )

    set_spines(ax)
    fig.tight_layout()

    base = save_dir / f"{sanitize_filename(target_name)}_{set_name}_{sanitize_filename(model_name)}_true_vs_pred"
    save_figure_multi(str(base))
    plt.close(fig)
    return str(base)


def main():
    save_dir = Path(SAVE_DIR).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    input_fp = Path(INPUT_FP).expanduser().resolve()
    if not input_fp.exists():
        raise SystemExit(f"Input file not found: {input_fp}")

    if str(input_fp).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_fp, sheet_name=INPUT_SHEET)
    else:
        df = pd.read_csv(input_fp)

    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    ssa_transform_info = {"enabled": bool(LOG_SSA), "applied": False, "column": None, "mode": None, "shift": 0.0}
    if LOG_SSA:
        ssa_names = {"ssa", "surface area", "surface_area", "specific surface area"}
        ssa_col = next((c for c in df.columns if c.strip().lower() in ssa_names), None)
        if ssa_col is not None:
            s = pd.to_numeric(df[ssa_col], errors="coerce")
            finite_mask = np.isfinite(s.to_numpy(float))
            if finite_mask.any():
                min_val = float(np.nanmin(s.to_numpy(float)))
                if min_val > 0:
                    df[ssa_col] = np.log(s)
                    ssa_transform_info = {
                        "enabled": True,
                        "applied": True,
                        "column": ssa_col,
                        "mode": "log",
                        "shift": 0.0,
                    }
                else:
                    shift = 1.0 - min_val
                    df[ssa_col] = np.log(s + shift)
                    ssa_transform_info = {
                        "enabled": True,
                        "applied": True,
                        "column": ssa_col,
                        "mode": "log_shift",
                        "shift": float(shift),
                    }

    if TARGET_COL not in df.columns:
        raise SystemExit(f"Target column not found: {TARGET_COL}")

    leak_cols = {
        "actual_Capacity",
        "predicted_Capacity",
        "abs_error",
        "split",
        "row_in_processed_table",
        "Unnamed: 41",
    }
    dynamic_leak_cols = [c for c in df.columns if c.lower().startswith("actual_") or c.lower().startswith("predicted_")]
    drop_cols = {TARGET_COL, "Electrolyte", "ICE", "No_Binder", "No Binder"} | leak_cols | set(dynamic_leak_cols)
    drop_cols = [c for c in drop_cols if c in df.columns]

    X_df = df.drop(columns=drop_cols, errors="ignore").copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    mask = y.notna()
    X_df = X_df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # Fill and drop all-NaN columns (if any).
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
    if all_nan_cols:
        X_df = X_df.drop(columns=all_nan_cols).copy()

    if X_df.shape[1] == 0:
        raise SystemExit("No usable feature columns after preprocessing.")

    feature_names = list(X_df.columns)
    row_idx = np.arange(len(X_df))

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X_df, y, row_idx, test_size=0.25, random_state=SEED
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.4, random_state=SEED
    )

    X_train_eval = X_train.copy()
    y_train_eval = y_train.copy()
    idx_train_eval = idx_train.copy()

    iqr_info = {"enabled": bool(IQR_ENABLED), "k": float(IQR_K), "removed_count": 0}
    if IQR_ENABLED:
        keep_mask, info = iqr_filter_mask(y_train, IQR_K)
        X_train = X_train.loc[keep_mask].reset_index(drop=True)
        y_train = y_train.loc[keep_mask].reset_index(drop=True)
        iqr_info = {"enabled": True, "k": float(IQR_K), **info}

    X_aug, y_aug = pairwise_augmentation(
        X_train, y_train, PAIRWISE_FRACTION, NOISE_FRACTION, SEED
    )
    if X_aug is not None:
        X_train = pd.concat([X_train, X_aug], ignore_index=True)
        y_train = pd.concat([y_train, y_aug], ignore_index=True)

    if USE_GRID:
        base = build_xgb(300, 0.12, 6, 0.8, 0.5, 1.0, SEED)
        grid = GridSearchCV(
            estimator=base,
            param_grid=GRID_PARAM,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        best_params = grid.best_params_
        pd.DataFrame(grid.cv_results_).to_csv(save_dir / "xgb_grid_trials.csv", index=False)
    else:
        model = build_xgb(
            N_ESTIMATORS,
            LEARNING_RATE,
            MAX_DEPTH,
            SUBSAMPLE,
            COLSAMPLE_BYTREE,
            REG_LAMBDA,
            SEED,
        )
        model.fit(X_train, y_train)
        best_params = {
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "max_depth": MAX_DEPTH,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "reg_lambda": REG_LAMBDA,
        }

    pred_tr = model.predict(X_train_eval)
    pred_va = model.predict(X_val)
    pred_te = model.predict(X_test)

    m_train = metrics(y_train_eval, pred_tr)
    m_val = metrics(y_val, pred_va)
    m_test = metrics(y_test, pred_te)

    # Refit on Train+Validation (with same IQR + augmentation policy), evaluate on Test.
    X_trainval = pd.concat([X_train_eval, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train_eval, y_val], ignore_index=True)
    iqr_info_trainval = {"enabled": bool(IQR_ENABLED), "k": float(IQR_K), "removed_count": 0}
    if IQR_ENABLED:
        keep_mask_tv, info_tv = iqr_filter_mask(y_trainval, IQR_K)
        X_trainval = X_trainval.loc[keep_mask_tv].reset_index(drop=True)
        y_trainval = y_trainval.loc[keep_mask_tv].reset_index(drop=True)
        iqr_info_trainval = {"enabled": True, "k": float(IQR_K), **info_tv}

    X_aug_tv, y_aug_tv = pairwise_augmentation(
        X_trainval, y_trainval, PAIRWISE_FRACTION, NOISE_FRACTION, SEED
    )
    if X_aug_tv is not None:
        X_trainval = pd.concat([X_trainval, X_aug_tv], ignore_index=True)
        y_trainval = pd.concat([y_trainval, y_aug_tv], ignore_index=True)

    model_refit = build_xgb(
        best_params["n_estimators"],
        best_params["learning_rate"],
        best_params["max_depth"],
        best_params["subsample"],
        best_params["colsample_bytree"],
        best_params["reg_lambda"],
        SEED,
    )
    model_refit.fit(X_trainval, y_trainval)
    pred_te_refit = model_refit.predict(X_test)
    m_test_refit = metrics(y_test, pred_te_refit)

    pred_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "row_in_processed_table": idx_train_eval,
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

    base_pred_xlsx = save_dir / "xgb_Capacity_train_val_test_actual_vs_pred.xlsx"
    base_pred_csv = save_dir / "xgb_Capacity_train_val_test_actual_vs_pred.csv"
    pred_xlsx = base_pred_xlsx
    pred_csv = base_pred_csv
    try:
        pred_df.to_excel(pred_xlsx, index=False)
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_xlsx = save_dir / f"xgb_Capacity_train_val_test_actual_vs_pred_{stamp}.xlsx"
        pred_df.to_excel(pred_xlsx, index=False)

    try:
        pred_df.to_csv(pred_csv, index=False)
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_csv = save_dir / f"xgb_Capacity_train_val_test_actual_vs_pred_{stamp}.csv"
        pred_df.to_csv(pred_csv, index=False)

    joblib.dump(model, save_dir / "xgb_Capacity.pkl")
    joblib.dump(model_refit, save_dir / "xgb_Capacity_refit_train_plus_val.pkl")
    joblib.dump(feature_names, save_dir / "feature_names.pkl")

    axis_settings = dict(AXIS_SETTINGS_TRUEPRED.get(TARGET_COL, {}))
    plot_files = {
        "train": plot_true_vs_pred(
            y_train_eval.to_numpy(float),
            pred_tr,
            "Train",
            TARGET_COL,
            MODEL_NAME,
            save_dir,
            **axis_settings,
        ),
        "validation": plot_true_vs_pred(
            y_val.to_numpy(float),
            pred_va,
            "Val",
            TARGET_COL,
            MODEL_NAME,
            save_dir,
            **axis_settings,
        ),
        "test": plot_true_vs_pred(
            y_test.to_numpy(float),
            pred_te,
            "Test",
            TARGET_COL,
            MODEL_NAME,
            save_dir,
            **axis_settings,
        ),
        "test_refit": plot_true_vs_pred(
            y_test.to_numpy(float),
            pred_te_refit,
            "Test_Refit",
            TARGET_COL,
            MODEL_NAME,
            save_dir,
            **axis_settings,
        ),
    }

    shap_outputs: Dict[str, Any] = {"enabled": bool(RUN_SHAP)}
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

        X_train_df_shap = pd.DataFrame(X_trainval, columns=feature_names)
        X_test_df_shap = pd.DataFrame(X_test, columns=feature_names)

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

        explainer_name = "shap.TreeExplainer(model_refit, data=X_bg)"
        try:
            explainer = shap.TreeExplainer(model_refit, data=X_bg)
            shap_exp = explainer(X_explain)
            shap_vals = np.asarray(shap_exp.values)
        except Exception as tree_exc:
            print("[SHAP] TreeExplainer failed; using PermutationExplainer:", tree_exc)
            max_evals = 2 * X_explain.shape[1] + 1
            explainer = shap.PermutationExplainer(model_refit.predict, X_bg)
            shap_exp = explainer(X_explain, max_evals=max_evals)
            shap_vals = np.asarray(shap_exp.values)
            explainer_name = "shap.PermutationExplainer(model_refit.predict, X_bg)"

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
        ax.tick_params(axis="x", labelsize=SHAP_PLOT_STYLE["tick_fs"])
        ax.tick_params(axis="y", labelsize=SHAP_PLOT_STYLE["ytick_fs"])
        set_spines_lw(ax, SHAP_PLOT_STYLE["spine_lw"])
        apply_shap_xaxis_controls(
            ax,
            SHAP_AXIS_SETTINGS_BAR,
            values=top["mean_abs_shap"].to_numpy(float),
            start_at_zero=True,
        )

        shap_bar_base = shap_dir / f"{target_safe}_{MODEL_NAME}_SHAP_bar_top_clean"
        save_shap_figure_multi(fig, str(shap_bar_base))
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

        shap_strip_base = shap_dir / f"{target_safe}_{MODEL_NAME}_SHAP_strip_top_clean"
        save_shap_figure_multi(fig, str(shap_strip_base))
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
        ax.tick_params(axis="x", labelsize=SHAP_PLOT_STYLE["tick_fs"])
        ax.tick_params(axis="y", labelsize=SHAP_PLOT_STYLE["ytick_fs"])
        set_spines_lw(ax, SHAP_PLOT_STYLE["spine_lw"])
        apply_shap_xaxis_controls(
            ax,
            SHAP_AXIS_SETTINGS_BAR,
            values=grouped_top["mean_abs_group_shap"].to_numpy(float),
            start_at_zero=True,
        )

        grouped_bar_base = shap_dir / f"{target_safe}_{MODEL_NAME}_SHAP_grouped_bar_top_clean"
        save_shap_figure_multi(fig, str(grouped_bar_base))
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

        ax.axvline(0, color="red", linestyle="--", linewidth=2)
        ax.set_yticks(range(len(grouped_top_names)))
        ax.set_yticklabels(grouped_top_names[::-1], fontsize=SHAP_PLOT_STYLE["ytick_fs"])
        ax.set_xlabel("Grouped SHAP value (sum within category)", fontsize=SHAP_PLOT_STYLE["label_fs"])
        ax.tick_params(axis="x", labelsize=SHAP_PLOT_STYLE["tick_fs"])
        set_spines_lw(ax, SHAP_PLOT_STYLE["spine_lw"])
        grouped_strip_axis_values = np.concatenate([group_signed_values[group_name] for group_name in grouped_top_names])
        apply_shap_xaxis_controls(
            ax,
            SHAP_AXIS_SETTINGS_STRIP,
            values=grouped_strip_axis_values,
            start_at_zero=False,
        )

        grouped_strip_base = shap_dir / f"{target_safe}_{MODEL_NAME}_SHAP_grouped_strip_top_clean"
        save_shap_figure_multi(fig, str(grouped_strip_base))
        plt.close(fig)
        print("[SHAP] Saved grouped:", str(grouped_strip_base) + ".png/.svg/.pdf")


        # ---------------- PDP: top grouped-SHAP 1D + main 2D design-window maps ----------------
        pdp_outputs: Dict[str, Any] = {"enabled": False}
        if RUN_PDP:
            pdp_dir = save_dir / "PDP_top3_grouped_SHAP"
            pdp_dir.mkdir(parents=True, exist_ok=True)

            pdp_records = []
            pdp_base = X_df.copy().reset_index(drop=True)
            rng_pdp = np.random.default_rng(PDP_RANDOM_SEED)
            if len(pdp_base) > PDP_MAX_BASE_ROWS:
                sampled_idx = rng_pdp.choice(len(pdp_base), size=PDP_MAX_BASE_ROWS, replace=False)
                pdp_base = pdp_base.iloc[sampled_idx].copy().reset_index(drop=True)

            def predict_pdp(X_work: pd.DataFrame) -> np.ndarray:
                X_work = X_work.reindex(columns=feature_names)
                return model_refit.predict(X_work).reshape(-1)

            # 1D PDP for top grouped SHAP groups: use the first resolved feature/category group.
            for rank_i, row in grouped_shap.head(PDP_TOP_N_GROUPS).iterrows():
                group_name = row["group"]
                members_text = str(row["source_features"])
                members = [m.strip() for m in members_text.split(";") if m.strip() and m.strip() in pdp_base.columns]

                if not members:
                    continue

                group_axis = {
                    "group": group_name,
                    "members": members,
                }

                kind = infer_pdp_axis_kind(pdp_base, members)

                if kind in ("continuous", "multi_numeric_use_first"):
                    feature = members[0]
                    vals = pd.to_numeric(pdp_base[feature], errors="coerce").dropna().to_numpy(float)
                    if vals.size < 2:
                        continue
                    lo = np.nanquantile(vals, PDP_PERCENTILE_RANGE[0])
                    hi = np.nanquantile(vals, PDP_PERCENTILE_RANGE[1])
                    if np.isclose(lo, hi):
                        continue
                    grid_vals = np.linspace(lo, hi, PDP_GRID_POINTS)
                    y_means = []
                    for v in grid_vals:
                        X_tmp = pdp_base.copy()
                        X_tmp[feature] = v
                        y_means.append(np.mean(predict_pdp(X_tmp)))

                    out_base = pdp_dir / f"rank{rank_i+1}_{sanitize_filename(group_name)}_{sanitize_filename(feature)}_PDP"
                    save_pdp_plot_numeric(
                        grid_vals,
                        np.asarray(y_means),
                        group_name,
                        f"Mean predicted {TARGET_COL}",
                        f"({rank_i+1}) {group_name}",
                        out_base,
                    )
                    pdp_records.append({
                        "rank": int(rank_i + 1),
                        "group": group_name,
                        "feature": feature,
                        "type": "continuous_1D",
                        "output_basepath": str(out_base),
                    })

                elif kind in ("categorical_group", "binary_single"):
                    if kind == "binary_single":
                        categories = members
                        values = [0.0, 1.0]
                        labels = [f"{group_name}=0", f"{group_name}=1"]
                    else:
                        categories = members[:MAIN_2D_PDP_MAX_CATEGORIES]
                        values = list(range(len(categories)))
                        labels = [display_name_by_source.get(m, m) for m in categories]

                    y_means = []
                    for pos, _ in enumerate(values):
                        X_tmp = pdp_base.copy()
                        if kind == "binary_single":
                            X_tmp[categories[0]] = float(pos)
                        else:
                            for m in categories:
                                X_tmp[m] = 0.0
                            X_tmp[categories[pos]] = 1.0
                        y_means.append(np.mean(predict_pdp(X_tmp)))

                    out_base = pdp_dir / f"rank{rank_i+1}_{sanitize_filename(group_name)}_category_response"
                    save_pdp_plot_categorical(
                        labels,
                        np.asarray(y_means),
                        group_name,
                        f"Mean predicted {TARGET_COL}",
                        f"({rank_i+1}) {group_name}",
                        out_base,
                    )
                    pdp_records.append({
                        "rank": int(rank_i + 1),
                        "group": group_name,
                        "features": "; ".join(members),
                        "type": "categorical_1D",
                        "output_basepath": str(out_base),
                    })

            pdp_summary_csv = pdp_dir / f"{sanitize_filename(TARGET_COL)}_{MODEL_NAME}_PDP_top3_grouped_summary.csv"
            pd.DataFrame(pdp_records).to_csv(pdp_summary_csv, index=False)

            main_2d_outputs: Dict[str, Any] = {"enabled": False}
            if RUN_MAIN_2D_PDP:
                main_2d_dir = save_dir / "PDP_main_2D_design_windows"
                main_2d_dir.mkdir(parents=True, exist_ok=True)

                main_2d_records = []
                main_2d_panels = []

                for pair_i, pair in enumerate(MAIN_2D_PDP_PAIRS, start=1):
                    x_group = pair["x_group"]
                    y_group = pair["y_group"]
                    pair_label = pair.get("label", f"{x_group} × {y_group}")
                    panel_label = f"({chr(96 + pair_i)}) {pair_label}"

                    panel = make_2d_pdp_panel(
                        x_group=x_group,
                        y_group=y_group,
                        panel_label=panel_label,
                        feature_names=feature_names,
                        X_base=pdp_base,
                        predict_fn=predict_pdp,
                        grid_points=MAIN_2D_PDP_GRID_POINTS,
                        percentile_range=MAIN_2D_PDP_PERCENTILE_RANGE,
                        max_categories=MAIN_2D_PDP_MAX_CATEGORIES,
                    )

                    if panel is None:
                        main_2d_records.append({
                            "rank": pair_i,
                            "x_group": x_group,
                            "y_group": y_group,
                            "label": pair_label,
                            "status": "skipped",
                            "output_basepath": None,
                        })
                        continue

                    out_base = main_2d_dir / f"panel{pair_i}_{sanitize_filename(TARGET_COL)}_{sanitize_filename(pair_label)}_2D_PDP"
                    save_2d_pdp_panel(panel, target_col=TARGET_COL, out_base=out_base)
                    main_2d_panels.append(panel)

                    main_2d_records.append({
                        "rank": pair_i,
                        "x_group": x_group,
                        "y_group": y_group,
                        "label": pair_label,
                        "x_axis_kind": panel["x_axis"]["kind"],
                        "y_axis_kind": panel["y_axis"]["kind"],
                        "x_members": "; ".join(panel["x_axis"]["members"]),
                        "y_members": "; ".join(panel["y_axis"]["members"]),
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
            "model_output": "Capacity prediction from train+validation refit model",
            "grouped": {
                "mean_abs_shap_csv": str(grouped_shap_csv),
                "bar_top_basepath": str(grouped_bar_base),
                "strip_top_basepath": str(grouped_strip_base),
                "plot_file_formats": ["png", "svg", "pdf"],
                "top_n": grouped_top_n,
                "group_definitions": SHAP_GROUP_DEFINITIONS,
                "group_display_map": SHAP_GROUP_DISPLAY_MAP,
                "groups_resolved": grouped_features,
                "importance_definition": "mean over samples of abs(sum(signed SHAP values)) within each group; this avoids over-weighting one-hot groups with many dummy columns",
                "audit_column": "mean_sum_abs_member_shap_audit keeps the older sum(abs(member SHAP)) value for comparison only",
                "strip_definition": "sum of signed SHAP values within each group for each explained sample",
            },
            "pdp_outputs": pdp_outputs,
        }
        print("[SHAP] Done.")

    metrics_out = {
        "input_file": str(input_fp),
        "input_sheet": INPUT_SHEET,
        "target": TARGET_COL,
        "seed": SEED,
        "model": "XGBRegressor",
        "best_params": best_params,
        "augmentation": {"pairwise_fraction": PAIRWISE_FRACTION, "noise_fraction": NOISE_FRACTION},
        "iqr": {
            "train_fit": iqr_info,
            "trainval_refit_fit": iqr_info_trainval,
        },
        "rows": {
            "n_total_after_target_filter": int(len(X_df)),
            "train_eval_rows": int(len(X_train_eval)),
            "train_fit_rows_after_iqr_before_aug": int(len(X_train) - (0 if X_aug is None else len(X_aug))),
            "train_fit_rows_after_aug": int(len(X_train)),
            "trainval_fit_rows_after_iqr_before_aug": int(len(X_trainval) - (0 if X_aug_tv is None else len(X_aug_tv))),
            "trainval_fit_rows_after_aug": int(len(X_trainval)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        "metrics": {
            "train": m_train,
            "val": m_val,
            "test": m_test,
            "test_refit_train_plus_val": m_test_refit,
        },
        "prediction_files": {"xlsx": str(pred_xlsx), "csv": str(pred_csv)},
        "model_files": {
            "train_model": str(save_dir / "xgb_Capacity.pkl"),
            "trainval_refit_model": str(save_dir / "xgb_Capacity_refit_train_plus_val.pkl"),
        },
        "axis_settings_used": axis_settings,
        "plot_style_used": {
            "scatter_size": PLOT_STYLE["scatter_s"],
            "scatter_alpha": PLOT_STYLE["scatter_alpha"],
            "scatter_color": PLOT_STYLE["scatter_color"],
            "ann_bbox": PLOT_STYLE["ann_bbox"],
        },
        "plot_files_basepath": plot_files,
        "shap_outputs": shap_outputs,
        "pdp_plot_style": PDP_PLOT_STYLE,
        "ssa_transform": ssa_transform_info,
        "features_used": feature_names,
        "dropped_input_columns": drop_cols,
        "all_nan_feature_columns_dropped": all_nan_cols,
    }
    (save_dir / "xgb_Capacity_metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    print("\n=== XGB Capacity Results ===")
    print(f"Train: R2={m_train['r2']:.3f}, RMSE={m_train['rmse']:.3f}")
    print(f"Val:   R2={m_val['r2']:.3f}, RMSE={m_val['rmse']:.3f}")
    print(f"Test:  R2={m_test['r2']:.3f}, RMSE={m_test['rmse']:.3f}")
    print(f"Test_Refit: R2={m_test_refit['r2']:.3f}, RMSE={m_test_refit['rmse']:.3f}")
    print(f"IQR removed rows from train fit: {iqr_info.get('removed_count', 0)}")
    print(f"IQR removed rows from train+val refit fit: {iqr_info_trainval.get('removed_count', 0)}")
    print(f"\nSaved artifacts to: {save_dir}")
    print(f"Predictions Excel: {pred_xlsx}")
    print(f"Model train: {save_dir / 'xgb_Capacity.pkl'}")
    print(f"Model train+val refit: {save_dir / 'xgb_Capacity_refit_train_plus_val.pkl'}")
    for k, base in plot_files.items():
        print(f"plot_{k}: {base}.png/.svg/.pdf")


if __name__ == "__main__":
    main()

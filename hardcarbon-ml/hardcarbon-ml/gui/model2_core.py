"""
Core prediction logic for Model 02 (ICE, Capacity, Slope, HDR).

No GUI dependency here on purpose (importable/testable without tkinter).

WHY THIS IS A SEPARATE MODULE FROM hard_carbon_core.py
----------------------------------------------------------
Model 02's four training scripts are architecturally different from Model 01's
seven scripts in every way that matters for a prediction GUI:

- No RobustScaler, no Box-Cox anywhere. CatBoost/XGB/GBDT/Stacking all fit on
  raw feature values and predict the target in raw units directly. There is
  nothing to inverse-transform.
- Categoricals are ALREADY one-hot columns in the source Excel (e.g. a "PVDF"
  column that is 1 for PVDF-binder rows and 0 otherwise), not a single
  integer-coded column. There is no Table S3 - the "encoding" is just
  "which of these columns is 1".
- Every script saves a plain `feature_names.pkl` (the exact column order the
  model was fit on) plus `model.pkl` (or a target-specific filename) - no
  structured `artifacts.pkl` with `feature_order`/`boxcox_params` like Model 01.

WHAT THIS GUI CANNOT KNOW WITHOUT YOUR REAL EXCEL FILES
----------------------------------------------------------
Your scripts identify one-hot groups (Binder type, Solvent type, Salt type,
Separator type, Electrolyte type) by FUZZY-MATCHING column names against
token lists (`SHAP_GROUP_DEFINITIONS`, ported verbatim below) - the actual
literal column names depend on your Excel headers, which I have not seen.
So this module does NOT hardcode a fixed input form the way Model 01's did
against Table S3. Instead, at load time it reads each model's own saved
`feature_names.pkl` and runs your own grouping logic against it, so the form
always matches whatever your real files actually contain.

"Electrolyte type" mixes two different kinds of column in your token list -
broad-class flags (Aqueous/Organic/Carbonate/Ether/Ionic liquid, likely
one-hot) and solvent-component fractions (EC/DMC/DEC/EMC/FEC/PC/..., which
can co-occur in a blend and are NOT mutually exclusive). Since I can't tell
these apart without your data, every member of that group is exposed as its
own numeric field rather than forced into a single dropdown - see
ELECTROLYTE_TYPE_GROUP_KEY below.
"""

import os
import re
import glob
import math

import joblib
import numpy as np
import pandas as pd


# =====================================================================
# MODEL CONFIG - one entry per target. Edit paths/filenames here if your
# artifact folders move. `model_filename_candidates` is tried in order;
# if none exist, the loader falls back to scanning save_dir for a
# plausible .pkl (see find_model_file()).
# =====================================================================
# Model folders are written by the training scripts in model_02/ (see README).
_OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "outputs", "model_02")
MODEL_CONFIGS = {
    "ICE": {
        "save_dir": os.path.join(_OUTPUTS, "ice_catboost"),
        "model_filename_candidates": [
            "catboost_ICE_refit_train_plus_val.pkl",
            "catboost_ICE.pkl",
        ],
        "display": "ICE (%)",
    },
    "Capacity": {
        "save_dir": os.path.join(_OUTPUTS, "capacity_xgb"),
        "model_filename_candidates": [
            "xgb_Capacity_refit_train_plus_val.pkl",
            "xgb_Capacity.pkl",
        ],
        "display": "Initial Capacity (mAh/g)",
    },
    "Slope": {
        "save_dir": os.path.join(_OUTPUTS, "slope_gbdt"),
        "model_filename_candidates": [
            "gbdt_slop_best.pkl",
            "gbdt_slope_best.pkl",
            "gbdt_slop_frac_best.pkl",
            "gbdt_slop_fraction_best.pkl",
        ],
        "display": "Slope Capacity (mAh/g)",
    },
    "HDR": {
        "save_dir": os.path.join(_OUTPUTS, "hdr_stacking"),
        "model_filename_candidates": [
            "stacking_hdr_capacity_logcurrent.pkl",
            "stacking_hdr_capacity_rawcurrent.pkl",
            "stacking_hdr_capacity_bothcurrent.pkl",
        ],
        "display": "HDR Capacity (mAh/g) @ 0.5C/1C/2C",
    },
}
TARGET_ORDER = ["ICE", "Capacity", "Slope", "HDR"]

FEATURE_NAMES_FILENAME = "feature_names.pkl"

# Derived-current column names your Slope/HDR scripts create internally.
CURRENT_MA_COL = "Current (mA)"
CURRENT_LOG_COL = "log10 current density"

# =====================================================================
# Grouping logic - ported verbatim from your training scripts (identical
# across all 4) so the GUI clusters columns exactly the way your own SHAP
# grouping does.
# =====================================================================
SHAP_GROUP_DEFINITIONS = {
    "Active material ratio": ["Active material ratio", "Active_Material_Ratio", "ActiveMaterialRatio", "Active ratio", "AM ratio"],
    "Carbon ratio": ["Carbon ratio", "Carbon_Ratio", "CarbonRatio", "Conductive carbon ratio"],
    "Binder ratio": ["Binder ratio", "Binder_Ratio", "BinderRatio"],
    "Separator type": ["GF", "PE", "PP"],
    "Binder type": [
        "CMC", "CMC+SBR", "CMC_SBR", "CMC SBR", "No Binder", "No_Binder",
        "PAA", "PANA", "PVDF", "SBR", "Sodium alginate", "Sodium_Alginate",
        "PTFE", "PVF",
    ],
    "Solvent type": ["NMP", "No Solvent", "No_Solvent", "Water", "Ethanol"],
    "Salt type": ["NaClO4", "NaCLO4", "NaOTf", "NaOTF", "NaPF6", "NaTFSI", "NaBF4"],
    "Salt concentration": ["Salt concentration", "Salt_Concentration", "SaltConcentration", "Salt conc", "Salt_conc"],
    "Additive concentration": [
        "Additive concentration", "Additive_Concentration", "AdditiveConcentration",
        "Additive conc", "Additive_conc", "Additive", "Additive%", "Additive_percent",
        "Additive percentage", "Additive wt%", "Additive_wt", "Additive amount",
    ],
    "Electrolyte type": [
        "Electrolyte type", "Electrolyte_Type", "ElectrolyteType",
        "Aqueous", "Organic", "Carbonate", "Ether", "Ionic liquid", "Ionic_Liquid",
        "PC", "PC fraction", "PC_Fraction", "PCFraction",
        "EC", "EC fraction", "EC_Fraction", "ECFraction",
        "DMC", "DMC fraction", "DMC_Fraction", "DMCFraction",
        "DEC", "DEC fraction", "DEC_Fraction", "DECFraction",
        "EMC", "EMC fraction", "EMC_Fraction", "EMCFraction",
        "FEC", "FEC fraction", "FEC_Fraction", "FECFraction",
        "DME", "DME fraction", "DME_Fraction", "DMEFraction",
        "DOL", "DOL fraction", "DOL_Fraction", "DOLFraction",
        "TETRAGLYME", "TETRAGLYME fraction", "TETRAGLYME_Fraction", "TETRAGLYMEFraction",
        "DIGLYME", "DIGLYME fraction", "DIGLYME_Fraction", "DIGLYMEFraction",
        "MONOGLYME", "MONOGLYME fraction", "MONOGLYME_Fraction", "MONOGLYMEFraction",
        "TRIGLYME", "TRIGLYME fraction", "TRIGLYME_Fraction", "TRIGLYMEFraction",
    ],
    "La": ["La", "L_a", "La_nm", "La (nm)", "Lateral crystallite size", "Lateral_Crystallite_Size"],
    "D-S": ["D-S", "D_S", "D spacing", "D-spacing", "D_Spacing", "D Spacing", "d002", "d_002"],
    "Current density": [
        "Current density", "Current_Density", "CurrentDensity", "Current", "Curr",
        "Current (A)", "Current (mA)", "mA/g", "log10 current density",
    ],
    "Lc": ["Lc", "L_c", "Lc_nm", "Lc (nm)", "Crystallite height", "Crystallite_Height"],
    "Surface area": ["SSA", "Surface Area", "Surface area", "Specific surface area", "BET", "BET surface area", "BET_Surface_Area"],
    "Pore volume": ["PV", "Pore Volume", "Pore volume", "Pore_Volume"],
    "IG/ID": ["IG/ID", "Ig/Id", "ID/IG", "Id/Ig", "Raman"],
    "C": ["C", "Carbon%", "Carbon percent", "C%"],
}

SHAP_GROUP_DISPLAY_MAP = {
    "Solvent type": "Solvent type",
    "Electrolyte type": "Electrolyte type",
    "Binder type": "Binder type",
    "Salt type": "Salt type",
    "Separator type": "Separator type",
    "D-S": "D-Spacing",
    "Current density": "Current",
    "Carbon ratio": "Carbon ratio",
    "La": "La",
    "Lc": "Lc",
    "Surface area": "Surface area (SSA)",
    "Pore volume": "Pore volume (PV)",
    "Binder ratio": "Binder ratio",
    "Active material ratio": "Active material ratio",
    "Salt concentration": "Salt concentration",
    "Additive concentration": "Additive concentration",
    "IG/ID": "Raman IG/ID",
    "C": "Carbon% (elemental)",
}

# Groups that are genuinely mutually exclusive in a real cell (you use ONE
# binder, ONE solvent, ONE salt, ONE separator) - these become a single
# dropdown when a model's feature set has more than one member. Every other
# group (including Electrolyte type - see module docstring) is exposed as
# individual numeric fields.
# Fixed order so validation messages ("Please choose a ...") are identical on every launch;
# a bare set would iterate in a hash-seed-dependent order.
ONE_HOT_DROPDOWN_ORDER = ("Binder type", "Solvent type", "Salt type", "Separator type")
ONE_HOT_DROPDOWN_GROUPS = set(ONE_HOT_DROPDOWN_ORDER)
ELECTROLYTE_TYPE_GROUP_KEY = "Electrolyte type"

# Active material + conductive carbon + binder ratios make up the electrode
# composition and must sum to 100% (binder can legitimately be 0 for a
# binder-free electrode - see BINDER_TYPE_GROUP_KEY / SOLVENT_TYPE_GROUP_KEY
# freeze behaviour in the GUI, which locks Binder/Solvent type to "No
# Binder"/"No Solvent" whenever Binder ratio is 0).
ELECTRODE_COMPOSITION_GROUP_KEYS = ["Active material ratio", "Carbon ratio", "Binder ratio"]
BINDER_RATIO_GROUP_KEY = "Binder ratio"
BINDER_TYPE_GROUP_KEY = "Binder type"
SOLVENT_TYPE_GROUP_KEY = "Solvent type"
NO_BINDER_LABEL_CANDIDATES = ("No Binder", "No_Binder")
NO_SOLVENT_LABEL_CANDIDATES = ("No Solvent", "No_Solvent")

# Electrolyte solvent-component fractions must sum to 1 (100% of the blend).
ELECTROLYTE_FRACTION_TOTAL = 1.0


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

    if feature_key in {f"{token_key}fraction", f"{token_key}percent", f"{token_key}percentage"}:
        return True
    if len(token_key) >= 4 and token_key in feature_key:
        return True
    return False


def build_shap_groups(feature_names, display_name_by_source=None):
    """Cluster feature_names into named groups, exactly mirroring the
    training scripts' own SHAP grouping. Returns {group_key: [members]}
    using SHAP_GROUP_DEFINITIONS keys (not the cosmetic display names) so
    ONE_HOT_DROPDOWN_GROUPS / ELECTROLYTE_TYPE_GROUP_KEY can match directly.
    """
    display_name_by_source = display_name_by_source or {}
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
            grouped[group_name] = members
            assigned.update(members)

    for feature_name in feature_names:
        if feature_name not in assigned:
            grouped[display_name_by_source.get(feature_name, feature_name)] = [feature_name]

    return grouped


# =====================================================================
# ARTIFACT LOADING
# =====================================================================
def find_model_file(save_dir, candidates):
    """Try each candidate filename in order; fall back to scanning save_dir
    for a plausible model .pkl (anything that isn't feature_names.pkl,
    preferring names containing 'refit', 'best', or 'stacking')."""
    for name in candidates:
        fp = os.path.join(save_dir, name)
        if os.path.exists(fp):
            return fp

    pattern = os.path.join(save_dir, "*.pkl")
    found = [f for f in glob.glob(pattern) if os.path.basename(f) != FEATURE_NAMES_FILENAME]
    if not found:
        return None

    def score(fp):
        name = os.path.basename(fp).lower()
        return (
            0 if "refit" in name or "best" in name or "stacking" in name else 1,
            name,
        )

    found.sort(key=score)
    return found[0]


def load_model_bundle(target_key, configs=None):
    configs = configs or MODEL_CONFIGS
    cfg = configs[target_key]
    save_dir = cfg["save_dir"]

    feature_names_fp = os.path.join(save_dir, FEATURE_NAMES_FILENAME)
    if not os.path.exists(feature_names_fp):
        raise FileNotFoundError(f"{FEATURE_NAMES_FILENAME} not found in:\n{save_dir}")
    feature_names = list(joblib.load(feature_names_fp))

    model_fp = find_model_file(save_dir, cfg["model_filename_candidates"])
    if model_fp is None:
        raise FileNotFoundError(
            f"No model .pkl found in:\n{save_dir}\n"
            f"Tried: {cfg['model_filename_candidates']} and a directory scan."
        )
    model = joblib.load(model_fp)

    grouped = build_shap_groups(feature_names)

    return {
        "target_key": target_key,
        "model": model,
        "model_file": model_fp,
        "feature_names": feature_names,
        "grouped_features": grouped,
        # Data-driven, not a hardcoded per-target guess: True whenever this
        # specific model's OWN feature_names contains anything that fuzzy-
        # matches the "Current density" tokens - covers ICE/Capacity too if
        # their real data has a current column your training scripts never
        # explicitly transformed (see CURRENT (A) section below).
        "needs_current": bool(grouped.get("Current density")),
    }


class BundleCache:
    def __init__(self, configs=None):
        self.configs = configs or MODEL_CONFIGS
        self._cache = {}

    def get(self, target_key):
        if target_key not in self._cache:
            self._cache[target_key] = load_model_bundle(target_key, self.configs)
        return self._cache[target_key]

    def clear(self):
        self._cache.clear()

    def get_all(self, target_order=None):
        """Load every target, collecting errors instead of raising, so the
        GUI can still build a form from whichever models DID load."""
        target_order = target_order or TARGET_ORDER
        bundles, errors = {}, {}
        for key in target_order:
            try:
                bundles[key] = self.get(key)
            except Exception as exc:
                errors[key] = str(exc)
        return bundles, errors


# =====================================================================
# C-RATE -> CURRENT (A). Per Niroshan: the "Current (A)" value entered in
# the GUI IS the 1C reference for that cell (e.g. entering 0.3 A means
# "0.3 A is 1C for this cell"); other rates are simple scalar multiples of
# it - 2C = 2 x (entered value), 0.5C = 0.5 x (entered value), etc. There is
# no separate theoretical-capacity calculation involved.
#
# ICE/Initial Capacity are measured at a slow 0.05C formation rate. HDR is
# explicitly evaluated at three rates (0.5C, 1C, 2C) and reported as all
# three, not a single value - see TARGET_C_RATE_MULTIPLIERS below, where
# HDR maps to a list instead of a single float.
#
# UNCONFIRMED: Slope's rate convention hasn't been stated. Defaulted to 1.0
# (use the entered value unscaled, i.e. "at 1C") until told otherwise -
# this is a flagged assumption, not a confirmed fact like the other three.
# =====================================================================
TARGET_C_RATE_MULTIPLIERS = {
    "ICE": 0.05,
    "Capacity": 0.05,
    "Slope": 1.0,       # UNCONFIRMED - assumption, ask before trusting this
    "HDR": [0.5, 1.0, 2.0],
}


def compute_target_current_A(target_key, current_1C_A):
    """Scales the user's entered 1C reference current for a specific
    target. Returns a single float for single-rate targets, or a list of
    floats for multi-rate targets (currently just HDR)."""
    current_1C_A = float(current_1C_A)
    mult = TARGET_C_RATE_MULTIPLIERS.get(target_key, 1.0)
    if isinstance(mult, list):
        return [current_1C_A * m for m in mult]
    return current_1C_A * mult


# =====================================================================
# CURRENT (A -> mA -> log10) - now fully data-driven per model instead of
# a hardcoded "only Slope/HDR use current" assumption. Your Slope/HDR
# scripts explicitly convert A -> mA -> log10(mA) and drop the raw column;
# ICE/Capacity's scripts never touch current at all, which does NOT mean
# their Excel lacks a current column - it may simply flow through
# untransformed as an ordinary numeric feature under some other name. This
# function checks each model's OWN "Current density" group (detected via
# the same token-matching used for everything else) and supplies whatever
# representation(s) that specific model actually has.
# =====================================================================
def resolve_current_inputs(bundle, current_A):
    members = bundle["grouped_features"].get("Current density", [])
    if not members:
        return {}

    current_A = float(current_A)
    if current_A <= 0:
        raise ValueError("Current (A) must be positive.")
    current_mA = current_A * 1000.0

    out = {}
    for m in members:
        if m == CURRENT_MA_COL:
            out[m] = current_mA
        elif m == CURRENT_LOG_COL:
            out[m] = math.log10(current_mA)
        else:
            # An untransformed current-like column under some other literal
            # name (this is the ICE/Capacity case if their real data has
            # one). Best guess: supply raw Amps directly, since that's the
            # natural unit for the shared input - flagged here so you can
            # verify it against what that specific column actually expects.
            out[m] = current_A
    return out


# =====================================================================
# FEATURE VECTOR CONSTRUCTION
# =====================================================================
def build_feature_row(bundle, raw_inputs):
    """raw_inputs: dict of {feature_name: float value}, already fully
    resolved (one-hot groups expanded to 0/1 per member, current already
    derived to Current (mA) / log10 current density as needed). Missing
    keys raise rather than silently defaulting to 0."""
    feature_names = bundle["feature_names"]
    row = {}
    for name in feature_names:
        if name not in raw_inputs:
            raise ValueError(f"Missing input '{name}' for target '{bundle['target_key']}'.")
        row[name] = float(raw_inputs[name])
    X = pd.DataFrame([row])[feature_names].astype(float)
    return X


def predict_one(cache: BundleCache, target_key, raw_inputs):
    bundle = cache.get(target_key)
    X = build_feature_row(bundle, raw_inputs)
    y_pred = bundle["model"].predict(X.values)
    return float(np.asarray(y_pred).reshape(-1)[0])

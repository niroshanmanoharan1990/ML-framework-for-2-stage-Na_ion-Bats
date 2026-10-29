"""
Core prediction logic for the 7 hard-carbon property models.
No GUI dependency here on purpose, so this half can be imported and unit
tested (or reused from a notebook / batch script) independently of tkinter.
"""

import os
import math

import joblib
import numpy as np
import pandas as pd

# Needed so joblib can unpickle the saved estimators.
from sklearn.preprocessing import RobustScaler  # noqa: F401
from sklearn.ensemble import (  # noqa: F401
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
try:
    from xgboost import XGBRegressor  # noqa: F401
except ImportError:
    pass  # only needed to unpickle the Carbon model


# Model folders are written by the training scripts in model_01/ (see README).
_OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "outputs", "model_01")
MODEL_SAVE_DIRS = {
    "Carbon":    os.path.join(_OUTPUTS, "carbon_xgb"),
    "D-Spacing": os.path.join(_OUTPUTS, "dspacing_histgb"),
    "La":        os.path.join(_OUTPUTS, "la_gbdt"),
    "Lc":        os.path.join(_OUTPUTS, "lc_rf"),
    "PV":        os.path.join(_OUTPUTS, "pv_gbdt"),
    "SSA":       os.path.join(_OUTPUTS, "ssa_gbdt"),
    "IG/ID":     os.path.join(_OUTPUTS, "raman_histgb"),
}

MODEL_ORDER = ["Carbon", "D-Spacing", "La", "Lc", "PV", "SSA", "IG/ID"]

CATEGORICAL_COLS = ["Prewash", "Postwash", "Type", "Method_01", "Activator_01", "Method_02", "Activator_02"]

# Domain-grouped categorical encoding from Table S3 of your supplementary
# info ("Domain based grouping and numerical encoding of categorical
# synthesis descriptors used in Model 01"). List order = code (index 0 -> 0,
# index 1 -> 1, ...). This is the encoding baked into the training Excel for
# the six "raw numeric code" models (Carbon, D-Spacing, La, Lc, PV, Raman).
#
# SSA is the odd one out: its script factorizes these same domain-grouped
# labels itself at train time (pd.factorize(..., sort=True), which happens
# to also be alphabetical - matching this table in the common case), so SSA
# uses its OWN saved cat_maps rather than this table. See
# resolve_categorical_code() below for exactly how the two are reconciled.
CATEGORICAL_ENCODING = {
    "Prewash": [
        "Conc Acid wash", "Mild Acid wash", "Mild Alkali wash",
        "Moderate Acid wash", "No wash", "Organic wash", "Water wash",
    ],
    "Postwash": [
        "Acid wash", "Mild Acid wash", "Mild Alkali wash", "Moderate Acid wash",
        "Moderate Alkali wash", "No wash", "Organic wash", "Water wash",
    ],
    "Type": ["One step", "Two step"],
    "Method_01": ["HTC", "Pyrolysis"],
    "Method_02": ["No", "Pyrolysis"],
    "Activator_01": [
        "Mild Acid", "Mild Base", "Moderate Acid", "Moderate Base",
        "No", "Strong Acid", "Strong Base",
    ],
    "Activator_02": [
        "Mild Acid + Base Mix", "Mild Base", "Moderate Acid", "Moderate Base",
        "Nitrogen Precursor", "No", "Oxidizing Agent", "Reducing Agent", "Strong Base",
    ],
}

# Same lists, exposed under the name the GUI uses to populate dropdowns
# (already in code order, i.e. index 0 first).
CATEGORICAL_LABEL_ORDER = CATEGORICAL_ENCODING


def resolve_categorical_code(bundle, col, label):
    """Map a human-readable label to the numeric code a given model expects.

    CONFIRMED against your actual SSA_imputated_corrected.xlsx: SSA's Excel
    stores categoricals as the SAME Table S3 integer codes as the other six
    models (e.g. Activator_01 column contains 0/1/4/5, not text). SSA's
    training script factorizes those already-numeric values
    (pd.factorize(sort=True)), which just reproduces the same integers, so
    SSA's saved cat_maps is a list like [0, 1, 4, 5] - not label strings.

    So for every model (SSA included) the label is first translated to its
    Table S3 code. For SSA specifically, that code is then checked against
    what SSA's own training data actually contained (cat_maps) - if SSA's
    training set never saw that code for this column, it's genuinely an
    unseen category for that model and is encoded as -1, matching the
    documented convention (Table S4: "unseen categories assigned -1").
    A string-match fallback is kept in case some other dataset really does
    store text labels in cat_maps.
    """
    if col not in CATEGORICAL_ENCODING or label not in CATEGORICAL_ENCODING[col]:
        raise ValueError(f"Unknown label '{label!r}' for column '{col}'.")
    code = CATEGORICAL_ENCODING[col].index(label)

    cat_maps = bundle.get("cat_maps")
    if cat_maps is not None and col in cat_maps:
        raw_values = cat_maps[col]
        numeric_seen = set()
        for v in raw_values:
            try:
                numeric_seen.add(int(v))
            except (TypeError, ValueError):
                pass
        if numeric_seen:
            if code in numeric_seen:
                return code
            print(f"[{bundle['model_key']}] '{label}' (Table S3 code {code}) was "
                  f"not among the values this model's training data actually "
                  f"contained for '{col}' - encoding as unseen (-1).")
            return -1
        # No numeric entries at all - cat_maps must hold text labels instead.
        str_labels = [str(v) for v in raw_values if v is not None]
        if label in str_labels:
            return str_labels.index(label)
        print(f"[{bundle['model_key']}] '{label}' was not present in this "
              f"model's own training categories for '{col}' - encoding as "
              f"unseen (-1).")
        return -1

    return code


# =====================================================================
# BOX-COX FORWARD / INVERSE with a KNOWN, already-fit lambda
# (matches boxcox_forward / inverse_boxcox_feature in every training script)
# =====================================================================
def boxcox_forward_known_lambda(raw_value, lam, shift):
    x = float(raw_value) + float(shift or 0.0)
    if x <= 0:
        x = 1e-9
    if lam is None or abs(lam) < 1e-12:
        return math.log(x)
    return (x ** lam - 1.0) / lam


def boxcox_inverse_known_lambda(bc_value, lam, shift=0.0, clip=True):
    if lam is None or abs(lam) < 1e-12:
        return math.exp(bc_value) - shift
    base = lam * bc_value + 1.0
    if clip:
        base = max(base, 1e-12)
    return (base ** (1.0 / lam)) - shift


# =====================================================================
# ARTIFACT LOADING - fully data-driven from each model's artifacts.pkl
# =====================================================================
def _safe_load(path):
    return joblib.load(path) if os.path.exists(path) else None


def load_model_bundle(model_key, save_dirs=None):
    save_dirs = save_dirs or MODEL_SAVE_DIRS
    save_dir = save_dirs[model_key]
    artifacts_fp = os.path.join(save_dir, "artifacts.pkl")
    if not os.path.exists(artifacts_fp):
        raise FileNotFoundError(f"artifacts.pkl not found in:\n{save_dir}")
    artifacts = joblib.load(artifacts_fp)

    model = _safe_load(os.path.join(save_dir, "model.pkl"))
    scaler_X = _safe_load(os.path.join(save_dir, "scaler_X.pkl"))
    scaler_y = _safe_load(os.path.join(save_dir, "scaler_y.pkl"))
    if model is None:
        raise FileNotFoundError(f"model.pkl not found in:\n{save_dir}")
    if scaler_y is None:
        raise FileNotFoundError(f"scaler_y.pkl not found in:\n{save_dir}")

    feature_order = artifacts.get("feature_order", {})
    continuous_actual = list(feature_order.get("continuous", []))
    boxcox_actual = list(feature_order.get("boxcox_transformed", []))
    categorical_actual = list(feature_order.get("categorical", []))
    boxcox_params_X = artifacts.get("boxcox_params_X", {})

    boxcox_params_y = artifacts.get("boxcox_params_y")
    if boxcox_params_y is None:
        boxcox_params_y = _safe_load(os.path.join(save_dir, "boxcox_params_y.pkl"))

    cat_maps = artifacts.get("cat_maps")  # only populated for SSA

    return {
        "model_key": model_key,
        "model": model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "continuous_actual": continuous_actual,
        "boxcox_actual": boxcox_actual,
        "categorical_actual": categorical_actual,
        "boxcox_params_X": boxcox_params_X,
        "boxcox_params_y": boxcox_params_y,
        "cat_maps": cat_maps,
    }


class BundleCache:
    def __init__(self, save_dirs=None):
        self.save_dirs = save_dirs or MODEL_SAVE_DIRS
        self._cache = {}

    def get(self, model_key):
        if model_key not in self._cache:
            self._cache[model_key] = load_model_bundle(model_key, self.save_dirs)
        return self._cache[model_key]

    def clear(self):
        self._cache.clear()


# =====================================================================
# FEATURE VECTOR CONSTRUCTION (mirrors gather_inputs() in the training scripts)
# =====================================================================
def build_feature_row(bundle, raw_inputs):
    """
    raw_inputs: dict of {column_name: value}.
    Continuous / temperature columns -> float.
    Categorical columns -> the human-readable label string (as shown in
    the shared dropdowns); resolved to each model's own numeric code via
    resolve_categorical_code().
    """
    cont = bundle["continuous_actual"]
    boxc = bundle["boxcox_actual"]
    cat = bundle["categorical_actual"]

    row = {}

    if cont:
        cont_vals = []
        for c in cont:
            v = raw_inputs.get(c)
            if v is None:
                raise ValueError(f"Missing continuous input '{c}'.")
            cont_vals.append(float(v))
        cont_vals = [cont_vals]
        if bundle["scaler_X"] is not None:
            cont_scaled = bundle["scaler_X"].transform(pd.DataFrame(cont_vals, columns=cont))[0]
        else:
            cont_scaled = cont_vals[0]
        for c, v in zip(cont, cont_scaled):
            row[c] = v

    for bc_col in boxc:
        base_col = bc_col[:-len("_boxcox")] if bc_col.endswith("_boxcox") else bc_col
        params = bundle["boxcox_params_X"].get(base_col, {"lam": None, "shift": 0.0})
        raw_val = raw_inputs.get(base_col)
        if raw_val is None:
            raise ValueError(f"Missing input '{base_col}' (needed for {bc_col}).")
        row[bc_col] = boxcox_forward_known_lambda(float(raw_val), params.get("lam"), params.get("shift", 0.0))

    for c in cat:
        label = raw_inputs.get(c)
        if label is None or label == "":
            raise ValueError(f"Missing categorical input '{c}'.")
        row[c] = resolve_categorical_code(bundle, c, label)

    ordered_cols = cont + boxc + cat
    if not ordered_cols:
        raise ValueError("Model has no usable features in its feature_order - artifacts.pkl looks malformed.")
    X = pd.DataFrame([{col: row.get(col, 0.0) for col in ordered_cols}])[ordered_cols].astype(float)
    return X


# =====================================================================
# TARGET INVERSE TRANSFORM
#    - Carbon: custom target_transform_carbon = log(y)^2  =>  inverse below
#    - SSA: Box-Cox fit on train-only y (artifacts["boxcox_params_y"])
#    - everything else: plain RobustScaler round trip
# =====================================================================
def inverse_target(bundle, y_scaled_pred):
    y_unscaled = bundle["scaler_y"].inverse_transform(np.array(y_scaled_pred).reshape(-1, 1))[0, 0]

    if bundle["model_key"] == "Carbon":
        # target_inverse_carbon(y) = exp(sqrt(max(y, 0)))
        return float(np.exp(np.sqrt(max(y_unscaled, 0.0))))

    if bundle["boxcox_params_y"] is not None:
        lam = bundle["boxcox_params_y"].get("lam")
        y_raw = boxcox_inverse_known_lambda(y_unscaled, lam, shift=0.0, clip=True)
        return float(max(y_raw, 0.0))

    return float(y_unscaled)


def predict_one(cache: BundleCache, model_key, raw_inputs):
    bundle = cache.get(model_key)
    X = build_feature_row(bundle, raw_inputs)
    y_scaled_pred = bundle["model"].predict(X.values)
    return inverse_target(bundle, y_scaled_pred)

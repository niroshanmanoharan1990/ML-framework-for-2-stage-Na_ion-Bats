"""
Synthetic artifact folders that match the structure of your real saved models,
so the GUI can be exercised end to end without your trained files.

Model 01: artifacts.pkl + model.pkl + scaler_X.pkl + scaler_y.pkl per target
Model 02: feature_names.pkl + one model .pkl per target
"""

import os
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler

import hard_carbon_core as core01
import model2_core as core02

# ---------------------------------------------------------------- Model 01
CONT_ALL = ["Cellulose%", "Hemi-cellulose%", "Lignin%", "Volatile Matter%", "Ash%", "Fixed Carbon%",
            "Carbon%", "Oxygen%", "Hydrogen%", "Time_01", "Heating rate_01", "Time_02", "Heating rate_02"]
CONT_CARBON = [c for c in CONT_ALL if c not in ("Time_02", "Heating rate_02")]  # Carbon model: 11 features
BOXCOX = ["Temperature_01_boxcox", "Temperature_02_boxcox"]
BOXCOX_PARAMS = {"Temperature_01": {"lam": 0.5, "shift": 0.0}, "Temperature_02": {"lam": 0.5, "shift": 1.0}}
CATS = ["Prewash", "Postwash", "Type", "Method_01", "Activator_01", "Method_02", "Activator_02"]
CAT_RANGE = {c: len(core01.CATEGORICAL_ENCODING[c]) for c in CATS}

RAW_RANGE = {
    "Cellulose%": (30, 45), "Hemi-cellulose%": (15, 30), "Lignin%": (15, 30),
    "Volatile Matter%": (60, 80), "Ash%": (0.5, 5), "Fixed Carbon%": (10, 25),
    "Carbon%": (40, 50), "Oxygen%": (35, 50), "Hydrogen%": (5, 7),
    "Time_01": (30, 180), "Heating rate_01": (1, 20), "Time_02": (0, 120), "Heating rate_02": (0, 20),
}

# (low, high) of each Model 01 target in real units
TARGET_RANGE = {
    "Carbon": (60, 95), "D-Spacing": (0.36, 0.42), "La": (2.0, 6.0), "Lc": (0.8, 2.0),
    "PV": (0.02, 0.4), "SSA": (5, 800), "IG/ID": (0.8, 1.4),
}


def build_model01_fixtures(root_dir, seed=0):
    rng = np.random.default_rng(seed)
    save_dirs = {}
    n = 150
    for key in core01.MODEL_ORDER:
        d = os.path.join(root_dir, "m01_" + key.replace("/", "_"))
        os.makedirs(d, exist_ok=True)
        cont = CONT_CARBON if key == "Carbon" else CONT_ALL

        raw = {c: rng.uniform(*RAW_RANGE[c], size=n) for c in cont}
        t1 = rng.uniform(300, 1500, size=n)
        t2 = rng.uniform(0, 1500, size=n)
        cats = {c: rng.integers(0, CAT_RANGE[c], size=n) for c in CATS}

        raw_df = pd.DataFrame(raw)
        scaler_X = RobustScaler().fit(raw_df)
        Xc = scaler_X.transform(raw_df)
        bc1 = np.array([core01.boxcox_forward_known_lambda(v, 0.5, 0.0) for v in t1])
        bc2 = np.array([core01.boxcox_forward_known_lambda(v, 0.5, 1.0) for v in t2])
        X = np.column_stack([Xc, bc1, bc2] + [cats[c] for c in CATS]).astype(float)

        z = (X - X.mean(0)) / (X.std(0) + 1e-9)
        score = z @ rng.normal(size=X.shape[1])
        score = (score - score.min()) / (score.max() - score.min())
        lo, hi = TARGET_RANGE[key]
        y_real = lo + (hi - lo) * score
        y_train = (np.log(y_real) ** 2) if key == "Carbon" else y_real   # Carbon uses the log^2 target transform

        scaler_y = RobustScaler().fit(y_train.reshape(-1, 1))
        y_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
        model = GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=0).fit(X, y_scaled)

        cat_maps = None
        if key == "SSA":   # SSA stores its own training-time categories as bare ints
            cat_maps = {"Prewash": list(range(7)), "Postwash": list(range(8)), "Type": [0, 1],
                        "Method_01": [0, 1], "Activator_01": [0, 1, 4, 5], "Method_02": [0, 1],
                        "Activator_02": [0, 1, 5, 8]}

        artifacts = {
            "feature_order": {"continuous": cont, "boxcox_transformed": BOXCOX, "categorical": CATS},
            "boxcox_params_X": BOXCOX_PARAMS,
            "boxcox_params_y": None,
            "cat_maps": cat_maps,
        }
        joblib.dump(artifacts, os.path.join(d, "artifacts.pkl"))
        joblib.dump(model, os.path.join(d, "model.pkl"))
        joblib.dump(scaler_X, os.path.join(d, "scaler_X.pkl"))
        joblib.dump(scaler_y, os.path.join(d, "scaler_y.pkl"))
        save_dirs[key] = d
    return save_dirs


# ---------------------------------------------------------------- Model 02
DESCRIPTORS = ["La", "D-S", "Lc", "SSA", "PV", "Carbon%", "IG/ID"]
ELECTRODE = ["Active material ratio", "Carbon ratio", "Binder ratio"]
SEPARATOR = ["GF", "PE", "PP"]
BINDER = ["PVDF", "CMC", "SBR", "No Binder"]
SOLVENT = ["NMP", "Water", "No Solvent"]
SALT = ["NaPF6", "NaClO4", "NaTFSI"]
ELECTROLYTE = ["Aqueous", "Organic", "EC fraction", "DMC fraction"]

BASE02 = (DESCRIPTORS + ELECTRODE + SEPARATOR + BINDER + SOLVENT + SALT
          + ["Salt concentration", "Additive concentration"] + ELECTROLYTE)
FEATURES02 = {
    "ICE": BASE02 + ["Current density"],   # raw, untransformed current-like column
    "Capacity": [f for f in BASE02 if f != "Additive concentration"],
    "Slope": [f for f in BASE02 if f not in ("Carbon ratio",) and f not in ELECTROLYTE] + ["log10 current density"],
    "HDR": BASE02 + ["log10 current density"],
}
RANGE02 = {"ICE": (40, 90), "Capacity": (150, 350), "Slope": (50, 150), "HDR": (100, 300)}


def build_model02_fixtures(root_dir, seed=1):
    rng = np.random.default_rng(seed)
    configs = {k: dict(v) for k, v in core02.MODEL_CONFIGS.items()}
    n = 120
    groups = [SEPARATOR, BINDER, SOLVENT, SALT]
    for key, names in FEATURES02.items():
        d = os.path.join(root_dir, "m02_" + key)
        os.makedirs(d, exist_ok=True)
        X = pd.DataFrame(rng.uniform(0, 1, size=(n, len(names))), columns=names)
        for grp in groups:
            if all(m in names for m in grp):
                codes = rng.integers(0, len(grp), size=n)
                for i, m in enumerate(grp):
                    X[m] = (codes == i).astype(float)
        if "log10 current density" in names:
            X["log10 current density"] = np.log10(rng.uniform(0.01, 2.0, size=n) * 1000)
        if "Current density" in names:
            X["Current density"] = rng.uniform(0.005, 0.1, size=n)
        score = X.to_numpy() @ rng.uniform(0.5, 3.0, size=len(names))
        score = (score - score.min()) / (score.max() - score.min())
        lo, hi = RANGE02[key]
        model = GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=0).fit(X.values, lo + (hi - lo) * score)
        joblib.dump(list(names), os.path.join(d, "feature_names.pkl"))
        joblib.dump(model, os.path.join(d, core02.MODEL_CONFIGS[key]["model_filename_candidates"][0]))
        configs[key]["save_dir"] = d
    return configs


def fresh_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path

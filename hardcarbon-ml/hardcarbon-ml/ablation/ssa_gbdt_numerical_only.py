"""
Ablation test for Model 01, target = SSA (GradientBoosting / GBDT).

Three runs on the SAME samples and the SAME train/val/test split:
  1. Full model, original hyperparameters      -> should reproduce the paper's test R2 (0.659)
  2. No categoricals, original hyperparameters -> sensitivity check (Supplementary)
  3. No categoricals, hyperparameters re-tuned -> the number for the main text

Preprocessing and splitting are identical to the final SSA pipeline:
  - legacy header renames
  - SSA > 0, raw IQR outlier removal on SSA with k = 20 (before split)
  - Box-Cox on temperatures (global)
  - categoricals factorized on TRAIN only (unseen categories -> -1)
  - target: Box-Cox fitted on TRAIN only, then RobustScaler
  - predictions: inverse scaler -> inverse Box-Cox (clipped) -> SSA units, clipped at 0
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# ---- CONFIGURATION ----
# ---- Repository paths (relative to the repository root) ----
from pathlib import Path as _RepoPath
REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "SI_Model_01.xlsx"
DATA_SHEET = "Model_01_Surface_Area_Imputated"
OUTPUT_DIR = REPO_ROOT / "outputs" / "ablation/ssa"

input_fp = str(DATA_FILE)
save_dir = str(OUTPUT_DIR)
os.makedirs(save_dir, exist_ok=True)
SEED = 42
target_col = "SSA"
IQR_K_RAW = 20.0

continuous_cols = ["Cellulose%", "Hemi-cellulose%", "Lignin%", "Volatile Matter%", "Ash%",
                   "Fixed Carbon%", "Carbon%", "Oxygen%", "Hydrogen%",
                   "Time_01", "Heating rate_01", "Time_02", "Heating rate_02"]
boxcox_cols = ["Temperature_01", "Temperature_02"]
categorical_cols = ["Prewash", "Postwash", "Type", "Method_01", "Activator_01",
                    "Method_02", "Activator_02"]

# Original optimised hyperparameters (from the final pipeline)
original_params = {
    "n_estimators": 600, "learning_rate": 0.1, "max_depth": None, "subsample": 0.6,
    "max_features": "sqrt", "min_samples_leaf": 6, "min_samples_split": 3, "ccp_alpha": 1e-4,
}
fixed_params = {"random_state": SEED}

# Re-tuning grid for the ablated model. It contains the original point, so the
# re-tuned model can only match or beat the original settings in cross-validation.
retune_grid = {
    "n_estimators": [300,450, 600],
    "learning_rate": [0.01,0.05, 0.1],
    "max_depth": [None,2, 4, 8],
    "subsample": [0.6, 0.8],
    "max_features": ["sqrt", 0.5],
    "min_samples_leaf": [3, 6],
    "min_samples_split": [3],
    "ccp_alpha": [1e-4],
}

LEGACY_RENAME_MAP = {
    "Cellulose %": "Cellulose%", "Hemi-cellulose %": "Hemi-cellulose%", "Lignin %": "Lignin%",
    "Volatile Matter %": "Volatile Matter%", "Ash %": "Ash%", "Fixed Carbon": "Fixed Carbon%",
    "Carbon %": "Carbon%", "Oxygen %": "Oxygen%", "Hydrogen %": "Hydrogen%",
    "Time 01": "Time_01", "Heating rate 01": "Heating rate_01",
    "Temperature 01": "Temperature_01", "Temperature 02": "Temperature_02",
    "Method 01": "Method_01", "Activator 01": "Activator_01", "Method 02": "Method_02",
    "Activatior 02": "Activator_02", "Time 02": "Time_02", "Heating rate 02": "Heating rate_02",
}

# ---- 1. LOAD + PREPROCESS (identical to the final pipeline) ----
df = pd.read_excel(input_fp, sheet_name=DATA_SHEET)
df.columns = [str(c).strip() for c in df.columns]
df = df.rename(columns={c: LEGACY_RENAME_MAP[c] for c in df.columns if c in LEGACY_RENAME_MAP})
for c in continuous_cols + boxcox_cols + categorical_cols + [target_col]:
    if c not in df.columns:
        print(f"WARNING: column '{c}' missing - created as empty (as in the final pipeline)")
        df[c] = np.nan
for c in continuous_cols + boxcox_cols + [target_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for col in boxcox_cols:
    data = df[col].dropna().astype(float)
    shift = abs(data.min()) + 1.0 if data.min() <= 0 else 0.0
    transformed, _ = boxcox(data + shift)
    df[col + "_boxcox"] = np.nan
    df.loc[data.index, col + "_boxcox"] = transformed
boxcox_t = [c + "_boxcox" for c in boxcox_cols]

all_inputs = continuous_cols + boxcox_t + categorical_cols
final_df = df[all_inputs + [target_col]].copy()
final_df = final_df[final_df[target_col].notnull() & (final_df[target_col] > 0)].reset_index(drop=True)

# Raw IQR outlier removal on SSA (k = 20), as in the final pipeline
q1, q3 = final_df[target_col].quantile([0.25, 0.75])
iqr = q3 - q1
final_df = final_df[final_df[target_col].between(q1 - IQR_K_RAW * iqr, q3 + IQR_K_RAW * iqr)].reset_index(drop=True)

# ---- 2. ONE SPLIT, SHARED BY ALL RUNS (same order of operations as the final pipeline) ----
idx = np.arange(len(final_df))
idx_train, idx_temp = train_test_split(idx, test_size=0.25, random_state=SEED)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.4, random_state=SEED)

# Drop incomplete rows within each split using the FULL feature set, as the final
# pipeline does. The same rows are then used by every run, including the ablated ones.
complete = final_df[all_inputs + [target_col]].notna().all(axis=1).to_numpy()
idx_train, idx_val, idx_test = (ix[complete[ix]] for ix in (idx_train, idx_val, idx_test))
print(f"Samples used: train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")

# Factorize categoricals on TRAIN only (unseen -> -1), as in the final pipeline
enc = final_df.copy()
for c in categorical_cols:
    vals = enc.loc[idx_train, c].astype(object).fillna("__NA__")
    _, uniques = pd.factorize(vals, sort=True)
    mapping = {str(u): i for i, u in enumerate(uniques)}
    enc[c] = enc[c].astype(object).fillna("__NA__").astype(str).map(mapping).fillna(-1).astype(int)

# Target: Box-Cox fitted on TRAIN only
y_train_raw = final_df.loc[idx_train, target_col].to_numpy(dtype=float)
_, lam = boxcox(y_train_raw)


def bc_forward(y):
    y = np.asarray(y, dtype=float)
    return np.log(y) if abs(lam) < 1e-12 else (np.power(y, lam) - 1.0) / lam


def bc_inverse(y_bc):
    y_bc = np.asarray(y_bc, dtype=float)
    if abs(lam) < 1e-12:
        return np.exp(y_bc)
    return np.power(np.maximum(lam * y_bc + 1.0, 1e-12), 1.0 / lam)


print(f"Target Box-Cox lambda (train only) = {lam:.6f}")


def run(inputs, params=None, grid=None, label=""):
    """Fit one model on the given input columns and return train/val/test R2 and RMSE."""
    cont = [c for c in inputs if c in continuous_cols]
    rest = [c for c in inputs if c not in continuous_cols]

    sx = RobustScaler().fit(enc.loc[idx_train, cont])
    def build(ix):
        return np.concatenate([sx.transform(enc.loc[ix, cont]), enc.loc[ix, rest].to_numpy()], axis=1)
    X_tr, X_va, X_te = build(idx_train), build(idx_val), build(idx_test)

    sy = RobustScaler()
    y_tr = sy.fit_transform(bc_forward(y_train_raw).reshape(-1, 1)).ravel()

    if grid is not None:
        gs = GridSearchCV(GradientBoostingRegressor(**fixed_params), grid, cv=5,
                          scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_tr)
        params = gs.best_params_
    model = GradientBoostingRegressor(**fixed_params, **params).fit(X_tr, y_tr)

    out = {"run": label, "n_features": len(inputs)}
    for name, X, ix in (("train", X_tr, idx_train), ("val", X_va, idx_val), ("test", X_te, idx_test)):
        pred_bc = sy.inverse_transform(model.predict(X).reshape(-1, 1)).ravel()
        pred = np.maximum(bc_inverse(pred_bc), 0)       # back to SSA units
        true = final_df.loc[ix, target_col].to_numpy()
        out[f"r2_{name}"] = round(r2_score(true, pred), 3)
        out[f"rmse_{name}"] = round(float(np.sqrt(mean_squared_error(true, pred))), 3)
    out["params"] = params

    print(f"\n{label}")
    print(f"  Train R2 = {out['r2_train']:.3f}   Val R2 = {out['r2_val']:.3f}   "
          f"Test R2 = {out['r2_test']:.3f}   Test RMSE = {out['rmse_test']:.3f}")
    print("  Hyperparameters" + (" (best from grid search):" if grid is not None else " (fixed):"))
    for k, v in params.items():
        print(f"    {k:18s} = {v}")
    return out


no_cat = continuous_cols + boxcox_t
results = [
    run(all_inputs, params=original_params, label="1. Full model (original params)"),
    run(no_cat, params=original_params, label="2. No categoricals (original params)"),
    run(no_cat, grid=retune_grid, label="3. No categoricals (re-tuned)"),
]

out_fp = os.path.join(save_dir, "ablation_ssa_gbdt.csv")
pd.DataFrame(results).to_csv(out_fp, index=False)
print(f"\nSaved: {out_fp}")
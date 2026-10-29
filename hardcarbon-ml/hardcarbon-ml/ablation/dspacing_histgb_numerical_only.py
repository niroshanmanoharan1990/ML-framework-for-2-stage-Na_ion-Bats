"""
Ablation test for Model 01, target = D-Spacing (HistGradientBoosting).

Three runs on the SAME samples and the SAME train/val/test split:
  1. Full model, original hyperparameters      -> should reproduce the paper's test R2 (0.702)
  2. No categoricals, original hyperparameters -> sensitivity check (Supplementary)
  3. No categoricals, hyperparameters re-tuned -> the number for the main text

Preprocessing and splitting are identical to the final D-Spacing pipeline
(no target transform, no outlier removal, Box-Cox on temperatures).
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# ---- CONFIGURATION ----
# ---- Repository paths (relative to the repository root) ----
from pathlib import Path as _RepoPath
REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "SI_Model_01.xlsx"
DATA_SHEET = "Model_01_D-Spacing_Imputated"
OUTPUT_DIR = REPO_ROOT / "outputs" / "ablation/dspacing"

input_fp = str(DATA_FILE)
save_dir = str(OUTPUT_DIR)
os.makedirs(save_dir, exist_ok=True)
SEED = 42
target_col = "D-Spacing"

continuous_cols = ["Cellulose%", "Hemi-cellulose%", "Lignin%", "Volatile Matter%", "Ash%",
                   "Fixed Carbon%", "Carbon%", "Oxygen%", "Hydrogen%",
                   "Time_01", "Heating rate_01", "Time_02", "Heating rate_02"]
boxcox_cols = ["Temperature_01", "Temperature_02"]
categorical_cols = ["Prewash", "Postwash", "Type", "Method_01", "Activator_01",
                    "Method_02", "Activator_02"]

# Original optimised hyperparameters (from the final pipeline)
original_params = {
    "max_iter": 250, "learning_rate": 0.15, "max_leaf_nodes": 124, "max_depth": 8,
    "min_samples_leaf": 4, "l2_regularization": 0.1, "loss": "absolute_error", "max_bins": 64,
}
fixed_params = {"random_state": SEED, "early_stopping": False}

# Re-tuning grid for the ablated model. It contains the original point, so the
# re-tuned model can only match or beat the original settings in cross-validation.
retune_grid = {
    "max_iter": [150, 250, 400],
    "learning_rate": [0.05, 0.1, 0.15],
    "max_leaf_nodes": [31, 64, 124],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [2, 4, 8],
    "l2_regularization": [0.1, 1.0],
    "loss": ["absolute_error"],
    "max_bins": [64],
}

# ---- 1. LOAD + PREPROCESS (identical to the final pipeline) ----
df = pd.read_excel(input_fp, sheet_name=DATA_SHEET)
df.columns = [str(c).strip() for c in df.columns]
for c in continuous_cols + boxcox_cols + categorical_cols + [target_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for col in boxcox_cols:
    data = df[col].dropna()
    shift = abs(data.min()) + 1.0 if data.min() <= 0 else 0.0
    transformed, _ = boxcox(data + shift)
    df[col + "_boxcox"] = np.nan
    df.loc[data.index, col + "_boxcox"] = transformed
boxcox_t = [c + "_boxcox" for c in boxcox_cols]

all_inputs = continuous_cols + boxcox_t + categorical_cols
final_df = df[all_inputs + [target_col]].copy()
final_df = final_df[final_df[target_col].notnull() & (final_df[target_col] > 0)].reset_index(drop=True)

# ---- 2. ONE SPLIT, SHARED BY ALL RUNS (same order of operations as the final pipeline) ----
idx = np.arange(len(final_df))
idx_train, idx_temp = train_test_split(idx, test_size=0.25, random_state=SEED)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.4, random_state=SEED)

# Drop incomplete rows within each split using the FULL feature set, as the final
# pipeline does. The same rows are then used by every run, including the ablated ones.
complete = final_df[all_inputs + [target_col]].notna().all(axis=1).to_numpy()
idx_train, idx_val, idx_test = (ix[complete[ix]] for ix in (idx_train, idx_val, idx_test))
print(f"Samples used: train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")


def run(inputs, params=None, grid=None, label=""):
    """Fit one model on the given input columns and return train/val/test R2 and RMSE."""
    cont = [c for c in inputs if c in continuous_cols]
    rest = [c for c in inputs if c not in continuous_cols]

    sx = RobustScaler().fit(final_df.loc[idx_train, cont])
    def build(ix):
        return np.concatenate([sx.transform(final_df.loc[ix, cont]),
                               final_df.loc[ix, rest].to_numpy()], axis=1)
    X_tr, X_va, X_te = build(idx_train), build(idx_val), build(idx_test)

    sy = RobustScaler()
    y_tr = sy.fit_transform(final_df.loc[idx_train, [target_col]]).ravel()

    if grid is not None:
        gs = GridSearchCV(HistGradientBoostingRegressor(**fixed_params), grid, cv=5,
                          scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_tr)
        params = gs.best_params_
    model = HistGradientBoostingRegressor(**fixed_params, **params).fit(X_tr, y_tr)

    out = {"run": label, "n_features": len(inputs)}
    for name, X, ix in (("train", X_tr, idx_train), ("val", X_va, idx_val), ("test", X_te, idx_test)):
        pred = sy.inverse_transform(model.predict(X).reshape(-1, 1)).ravel()
        true = final_df.loc[ix, target_col].to_numpy()
        out[f"r2_{name}"] = round(r2_score(true, pred), 3)
        out[f"rmse_{name}"] = round(float(np.sqrt(mean_squared_error(true, pred))), 4)
    out["params"] = params

    print(f"\n{label}")
    print(f"  Train R2 = {out['r2_train']:.3f}   Val R2 = {out['r2_val']:.3f}   "
          f"Test R2 = {out['r2_test']:.3f}   Test RMSE = {out['rmse_test']:.4f}")
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

out_fp = os.path.join(save_dir, "ablation_dspacing_histgb.csv")
pd.DataFrame(results).to_csv(out_fp, index=False)
print(f"\nSaved: {out_fp}")
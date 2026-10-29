"""
Ablation test for Model 01, target = Lc (RandomForest).

Three runs on the SAME samples and the SAME train/val/test split:
  1. Full model, original grid                 -> should reproduce the paper's test R2 (0.888)
  2. No categoricals, original grid            -> sensitivity check (Supplementary)
  3. No categoricals, hyperparameters re-tuned -> the number for the main text

Your final Lc pipeline searched bootstrap = [True, False], so runs 1 and 2 use that
same small grid rather than a single fixed setting.

Preprocessing and splitting are identical to the final Lc pipeline
(legacy header renames, no target transform, IQR outlier removal on Lc with k = 1.5,
Box-Cox on temperatures).
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# ---- CONFIGURATION ----
# ---- Repository paths (relative to the repository root) ----
from pathlib import Path as _RepoPath
REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "SI_Model_01.xlsx"
DATA_SHEET = "Model_01_Lc_Imputated"
OUTPUT_DIR = REPO_ROOT / "outputs" / "ablation/lc"

input_fp = str(DATA_FILE)
save_dir = str(OUTPUT_DIR)
os.makedirs(save_dir, exist_ok=True)
SEED = 42
target_col = "Lc"

continuous_cols = ["Cellulose%", "Hemi-cellulose%", "Lignin%", "Volatile Matter%", "Ash%",
                   "Fixed Carbon%", "Carbon%", "Oxygen%", "Hydrogen%",
                   "Time_01", "Heating rate_01", "Time_02", "Heating rate_02"]
boxcox_cols = ["Temperature_01", "Temperature_02"]
categorical_cols = ["Prewash", "Postwash", "Type", "Method_01", "Activator_01",
                    "Method_02", "Activator_02"]

# Original grid from the final Lc pipeline (bootstrap was searched)
original_grid = {
    "n_estimators": [400], "max_depth": [8], "min_samples_leaf": [2],
    "min_samples_split": [2], "max_features": [0.5], "bootstrap": [True, False],
}
fixed_params = {"random_state": SEED, "n_jobs": -1}

# Re-tuning grid for the ablated model. It contains the original grid, so the
# re-tuned model can only match or beat the original settings in cross-validation.
retune_grid = {
    "n_estimators": [200, 400, 800],
    "max_depth": [6, 8, 12],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 4],
    "max_features": [0.3, 0.5, 0.8],
    "bootstrap": [True, False],
}

# Legacy header renames used by the final Lc pipeline
LEGACY_RENAME_MAP = {
    "Cellulose %": "Cellulose%", "Hemi-cellulose %": "Hemi-cellulose%", "Lignin %": "Lignin%",
    "Volatile Matter %": "Volatile Matter%", "Ash %": "Ash%", "Fixed Carbon": "Fixed Carbon%",
    "Carbon %": "Carbon%", "Oxygen %": "Oxygen%", "Hydrogen %": "Hydrogen%",
    "Time 01": "Time_01", "Heating rate 01": "Heating rate_01", "Heating Rate 01": "Heating rate_01",
    "Temperature 01": "Temperature_01", "Temperature 02": "Temperature_02",
    "Method 01": "Method_01", "Activator 01": "Activator_01", "Method 02": "Method_02",
    "Activatior 02": "Activator_02", "Time 02": "Time_02", "Heating rate 02": "Heating rate_02",
    "Heating Rate 02": "Heating rate_02", "Heating Rate_02": "Heating rate_02",
}

# ---- 1. LOAD + PREPROCESS (identical to the final pipeline) ----
df = pd.read_excel(input_fp, sheet_name=DATA_SHEET)
df.columns = [str(c).strip() for c in df.columns]
df = df.rename(columns={c: LEGACY_RENAME_MAP[c] for c in df.columns if c in LEGACY_RENAME_MAP})
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

# IQR outlier removal on the raw target (k = 1.5), as in the final pipeline
q1, q3 = final_df[target_col].quantile([0.25, 0.75])
iqr = q3 - q1
final_df = final_df[final_df[target_col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)].reset_index(drop=True)

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
        gs = GridSearchCV(RandomForestRegressor(**fixed_params), grid, cv=5,
                          scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_tr)
        params = gs.best_params_
    model = RandomForestRegressor(**fixed_params, **params).fit(X_tr, y_tr)

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
    run(all_inputs, grid=original_grid, label="1. Full model (original grid)"),
    run(no_cat, grid=original_grid, label="2. No categoricals (original grid)"),
    run(no_cat, grid=retune_grid, label="3. No categoricals (re-tuned)"),
]

out_fp = os.path.join(save_dir, "ablation_lc_rf.csv")
pd.DataFrame(results).to_csv(out_fp, index=False)
print(f"\nSaved: {out_fp}")
"""
Ablation test for Model 01, target = Carbon% (XGBoost).

Three runs on the SAME samples and the SAME train/val/test split:
  1. Full model, original hyperparameters      -> should reproduce the paper's test R2 (0.713)
  2. No categoricals, original hyperparameters -> sensitivity check (Supplementary)
  3. No categoricals, hyperparameters re-tuned -> the number for the main text

The only difference between runs is the input columns (and, for run 3, re-tuned
hyperparameters). Preprocessing, target transform, outlier removal and the split
are identical to the final Carbon pipeline.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# ---- CONFIGURATION ----
# ---- Repository paths (relative to the repository root) ----
from pathlib import Path as _RepoPath
REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "SI_Model_01.xlsx"
DATA_SHEET = "Model_01_Carbon_Imputated"
OUTPUT_DIR = REPO_ROOT / "outputs" / "ablation/carbon"

input_fp = str(DATA_FILE)
save_dir = str(OUTPUT_DIR)
os.makedirs(save_dir, exist_ok=True)
SEED = 42
target_col = "Carbon"

# Target transform. Your final (published) Carbon model used ln(C)^2.
# True  -> identical to the final pipeline; run 1 reproduces test R2 = 0.713.
# False -> raw Carbon; run 1 will NOT reproduce 0.713, and outlier removal
#          (done on the target) may keep a slightly different set of samples.
USE_LOG_TRANSFORM = True
transformed_col = "logC2" if USE_LOG_TRANSFORM else "Carbon_raw"

continuous_cols = ["Cellulose%", "Hemi-cellulose%", "Lignin%", "Volatile Matter%", "Ash%",
                   "Fixed Carbon%", "Carbon%", "Oxygen%", "Hydrogen%", "Time_01", "Heating rate_01"]
boxcox_cols = ["Temperature_01", "Temperature_02"]
categorical_cols = ["Prewash", "Postwash", "Type", "Method_01", "Activator_01",
                    "Method_02", "Activator_02"]

# Original optimised hyperparameters (from the final pipeline)
original_params = {
    "n_estimators": 550, "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 2,
    "subsample": 0.6, "colsample_bytree": 0.4, "colsample_bylevel": 0.8, "gamma": 0.0,
    "reg_alpha": 0.15, "reg_lambda": 0.1,
}
fixed_params = {"random_state": SEED, "objective": "reg:squarederror", "n_jobs": -1,
                "tree_method": "hist", "eval_metric": "rmse"}

# Re-tuning grid for the ablated model. It contains the original point, so the
# re-tuned model can only match or beat the original settings in cross-validation.
retune_grid = {
    "n_estimators": [300, 550, 800],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8],
    "min_child_weight": [1, 2, 4],
    "subsample": [0.6, 0.8],
    "colsample_bytree": [0.4, 0.7],
    "colsample_bylevel": [0.8], "gamma": [0.0], "reg_alpha": [0.15], "reg_lambda": [0.1],
}

# ---- 1. LOAD + PREPROCESS (identical to the final pipeline) ----
df = pd.read_excel(input_fp, sheet_name=DATA_SHEET)
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
final_df = final_df[final_df[target_col].notnull() & (final_df[target_col] > 0)]
if USE_LOG_TRANSFORM:
    final_df[transformed_col] = np.log(final_df[target_col]) ** 2
else:
    final_df[transformed_col] = final_df[target_col]
    print("NOTE: no target transform - full-model R2 will differ from the paper value (0.713).")

q1, q3 = final_df[transformed_col].quantile([0.25, 0.75])
iqr = q3 - q1
final_df = final_df[final_df[transformed_col].between(q1 - 1 * iqr, q3 + 1 * iqr)]

# Drop incomplete rows using the FULL feature set, before splitting.
# This guarantees the full and ablated models use exactly the same samples.
final_df = final_df.dropna(subset=all_inputs + [transformed_col]).reset_index(drop=True)
print(f"Samples used: {len(final_df)}")

# ---- 2. ONE SPLIT, SHARED BY ALL RUNS ----
idx = np.arange(len(final_df))
idx_train, idx_temp = train_test_split(idx, test_size=0.25, random_state=SEED)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.4, random_state=SEED)


def run(inputs, params=None, grid=None, label=""):
    """Fit one model on the given input columns and return train/val/test R2 and RMSE."""
    cont = [c for c in inputs if c in continuous_cols]
    rest = [c for c in inputs if c not in continuous_cols]

    def build(ix, scaler, fit=False):
        Xc = final_df.loc[ix, cont].to_numpy()
        Xc = scaler.fit_transform(Xc) if fit else scaler.transform(Xc)
        return np.concatenate([Xc, final_df.loc[ix, rest].to_numpy()], axis=1)

    sx = RobustScaler()
    X_tr = build(idx_train, sx, fit=True)
    X_va, X_te = build(idx_val, sx), build(idx_test, sx)

    sy = RobustScaler()
    y_tr = sy.fit_transform(final_df.loc[idx_train, [transformed_col]]).ravel()

    if grid is not None:
        gs = GridSearchCV(XGBRegressor(**fixed_params), grid, cv=5,
                          scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_tr)
        params = gs.best_params_
    model = XGBRegressor(**fixed_params, **params).fit(X_tr, y_tr)

    out = {"run": label, "n_features": len(inputs)}
    for name, X, ix in (("train", X_tr, idx_train), ("val", X_va, idx_val), ("test", X_te, idx_test)):
        pred_t = sy.inverse_transform(model.predict(X).reshape(-1, 1))
        if USE_LOG_TRANSFORM:
            pred = np.exp(np.sqrt(np.clip(pred_t, 0, None))).ravel()   # back to Carbon %
        else:
            pred = pred_t.ravel()
        true = final_df.loc[ix, target_col].to_numpy()
        out[f"r2_{name}"] = round(r2_score(true, pred), 3)
        out[f"rmse_{name}"] = round(float(np.sqrt(mean_squared_error(true, pred))), 3)
    out["params"] = params
    print(f"\n{label}")
    print(f"  Train R2 = {out['r2_train']:.3f}   Val R2 = {out['r2_val']:.3f}   Test R2 = {out['r2_test']:.3f}   Test RMSE = {out['rmse_test']:.3f}")
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

out_fp = os.path.join(save_dir, "ablation_carbon_xgb.csv")
pd.DataFrame(results).to_csv(out_fp, index=False)
print(f"\nSaved: {out_fp}")
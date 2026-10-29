"""
Ablation test for Model 02, target = initial Capacity (XGBoost).

Three runs on the SAME samples and the SAME train/val/test split:
  1. Full model, original hyperparameters             -> should reproduce the paper's test R2
  2. No categorical assembly variables, original params -> sensitivity check (Supplementary)
  3. No categorical assembly variables, re-tuned        -> the number for the main text

Preprocessing is identical to the final Capacity pipeline:
  - drop target, ICE, Electrolyte text column, No_Binder and leakage columns
  - median imputation, all-NaN columns dropped
  - 75/15/10 split (seed 42)
  - IQR filter on the TRAINING target only (k = 1.9)
  - pairwise augmentation on the training set only (10 %, 5 % noise)
  - each run reports test R2 for the train-only model AND the train+val refit
    (your pipeline reports both; use whichever matches the paper)

"Categorical assembly variables" = every column that your SHAP grouping assigns to
Separator, Binder type, Solvent type, Salt type or Electrolyte type, plus any other
0/1 column. The script PRINTS both lists - check them before trusting the result.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

# ---- CONFIGURATION ----
# ---- Repository paths (relative to the repository root) ----
from pathlib import Path as _RepoPath
REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "SI_Model_02.xlsx"
DATA_SHEET = "Capacity_Processed"
OUTPUT_DIR = REPO_ROOT / "outputs" / "ablation/capacity"

INPUT_FP = str(DATA_FILE)
INPUT_SHEET = DATA_SHEET
SAVE_DIR = str(OUTPUT_DIR)
TARGET_COL = "Capacity"
SEED = 42
PAIRWISE_FRACTION = 0.0
NOISE_FRACTION = 0.05
IQR_K = 20

original_params = {
    "n_estimators": 600, "learning_rate": 0.0575, "max_depth": 5,
    "subsample": 0.8, "colsample_bytree": 0.35, "reg_lambda": 2.5,
}
fixed_params = {"objective": "reg:squarederror", "random_state": SEED, "n_jobs": -1, "verbosity": 0}

# Re-tuning grid for the ablated model (contains the original point).
retune_grid = {
    "n_estimators": [300, 600,100],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [None,13, 5, 7],
    "subsample": [0.4,0.6,0.8],
    "colsample_bytree": [0.35, 0.6],
    "reg_lambda": [1.0, 2.5],
}

# Copied from your pipeline so the grouping is identical
SHAP_GROUP_DEFINITIONS = {
    "Active material ratio": ["Active material ratio", "Active_Material_Ratio", "ActiveMaterialRatio", "Active ratio", "AM ratio"],
    "Carbon ratio": ["Carbon ratio", "Carbon_Ratio", "CarbonRatio", "Conductive carbon ratio"],
    "Binder ratio": ["Binder ratio", "Binder_Ratio", "BinderRatio"],
    "Separator type": ["GF", "PE", "PP"],
    "Binder type": ["CMC", "CMC+SBR", "CMC_SBR", "CMC SBR", "No Binder", "No_Binder", "PAA", "PANA", "PVDF",
                    "SBR", "Sodium alginate", "Sodium_Alginate"],
    "Solvent type": ["NMP", "No Solvent", "No_Solvent", "Water"],
    "Salt type": ["NaClO4", "NaCLO4", "NaOTf", "NaOTF", "NaPF6", "NaTFSI"],
    "Salt concentration": ["Salt concentration", "Salt_Concentration", "SaltConcentration", "Salt conc", "Salt_conc"],
    "Additive concentration": ["Additive concentration", "Additive_Concentration", "AdditiveConcentration",
                               "Additive conc", "Additive_conc"],
    "Electrolyte type": ["Electrolyte type", "Electrolyte_Type", "ElectrolyteType", "Aqueous", "Organic", "Carbonate",
                         "Ether", "Ionic liquid", "Ionic_Liquid", "PC", "PC fraction", "PC_Fraction", "PCFraction",
                         "EC", "EC fraction", "EC_Fraction", "ECFraction", "DMC", "DMC fraction", "DMC_Fraction",
                         "DMCFraction", "DEC", "DEC fraction", "DEC_Fraction", "DECFraction", "EMC", "EMC fraction",
                         "EMC_Fraction", "EMCFraction", "FEC", "FEC fraction", "FEC_Fraction", "FECFraction", "DME",
                         "DME fraction", "DME_Fraction", "DMEFraction", "DOL", "DOL fraction", "DOL_Fraction",
                         "DOLFraction", "TETRAGLYME", "TETRAGLYME fraction", "TETRAGLYME_Fraction",
                         "TETRAGLYMEFraction"],
    "La": ["La", "L_a", "La_nm", "La (nm)", "Lateral crystallite size", "Lateral_Crystallite_Size"],
    "D-S": ["D-S", "D_S", "D spacing", "D-spacing", "D_Spacing", "D Spacing", "d002", "d_002"],
    "Current density": ["Current density", "Current_Density", "CurrentDensity", "Current", "Curr", "mA/g"],
    "Lc": ["Lc", "L_c", "Lc_nm", "Lc (nm)", "Crystallite height", "Crystallite_Height"],
    "Surface area": ["SSA", "Surface Area", "Surface area", "Specific surface area", "BET", "BET surface area",
                     "BET_Surface_Area"],
    "Pore volume": ["PV", "Pore Volume", "Pore volume", "Pore_Volume"],
}
CATEGORICAL_GROUPS = {"Separator type", "Binder type", "Solvent type", "Salt type", "Electrolyte type"}


def _norm(v):
    return "".join(ch for ch in str(v).lower() if ch.isalnum())


def _chunks(v):
    return [c for c in re.split(r"[^A-Za-z0-9]+", str(v).lower()) if c]


def matches(feature, token):
    fk, tk = _norm(feature), _norm(token)
    if not fk or not tk:
        return False
    if fk == tk:
        return True
    fc, tc = _chunks(feature), _chunks(token)
    if tc and tc == fc:
        return True
    if len(tc) == 1 and tc[0] in fc:
        return True
    if len(tc) > 1 and _norm("".join(fc)) == _norm("".join(tc)):
        return True
    if fk in {f"{tk}fraction", f"{tk}percent", f"{tk}percentage"}:
        return True
    return len(tk) >= 4 and tk in fk


def is_binary(s):
    v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    return v.size > 0 and bool(np.all(np.isin(np.round(np.unique(v), 12), [0.0, 1.0])))


def split_categorical(columns, X):
    """Assign columns to groups in your pipeline's order; return (categorical, continuous)."""
    assigned, cat = set(), []
    for g, tokens in SHAP_GROUP_DEFINITIONS.items():
        for f in columns:
            if f not in assigned and any(matches(f, t) for t in tokens):
                assigned.add(f)
                if g in CATEGORICAL_GROUPS:
                    cat.append(f)
    for f in columns:                      # any other 0/1 column is also categorical
        if f not in cat and is_binary(X[f]):
            cat.append(f)
    return cat, [f for f in columns if f not in cat]


def iqr_mask(y, k):
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    return ((y >= q1 - k * iqr) & (y <= q3 + k * iqr)).to_numpy()


def pairwise_augmentation(X_in, y_in, fraction, noise_fraction, random_state):
    """Identical to your pipeline."""
    rng = np.random.default_rng(random_state)
    n = len(X_in)
    n_aug = int(n * fraction)
    if n_aug <= 0:
        return None, None
    idx1 = rng.choice(n, n_aug, replace=False)
    idx2 = rng.choice(n, n_aug, replace=False)
    X1, X2 = X_in.iloc[idx1].to_numpy(float), X_in.iloc[idx2].to_numpy(float)
    X_syn = (X1 + X2) / 2.0
    binary_cols = [c for c in X_in.columns if is_binary(X_in[c])]
    b_idx = [X_in.columns.get_loc(c) for c in binary_cols]
    cont_cols = [c for c in X_in.columns if c not in binary_cols]
    c_idx = [X_in.columns.get_loc(c) for c in cont_cols]
    if b_idx:
        take_first = rng.integers(0, 2, size=(n_aug, len(b_idx))).astype(bool)
        X_syn[:, b_idx] = np.where(take_first, X1[:, b_idx], X2[:, b_idx])
    if c_idx:
        sd = X_in[cont_cols].std(axis=0, ddof=0).to_numpy(float)
        X_syn[:, c_idx] += rng.normal(0.0, noise_fraction * sd, size=(n_aug, len(c_idx)))
    y_arr = y_in.to_numpy(float)
    return pd.DataFrame(X_syn, columns=X_in.columns), pd.Series((y_arr[idx1] + y_arr[idx2]) / 2.0, name=y_in.name)


def fit_set(X, y):
    """IQR filter + augmentation, as applied to a training set in your pipeline."""
    keep = iqr_mask(y, IQR_K)
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
    X_aug, y_aug = pairwise_augmentation(X, y, PAIRWISE_FRACTION, NOISE_FRACTION, SEED)
    if X_aug is not None:
        X, y = pd.concat([X, X_aug], ignore_index=True), pd.concat([y, y_aug], ignore_index=True)
    return X, y


# ---- 1. LOAD + PREPROCESS (identical to the final pipeline) ----
save_dir = Path(SAVE_DIR)
save_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_excel(INPUT_FP, sheet_name=INPUT_SHEET)
df.columns = [str(c).strip() for c in df.columns]
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype(str).str.strip()

leak = {"actual_Capacity", "predicted_Capacity", "abs_error", "split", "row_in_processed_table", "Unnamed: 41"}
dyn = [c for c in df.columns if c.lower().startswith(("actual_", "predicted_"))]
drop_cols = [c for c in {TARGET_COL, "Electrolyte", "ICE", "No_Binder", "No Binder"} | leak | set(dyn) if c in df.columns]

X_df = df.drop(columns=drop_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET_COL], errors="coerce")
mask = y.notna()
X_df, y = X_df.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)
X_df = X_df.fillna(X_df.median(numeric_only=True))
X_df = X_df.drop(columns=X_df.columns[X_df.isna().all()].tolist())

# ---- 2. ONE SPLIT, SHARED BY ALL RUNS ----
X_train, X_temp, y_train, y_temp = train_test_split(X_df, y, test_size=0.25, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=SEED)
print(f"Samples: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")

all_features = list(X_df.columns)
cat_cols, cont_cols = split_categorical(all_features, X_df)
print(f"\nCategorical assembly columns REMOVED in ablation ({len(cat_cols)}):\n  {cat_cols}")
print(f"\nColumns KEPT in ablation ({len(cont_cols)}):\n  {cont_cols}")


def run(cols, params=None, grid=None, label=""):
    Xtr, ytr = fit_set(X_train[cols].reset_index(drop=True), y_train.reset_index(drop=True))
    if grid is not None:
        # tune on the IQR-filtered training set WITHOUT synthetic rows, so synthetic
        # samples cannot appear in both CV training and CV validation folds
        keep = iqr_mask(y_train.reset_index(drop=True), IQR_K)
        gs = GridSearchCV(XGBRegressor(**fixed_params), grid, cv=5,
                          scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=0)
        gs.fit(X_train[cols].reset_index(drop=True).loc[keep], y_train.reset_index(drop=True).loc[keep])
        params = gs.best_params_
    model = XGBRegressor(**fixed_params, **params).fit(Xtr, ytr)

    out = {"run": label, "n_features": len(cols)}
    for name, X, yt in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
        p = model.predict(X[cols])
        out[f"r2_{name}"] = round(r2_score(yt, p), 3)
        out[f"rmse_{name}"] = round(float(np.sqrt(mean_squared_error(yt, p))), 3)

    # refit on train + val (same IQR + augmentation policy), evaluate on test
    Xtv, ytv = fit_set(pd.concat([X_train[cols], X_val[cols]], ignore_index=True),
                       pd.concat([y_train, y_val], ignore_index=True))
    p = XGBRegressor(**fixed_params, **params).fit(Xtv, ytv).predict(X_test[cols])
    out["r2_test_refit"] = round(r2_score(y_test, p), 3)
    out["rmse_test_refit"] = round(float(np.sqrt(mean_squared_error(y_test, p))), 3)
    out["params"] = params

    print(f"\n{label}")
    print(f"  Train R2 = {out['r2_train']:.3f}   Val R2 = {out['r2_val']:.3f}   "
          f"Test R2 = {out['r2_test']:.3f}   Test RMSE = {out['rmse_test']:.3f}")
    print(f"  Test R2 (refit on train+val) = {out['r2_test_refit']:.3f}   RMSE = {out['rmse_test_refit']:.3f}")
    print("  Hyperparameters" + (" (best from grid search):" if grid is not None else " (fixed):"))
    for k, v in params.items():
        print(f"    {k:18s} = {v}")
    return out


results = [
    run(all_features, params=original_params, label="1. Full model (original params)"),
    run(cont_cols, params=original_params, label="2. No categorical assembly variables (original params)"),
    run(cont_cols, grid=retune_grid, label="3. No categorical assembly variables (re-tuned)"),
]
out_fp = save_dir / "ablation_capacity_xgb.csv"
pd.DataFrame(results).to_csv(out_fp, index=False)
print(f"\nSaved: {out_fp}")
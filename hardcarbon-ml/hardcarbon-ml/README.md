# Interpretable two-stage machine learning for biomass-derived hard carbon anodes in sodium-ion batteries

Code and data accompanying:

> N. Manoharan, N. Boase, K. Fairfull-Smith, D. Dubal. *Interpretable two-stage machine learning for biomass-derived hard carbon anodes in sodium-ion batteries.* (submitted, 2026).

The framework predicts hard carbon properties in two physically sequenced stages:

| Stage | Inputs | Outputs |
|---|---|---|
| **Model 01** – biomass → carbon | precursor composition (lignocellulosic, proximate, elemental), process conditions and categorical synthesis descriptors (route, method, activator, pre-/post-wash) | carbon content, d₀₀₂, L_a, L_c, pore volume, Raman I_G/I_D, specific surface area |
| **Model 02** – carbon → cell | the seven structural descriptors plus electrode ratios, binder, electrode-processing solvent, salt, electrolyte composition, additive, separator and current density | ICE, initial capacity, slope capacity, high-rate discharge (HDR) capacity |

Each script trains one target, reports train/validation/test metrics, and produces parity plots, SHAP attribution (individual and grouped) and one- and two-variable partial-dependence maps. A desktop interface chains the two models.

## Repository structure

```
hardcarbon-ml/
├── data/
│   ├── SI_Model_01.xlsx      Model 01 dataset (Supplementary Data 1)
│   └── SI_Model_02.xlsx      Model 02 dataset (Supplementary Data 2)
├── model_01/                 7 training pipelines (one per structural descriptor)
├── model_02/                 4 training pipelines (one per electrochemical metric)
├── ablation/                 11 scripts: numerical + categorical vs numerical-only inputs
├── gui/                      Tkinter interface (Modes 01, 02 and 03)
├── outputs/                  created when scripts run (models, plots, SHAP, PDP) – not tracked
├── requirements.txt
└── LICENSE
```

### Data files

| File | Sheet | Used by |
|---|---|---|
| `SI_Model_01.xlsx` | `Master Copy_Model_01` | complete compiled dataset (542 samples) |
| | `Reference` | source publications (140), referenced by the `REF` column |
| | `Model_01_<target>_Imputated` | training data for each Model 01 target |
| `SI_Model_02.xlsx` | `Master Copy_Model_02` | complete compiled dataset (930 half-cell records) |
| | `Reference` | source publications |
| | `ICE_Processed`, `Capacity_Processed`, `Slope_Processed`, `HDR_Processed` | training data for each Model 02 target |

Categorical codes used in Model 01 are defined in Supplementary Tables S1–S3 of the paper; Model 02 categorical variables are one-hot encoded and electrolyte composition is given as solvent volume fractions (Supplementary Table S15).

## Installation

```bash
git clone https://github.com/<username>/hardcarbon-ml.git
cd hardcarbon-ml
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run any script from the repository root; paths are resolved relative to the repository, so no editing is needed.

```bash
# Model 01 – structural descriptors
python model_01/carbon_xgb.py
python model_01/dspacing_histgb.py
python model_01/la_gbdt.py
python model_01/lc_rf.py
python model_01/pv_gbdt.py
python model_01/raman_histgb.py
python model_01/ssa_gbdt.py

# Model 02 – electrochemical performance
python model_02/ice_catboost.py
python model_02/capacity_xgb.py
python model_02/slope_gbdt.py
python model_02/hdr_stacking.py
```

Each script writes to `outputs/<model>/<target>/`: the fitted model and preprocessing objects (`.pkl`), parity plots, SHAP results (CSV and figures) and partial-dependence maps (PNG, SVG, PDF).

| Target | Script | Algorithm | Data sheet |
|---|---|---|---|
| Carbon content | `model_01/carbon_xgb.py` | XGBoost | `Model_01_Carbon_Imputated` |
| d₀₀₂ | `model_01/dspacing_histgb.py` | HistGradientBoosting | `Model_01_D-Spacing_Imputated` |
| L_a | `model_01/la_gbdt.py` | Gradient boosting | `Model_01_La_Imputated` |
| L_c | `model_01/lc_rf.py` | Random forest | `Model_01_Lc_Imputated` |
| Pore volume | `model_01/pv_gbdt.py` | Gradient boosting | `Model_01_Pore_Volume_Imputated` |
| Raman I_G/I_D | `model_01/raman_histgb.py` | HistGradientBoosting | `Model_01_ID_IG_Imputated` |
| SSA | `model_01/ssa_gbdt.py` | Gradient boosting | `Model_01_Surface_Area_Imputated` |
| ICE | `model_02/ice_catboost.py` | CatBoost | `ICE_Processed` |
| Initial capacity | `model_02/capacity_xgb.py` | XGBoost | `Capacity_Processed` |
| Slope capacity | `model_02/slope_gbdt.py` | Gradient boosting | `Slope_Processed` |
| HDR capacity | `model_02/hdr_stacking.py` | Stacking (RF + XGBoost + CatBoost) | `HDR_Processed` |

All scripts use a fixed 75:15:10 train/validation/test split (random seed 42). Train-only outlier filtering and pairwise augmentation are applied where stated in each script's configuration block.

### Numerical-only comparison (Supplementary Table S10)

Each script in `ablation/` retrains one model with and without its categorical inputs on the same samples and split, and prints test R² and the selected hyperparameters:

```bash
python ablation/ice_catboost_numerical_only.py
```

Results are saved as CSV in `outputs/ablation/<target>/`. Re-tuning uses grid search and can take from minutes to about an hour per target (the HDR stacking model is the slowest).

### Graphical interface

Train the models first (the interface loads them from `outputs/`), then:

```bash
cd gui
python combined_app.py
```

- **Mode 01** predicts the seven structural descriptors from biomass and process inputs.
- **Mode 02** predicts the electrochemical metrics from measured descriptors and cell-assembly conditions.
- **Mode 03** chains the two. Its predictions carry Model 01 error into Model 02 and are intended for screening rather than quantitative prediction.

See `gui/README.md` for details.

## Reproducibility notes

- Tree-based learners can give slightly different results across library versions and operating systems. Metrics reported in the paper were obtained with the package versions used at the time; small differences in R² (typically < 0.02) are expected with other versions.
- SHAP values for non-tree models (HistGradientBoosting, stacking) use the permutation explainer and are sampled; repeated runs give closely similar but not identical values.

## Citation

If you use this code or data, please cite the paper above. A Zenodo DOI will be added on release.

## License

Code: MIT License (see `LICENSE`). Data: please cite the original publications listed in the `Reference` sheets.

## Contact

Niroshan Manoharan · Deepak Dubal (deepak.dubal@qut.edu.au), Queensland University of Technology

# Hard Carbon Prediction Suite (GUI)

Run the training scripts in `model_01/` and `model_02/` first (see the main README); they write the fitted models to `outputs/`, which this interface loads automatically.

One window, three modes. Every mode is a single page: input cards on the left,
predictions on the right.

| Mode | Inputs | Predicts |
|---|---|---|
| **01** Biomass → Carbon | biomass composition + synthesis route | 7 carbon descriptors (Model 01) |
| **02** Carbon → Battery | measured descriptors + cell assembly + testing | ICE, initial capacity, slope capacity, HDR (Model 02) |
| **03** Biomass → Battery | biomass + synthesis + cell assembly + testing | Model 01's descriptors fill Model 02's inputs, then Model 02 predicts |

## Run

```
python combined_app.py
```

Model folders are read from `../outputs/model_01/` and `../outputs/model_02/` (set in
`hard_carbon_core.py` → `MODEL_SAVE_DIRS` and `model2_core.py` → `MODEL_CONFIGS`).

Needs `tkinter`, `joblib`, `pandas`, `numpy`, `scikit-learn`, plus whatever your
saved models need to unpickle (`xgboost`, `catboost`).

## Mode 03

There is no separate Model 01 readout. The seven predicted descriptors are
written straight into the **Carbon descriptors** card (amber), where they stay
editable.

- **Run chain** – Model 01 fills the descriptors, then Model 02 predicts.
- **Fill descriptors** – only the Model 01 step.
- **Predict only** – Model 02 on whatever the descriptor fields now hold.

Type a measured value into any amber field and its tint clears, so you can mix
measured and predicted descriptors. If any Model 01 target fails, its field is
left blank, the status bar names it, and Model 02 is not run on a partial set.
Predicted descriptors carry Model 01's error into Model 02: treat Mode 03 as a
screening estimate.

Mapping (Model 01 output → Model 02 field): Carbon → Carbon%, D-Spacing → D-S,
La → La, Lc → Lc, PV → pore volume, SSA → surface area, IG/ID → IG/ID
(`MODEL01_TO_MODEL02_GROUP` in `model02_form.py`).

## Design

- **Layout.** Cards are auto-balanced into 2–4 columns so each mode fits one
  window. Results sit on the right in every mode. Wider windows get roomier
  spacing; content is centred on very wide screens.
- **Colour.** One accent (indigo). A coloured top edge marks which model a card
  feeds: **indigo = Model 01 inputs, teal = Model 02 inputs**. Amber = predicted
  by Model 01. Red = needs attention.
- **Errors are inline.** A problem field gets a red outline and focus, the
  message goes in the status bar, and the outline clears as you type. No
  pop-ups.
- **Per-output ▶** replaces the old "target to predict individually" dropdown.
- **Enter** in any text field runs the page's main action (not on buttons or
  dropdowns).
- **Models pill** (top right) shows how many of the 11 saved models loaded;
  click it for the per-model report and a console dump of your real feature
  groups.
- **Density** is chosen automatically (compact up to an 864px-tall screen,
  comfortable above) and `ui_kit.py` holds every colour and spacing token if you
  want to restyle. On Windows the app asks for crisp (DPI-aware) rendering.

### Fit, as tested

Nothing needs a scrollbar and no widget is clipped in any mode, including the
busiest Mode 03 state (Two step, four electrolyte components, additive on), on:
1280×720 (Arial-like and a much wider font), 1366×768, 1536×864, 1920×1080, and
simulated Windows display scaling of 125% and 150%. If a screen genuinely cannot
hold a page, a scrollbar appears instead of clipping.

## Behaviour carried over

- Every numeric field is required; blank never becomes 0.
- Mode 01: composition groups capped at 100%; Step 2 locked until Type = Two step
  and reset to 0 / "No" when switched back.
- Mode 02: AM + carbon + binder = 100%; Binder/Solvent type locked to
  "No Binder"/"No Solvent" while Binder ratio is 0; electrolyte is pick → Add →
  fraction (sums to 1.00); additive is checkbox-gated.
- Current: the entered value IS 1C. ICE/Capacity use 0.05C, **Slope uses 1C
  (unconfirmed)**, HDR is predicted at 0.5C / 1C / 2C.
- Every prediction prints the exact encoded row to the console.

## Open items

- **Slope's rate** is still an unconfirmed 1C default.
- **Dropdowns start on their first option** (for example Pre-wash = Conc Acid
  wash), so an untouched one is sent silently. They could start blank and
  require a choice, like the numeric fields do.
- Tested on Linux with synthetic model files shaped like yours, not your trained
  models and not on Windows: Segoe UI rendering, the exact look of the `clam`
  theme and DPI behaviour on your machine are the things to eyeball.

## Tests (real tkinter)

```
xvfb-run -a -s "-screen 0 1280x720x24 -dpi 96" python3 test_combined_gui.py   # behaviour
xvfb-run -a -s "-screen 0 1920x1080x24 -dpi 96" python3 test_layout.py       # fit / clipping matrix
xvfb-run -a -s "-screen 0 1280x720x24 -dpi 96" python3 test_degraded.py      # wrong paths, missing model
```

On a normal desktop run them with plain `python`. `SHOT_DIR=somefolder` also
saves screenshots. The fixtures mimic your artifact structure; they are not your
trained models, so the tests check the GUI and the hand-off, not accuracy.

## Files

- `combined_app.py` – run this (shell, pages, Mode 03 chain)
- `ui_kit.py` – design system: colours, type, spacing, cards, field grid, tiles
- `form_base.py`, `model01_form.py`, `model02_form.py` – the forms and their logic
- `hard_carbon_core.py`, `model2_core.py` – prediction logic, no tkinter
- `diagnose_model2_features.py` – prints what your real `feature_names.pkl` files contain
- `test_*.py` – tests; `docs/` – screenshots (test data, not real predictions)

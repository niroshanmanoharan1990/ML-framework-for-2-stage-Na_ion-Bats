"""
Model 01 form: biomass + synthesis inputs -> 7 carbon structural descriptors.

The form owns the state (entries, dropdowns, outputs) and all validation /
prediction logic; `build_cards(parent)` creates its four input cards for a page
to arrange. Prediction logic itself lives in hard_carbon_core.py.

Behaviour (unchanged from the verified standalone version)
  - every numeric field is required (blank never silently becomes 0)
  - lignocellulosic, proximate and elemental groups are each capped at 100%
  - Step 2 is frozen (disabled, showing the one-step defaults) until
    Type = "Two step", and re-freezes to 0 / "No" when switched back
  - a console line per model shows the exact encoded row sent to it
"""

import traceback
import tkinter as tk

import hard_carbon_core as core
from form_base import FormBase, FieldError
from ui_kit import C, Card, FieldGrid, set_chip

MODEL_ORDER = core.MODEL_ORDER
CATEGORICAL_COLS = core.CATEGORICAL_COLS
CATEGORICAL_LABEL_ORDER = core.CATEGORICAL_LABEL_ORDER

# (key, tile label, colour name)
OUTPUT_TILES = [
    ("Carbon", "Carbon (%)", "blue"),
    ("D-Spacing", "D-spacing (nm)", "orange"),
    ("La", "La (nm)", "green"),
    ("Lc", "Lc (nm)", "rose"),
    ("PV", "PV (cm\u00b3/g)", "violet"),
    ("SSA", "SSA (m\u00b2/g)", "teal"),
    ("IG/ID", "Raman IG/ID", "amber"),
]

COMPOSITION_TOLERANCE = 0.5  # percentage points of headroom allowed above 100%

# All three groups are CAPS, not equalities: literature-compiled rows often
# don't close exactly to 100% (moisture reported separately, rounding across
# source papers), but none can legitimately exceed 100% of the sample.
COMPOSITION_GROUP_1 = ["Cellulose%", "Hemi-cellulose%", "Lignin%"]
COMPOSITION_GROUP_2 = ["Volatile Matter%", "Ash%", "Fixed Carbon%"]
ELEMENTAL_GROUP = ["Carbon%", "Oxygen%", "Hydrogen%"]

SHORT_LABEL = {
    "Cellulose%": "Cellulose", "Hemi-cellulose%": "Hemi-cellulose", "Lignin%": "Lignin",
    "Volatile Matter%": "Volatile matter", "Ash%": "Ash", "Fixed Carbon%": "Fixed carbon",
    "Carbon%": "Carbon", "Oxygen%": "Oxygen", "Hydrogen%": "Hydrogen",
}
GROUPS = [
    ("group1", "Lignocellulosic \u00b7 wt%", COMPOSITION_GROUP_1),
    ("group2", "Proximate \u00b7 wt%", COMPOSITION_GROUP_2),
    ("group3", "Elemental \u00b7 wt%", ELEMENTAL_GROUP),
]

# Long labels are used in validation messages; short ones on screen.
STEP1_FIELDS = [("Time_01", "Time, step 1 (min)"), ("Heating rate_01", "Heating rate, step 1 (\u00b0C/min)")]
STEP2_FIELDS = [("Time_02", "Time, step 2 (min)"), ("Heating rate_02", "Heating rate, step 2 (\u00b0C/min)")]
TEMPERATURE_FIELDS = [("Temperature_01", "Temperature, step 1 (\u00b0C)"),
                      ("Temperature_02", "Temperature, step 2 (\u00b0C)")]
ALL_SIMPLE_NUMERIC_FIELDS = STEP1_FIELDS + STEP2_FIELDS + TEMPERATURE_FIELDS
LONG_LABEL = dict(ALL_SIMPLE_NUMERIC_FIELDS)

STEP_FIELDS = {
    1: [("Time_01", "Time (min)"), ("Heating rate_01", "Rate (\u00b0C/min)"), ("Temperature_01", "Temp (\u00b0C)")],
    2: [("Time_02", "Time (min)"), ("Heating rate_02", "Rate (\u00b0C/min)"), ("Temperature_02", "Temp (\u00b0C)")],
}
STEP2_NUMERIC = ["Time_02", "Heating rate_02", "Temperature_02"]


class Model01Form(FormBase):
    def __init__(self, master, cache=None):
        super().__init__(master)
        self.cache = cache if cache is not None else core.BundleCache(core.MODEL_SAVE_DIRS)

        self.entries = {}
        self.combo_vars = {}
        self.combo_widgets = {}
        self.total_labels = {}
        self.total_texts = {}
        self.output_vars = {k: tk.StringVar(master=master) for k, _l, _c in OUTPUT_TILES}
        self.last_values = {}
        self.last_errors = {}
        self.step2_card = None

    # ================================================================= gathering / validation
    def get_required_float(self, col, label):
        entry = self.entries[col]
        val = entry.get().strip()
        if val == "":
            raise FieldError(f"'{label}' is required.", [entry])
        try:
            return float(val)
        except ValueError:
            raise FieldError(f"'{label}' must be a number.", [entry])

    def gather_raw_inputs(self):
        raw = {}
        for col in COMPOSITION_GROUP_1 + COMPOSITION_GROUP_2 + ELEMENTAL_GROUP:
            raw[col] = self.get_required_float(col, col)
        for col, label in ALL_SIMPLE_NUMERIC_FIELDS:
            raw[col] = self.get_required_float(col, label)
        for col in CATEGORICAL_COLS:
            val = self.combo_vars[col].get().strip()
            if val == "":
                raise FieldError(f"Please choose a value for '{col}'.", [self.combo_widgets[col]])
            raw[col] = val
        return raw

    def validate_composition_group(self, cols, mode="atmost", tolerance=COMPOSITION_TOLERANCE):
        """mode='equals': must sum to 100% (+/- tolerance).
        mode='atmost': must sum to no more than 100% (+ tolerance headroom)."""
        total = 0.0
        for c in cols:
            val = self.entries[c].get().strip()
            if val == "":
                raise FieldError(f"Please fill in '{c}'.", [self.entries[c]])
            try:
                total += float(val)
            except ValueError:
                raise FieldError(f"'{c}' must be a number.", [self.entries[c]])
        group = [self.entries[c] for c in cols]
        if mode == "equals" and abs(total - 100.0) > tolerance:
            raise FieldError(f"{' + '.join(cols)} must sum to 100% (currently {total:.2f}%).", group)
        if mode == "atmost" and total > 100.0 + tolerance:
            raise FieldError(f"{' + '.join(cols)} cannot exceed 100% (currently {total:.2f}%).", group)
        return total

    def validate_inputs(self):
        try:
            self.validate_composition_group(COMPOSITION_GROUP_1, mode="atmost")
            self.validate_composition_group(COMPOSITION_GROUP_2, mode="atmost")
            self.validate_composition_group(ELEMENTAL_GROUP, mode="atmost")
            self.gather_raw_inputs()  # raises on any other blank/invalid field
        except FieldError as e:
            return self.fail(e)
        except ValueError as e:
            return self.fail(FieldError(str(e)))
        return self.ok()

    # ================================================================= prediction
    def predict_with_trace(self, model_key, raw):
        bundle = self.cache.get(model_key)
        X = core.build_feature_row(bundle, raw)
        print(f"[{model_key}] encoded row -> {X.to_dict(orient='records')[0]}")
        y_scaled_pred = bundle["model"].predict(X.values)
        return core.inverse_target(bundle, y_scaled_pred)

    def run_predictions(self, keys=None):
        """Validate, then predict each requested model and write the outputs.
        Returns {model_key: float or None}, or None if validation failed."""
        if not self.validate_inputs():
            return None
        raw = self.gather_raw_inputs()
        results = {}
        for model_key in (keys or MODEL_ORDER):
            try:
                value = self.predict_with_trace(model_key, raw)
                self.output_vars[model_key].set(f"{value:.4f}")
                self.last_values[model_key] = value
                self.last_errors.pop(model_key, None)
                results[model_key] = value
            except Exception as exc:
                self.output_vars[model_key].set("Error")
                self.last_errors[model_key] = str(exc)
                results[model_key] = None
                print(f"[{model_key}] prediction failed: {exc}")
                traceback.print_exc()
        return results

    def predict_all(self):
        results = self.run_predictions()
        if results is not None:
            n_ok = sum(v is not None for v in results.values())
            if n_ok == len(results):
                self.notify("ok", f"Predicted all {n_ok} carbon descriptors.")
            else:
                bad = ", ".join(k for k, v in results.items() if v is None)
                self.notify("error", f"{n_ok}/{len(results)} predicted. Failed: {bad} "
                                     f"(check the model paths).")
        return results

    def predict_selected(self, model_key):
        results = self.run_predictions([model_key])
        if results is not None:
            if results.get(model_key) is None:
                self.notify("error", f"{model_key} failed: {self.last_errors.get(model_key, 'unknown error')}")
            else:
                self.notify("ok", f"Predicted {model_key}.")
        return results

    def reset_fields(self):
        self.clear_flags()
        for w in self.entries.values():
            w.config(state="normal")
            w.delete(0, tk.END)
        for widget in self.combo_widgets.values():
            if str(widget.cget("state")) != "disabled":
                widget.current(0)
        self.on_type_change()  # re-applies the frozen Step 2 state if Type resets to One step
        for key, cols in ((k, c) for k, _t, c in GROUPS):
            self.update_composition_total(cols, key)
        for var in self.output_vars.values():
            var.set("")
        self.last_values.clear()
        self.last_errors.clear()

    def check_model_paths(self, reload=True):
        if reload:
            self.cache.clear()
        lines = []
        for model_key in MODEL_ORDER:
            try:
                self.cache.get(model_key)
                lines.append(f"OK    Model 01 / {model_key}")
            except Exception as exc:
                lines.append(f"FAIL  Model 01 / {model_key}: {exc}")
        return lines

    # ================================================================= live behaviour
    def update_composition_total(self, cols, total_key, mode="atmost"):
        label = self.total_labels.get(total_key)
        if label is None:
            return
        total, ok = 0.0, True
        for c in cols:
            val = self.entries[c].get().strip()
            if val == "":
                ok = False
                continue
            try:
                total += float(val)
            except ValueError:
                ok = False
        if not ok:
            text, color = "\u03a3  \u2014", C.faint
        else:
            fine = (abs(total - 100.0) <= COMPOSITION_TOLERANCE) if mode == "equals" else (total <= 100.0 + COMPOSITION_TOLERANCE)
            if fine:
                text, color = f"\u03a3 {total:.1f}%  \u2713", C.success
            else:
                text, color = f"\u03a3 {total:.1f}%  exceeds 100%", C.danger
        self.total_texts[total_key] = text
        label.config(text=text, fg=color)

    def on_type_change(self, event=None):
        is_two_step = self.combo_vars["Type"].get() == "Two step"

        if self.step2_card is not None and self.step2_card.chip is not None:
            set_chip(self.step2_card.chip, "ACTIVE" if is_two_step else "LOCKED", "teal" if is_two_step else "muted")

        for col in ("Method_02", "Activator_02"):
            widget = self.combo_widgets[col]
            if is_two_step:
                widget.config(state="readonly")
                self.combo_vars[col].set("")
            else:
                widget.config(state="readonly")
                self.combo_vars[col].set("No")
                widget.config(state="disabled")

        # Step-2 numerics are fed to every model regardless of Type (Type does
        # not change which columns a trained model reads), so while frozen for
        # one-step they still need a value: 0 is the placeholder. A disabled
        # tk.Entry ignores programmatic insert/delete, so unlock, edit, then
        # re-apply the target state.
        for col in STEP2_NUMERIC:
            self.entries[col].config(state="normal")
            if is_two_step:
                if self.entries[col].get().strip() == "0":
                    self.entries[col].delete(0, tk.END)   # drop the one-step placeholder
            else:
                # ALWAYS reset to 0 when freezing - otherwise values typed during
                # a previous Two-step session would stay (invisibly, greyed out)
                # and be sent to the models for a one-step sample.
                self.entries[col].delete(0, tk.END)
                self.entries[col].insert(0, "0")
                self.entries[col].config(state="disabled")

    # ================================================================= cards
    def _dropdown(self, grid, col, label, span):
        var = tk.StringVar(master=self.master)
        combo = grid.dropdown(label, CATEGORICAL_LABEL_ORDER[col], var, span=span)
        combo.current(0)
        self.combo_vars[col] = var
        self.combo_widgets[col] = combo
        return combo

    def _card_composition(self, parent):
        card = Card(parent, "Biomass composition", rule=C.accent)
        g = FieldGrid(card.body)
        g.pack(fill="x")
        for key, title, cols in GROUPS:
            _left, right = g.subheader(title)
            self.total_labels[key] = right
            for c in cols:
                e = g.numeric(SHORT_LABEL[c])
                e.bind("<KeyRelease>", lambda ev, cols=cols, key=key: self.update_composition_total(cols, key), add="+")
                self.entries[c] = e
            self.update_composition_total(cols, key)
        return card

    def _card_route(self, parent):
        card = Card(parent, "Synthesis route", rule=C.accent)
        g = FieldGrid(card.body)
        g.pack(fill="x")
        type_combo = self._dropdown(g, "Type", "Type", 2)
        type_combo.bind("<<ComboboxSelected>>", self.on_type_change)
        self._dropdown(g, "Prewash", "Pre-wash", 4)
        self._dropdown(g, "Postwash", "Post-wash", 4)
        return card

    def _card_step(self, parent, n):
        chip_text, chip_kind = (("LOCKED", "muted") if n == 2 else (None, "accent"))
        card = Card(parent, f"Step {n} pyrolysis", chip_text, chip_kind, rule=C.accent)
        g = FieldGrid(card.body)
        g.pack(fill="x")
        for col, label in STEP_FIELDS[n]:
            self.entries[col] = g.numeric(label)
        self._dropdown(g, f"Method_0{n}", "Method", 2)
        self._dropdown(g, f"Activator_0{n}", "Activator", 4)
        if n == 2:
            self.step2_card = card
        return card

    def build_cards(self, parent):
        """Create the four input cards (not yet placed) in reading order."""
        cards = [
            self._card_composition(parent),
            self._card_route(parent),
            self._card_step(parent, 1),
            self._card_step(parent, 2),
        ]
        self.on_type_change()  # apply the initial One-step (frozen) defaults
        return cards

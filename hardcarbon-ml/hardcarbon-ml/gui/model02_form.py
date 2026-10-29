"""
Model 02 form: carbon descriptors + cell assembly + testing -> ICE, initial
capacity, slope capacity, HDR capacity.

The form owns the state and all validation / prediction logic;
`build_cards(parent)` creates its input cards for a page to arrange. The input
set is built at runtime from each target's real feature_names.pkl (see
model2_core.py), so it reflects whatever columns your saved models contain.

Behaviour (unchanged from the verified standalone version)
  - AM + conductive carbon + binder ratio must total 100%
  - Binder type / Solvent type freeze to "No Binder"/"No Solvent" while the
    Binder ratio is 0 or blank
  - electrolyte composition is pick -> Add -> fraction; fractions sum to 1.00
  - additive is checkbox-gated
  - "Current at 1C (A)" is the 1C reference: ICE/Capacity use 0.05C, Slope 1C
    (unconfirmed default), HDR is predicted at 0.5C/1C/2C
  - every numeric field is required; a console line shows each encoded row

Chain hook: apply_predicted_descriptors() writes Model 01 predictions into the
carbon-descriptor fields. They stay editable; filled fields are tinted so
predicted values are visually distinct from values you typed.
"""

import traceback
import numpy as np
import tkinter as tk
from tkinter import ttk

import model2_core as core
from form_base import FormBase, FieldError
from ui_kit import (C, F, D, px, Card, FieldGrid, set_chip, ghost_button, icon_button, make_entry,
                    make_combo, combo_values)

TARGET_ORDER = core.TARGET_ORDER

OUTPUT_TILES = [
    ("ICE", "ICE (%)", "blue"),
    ("Capacity", "Initial capacity (mAh/g)", "orange"),
    ("Slope", "Slope capacity (mAh/g)", "green"),
    ("HDR", "HDR capacity (mAh/g)", "rose"),
]

# Long names appear in validation messages; short ones on screen.
DISPLAY_NAME = {
    "Active material ratio": "Active material ratio (%)",
    "Carbon ratio": "Conductive carbon ratio (%)",
    "Binder ratio": "Binder ratio (%)",
    "Salt concentration": "Salt concentration (M)",
    "La": "La (nm)",
    "D-S": "D-Spacing (nm)",
    "Lc": "Lc (nm)",
    "SSA": "Surface area, SSA (m\u00b2/g)",
    "PV": "Pore volume (cm\u00b3/g)",
    "Carbon%": "Carbon% (elemental)",
    "IG/ID": "Raman IG/ID",
}
SHORT_NAME = {
    "Active material ratio": "Active material",
    "Carbon ratio": "Conductive C",
    "Binder ratio": "Binder",
    "Salt concentration": "Conc. (M)",
    "La": "La (nm)",
    "D-S": "D-spacing (nm)",
    "Lc": "Lc (nm)",
    "SSA": "SSA (m\u00b2/g)",
    "PV": "PV (cm\u00b3/g)",
    "Carbon%": "Carbon (%)",
    "IG/ID": "IG/ID",
}

# Model 02 group keys (from model2_core.SHAP_GROUP_DEFINITIONS) of the seven
# carbon descriptors, in display order.
DESCRIPTOR_GROUP_KEYS = ["La", "D-S", "Lc", "Surface area", "Pore volume", "C", "IG/ID"]

# Model 01 output name -> Model 02 descriptor group. Per the manuscript Model 02
# takes "the seven carbon structural descriptors from Model 01"; the two
# datasets were matched on exactly these seven properties, so units agree.
MODEL01_TO_MODEL02_GROUP = {
    "Carbon": "C",
    "D-Spacing": "D-S",
    "La": "La",
    "Lc": "Lc",
    "PV": "Pore volume",
    "SSA": "Surface area",
    "IG/ID": "IG/ID",
}

CURRENT_A_LABEL = "Current at 1C (A)"
ELECTRODE_TOTAL_TOLERANCE = 0.5          # percentage points either side of 100%
ELECTROLYTE_FRACTION_TOLERANCE = 0.01    # either side of 1.00


def format_c_rate(mult):
    return f"{int(mult)}C" if float(mult).is_integer() else f"{mult:g}C"


class Model02Form(FormBase):
    def __init__(self, master, cache=None):
        super().__init__(master)
        self.cache = cache if cache is not None else core.BundleCache(core.MODEL_CONFIGS)

        self.entries = {}             # feature_name -> Entry (plain numeric fields)
        self.onehot_vars = {}         # group_key -> StringVar (Binder/Solvent/Salt/Separator type)
        self.onehot_widgets = {}      # group_key -> Combobox
        self.current_entry = None
        self.output_vars = {k: tk.StringVar(master=master) for k, _l, _c in OUTPUT_TILES}
        self.last_values = {}
        self.last_errors = {}

        self.electrode_total_label = None
        self.electrode_total_text = ""
        self.electrolyte_rows = {}    # component -> (row_frame, fraction_entry)
        self.electrolyte_all_members = []
        self.electrolyte_picker_var = None
        self.electrolyte_picker_combo = None
        self.electrolyte_list_frame = None
        self.electrolyte_total_label = None
        self.electrolyte_total_text = ""

        self.additive_used_var = None
        self.additive_name_entry = None
        self.additive_pct_entry = None

        self.predicted_fields = set()  # descriptor entries currently holding Model 01 values
        self._filling = False
        self._tracked_vars = {}        # name -> StringVar; MUST stay referenced or tkinter garbage-collects
                                       # the variable (and the edit trace on it) as soon as the builder returns
        self.descriptor_chip = None

        self.load_errors = {}
        self.union_grouped = {}
        self._scan_models()

    # ================================================================= model scan
    def _scan_models(self):
        bundles, self.load_errors = self.cache.get_all(TARGET_ORDER)
        union_features, seen = [], set()
        for key in TARGET_ORDER:
            bundle = bundles.get(key)
            if bundle is None:
                continue
            for f in bundle["feature_names"]:
                if f in (core.CURRENT_MA_COL, core.CURRENT_LOG_COL):
                    continue
                if f not in seen:
                    seen.add(f)
                    union_features.append(f)
        self.union_grouped = core.build_shap_groups(union_features)
        self.electrolyte_all_members = list(self.union_grouped.get(core.ELECTROLYTE_TYPE_GROUP_KEY, []))
        self.needs_current_any = any(b["needs_current"] for b in bundles.values())

    # ================================================================= gathering / validation
    def get_required_float(self, name):
        label = DISPLAY_NAME.get(name, name)
        if name not in self.entries:
            raise FieldError(f"No input widget found for '{label}' - internal error.")
        entry = self.entries[name]
        val = entry.get().strip()
        if val == "":
            raise FieldError(f"'{label}' is required.", [entry])
        try:
            return float(val)
        except ValueError:
            raise FieldError(f"'{label}' must be a number.", [entry])

    def get_current_A(self):
        val = self.current_entry.get().strip()
        if val == "":
            raise FieldError(f"'{CURRENT_A_LABEL}' is required.", [self.current_entry])
        try:
            return float(val)
        except ValueError:
            raise FieldError(f"'{CURRENT_A_LABEL}' must be a number.", [self.current_entry])

    def validate_electrode_composition_total(self):
        """AM + Carbon + Binder = 100%. Checked once per predict action,
        independent of the target, since it describes the physical electrode."""
        members = []
        for key in core.ELECTRODE_COMPOSITION_GROUP_KEYS:
            members.extend(self.union_grouped.get(key, []))
        if not members:
            return
        total = 0.0
        for m in members:
            total += self.get_required_float(m)
        if abs(total - 100.0) > ELECTRODE_TOTAL_TOLERANCE:
            raise FieldError(
                f"Active material + Conductive carbon + Binder ratios must sum to 100% "
                f"(currently {total:.2f}%).", [self.entries[m] for m in members])

    def gather_raw_inputs_for_model(self, bundle, current_A_override=None):
        raw = {}
        grouped = bundle["grouped_features"]

        for group_key in core.ONE_HOT_DROPDOWN_ORDER:
            members = grouped.get(group_key, [])
            if len(members) > 1:
                var = self.onehot_vars.get(group_key)
                selected = var.get().strip() if var is not None else ""
                if selected == "":
                    widget = self.onehot_widgets.get(group_key)
                    raise FieldError(f"Please choose a {group_key}.", [widget] if widget is not None else [])
                for m in members:
                    raw[m] = 1.0 if m == selected else 0.0
            elif len(members) == 1:
                raw[members[0]] = self.get_required_float(members[0])

        for group_key in core.ELECTRODE_COMPOSITION_GROUP_KEYS:
            for m in grouped.get(group_key, []):
                raw[m] = self.get_required_float(m)

        handled = (
            set(core.ONE_HOT_DROPDOWN_GROUPS)
            | set(core.ELECTRODE_COMPOSITION_GROUP_KEYS)
            | {core.ELECTROLYTE_TYPE_GROUP_KEY, "Additive concentration", "Current density"}
        )
        for group_key, members in grouped.items():
            if group_key in handled:
                continue
            for m in members:
                raw[m] = self.get_required_float(m)

        electrolyte_members = grouped.get(core.ELECTROLYTE_TYPE_GROUP_KEY, [])
        if electrolyte_members:
            if not self.electrolyte_rows:
                raise FieldError("Please add at least one electrolyte component (Cell assembly card).",
                                 [self.electrolyte_picker_combo] if self.electrolyte_picker_combo is not None else [])
            total = 0.0
            fractions = {}
            for name, (_row, e) in self.electrolyte_rows.items():
                val = e.get().strip()
                if val == "":
                    raise FieldError(f"Please enter a fraction for electrolyte component '{name}'.", [e])
                try:
                    fval = float(val)
                except ValueError:
                    raise FieldError(f"Electrolyte fraction for '{name}' must be a number.", [e])
                fractions[name] = fval
                total += fval
            if abs(total - core.ELECTROLYTE_FRACTION_TOTAL) > ELECTROLYTE_FRACTION_TOLERANCE:
                raise FieldError(f"Electrolyte component fractions must sum to 1.00 (currently {total:.3f}).",
                                 [e for _r, e in self.electrolyte_rows.values()])
            for m in electrolyte_members:
                raw[m] = fractions.get(m, 0.0)

        additive_members = grouped.get("Additive concentration", [])
        if additive_members:
            m = additive_members[0]
            if self.additive_used_var is not None and self.additive_used_var.get():
                val = self.additive_pct_entry.get().strip()
                if val == "":
                    raise FieldError("Additive is ticked - please enter its concentration.", [self.additive_pct_entry])
                try:
                    raw[m] = float(val)
                except ValueError:
                    raise FieldError("Additive concentration must be a number.", [self.additive_pct_entry])
            else:
                raw[m] = 0.0

        if bundle["needs_current"]:
            if current_A_override is not None:
                current_A = current_A_override
            else:
                current_A = core.compute_target_current_A(bundle["target_key"], self.get_current_A())
                if isinstance(current_A, list):
                    raise FieldError(
                        f"{bundle['target_key']} uses multiple C-rates - use the dedicated "
                        f"multi-rate prediction path instead of a single gather call.")
            try:
                raw.update(core.resolve_current_inputs(bundle, current_A))
            except ValueError as e:
                raise FieldError(str(e), [self.current_entry])

        return raw

    def validate_and_gather(self, target_key):
        bundle = self.cache.get(target_key)
        try:
            self.validate_electrode_composition_total()
            raw = self.gather_raw_inputs_for_model(bundle)
        except FieldError as e:
            self.fail(e)
            return bundle, None
        except ValueError as e:
            self.fail(FieldError(str(e)))
            return bundle, None
        self.ok()
        return bundle, raw

    def _gather_hdr_all_rates(self):
        """HDR is reported at 0.5C/1C/2C: the entered current IS 1C, the other
        rates are scalar multiples of it. Returns [(rate, raw_dict)] or None
        (error already shown)."""
        bundle = self.cache.get("HDR")
        try:
            self.validate_electrode_composition_total()
            current_1C_A = self.get_current_A()
            results = []
            for rate_mult in core.TARGET_C_RATE_MULTIPLIERS["HDR"]:
                raw = self.gather_raw_inputs_for_model(bundle, current_A_override=current_1C_A * rate_mult)
                results.append((rate_mult, raw))
        except FieldError as e:
            self.fail(e)
            return None
        except ValueError as e:
            self.fail(FieldError(str(e)))
            return None
        self.ok()
        return results

    # ================================================================= prediction
    def predict_with_trace(self, target_key, raw):
        bundle = self.cache.get(target_key)
        X = core.build_feature_row(bundle, raw)
        print(f"[{target_key}] encoded row -> {X.to_dict(orient='records')[0]}")
        y_pred = bundle["model"].predict(X.values)
        return float(np.asarray(y_pred).reshape(-1)[0])

    def _predict_hdr_rates(self):
        gathered = self._gather_hdr_all_rates()
        if gathered is None:
            return None
        out = []
        for rate_mult, raw in gathered:
            try:
                out.append((rate_mult, self.predict_with_trace("HDR", raw)))
            except Exception as exc:
                out.append((rate_mult, None))
                self.last_errors["HDR"] = str(exc)
                print(f"[HDR @ {format_c_rate(rate_mult)}] prediction failed: {exc}")
                traceback.print_exc()
        return out

    @staticmethod
    def format_hdr(rates):
        return "\n".join(
            f"{format_c_rate(r)}: {v:.2f}" if v is not None else f"{format_c_rate(r)}: Error"
            for r, v in rates
        )

    def run_predictions(self, targets=None):
        """Predict each requested target and write the outputs. Returns
        {target: float | [(rate, float)] | None}, or None if validation failed."""
        results = {}
        for target_key in (targets or TARGET_ORDER):
            if target_key in self.load_errors:
                self.output_vars[target_key].set("N/A")
                results[target_key] = None
                continue
            if target_key == "HDR":
                rates = self._predict_hdr_rates()
                if rates is None:
                    return None
                self.output_vars["HDR"].set(self.format_hdr(rates))
                results["HDR"] = rates
                self.last_values["HDR"] = rates
                continue
            _bundle, raw = self.validate_and_gather(target_key)
            if raw is None:
                return None
            try:
                value = self.predict_with_trace(target_key, raw)
                self.output_vars[target_key].set(f"{value:.4f}")
                self.last_values[target_key] = value
                self.last_errors.pop(target_key, None)
                results[target_key] = value
            except Exception as exc:
                self.output_vars[target_key].set("Error")
                self.last_errors[target_key] = str(exc)
                results[target_key] = None
                print(f"[{target_key}] prediction failed: {exc}")
                traceback.print_exc()
        return results

    def predict_all(self):
        results = self.run_predictions()
        self._report(results)
        return results

    def predict_selected(self, target_key):
        results = self.run_predictions([target_key])
        self._report(results)
        return results

    def _report(self, results):
        if results is None:
            return
        failed = [k for k, v in results.items() if v is None]
        if failed:
            detail = "; ".join(f"{k}: {self.load_errors.get(k) or self.last_errors.get(k, 'failed')}" for k in failed)
            self.notify("error", f"Not predicted - {detail}")
        else:
            self.notify("ok", "Predicted " + ", ".join(results) + ".")

    def reset_fields(self):
        self.clear_flags()
        self._filling = True
        try:
            for w in self.entries.values():
                w.config(state="normal")
                w.delete(0, tk.END)
                w.config(bg=C.field)
        finally:
            self._filling = False
        self.predicted_fields.clear()
        if self.current_entry is not None:
            self.current_entry.delete(0, tk.END)
        for widget in self.onehot_widgets.values():
            widget.config(state="readonly")
            widget.current(0)
        for name in list(self.electrolyte_rows.keys()):
            self.remove_electrolyte_component(name)
        if self.additive_used_var is not None:
            self.additive_used_var.set(False)
            self.on_additive_toggle()
        self.on_binder_ratio_change()
        for var in self.output_vars.values():
            var.set("")
        self.last_values.clear()
        self.last_errors.clear()

    def check_model_paths(self, reload=True):
        if reload:
            self.cache.clear()
        lines = []
        if reload:
            print("\n" + "=" * 70)
            print("CHECK MODEL PATHS (Model 02)")
            print("=" * 70)
        for target_key in TARGET_ORDER:
            try:
                bundle = self.cache.get(target_key)
                lines.append(f"OK    Model 02 / {target_key}  ({len(bundle['feature_names'])} features, "
                             f"needs_current={bundle['needs_current']}, model={bundle['model_file']})")
                if reload:
                    print(f"\n[{target_key}] {len(bundle['feature_names'])} features -> {bundle['model_file']}")
                    for group_key, members in bundle["grouped_features"].items():
                        tag = " <- ONE-HOT DROPDOWN" if group_key in core.ONE_HOT_DROPDOWN_GROUPS and len(members) > 1 else ""
                        print(f"    {group_key:24s}: {members}{tag}")
            except Exception as exc:
                lines.append(f"FAIL  Model 02 / {target_key}: {exc}")
                if reload:
                    print(f"\n[{target_key}] FAILED TO LOAD: {exc}")

        all_found = set()
        for target_key in TARGET_ORDER:
            try:
                all_found |= set(self.cache.get(target_key)["grouped_features"].keys())
            except Exception:
                pass
        if reload:
            print("\nOptional sections (only shown if found in at least one loaded model):")
            for key in ["Additive concentration", core.ELECTROLYTE_TYPE_GROUP_KEY, "Current density"]:
                print(f"    {key:24s}: {'FOUND' if key in all_found else 'NOT FOUND - no widget for this in the form'}")
            print("=" * 70)
        return lines

    # ================================================================= chain hook
    def descriptor_members(self):
        """{model01_target: [model02 column names]} for descriptors present in this form."""
        out = {}
        for m01_key, group_key in MODEL01_TO_MODEL02_GROUP.items():
            members = [m for m in self.union_grouped.get(group_key, []) if m in self.entries]
            if members:
                out[m01_key] = members
        return out

    def apply_predicted_descriptors(self, predicted):
        """Write Model 01 predictions ({model01_target: float|None}) into the
        descriptor fields. Returns (filled, not_in_model02, missing_values).
        Filled fields are tinted and stay editable."""
        targets = self.descriptor_members()
        filled, not_in_model02, missing = [], [], []
        self._filling = True
        try:
            for m01_key, value in predicted.items():
                members = targets.get(m01_key)
                if value is None:
                    missing.append(m01_key)
                    continue
                if not members:
                    not_in_model02.append(m01_key)
                    continue
                for name in members:
                    entry = self.entries[name]
                    entry.config(state="normal")
                    entry.delete(0, tk.END)
                    entry.insert(0, f"{float(value):.6g}")
                    entry.config(bg=C.predicted)
                    self.predicted_fields.add(name)
                filled.append(m01_key)
        finally:
            self._filling = False
        if self.descriptor_chip is not None and filled:
            set_chip(self.descriptor_chip, "AUTO-FILLED", "warn")
        return filled, not_in_model02, missing

    def _on_tracked_edit(self, name):
        """A descriptor field changed. If the user typed it (not a programmatic
        fill), it is no longer a Model 01 prediction."""
        if self._filling:
            return
        if name in self.predicted_fields:
            self.predicted_fields.discard(name)
            self.entries[name].config(bg=C.field)

    # ================================================================= live behaviour
    def on_binder_ratio_change(self, event=None):
        binder_entry = self.entries.get("Binder ratio")
        binder_val = None
        if binder_entry is not None:
            try:
                binder_val = float(binder_entry.get().strip())
            except ValueError:
                binder_val = None
        is_used = binder_val is not None and binder_val > 0

        for group_key, no_choice_names in (
            (core.BINDER_TYPE_GROUP_KEY, core.NO_BINDER_LABEL_CANDIDATES),
            (core.SOLVENT_TYPE_GROUP_KEY, core.NO_SOLVENT_LABEL_CANDIDATES),
        ):
            widget = self.onehot_widgets.get(group_key)
            var = self.onehot_vars.get(group_key)
            if widget is None or var is None:
                continue
            available = combo_values(widget)
            if is_used:
                widget.config(state="readonly")
                var.set("")
            else:
                no_choice = next((v for v in available if v in no_choice_names), None)
                var.set(no_choice if no_choice is not None else (available[0] if available else ""))
                widget.config(state="readonly")
                widget.config(state="disabled")

        self.update_electrode_total()

    def update_electrode_total(self):
        label = self.electrode_total_label
        if label is None:
            return
        members = []
        for key in core.ELECTRODE_COMPOSITION_GROUP_KEYS:
            members.extend(self.union_grouped.get(key, []))
        total, ok = 0.0, bool(members)
        for m in members:
            val = self.entries[m].get().strip() if m in self.entries else ""
            if val == "":
                ok = False
                continue
            try:
                total += float(val)
            except ValueError:
                ok = False
        if not ok:
            text, color = "\u03a3  \u2014   (must be 100)", C.faint
        elif abs(total - 100.0) <= ELECTRODE_TOTAL_TOLERANCE:
            text, color = f"\u03a3 {total:.1f}%  \u2713", C.success
        else:
            text, color = f"\u03a3 {total:.1f}%  need 100", C.danger
        self.electrode_total_text = text
        label.config(text=text, fg=color)

    # ----------------------------------------------------------- electrolyte
    def refresh_electrolyte_picker_options(self):
        remaining = sorted(m for m in self.electrolyte_all_members if m not in self.electrolyte_rows)
        self.electrolyte_picker_combo["values"] = remaining
        self.electrolyte_picker_var.set(remaining[0] if remaining else "")
        self.electrolyte_picker_combo.config(state="readonly" if remaining else "disabled")

    def add_electrolyte_component(self):
        name = self.electrolyte_picker_var.get().strip()
        if not name or name in self.electrolyte_rows:
            return
        row = tk.Frame(self.electrolyte_list_frame, bg=C.surface)
        row.pack(fill="x", pady=(0, px(3)))
        tk.Label(row, text=name, font=F.base, fg=C.text, bg=C.surface, anchor="w").pack(side="left")
        icon_button(row, "\u00d7", lambda n=name: self.remove_electrolyte_component(n)).pack(side="right")
        e = make_entry(row)
        e.config(width=7)
        e.pack(side="right", padx=(px(6), px(4)))
        e.bind("<KeyRelease>", lambda ev: self.update_electrolyte_total(), add="+")
        self.electrolyte_rows[name] = (row, e)
        self.refresh_electrolyte_picker_options()
        self.update_electrolyte_total()
        e.focus_set()

    def remove_electrolyte_component(self, name):
        if name not in self.electrolyte_rows:
            return
        row, _e = self.electrolyte_rows.pop(name)
        row.destroy()
        self.refresh_electrolyte_picker_options()
        self.update_electrolyte_total()

    def update_electrolyte_total(self):
        label = self.electrolyte_total_label
        if label is None:
            return
        if not self.electrolyte_rows:
            text, color = "none added  (\u03a3 must be 1.00)", C.faint
        else:
            total, ok = 0.0, True
            for _name, (_row, e) in self.electrolyte_rows.items():
                val = e.get().strip()
                if val == "":
                    ok = False
                    continue
                try:
                    total += float(val)
                except ValueError:
                    ok = False
            if not ok:
                text, color = "\u03a3  \u2014", C.faint
            elif abs(total - core.ELECTROLYTE_FRACTION_TOTAL) <= ELECTROLYTE_FRACTION_TOLERANCE:
                text, color = f"\u03a3 {total:.3f}  \u2713", C.success
            else:
                text, color = f"\u03a3 {total:.3f}  need 1.00", C.danger
        self.electrolyte_total_text = text
        label.config(text=text, fg=color)

    # ----------------------------------------------------------- additive
    def on_additive_toggle(self):
        is_used = self.additive_used_var.get()
        for w in (self.additive_name_entry, self.additive_pct_entry):
            w.config(state="normal")
            if not is_used:
                w.delete(0, tk.END)
            w.config(state="normal" if is_used else "disabled")

    # ================================================================= cards
    def _label(self, name):
        return SHORT_NAME.get(name, DISPLAY_NAME.get(name, name))

    def _dropdown(self, grid, group_key, members, span):
        var = tk.StringVar(master=self.master)
        combo = grid.dropdown(group_key.replace(" type", ""), sorted(members), var, span=span)
        combo.current(0)
        self.onehot_vars[group_key] = var
        self.onehot_widgets[group_key] = combo
        return combo

    def _single_or_dropdown(self, grid, group_key, span):
        members = self.union_grouped.get(group_key, [])
        if len(members) > 1:
            self._dropdown(grid, group_key, members, span)
            return True
        if len(members) == 1:
            self.entries[members[0]] = grid.numeric(self._label(members[0]), span=2)
            return True
        return False

    def _card_descriptors(self, parent, chip_text, chip_kind, hint):
        members = [(g, m) for g in DESCRIPTOR_GROUP_KEYS for m in self.union_grouped.get(g, [])]
        if not members:
            return None
        card = Card(parent, "Carbon descriptors", chip_text, chip_kind, rule=C.teal)
        self.descriptor_chip = card.chip
        g = FieldGrid(card.body)
        g.pack(fill="x")
        for _group, name in members:
            var = tk.StringVar(master=self.master)
            self._tracked_vars[name] = var
            e = g.numeric(self._label(name), textvariable=var)
            var.trace_add("write", lambda *_a, n=name: self._on_tracked_edit(n))
            self.entries[name] = e
        if hint:
            g.note(hint, wrap=260)
        return card

    def _card_electrode(self, parent):
        ratio_members = [m for k in core.ELECTRODE_COMPOSITION_GROUP_KEYS for m in self.union_grouped.get(k, [])]
        has_types = any(self.union_grouped.get(k) for k in (core.BINDER_TYPE_GROUP_KEY, core.SOLVENT_TYPE_GROUP_KEY))
        if not ratio_members and not has_types:
            return None
        card = Card(parent, "Electrode preparation", rule=C.teal)
        g = FieldGrid(card.body)
        g.pack(fill="x")
        if ratio_members:
            _l, right = g.subheader("Composition \u00b7 %")
            self.electrode_total_label = right
            for group_key in core.ELECTRODE_COMPOSITION_GROUP_KEYS:
                for m in self.union_grouped.get(group_key, []):
                    e = g.numeric(self._label(m))
                    e.bind("<KeyRelease>", self.on_binder_ratio_change if m == "Binder ratio"
                           else (lambda ev: self.update_electrode_total()), add="+")
                    self.entries[m] = e
            g.subheader("Binder system")
        for group_key in (core.BINDER_TYPE_GROUP_KEY, core.SOLVENT_TYPE_GROUP_KEY):
            self._single_or_dropdown(g, group_key, 3)
        return card

    def _card_cell(self, parent):
        grouped = self.union_grouped
        has = any(grouped.get(k) for k in ("Salt type", "Salt concentration", "Separator type")) \
            or self.electrolyte_all_members or grouped.get("Additive concentration")
        if not has:
            return None
        card = Card(parent, "Cell assembly", rule=C.teal)
        g = FieldGrid(card.body)
        g.pack(fill="x")

        if grouped.get("Salt type"):
            self._single_or_dropdown(g, "Salt type", 4)
        for member in grouped.get("Salt concentration", []):
            self.entries[member] = g.numeric(self._label(member), span=2)
        if grouped.get("Separator type"):
            self._single_or_dropdown(g, "Separator type", 4)

        if self.electrolyte_all_members:
            _l, self.electrolyte_total_label = g.subheader("Electrolyte \u00b7 fractions")
            picker = g.full(pady=(0, px(4)))
            self.electrolyte_picker_var = tk.StringVar(master=self.master)
            self.electrolyte_picker_combo = make_combo(picker, sorted(self.electrolyte_all_members),
                                                       self.electrolyte_picker_var)
            self.electrolyte_picker_combo.pack(side="left", fill="x", expand=True)
            ghost_button(picker, "+ Add", self.add_electrolyte_component).pack(side="left", padx=(px(6), 0))
            self.electrolyte_list_frame = g.full(pady=(0, px(2)))
            self.refresh_electrolyte_picker_options()
            self.update_electrolyte_total()

        if grouped.get("Additive concentration"):
            g.subheader("Additive")
            self.additive_used_var = tk.BooleanVar(master=self.master, value=False)
            box = g.full(pady=(0, px(D.row_gap)))
            ttk.Checkbutton(box, text="Additive used", variable=self.additive_used_var, style="Card.TCheckbutton",
                            command=self.on_additive_toggle).pack(anchor="w")
            self.additive_name_entry = g.numeric("Name (note only)", span=3)
            self.additive_pct_entry = g.numeric("Concentration (%)", span=3)
            self.additive_name_entry.config(state="disabled")
            self.additive_pct_entry.config(state="disabled")
        return card

    def _card_testing(self, parent):
        card = Card(parent, "Testing conditions", rule=C.teal)
        g = FieldGrid(card.body)
        g.pack(fill="x")
        self.current_entry = g.numeric(CURRENT_A_LABEL, span=3)
        g.note("Enter the current that IS 1C for this cell. ICE and initial capacity use 0.05C, slope "
               "uses 1C (unconfirmed), HDR is predicted at 0.5C, 1C and 2C.", wrap=270)
        return card

    def build_cards(self, parent, descriptor_chip=("MEASURED", "muted"), descriptor_hint=None):
        """Create the input cards (not yet placed) in reading order."""
        cards = [
            self._card_descriptors(parent, descriptor_chip[0], descriptor_chip[1], descriptor_hint),
            self._card_electrode(parent),
            self._card_cell(parent),
            self._card_testing(parent),
        ]
        cards = [c for c in cards if c is not None]
        self.on_binder_ratio_change()  # apply the initial frozen state now that all widgets exist
        return cards

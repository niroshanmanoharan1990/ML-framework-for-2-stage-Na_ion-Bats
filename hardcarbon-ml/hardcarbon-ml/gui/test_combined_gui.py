"""
Functional GUI tests against REAL tkinter (not a stub).

Run:   xvfb-run -a -s "-screen 0 1280x720x24 -dpi 96" python3 test_combined_gui.py
(on a normal desktop just:  python test_combined_gui.py)
"""

import math
import os
import time

import hard_carbon_core as core01
import model2_core as core02
from model02_form import MODEL01_TO_MODEL02_GROUP
from ui_kit import C
from test_support import (DIALOGS, PREDICT, RUN_CHAIN, RUN_TILE, REMOVE, make_app, pump, find_all,
                          find_button, click, put, type_into, pick, state_of, flagged, status, ok, fill_m01,
                          fill_m02_cell, screenshot)
from tkinter import ttk

SHOT_DIR = os.environ.get("SHOT_DIR", "")

app = make_app(auto_check=False)
root = app.root
root.focus_force()
pump(root)

print("=== A: shell ===")
assert [k for k, _n, _t in app.PAGES] == ["m1", "m2", "m3"]
for key in ("m1", "m2", "m3"):
    app.show(key); pump(root, 3)
    assert app.hosts[key].winfo_ismapped()
    assert all(not h.winfo_ismapped() for k, h in app.hosts.items() if k != key), "hidden pages must be un-mapped"
    assert not app.hosts[key].needs_v and not app.hosts[key].needs_h, f"{key} needs scrolling at the test size"
ok("three pages; only the selected one is mapped; none needs a scrollbar at 1280x680")

p1, p2, p3 = app.pages["m1"], app.pages["m2"], app.pages["m3"]
for key in ("m3", "m1", "m2"):
    app.show(key)
    assert status(app) == ("info", app.pages[key].hint)
ok("switching pages puts that page's guidance in the status bar (and clears any message left by another page)")
assert p2.form.entries["La"] is not p3.m02.entries["La"] and p1.form.entries["Ash%"] is not p3.m01.entries["Ash%"]
assert p1.form.cache is p3.m01.cache and p2.form.cache is p3.m02.cache
ok("Mode 02/03 and Mode 01/03 forms have independent widgets but share one model cache each")
assert set(p3.results.tiles) == set(core02.TARGET_ORDER), "Mode 03 must show only the Model 02 outputs"
assert len(p1.results.tiles) == 7 and len(p2.results.tiles) == 4
ok("Mode 03 shows only the 4 Model 02 tiles (no Model 01 readout); Mode 01 shows 7, Mode 02 shows 4")

print("\n=== B: Mode 01 ===")
app.show("m1"); pump(root, 3)
f1 = p1.form
assert state_of(f1.entries["Time_02"]) == "disabled" and f1.entries["Time_02"].get() == "0"
assert state_of(f1.combo_widgets["Method_02"]) == "disabled" and f1.combo_vars["Method_02"].get() == "No"
assert f1.step2_card.chip.cget("text") == "LOCKED"
ok("Step 2 starts frozen: disabled, Method_02='No', numerics='0', chip says LOCKED")

click(root, p1, PREDICT)
kind, text = status(app)
assert kind == "error" and "Please fill in 'Cellulose%'" in text, (kind, text)
assert flagged(f1.entries["Cellulose%"]) and not flagged(f1.entries["Ash%"])
assert all(v.get() == "" for v in f1.output_vars.values())
assert not [d for d in DIALOGS if d[0] == "error"], "validation must not open a dialog"
ok(f"blank Predict -> red outline on Cellulose, status bar: {text!r}, no dialog, no outputs")

e = f1.entries["Cellulose%"]
e.focus_force(); pump(root, 1)
e.event_generate("<Key>", keysym="4"); pump(root, 1)
assert not flagged(e)
ok("typing into a flagged field clears its red outline")

fill_m01(root, f1)
assert "\u2713" in f1.total_labels["group1"].cget("text")
click(root, p1, PREDICT)
vals = {k: float(v.get()) for k, v in f1.output_vars.items()}
assert set(vals) == set(core01.MODEL_ORDER) and all(math.isfinite(v) for v in vals.values())
assert 55 < vals["Carbon"] < 100
assert status(app)[0] == "ok" and "all 7" in status(app)[1]
for k, tile in p1.results.tiles.items():
    assert tile.value_label.cget("text") == f1.output_vars[k].get()
ok("valid inputs -> 7 finite descriptors shown in the tiles: " + ", ".join(f"{k}={v:.3g}" for k, v in vals.items()))

for v in f1.output_vars.values():
    v.set("")
find_button(p1.results.tiles["SSA"], RUN_TILE).invoke(); pump(root)
assert f1.output_vars["SSA"].get() != "" and all(f1.output_vars[k].get() == "" for k in f1.output_vars if k != "SSA")
assert status(app) == ("ok", "Predicted SSA.")
ok("per-tile \u25b6 runs just that target (SSA) and leaves the others empty")

type_into(root, f1.entries["Cellulose%"], 70)
assert "exceeds 100%" in f1.total_labels["group1"].cget("text") and str(f1.total_labels["group1"].cget("fg")).lower() == C.danger
click(root, p1, PREDICT)
assert "cannot exceed 100%" in status(app)[1]
assert all(flagged(f1.entries[c]) for c in ("Cellulose%", "Hemi-cellulose%", "Lignin%")) and not flagged(f1.entries["Ash%"])
type_into(root, f1.entries["Cellulose%"], 40)
ok("composition cap: live total turns red; Predict outlines exactly the three offending fields")

pick(root, f1.combo_widgets["Type"], "Two step")
assert state_of(f1.entries["Time_02"]) == "normal" and f1.entries["Time_02"].get() == ""
assert state_of(f1.combo_widgets["Method_02"]) == "readonly" and f1.combo_vars["Method_02"].get() == ""
assert f1.step2_card.chip.cget("text") == "ACTIVE"
click(root, p1, PREDICT)
assert "step 2" in status(app)[1].lower() and flagged(f1.entries["Time_02"])
ok("Two step unlocks Step 2 (chip ACTIVE); blank Step 2 is rejected and Time_02 is outlined")

put(f1.entries["Time_02"], 30); put(f1.entries["Heating rate_02"], 5); put(f1.entries["Temperature_02"], 600)
pick(root, f1.combo_widgets["Method_02"], "Pyrolysis")
pick(root, f1.combo_widgets["Activator_02"], "Mild Acid + Base Mix")
click(root, p1, PREDICT)
assert all(math.isfinite(float(v.get())) for v in f1.output_vars.values())
ok("filled Step 2 (incl. the longest dropdown label) predicts")

pick(root, f1.combo_widgets["Type"], "One step")
assert [f1.entries[c].get() for c in ("Time_02", "Heating rate_02", "Temperature_02")] == ["0", "0", "0"]
assert state_of(f1.entries["Time_02"]) == "disabled" and f1.step2_card.chip.cget("text") == "LOCKED"
ok("back to One step re-freezes AND resets Step 2 to 0 (no stale two-step values reach the models)")

click(root, p1, "Reset")
assert status(app) == ("info", p1.hint)
assert all(v.get() == "" for v in f1.output_vars.values()) and f1.entries["Cellulose%"].get() == ""
assert f1.entries["Time_02"].get() == "0" and "\u2014" in f1.total_labels["group1"].cget("text")
assert all(t.value_label.cget("text") == "\u2014" for t in p1.results.tiles.values())
ok("Reset clears inputs and tiles and re-freezes Step 2")

print("\n=== C: Mode 02 ===")
app.show("m2"); pump(root, 3)
f2 = p2.form
assert state_of(f2.onehot_widgets["Binder type"]) == "disabled" and f2.onehot_vars["Binder type"].get() == "No Binder"
assert f2.onehot_vars["Solvent type"].get() == "No Solvent"
ok("Binder/Solvent type start frozen to 'No Binder'/'No Solvent'")

type_into(root, f2.entries["Binder ratio"], 10)
assert state_of(f2.onehot_widgets["Binder type"]) == "readonly" and f2.onehot_vars["Binder type"].get() == ""
type_into(root, f2.entries["Binder ratio"], 0)
assert state_of(f2.onehot_widgets["Binder type"]) == "disabled" and f2.onehot_vars["Binder type"].get() == "No Binder"
ok("Binder ratio > 0 unlocks and = 0 re-freezes Binder/Solvent type through the real <KeyRelease> binding")

type_into(root, f2.entries["Active material ratio"], 80)
type_into(root, f2.entries["Carbon ratio"], 10)
type_into(root, f2.entries["Binder ratio"], 5)
assert "need 100" in f2.electrode_total_label.cget("text")
type_into(root, f2.entries["Binder ratio"], 10)
assert "100.0%" in f2.electrode_total_label.cget("text") and "\u2713" in f2.electrode_total_label.cget("text")
ok("electrode total label: " + f2.electrode_total_label.cget("text"))

click(root, p2, PREDICT)
assert "Please choose a Binder type" in status(app)[1] and flagged(f2.onehot_widgets["Binder type"])
ok("Binder ratio > 0 with no Binder type -> the dropdown is outlined and named in the status bar")

f2.reset_fields()
fill_m02_cell(root, p2, f2, with_descriptors=False)
click(root, p2, PREDICT)
assert "'La (nm)' is required" in status(app)[1] and flagged(f2.entries["La"])
ok("everything filled except descriptors -> La is outlined, status names it")

for name, val in {"La": 3.1, "D-S": 0.385, "Lc": 1.2, "SSA": 40, "PV": 0.05, "Carbon%": 92, "IG/ID": 1.1}.items():
    put(f2.entries[name], val)
assert "\u03a3 1.000" in f2.electrolyte_total_label.cget("text")
assert "EC fraction" not in list(f2.electrolyte_picker_combo.cget("values"))
ok("electrolyte: added components leave the picker; total reads " + f2.electrolyte_total_label.cget("text"))

type_into(root, f2.electrolyte_rows["DMC fraction"][1], 0.3)
assert "need 1.00" in f2.electrolyte_total_label.cget("text")
click(root, p2, PREDICT)
assert "sum to 1.00" in status(app)[1] and all(flagged(e) for _r, e in f2.electrolyte_rows.values())
type_into(root, f2.electrolyte_rows["DMC fraction"][1], 0.5)
ok("fractions not summing to 1 are rejected and both fraction fields are outlined")

check = [c for c in find_all(p2, ttk.Checkbutton) if c.cget("text") == "Additive used"][0]
assert state_of(f2.additive_pct_entry) == "disabled"
check.invoke(); pump(root)
assert state_of(f2.additive_pct_entry) == "normal"
click(root, p2, PREDICT)
assert "Additive is ticked" in status(app)[1] and flagged(f2.additive_pct_entry)
put(f2.additive_pct_entry, 2.5)
assert f2.gather_raw_inputs_for_model(f2.cache.get("ICE"))["Additive concentration"] == 2.5
check.invoke(); pump(root)
assert state_of(f2.additive_pct_entry) == "disabled" and f2.additive_pct_entry.get() == ""
ok("additive checkbox gates the % field (ticked+blank rejected and outlined, 2.5 passes through, unticking clears)")

click(root, p2, PREDICT)
out = {k: v.get() for k, v in f2.output_vars.items()}
for k in ("ICE", "Capacity", "Slope"):
    assert math.isfinite(float(out[k])) and p2.results.tiles[k].value_label.cget("text") == out[k]
mini = p2.results.tiles["HDR"].mini
assert [r.cget("text") for r, _v in mini] == ["0.5C", "1C", "2C"]
assert all(math.isfinite(float(v.cget("text"))) for _r, v in mini)
ok("Predict -> ICE/Capacity/Slope tiles and HDR mini-stats " + " | ".join(f"{r.cget('text')} {v.cget('text')}" for r, v in mini))

assert math.isclose(f2.gather_raw_inputs_for_model(f2.cache.get("ICE"))["Current density"], 0.3 * 0.05)
assert math.isclose(f2.gather_raw_inputs_for_model(f2.cache.get("Slope"))["log10 current density"], math.log10(300))
rates = f2._gather_hdr_all_rates()
assert all(math.isclose(r["log10 current density"], math.log10(x)) for (_m, r), x in zip(rates, (150, 300, 600)))
ok("C-rate scaling: entered 0.3 A = 1C -> ICE 0.05C, Slope 1C, HDR 0.5C/1C/2C (150/300/600 mA)")

for v in f2.output_vars.values():
    v.set("")
find_button(p2.results.tiles["Capacity"], RUN_TILE).invoke(); pump(root)
assert f2.output_vars["Capacity"].get() != "" and f2.output_vars["ICE"].get() == ""
ok("per-tile \u25b6 works in Mode 02 too")

click(root, f2.electrolyte_list_frame, REMOVE)
assert len(f2.electrolyte_rows) == 1
ok("the \u00d7 button removes an electrolyte component")

calls = []
orig = p2.primary_action
p2.primary_action = lambda: calls.append(1)
f2.current_entry.focus_force(); pump(root, 1)
f2.current_entry.event_generate("<Return>"); pump(root, 1)
n_entry = len(calls)
find_button(p2, "Reset").focus_force(); pump(root, 1)
find_button(p2, "Reset").event_generate("<Return>"); pump(root, 1)
p2.primary_action = orig
assert n_entry == 1 and len(calls) == 1, (n_entry, calls)
ok("Enter in a text field runs the page's main action; Enter on a button does not")

click(root, p2, "Reset")
assert not f2.electrolyte_rows and all(v.get() == "" for v in f2.output_vars.values())
assert state_of(f2.onehot_widgets["Binder type"]) == "disabled" and f2.entries["La"].get() == ""
ok("Reset clears everything and re-freezes Binder/Solvent type")

print("\n=== D: Mode 03 chain ===")
app.show("m3"); pump(root, 3)
s1, s2 = p3.m01, p3.m02
fill_m01(root, s1)
fill_m02_cell(root, p3, s2, with_descriptors=False)

find_button(p3, "Predict only").invoke(); pump(root)
assert "'La (nm)' is required" in status(app)[1] and flagged(s2.entries["La"])
ok("Predict only before filling descriptors -> La is outlined and named")

find_button(p3, "Fill descriptors").invoke(); pump(root)
assert status(app)[0] == "ok" and "7/7" in status(app)[1]
expected = {}
for m01_key, group_key in MODEL01_TO_MODEL02_GROUP.items():
    name = s2.union_grouped[group_key][0]
    expected[name] = s1.last_values[m01_key]
    assert math.isclose(float(s2.entries[name].get()), expected[name], rel_tol=1e-5)
    assert str(s2.entries[name].cget("bg")).lower() == C.predicted.lower()
assert len(s2.predicted_fields) == 7 and s2.descriptor_chip.cget("text") == "AUTO-FILLED"
ok("Fill descriptors wrote all 7 Model 01 predictions into Model 02's fields (amber, chip AUTO-FILLED): "
   + ", ".join(f"{k}={float(s2.entries[k].get()):.4g}" for k in expected))
assert app.pages["m2"].form.entries["La"].get() == ""
ok("the Mode 02 tab's fields were not touched")

put(s2.entries["SSA"], 950); pump(root)
assert str(s2.entries["SSA"].cget("bg")).lower() == C.field.lower() and "SSA" not in s2.predicted_fields
ok("overriding one descriptor clears its tint (6 predicted + 1 entered)")

find_button(p3, "Predict only").invoke(); pump(root)
assert all(v.get() != "" for v in s2.output_vars.values())
assert s2.gather_raw_inputs_for_model(s2.cache.get("ICE"))["SSA"] == 950
assert "6 descriptor(s) from Model 01 and 1 entered" in status(app)[1], status(app)
ok("Predict only used the edited SSA=950 (hybrid run). Status: " + status(app)[1])

click(root, p3, "Reset all")
assert s1.entries["Cellulose%"].get() == "" and s2.entries["SSA"].get() == "" and not s2.predicted_fields
assert str(s2.entries["La"].cget("bg")).lower() == C.field.lower()
fill_m01(root, s1); fill_m02_cell(root, p3, s2)
click(root, p3, RUN_CHAIN)
assert all(v.get() != "" for v in s2.output_vars.values()) and len(s2.predicted_fields) == 7
assert "0 entered" in status(app)[1] and status(app)[0] == "ok"
ok("Run chain: Model 01 -> descriptor hand-off -> Model 02 outputs in one click")

calls = []
orig = p3.primary_action
p3.primary_action = lambda: calls.append(1)
s1.entries["Ash%"].focus_force(); pump(root, 1)
s1.entries["Ash%"].event_generate("<Return>"); pump(root, 1)
p3.primary_action = orig
assert calls == [1]
ok("Enter in Mode 03 triggers the chain")

# failure path: the SSA model becomes unavailable
click(root, p3, "Reset all")
fill_m01(root, s1); fill_m02_cell(root, p3, s2)
real_get = s1.cache.get
s1.cache.get = lambda k: (_ for _ in ()).throw(RuntimeError("SSA model file missing")) if k == "SSA" else real_get(k)
click(root, p3, RUN_CHAIN)
s1.cache.get = real_get
kind, text = status(app)
assert kind == "error" and "FAILED" in text and "SSA" in text
assert s2.entries["SSA"].get() == "" and len(s2.predicted_fields) == 6
assert all(v.get() == "" for v in s2.output_vars.values()), "Model 02 must not run on a partial chain"
ok("one failed Model 01 target: its field stays blank, the status bar names it, Model 02 is NOT run on a partial chain")

print("\n=== E: model status ===")
app.check_models(show_dialog=False, reload=True)
assert "11/11" in app.models_pill.cget("text") and status(app)[0] == "ok"
n = len(DIALOGS)
app.models_pill.event_generate("<Button-1>"); pump(root)
assert DIALOGS[n][0] == "info" and "OK    Model 01 / Carbon" in DIALOGS[n][2] and "Model 02 / HDR" in DIALOGS[n][2]
ok("header pill reports 11/11 and opens the per-model report when clicked")
root.destroy()

app2 = make_app(auto_check=True)
for _ in range(40):
    app2.root.update(); time.sleep(0.01)
assert "11/11" in app2.models_pill.cget("text")
ok("the startup check runs by itself and turns the pill green")

if SHOT_DIR:
    os.makedirs(SHOT_DIR, exist_ok=True)
    root2 = app2.root
    app2.show("m1"); fill_m01(root2, app2.pages["m1"].form); app2.pages["m1"].primary_action()
    app2.show("m2"); fill_m02_cell(root2, app2.pages["m2"], app2.pages["m2"].form, with_descriptors=True); app2.pages["m2"].primary_action()
    app2.show("m3")
    fill_m01(root2, app2.pages["m3"].m01); fill_m02_cell(root2, app2.pages["m3"], app2.pages["m3"].m02)
    app2.pages["m3"].run_chain()
    for key in ("m1", "m2", "m3"):
        app2.show(key); pump(root2, 3)
        screenshot(root2, os.path.join(SHOT_DIR, f"{key}.png"))
app2.root.destroy()

print("\nALL FUNCTIONAL GUI TESTS PASSED")

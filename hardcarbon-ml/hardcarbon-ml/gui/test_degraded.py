"""
Degraded-setup tests (real tkinter): what a first launch looks like when paths are
wrong or a model folder is missing. Nothing should crash, and the screen should
say what is wrong.

Run:  xvfb-run -a -s "-screen 0 1280x720x24 -dpi 96" python3 test_degraded.py
"""

import os
import shutil
import tempfile

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

import hard_carbon_core as core01
import model2_core as core02
from ui_kit import C
from test_support import (DIALOGS, PREDICT, RUN_CHAIN, make_app, pump, click, find_button, put, type_into, state_of,
                          status, ok, fill_m01, fill_m02_cell, fixture_dirs)

tmp = tempfile.mkdtemp(prefix="suite_deg_")

print("=== 1: every path wrong (first launch before editing the paths) ===")
bad01 = {k: os.path.join(tmp, "nope01", k.replace("/", "_")) for k in core01.MODEL_ORDER}
bad02 = {k: dict(v, save_dir=os.path.join(tmp, "nope02", k)) for k, v in core02.MODEL_CONFIGS.items()}
app = make_app(cache01=core01.BundleCache(bad01), cache02=core02.BundleCache(bad02), auto_check=True)
root = app.root
for _ in range(10):
    root.update()
import time
time.sleep(0.15)
for _ in range(10):
    root.update()
assert len(app.pages["m2"].form.load_errors) == 4
assert "0/11" in app.models_pill.cget("text") and str(app.models_pill.cget("fg")).lower() == C.danger.lower()
kind, text = status(app)
assert kind == "error" and "No models could be loaded" in text
ok(f"app opens; header pill reads {app.models_pill.cget('text').split(chr(9679))[-1].strip()!r} in red; status bar: {text[:60]}...")

app.show("m1"); pump(root, 3)
fill_m01(root, app.pages["m1"].form)
click(root, app.pages["m1"], PREDICT)
assert all(v.get() == "Error" for v in app.pages["m1"].form.output_vars.values())
kind, text = status(app)
assert kind == "error" and "0/7 predicted" in text, (kind, text)
tile = app.pages["m1"].results.tiles["Carbon"]
assert tile.value_label.cget("text") == "Error" and str(tile.value_label.cget("fg")).lower() == C.danger.lower()
ok("Mode 01 Predict with no models -> every tile reads 'Error' in red, status explains, no exception")

app.show("m2"); pump(root, 3)
click(root, app.pages["m2"], PREDICT)
assert all(v.get() == "N/A" for v in app.pages["m2"].form.output_vars.values())
assert status(app)[0] == "error"
ok("Mode 02 Predict with no models -> 'N/A' everywhere, error in the status bar")

app.show("m3"); pump(root, 3)
p3 = app.pages["m3"]
fill_m01(root, p3.m01)
click(root, p3, RUN_CHAIN)
kind, text = status(app)
assert kind == "error" and "FAILED" in text
assert all(v.get() == "" for v in p3.m02.output_vars.values())
ok("Mode 03 chain with no models -> status says FAILED; Model 02 is not run")
n = len(DIALOGS)
app.models_pill.event_generate("<Button-1>"); pump(root)
assert DIALOGS[n][0] == "info" and DIALOGS[n][2].count("FAIL") == 11
ok("clicking the pill lists all 11 failures")
root.destroy()

print("\n=== 2: only the HDR folder missing ===")
d1, d2, _ = fixture_dirs()
good02 = {k: dict(v) for k, v in core02.MODEL_CONFIGS.items()}
for k in good02:
    good02[k]["save_dir"] = d2[k]["save_dir"]
broken = tempfile.mkdtemp(prefix="suite_hdr_")
shutil.copytree(os.path.dirname(good02["HDR"]["save_dir"]), os.path.join(broken, "m02"))
shutil.rmtree(os.path.join(broken, "m02", "m02_HDR"))
for k in good02:
    good02[k]["save_dir"] = os.path.join(broken, "m02", f"m02_{k}")
app = make_app(cache02=core02.BundleCache(good02), auto_check=True)
root = app.root
time.sleep(0.15)
for _ in range(10):
    root.update()
assert list(app.pages["m2"].form.load_errors) == ["HDR"]
assert "10/11" in app.models_pill.cget("text") and str(app.models_pill.cget("fg")).lower() == C.warn.lower()
ok("form builds from the 3 targets that loaded; pill is amber: " + app.models_pill.cget("text").split(chr(9679))[-1].strip())

app.show("m3"); pump(root, 3)
p3 = app.pages["m3"]
fill_m01(root, p3.m01)
fill_m02_cell(root, p3, p3.m02)
click(root, p3, RUN_CHAIN)
out = p3.m02.output_vars
assert out["HDR"].get() == "N/A" and all(out[k].get() not in ("", "N/A", "Error") for k in ("ICE", "Capacity", "Slope"))
kind, text = status(app)
assert kind in ("warn", "error") and "HDR" in text
ok("chain runs: ICE/Capacity/Slope predicted, HDR tile reads N/A. Status: " + text[:90])
root.destroy()

print("\n=== 3: Model 02 feature set with NO electrolyte, additive, current or one-hot columns ===")
minimal = {k: dict(v) for k, v in core02.MODEL_CONFIGS.items()}
names = ["La", "D-S", "Lc", "SSA", "PV", "Carbon%", "IG/ID", "Active material ratio", "Carbon ratio", "Binder ratio"]
for k in minimal:
    d = os.path.join(tmp, "min02_" + k)
    os.makedirs(d)
    X = np.random.default_rng(0).uniform(0, 1, size=(40, len(names)))
    joblib.dump(names, os.path.join(d, "feature_names.pkl"))
    joblib.dump(GradientBoostingRegressor(n_estimators=10).fit(X, X.sum(1)),
                os.path.join(d, core02.MODEL_CONFIGS[k]["model_filename_candidates"][0]))
    minimal[k]["save_dir"] = d
app = make_app(cache02=core02.BundleCache(minimal))
root = app.root
app.show("m2"); pump(root, 3)
p2 = app.pages["m2"]
f2 = p2.form
assert f2.electrolyte_picker_combo is None and f2.additive_used_var is None and not f2.onehot_widgets
for name, v in {"La": 3, "D-S": 0.38, "Lc": 1.2, "SSA": 50, "PV": 0.1, "Carbon%": 90, "IG/ID": 1.1,
                "Active material ratio": 90, "Carbon ratio": 5}.items():
    put(f2.entries[name], v)
type_into(root, f2.entries["Binder ratio"], 5)
find_button(p2.results.tiles["ICE"], "\u25b6").invoke(); pump(root)
assert f2.output_vars["ICE"].get() not in ("", "Error"), status(app)
ok("a minimal feature set builds (no Cell/electrolyte/additive widgets) and predicts")
click(root, p2, "Reset")
assert f2.entries["La"].get() == ""
ok("Reset works with that minimal form")
assert state_of(f2.current_entry) == "normal"
root.destroy()

print("\nALL DEGRADED-SETUP TESTS PASSED")

"""
Helpers shared by the real-tkinter test files.

Dialogs (messagebox) are recorded instead of shown so nothing blocks. Validation
problems no longer use dialogs at all: they appear as red field outlines plus a
message in the status bar, which the tests read back.
"""

import os
import sys
import tempfile
import tkinter as tk
from tkinter import messagebox

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import hard_carbon_core as core01
import model2_core as core02
import test_fixtures as fx
from combined_app import CombinedApp
from ui_kit import C

DIALOGS = []


def _recorder(kind):
    def f(title=None, message=None, **kw):
        DIALOGS.append((kind, title, message))
        return "ok"
    return f


def install_dialog_recorder():
    messagebox.showerror = _recorder("error")
    messagebox.showinfo = _recorder("info")
    messagebox.showwarning = _recorder("warning")


install_dialog_recorder()

PREDICT = "Predict  \u25b6"
RUN_CHAIN = "Run chain  \u25b6"
RUN_TILE = "\u25b6"
REMOVE = "\u00d7"

_FIXTURE_DIR = None


def fixture_dirs():
    """Build the synthetic model folders once per process."""
    global _FIXTURE_DIR
    if _FIXTURE_DIR is None:
        tmp = tempfile.mkdtemp(prefix="suite_fx_")
        _FIXTURE_DIR = (fx.build_model01_fixtures(os.path.join(tmp, "m01")),
                        fx.build_model02_fixtures(os.path.join(tmp, "m02")), tmp)
    return _FIXTURE_DIR


def make_app(density="compact", geometry="1280x680+0+0", family=None, dpi=1.0, auto_check=False,
             cache01=None, cache02=None):
    d1, d2, _tmp = fixture_dirs()
    root = tk.Tk()
    # Tk stores display scaling per DISPLAY, so a value set by an earlier test would leak into this
    # root: always set it explicitly (this is also what Windows display scaling does to Tk).
    root.tk.call("tk", "scaling", dpi * 96 / 72)
    app = CombinedApp(cache01=cache01 or core01.BundleCache(d1), cache02=cache02 or core02.BundleCache(d2),
                      root=root, density=density, geometry=geometry, family=family, auto_check=auto_check)
    pump(root)
    return app


# ------------------------------------------------------------------ events / widgets
def pump(root, n=2):
    for _ in range(n):
        root.update_idletasks()
        root.update()


def find_all(widget, cls):
    out, stack = [], [widget]
    while stack:
        w = stack.pop()
        if isinstance(w, cls):
            out.append(w)
        stack.extend(w.winfo_children())
    return out


def find_button(scope, text):
    for b in find_all(scope, tk.Button):
        if b.cget("text") == text:
            return b
    raise AssertionError(f"button {text!r} not found")


def click(root, scope, text):
    find_button(scope, text).invoke()
    pump(root)


def put(entry, text):
    """Programmatic edit that also works on a disabled entry."""
    prev = str(entry.cget("state"))
    entry.config(state="normal")
    entry.delete(0, "end")
    entry.insert(0, str(text))
    entry.config(state=prev)


def type_into(root, entry, text):
    """Like typing: set the text, then fire <KeyRelease> so live bindings run.
    The entry's page must be the visible one."""
    put(entry, text)
    entry.focus_force()
    pump(root, 1)
    entry.event_generate("<KeyRelease>")
    pump(root, 1)


def pick(root, combo, value):
    combo.set(value)
    combo.event_generate("<<ComboboxSelected>>")
    pump(root)


def state_of(w):
    return str(w.cget("state"))


def flagged(w):
    """True if the widget is showing the inline error outline."""
    if hasattr(w, "cget") and str(w.winfo_class()) == "TCombobox":
        return str(w.cget("style")) == "Err.TCombobox"
    return str(w.cget("highlightbackground")).lower() == C.danger.lower()


def status(app):
    """(kind, text) as currently shown in the status bar."""
    colour = str(app.status.dot.cget("fg")).lower()
    kind = {C.faint: "info", C.success: "ok", C.warn: "warn", C.danger: "error"}.get(colour, colour)
    return kind, app.status.text_var.get()


def ok(msg):
    print(f"OK - {msg}")


# ------------------------------------------------------------------ data
M01_VALID = {"Cellulose%": 40, "Hemi-cellulose%": 25, "Lignin%": 25,
             "Volatile Matter%": 70, "Ash%": 2, "Fixed Carbon%": 20,
             "Carbon%": 45, "Oxygen%": 42, "Hydrogen%": 6,
             "Time_01": 60, "Heating rate_01": 5, "Temperature_01": 1000}

DESCRIPTOR_VALUES = {"La": 3.1, "D-S": 0.385, "Lc": 1.2, "SSA": 40, "PV": 0.05, "Carbon%": 92, "IG/ID": 1.1}


def fill_m01(root, form):
    for col, val in M01_VALID.items():
        type_into(root, form.entries[col], val)


def fill_m02_cell(root, page, form, with_descriptors=False, electrolytes=(("EC fraction", 0.5), ("DMC fraction", 0.5))):
    """Everything in Model 02 except (optionally) the carbon descriptors."""
    if with_descriptors:
        for name, val in DESCRIPTOR_VALUES.items():
            put(form.entries[name], val)
    type_into(root, form.entries["Active material ratio"], 80)
    type_into(root, form.entries["Carbon ratio"], 10)
    type_into(root, form.entries["Binder ratio"], 10)
    form.onehot_vars["Binder type"].set("PVDF")
    form.onehot_vars["Solvent type"].set("NMP")
    form.onehot_vars["Salt type"].set("NaPF6")
    form.onehot_vars["Separator type"].set("GF")
    put(form.entries["Salt concentration"], 1.0)
    for comp, frac in electrolytes:
        form.electrolyte_picker_var.set(comp)
        click(root, page, "+ Add")
        type_into(root, form.electrolyte_rows[comp][1], frac)
    put(form.current_entry, 0.3)
    pump(root)


def screenshot(root, path):
    try:
        from PIL import ImageGrab
        pump(root)
        x, y, w, h = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(path)
        print(f"   screenshot -> {os.path.basename(path)}")
    except Exception as exc:
        print(f"   (screenshot skipped: {exc})")

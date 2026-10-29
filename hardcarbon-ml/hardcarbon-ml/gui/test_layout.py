"""
Layout tests (real tkinter): "everything fits one page" is checked, not assumed.

For every configuration (window size x font x display scaling) and every mode:
  * the page needs no scrollbar
  * no widget is clipped by its parent (a generic geometry check, which is what
    would catch a button cut off at a card edge)
and for the busiest realistic state (Two step, four electrolyte components,
additive on) the Mode 03 page must still fit.

Run:  xvfb-run -a -s "-screen 0 1920x1080x24 -dpi 96" python3 test_layout.py
Set SHOT_DIR to also save a screenshot per configuration and mode.
"""

import os
import tkinter as tk

import ui_kit as K
from test_support import make_app, pump, put, pick, ok, screenshot

SHOT_DIR = os.environ.get("SHOT_DIR", "")

CONFIGS = [
    # name, window (physical px), density, font, display scaling
    ("1280x720 screen, compact, Arial-like font", "1280x680", "compact", "Liberation Sans", 1.0),
    ("1280x720 screen, compact, very wide font", "1280x680", "compact", "DejaVu Sans", 1.0),
    ("1366x768 screen, compact", "1366x728", "compact", "Liberation Sans", 1.0),
    ("1536x864 screen (125% of 1080p), compact", "1536x824", "compact", "Liberation Sans", 1.0),
    ("1920x1080 screen, comfortable", "1920x1040", "comfortable", "Liberation Sans", 1.0),
    ("1920x1080 screen, comfortable, very wide font", "1920x1040", "comfortable", "DejaVu Sans", 1.0),
    ("Windows 125% display scaling on 1080p", "1920x1030", "compact", "Liberation Sans", 1.25),
    ("Windows 150% display scaling on 1080p", "1920x1020", "compact", "Liberation Sans", 1.5),
]


def clipped_widgets(page, tol=2):
    """Widgets that extend beyond their parent's visible rectangle."""
    bad = []
    stack = [page]
    while stack:
        parent = stack.pop()
        if not parent.winfo_ismapped():
            continue
        px0, py0 = parent.winfo_rootx(), parent.winfo_rooty()
        px1, py1 = px0 + parent.winfo_width(), py0 + parent.winfo_height()
        for w in parent.winfo_children():
            if not w.winfo_ismapped() or w.winfo_width() <= 1 or w.winfo_height() <= 1:
                continue
            x0, y0 = w.winfo_rootx(), w.winfo_rooty()
            x1, y1 = x0 + w.winfo_width(), y0 + w.winfo_height()
            if x0 < px0 - tol or y0 < py0 - tol or x1 > px1 + tol or y1 > py1 + tol:
                bad.append((w.winfo_class(), str(w), (x0, y0, x1, y1), (px0, py0, px1, py1)))
            stack.append(w)
    return bad


def check_pages(app, label):
    problems = []
    for key in ("m1", "m2", "m3"):
        app.show(key)
        pump(app.root, 5)
        host, page = app.hosts[key], app.pages[key]
        if host.needs_v or host.needs_h:
            problems.append(f"{key}: needs scrollbar (page {page.winfo_reqwidth()}x{page.winfo_reqheight()} in "
                            f"{host.canvas.winfo_width()}x{host.canvas.winfo_height()})")
        bad = clipped_widgets(page)
        if bad:
            problems.append(f"{key}: {len(bad)} clipped widget(s), first: {bad[0]}")
        if SHOT_DIR:
            os.makedirs(SHOT_DIR, exist_ok=True)
            tag = "".join(ch if ch.isalnum() else "_" for ch in label)[:48]
            screenshot(app.root, os.path.join(SHOT_DIR, f"{tag}_{key}.png"))
    return problems


def stress_mode3(app):
    """Two step + four electrolyte components + additive on: the busiest realistic Mode 03 page."""
    app.show("m3")
    pump(app.root, 3)
    page = app.pages["m3"]
    m1, m2 = page.m01, page.m02
    pick(app.root, m1.combo_widgets["Type"], "Two step")
    m1.combo_vars["Prewash"].set("Moderate Acid wash")
    m1.combo_vars["Postwash"].set("Moderate Alkali wash")
    m1.combo_vars["Method_02"].set("Pyrolysis")
    m1.combo_vars["Activator_02"].set("Mild Acid + Base Mix")
    for comp in ("EC fraction", "DMC fraction", "Aqueous", "Organic"):
        m2.electrolyte_picker_var.set(comp)
        m2.add_electrolyte_component()
    m2.additive_used_var.set(True)
    m2.on_additive_toggle()
    put(m2.additive_name_entry, "FEC")
    pump(app.root, 5)


problems_total = 0
for name, geometry, density, font, dpi in CONFIGS:
    app = make_app(density=density, geometry=geometry + "+0+0", family=font, dpi=dpi)
    problems = check_pages(app, name)
    stress_mode3(app)
    host = app.hosts["m3"]
    if host.needs_v or host.needs_h:
        problems.append(f"m3 stress state needs a scrollbar "
                        f"(page {app.pages['m3'].winfo_reqwidth()}x{app.pages['m3'].winfo_reqheight()} in "
                        f"{host.canvas.winfo_width()}x{host.canvas.winfo_height()})")
    bad = clipped_widgets(app.pages["m3"])
    if bad:
        problems.append(f"m3 stress state: clipped widget {bad[0]}")
    cols = app.pages["m3"].ncols
    if problems:
        problems_total += len(problems)
        print(f"FAIL - {name}  (Mode 03 uses {cols} columns)")
        for p in problems:
            print("       " + p)
    else:
        ok(f"{name}: all three modes fit one page, nothing clipped, Mode 03 stress state fits "
           f"({cols} card columns)")
    app.root.destroy()

print()
# density rule: compact on small/laptop screens, comfortable on large ones
for screen_h, expected in ((720, "compact"), (768, "compact"), (864, "compact"), (900, "comfortable"),
                           (1080, "comfortable"), (1440, "comfortable")):
    root = tk.Tk()
    root.tk.call("tk", "scaling", 96 / 72)      # 100% (Tk keeps scaling per display, see test_support)
    root.winfo_screenheight = lambda h=screen_h: h
    got = K.init_theme(root)
    root.destroy()
    assert got == expected, (screen_h, got, expected)
ok("auto density: compact up to 864px screen height, comfortable from 900px (720/768/864 -> compact; 900/1080/1440 -> comfortable)")

assert problems_total == 0, f"{problems_total} layout problem(s)"
print("\nALL LAYOUT TESTS PASSED")

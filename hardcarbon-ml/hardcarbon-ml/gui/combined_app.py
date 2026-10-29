"""
Hard Carbon Prediction Suite - Mode 01 / Mode 02 / Mode 03 in one window
=========================================================================
Run this file:    python combined_app.py

  Mode 01  Biomass & synthesis inputs        -> 7 carbon descriptors      (Model 01)
  Mode 02  Measured descriptors + cell inputs -> ICE, capacity, slope, HDR  (Model 02)
  Mode 03  Biomass & synthesis + cell inputs  -> descriptors -> performance
           (Model 01 predictions fill Model 02's descriptor fields)

Every mode is one page: input cards on the left, predictions on the right.

Mode 03
-------
  Run chain         Model 01 predicts the seven descriptors, writes them into the
                    "Carbon descriptors" card (tinted yellow) and Model 02 predicts.
  Fill descriptors  Only the Model 01 step. Descriptors stay editable: type a
                    measured value into any of them (its tint clears), then...
  Predict only      ...runs Model 02 on whatever the descriptor fields now hold,
                    so you can mix measured and predicted descriptors.

Paths to your saved models are configured where they always were:
  hard_carbon_core.MODEL_SAVE_DIRS   (Model 01, seven folders)
  model2_core.MODEL_CONFIGS          (Model 02, four folders)

Mode 03 passes Model 01's *predicted* descriptors to a model trained on
*measured* ones, so Model 01's errors propagate into Model 02. Treat Mode 03
output as a screening estimate.

Enter runs the current page's main action.
"""

import os
import sys

# --- pre-flight -----------------------------------------------------------------------------
# This file imports its sibling modules, so they must all sit in the same folder. The usual way
# that goes wrong is double-clicking/running combined_app.py from INSIDE the zip: Windows then
# extracts only that one file to a temp folder. Say so plainly instead of a bare ImportError.
_REQUIRED = ("hard_carbon_core.py", "model2_core.py", "ui_kit.py", "form_base.py",
             "model01_form.py", "model02_form.py")
_HERE = os.path.dirname(os.path.abspath(__file__))
_MISSING = [f for f in _REQUIRED if not os.path.exists(os.path.join(_HERE, f))]
if _MISSING:
    sys.exit(
        "\ncombined_app.py cannot start: these files are not next to it:\n  "
        + "\n  ".join(_MISSING)
        + f"\n\nIt is running from:\n  {_HERE}\n\n"
        "All the .py files must be in ONE folder. If you opened it from inside the zip, extract the\n"
        "whole zip first (right-click the zip -> Extract All), then run combined_app.py from the\n"
        "extracted hard_carbon_suite folder.\n")

import tkinter as tk
from tkinter import ttk, messagebox

import hard_carbon_core as core01
import model2_core as core02
from model01_form import Model01Form, OUTPUT_TILES as TILES01
from model02_form import Model02Form, OUTPUT_TILES as TILES02
import ui_kit as K
from ui_kit import C, F, D, px, ResultsPanel, SegmentedControl, StatusBar, AutoScrollFrame, partition

MAX_CONTENT_WIDTH = 1560


def _tiles(spec, multi=("HDR",)):
    return [dict(key=k, label=l, color=c, multi=(k in multi)) for k, l, c in spec]


# ============================================================================ pages
class BasePage(tk.Frame):
    """A page = a grid of input cards (auto-balanced into columns) next to a
    results panel."""

    RESULTS_WIDTH = 250
    hint = ""

    def __init__(self, parent, app):
        super().__init__(parent, bg=C.bg)
        self.app = app
        self.outer = tk.Frame(self, bg=C.bg)
        self.outer.pack(fill="both", expand=True, padx=px(14), pady=px(12))
        self.outer.grid_columnconfigure(0, weight=1)
        self.outer.grid_rowconfigure(0, weight=1)
        self.cards_area = tk.Frame(self.outer, bg=C.bg)
        self.cards_area.grid(row=0, column=0, sticky="nsew")
        self.columns = []

    def layout_cards(self, cards, ncols):
        """Distribute cards (in reading order) over `ncols` columns so the tallest
        column is as short as possible, then place them. Safe to call again."""
        self._cards = cards
        for c in cards:
            c.pack_forget()
        for col in self.columns:
            col.destroy()
        self.columns = []
        for i in range(8):
            self.cards_area.grid_columnconfigure(i, weight=0, uniform="", minsize=0)
        self.update_idletasks()
        heights = [c.winfo_reqheight() for c in cards]
        groups = partition(heights, ncols)
        for gi, idxs in enumerate(groups):
            col = tk.Frame(self.cards_area, bg=C.bg)
            col.grid(row=0, column=gi, sticky="new", padx=(0 if gi == 0 else px(D.gap), 0))
            # equal *extra* space, not equal widths: forcing equal widths would make every
            # column as wide as the widest card and waste ~130px on narrower screens
            self.cards_area.grid_columnconfigure(gi, weight=1)
            self.columns.append(col)
            for i in idxs:
                cards[i].pack(in_=col, fill="x", pady=(0, px(D.gap)))
                cards[i].lift()
        self.update_idletasks()

    def fit_columns(self, cards, max_cols, avail_px):
        """Use as many columns as fit the available width (more columns = shorter page)."""
        for n in range(max_cols, 1, -1):
            self.layout_cards(cards, n)
            if self.winfo_reqwidth() <= avail_px:
                return n
        return 2

    def place_results(self, panel):
        self.results = panel
        panel.grid(in_=self.outer, row=0, column=1, sticky="new", padx=(px(D.gap), 0))
        panel.lift()
        self.outer.grid_columnconfigure(1, weight=0, minsize=px(self.RESULTS_WIDTH))

    # subclasses implement
    def primary_action(self):
        raise NotImplementedError

    def forms(self):
        return []


class Mode1Page(BasePage):
    hint = "Mode 01: enter the biomass and synthesis details, then press Predict (or Enter)."

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.form = Model01Form(self, cache=app.cache01)
        self.form.notify = app.notify
        cards = self.form.build_cards(self.cards_area)
        self.layout_cards(cards, ncols=2)
        self.place_results(ResultsPanel(
            self.outer, "Predicted descriptors", "Model 01 \u00b7 biomass \u2192 7 carbon descriptors",
            _tiles(TILES01, multi=()), self.form.output_vars,
            primary=("Predict  \u25b6", self.form.predict_all),
            secondary=[("Reset", self.reset)],
            on_tile_run=self.form.predict_selected, width=self.RESULTS_WIDTH, row_tiles=True))

    def reset(self):
        self.form.reset_fields()
        self.app.notify("info", self.hint)

    def primary_action(self):
        self.form.predict_all()

    def forms(self):
        return [self.form]


class Mode2Page(BasePage):
    hint = "Mode 02: enter your measured carbon descriptors and the cell details, then press Predict (or Enter)."

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.form = Model02Form(self, cache=app.cache02)
        self.form.notify = app.notify
        cards = self.form.build_cards(self.cards_area, descriptor_chip=("MEASURED", "muted"))
        self.layout_cards(cards, ncols=2)
        self.place_results(ResultsPanel(
            self.outer, "Predicted performance", "Model 02 \u00b7 carbon + cell \u2192 4 electrochemical targets",
            _tiles(TILES02), self.form.output_vars,
            primary=("Predict  \u25b6", self.form.predict_all),
            secondary=[("Reset", self.reset)],
            on_tile_run=self.form.predict_selected, width=self.RESULTS_WIDTH))

    def reset(self):
        self.form.reset_fields()
        self.app.notify("info", self.hint)

    def primary_action(self):
        self.form.predict_all()

    def forms(self):
        return [self.form]


class Mode3Page(BasePage):
    hint = ("Mode 03: fill both stages, then Run chain. Model 01 fills the carbon descriptors, "
            "Model 02 predicts the performance.")

    def __init__(self, parent, app, max_cols=4, avail_px=1300):
        super().__init__(parent, app)
        self.m01 = Model01Form(self, cache=app.cache01)
        self.m02 = Model02Form(self, cache=app.cache02)
        self.m01.notify = app.notify
        self.m02.notify = app.notify
        cards = (self.m01.build_cards(self.cards_area)
                 + self.m02.build_cards(self.cards_area, descriptor_chip=("FROM MODEL 01", "accent"),
                                        descriptor_hint="Filled by Model 01. Edit any value to override it."))
        # The results panel is placed first so the width test below includes it.
        self.place_results(ResultsPanel(
            self.outer, "Predicted performance", "Model 01 fills the descriptors, Model 02 predicts",
            _tiles(TILES02), self.m02.output_vars,
            primary=("Run chain  \u25b6", self.run_chain),
            secondary=[[("Fill descriptors", self.fill_descriptors), ("Predict only", self.predict_performance)],
                       [("Reset all", self.reset_all)]],
            on_tile_run=self.m02.predict_selected,
            footnote="Predicted descriptors carry Model 01's error into Model 02: a screening estimate.",
            legend=[(C.accent, "Model 01 inputs"), (C.teal, "Model 02 inputs")],
            width=self.RESULTS_WIDTH))
        self.ncols = self.fit_columns(cards, max_cols, avail_px)

    def primary_action(self):
        self.run_chain()

    def forms(self):
        return [self.m01, self.m02]

    def fill_descriptors(self):
        """Step 1: Model 01 -> descriptor fields. True if all seven were produced."""
        values = self.m01.run_predictions()
        if values is None:
            return False
        filled, not_in_model02, missing = self.m02.apply_predicted_descriptors(values)

        text = (f"Model 01 predicted {len(values) - len(missing)}/{len(values)} descriptors; "
                f"{len(filled)} written into the descriptor card.")
        if not_in_model02:
            text += f" Not used by your Model 02 feature set: {', '.join(not_in_model02)}."
        if missing:
            detail = "; ".join(f"{k}: {self.m01.last_errors.get(k, 'failed')}" for k in missing)
            self.app.notify("error", f"{text} FAILED: {detail}. Enter those by hand, or fix the model.")
            return False
        self.app.notify("ok", text)
        return True

    def predict_performance(self):
        """Step 2: Model 02 on whatever the descriptor fields hold."""
        results = self.m02.run_predictions()
        if results is None:
            return False
        total = sum(len(v) for v in self.m02.descriptor_members().values())
        from_model01 = len(self.m02.predicted_fields)
        failed = [k for k, v in results.items() if v is None]
        text = (f"Model 02 run with {from_model01} descriptor(s) from Model 01 and {total - from_model01} "
                f"entered by you.")
        if failed:
            self.app.notify("warn", f"{text} Not predicted: {', '.join(failed)}.")
        else:
            self.app.notify("ok", text)
        return not failed

    def run_chain(self):
        if self.fill_descriptors():
            self.predict_performance()

    def reset_all(self):
        self.m01.reset_fields()
        self.m02.reset_fields()
        self.app.notify("info", self.hint)


# ============================================================================ app shell
class CombinedApp:
    PAGES = [("m1", "01", "Biomass \u2192 Carbon"), ("m2", "02", "Carbon \u2192 Battery"),
             ("m3", "03", "Biomass \u2192 Battery")]

    def __init__(self, cache01=None, cache02=None, root=None, density=None, geometry=None, auto_check=True,
                 family=None):
        if root is None:
            K.enable_dpi_awareness()
            root = tk.Tk()
        self.root = root
        self.root.title("Hard Carbon Prediction Suite")
        self.density = K.init_theme(self.root, density, family=family)

        screen_w = self.root.winfo_screenwidth() / K.SCALE
        screen_h = self.root.winfo_screenheight() / K.SCALE
        if geometry is None:
            w = min(screen_w - 40, MAX_CONTENT_WIDTH + 30)
            h = min(screen_h - 100, 860)
            geometry = f"{int(w * K.SCALE)}x{int(h * K.SCALE)}+{int(20 * K.SCALE)}+{int(10 * K.SCALE)}"
        self.root.geometry(geometry)
        self.root.minsize(px(900), px(520))

        self.cache01 = cache01 if cache01 is not None else core01.BundleCache(core01.MODEL_SAVE_DIRS)
        self.cache02 = cache02 if cache02 is not None else core02.BundleCache(core02.MODEL_CONFIGS)

        self._build_header()
        self.status = StatusBar(self.root)
        self.status.pack(fill="x", side="bottom")
        self.container = tk.Frame(self.root, bg=C.bg)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        # Show the window now: building the pages loads the Model 02 files, which can take a few
        # seconds, and a blank window that does not repaint looks like a crash.
        self.notify("info", "Starting \u2013 loading the saved models\u2026")
        self.root.update()

        # Mode 03 holds both stages: it uses as many card columns (up to four) as the window width allows.
        window_w = int(geometry.split("x")[0])
        self.hosts, self.pages = {}, {}
        for key, builder in (("m1", lambda p: Mode1Page(p, self)),
                             ("m2", lambda p: Mode2Page(p, self)),
                             ("m3", lambda p: Mode3Page(p, self, max_cols=4, avail_px=window_w))):
            host = AutoScrollFrame(self.container, max_width=MAX_CONTENT_WIDTH)
            host.grid(row=0, column=0, sticky="nsew")
            page = builder(host.inner)
            page.pack(fill="both", expand=True)
            self.hosts[key], self.pages[key] = host, page

        self.current = None
        self.last_check = []
        self.show("m1")
        for host in self.hosts.values():
            host.refresh()
        self.root.bind("<Return>", self._on_return)
        self._check_id = None
        if auto_check:
            self.notify("info", "Loading models\u2026")
            self._check_id = self.root.after(80, lambda: self.check_models(show_dialog=False, reload=False))
        self.root.bind("<Destroy>", self._on_destroy, add="+")

    def _on_destroy(self, event):
        if event.widget is self.root and self._check_id:
            try:
                self.root.after_cancel(self._check_id)
            except tk.TclError:
                pass

    # ------------------------------------------------------------ chrome
    def _build_header(self):
        header = tk.Frame(self.root, bg=C.surface)
        header.pack(fill="x", side="top")
        header.grid_columnconfigure(1, weight=1)

        brand = tk.Frame(header, bg=C.surface)
        brand.grid(row=0, column=0, sticky="w", padx=(px(14), 0), pady=px(8))
        tk.Label(brand, text="HC", font=F.title, fg="#ffffff", bg=C.accent, padx=px(7), pady=px(3)).pack(side="left")
        tk.Label(brand, text="Hard Carbon Suite", font=F.app_title, fg=C.text, bg=C.surface).pack(side="left", padx=px(10))

        self.nav = SegmentedControl(header, self.PAGES, self.show)
        self.nav.grid(row=0, column=1)

        self.models_pill = tk.Label(header, text="\u25cf  Models: checking\u2026", font=F.chip, fg=C.muted, bg=C.disabled,
                                    padx=px(9), pady=px(4), cursor="hand2")
        self.models_pill.grid(row=0, column=2, sticky="e", padx=(0, px(14)))
        self.models_pill.bind("<Button-1>", lambda e: self.check_models(show_dialog=True, reload=True))
        tk.Frame(self.root, bg=C.border, height=1).pack(fill="x", side="top")

    def show(self, key):
        """Show one page. Hidden pages are un-mapped (not just covered) so the Tab key can
        never move focus into a field you cannot see."""
        self.current = key
        self.nav.select(key)
        for k, host in self.hosts.items():
            if k == key:
                host.grid()
            else:
                host.grid_remove()
        self.hosts[key].refresh()
        self.notify("info", self.pages[key].hint)

    def notify(self, kind, text):
        self.status.set(kind, text)

    def _on_return(self, event=None):
        # Enter on a button / dropdown / checkbox has its own meaning; only text fields run the page.
        if event is not None and isinstance(event.widget, (tk.Button, ttk.Combobox, ttk.Checkbutton)):
            return
        self.pages[self.current].primary_action()

    # ------------------------------------------------------------ model status
    def check_models(self, show_dialog=True, reload=True):
        page1, page2 = self.pages["m1"], self.pages["m2"]
        lines = page1.form.check_model_paths(reload=reload) + page2.form.check_model_paths(reload=reload)
        ok = sum(l.startswith("OK") for l in lines)
        total = len(lines)
        if ok == total:
            self.models_pill.config(text=f"\u25cf  Models ready  {ok}/{total}", fg=C.success, bg=C.success_soft)
            self.notify("ok", f"All {total} models loaded.")
        elif ok == 0:
            self.models_pill.config(text=f"\u25cf  No models found  0/{total}", fg=C.danger, bg=C.danger_soft)
            self.notify("error", "No models could be loaded - check hard_carbon_core.MODEL_SAVE_DIRS and "
                                 "model2_core.MODEL_CONFIGS.")
        else:
            self.models_pill.config(text=f"\u25cf  Models  {ok}/{total}", fg=C.warn, bg=C.warn_soft)
            self.notify("warn", f"{total - ok} of {total} models could not be loaded - click the Models pill for details.")
        self.last_check = lines
        if show_dialog:
            messagebox.showinfo("Model artifact check", "\n".join(lines))
        return lines

    # ------------------------------------------------------------ run
    def run(self):
        self.root.mainloop()


def main():
    CombinedApp().run()


if __name__ == "__main__":
    main()

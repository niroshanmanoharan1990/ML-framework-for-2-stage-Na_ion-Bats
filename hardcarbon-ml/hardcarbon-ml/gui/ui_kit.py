"""
ui_kit.py - design system for the Hard Carbon Prediction Suite.

Everything visual lives here so the forms and pages stay free of styling code:

  * colour tokens (C), fonts (F), spacing/density (D), DPI-aware px()
  * init_theme(root)  - fonts, ttk styles, combobox popup styling
  * Card, FieldGrid   - the building blocks of every input form
  * OutputTile, ResultsPanel - the prediction readouts
  * SegmentedControl, StatusBar, AutoScrollFrame, chips and buttons

Design rules used throughout
  * one accent colour; semantic colours only for state (ok / warn / error)
  * white cards on a pale grey canvas, 1px borders, no shadows
  * labels sit above inputs (small, muted); inputs are flat with a focus ring
  * the layout is sized so each mode fits one window; AutoScrollFrame only
    shows scrollbars if a small screen genuinely cannot hold it
"""

import sys
import types
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

SCALE = 1.0


def px(n):
    """Scale a design pixel to the current display scaling."""
    return max(1, int(round(n * SCALE)))


def enable_dpi_awareness():
    """Windows only: ask for crisp (non-bitmap-scaled) rendering. Must run
    before the first Tk() is created. Harmless elsewhere."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# ------------------------------------------------------------------ tokens
class C:
    bg = "#f2f4f9"
    surface = "#ffffff"
    border = "#e0e4ee"
    border_strong = "#c8cfdf"
    text = "#1a2036"
    muted = "#626d86"
    faint = "#9aa3b8"
    accent = "#3b5bdb"
    accent_hover = "#2e49b8"
    accent_soft = "#e6ebff"
    teal = "#0e8a8c"
    teal_soft = "#dbf3f1"
    success = "#168a5c"
    success_soft = "#dff4ea"
    warn = "#a8680a"
    warn_soft = "#fff1cf"
    danger = "#cf3a3a"
    danger_soft = "#fde8e8"
    field = "#ffffff"
    disabled = "#edf0f6"
    predicted = "#fff0b3"
    tile_colors = {
        "blue": "#2f6fed", "orange": "#e8710a", "green": "#1f9d57", "rose": "#d6336c",
        "violet": "#7048e8", "teal": "#0e8a8c", "amber": "#c98a00",
    }


F = types.SimpleNamespace()   # fonts, filled by init_theme
D = types.SimpleNamespace()   # spacing / density, filled by init_theme

DENSITY = {
    "comfortable": dict(base=11, label=9, title=12, sub=8, value=26, value_small=17, button=11, chip=8,
                        pad=16, pad_top=14, gap=12, field_gap=12, row_gap=9, entry_bd=5, col_min=54, tile_pad=11),
    "compact": dict(base=9, label=8, title=10, sub=8, value=17, value_small=13, button=9, chip=7,
                    pad=10, pad_top=8, gap=8, field_gap=7, row_gap=4, entry_bd=3, col_min=40, tile_pad=6),
}


def _pick_family(root):
    avail = set(tkfont.families(root))
    for fam in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Roboto", "Arial", "Liberation Sans",
                "DejaVu Sans"):
        if fam in avail:
            return fam
    return tkfont.nametofont("TkDefaultFont").actual("family")


def init_theme(root, density=None, family=None):
    """Create fonts, ttk styles and option-database defaults. Returns the
    density name in use."""
    global SCALE
    SCALE = root.winfo_fpixels("1i") / 96.0

    if density is None:
        usable_h = root.winfo_screenheight() / SCALE - 110     # minus taskbar + title bar
        density = "comfortable" if usable_h >= 760 else "compact"
    spec = DENSITY[density]
    D.__dict__.update(spec)
    D.name = density

    fam = family or _pick_family(root)
    F.family = fam
    F.base = tkfont.Font(root=root, family=fam, size=spec["base"])
    F.base_bold = tkfont.Font(root=root, family=fam, size=spec["base"], weight="bold")
    F.label = tkfont.Font(root=root, family=fam, size=spec["label"])
    F.title = tkfont.Font(root=root, family=fam, size=spec["title"], weight="bold")
    F.sub = tkfont.Font(root=root, family=fam, size=spec["sub"], weight="bold")
    F.value = tkfont.Font(root=root, family=fam, size=spec["value"], weight="bold")
    F.value_small = tkfont.Font(root=root, family=fam, size=spec["value_small"], weight="bold")
    F.button = tkfont.Font(root=root, family=fam, size=spec["button"], weight="bold")
    F.chip = tkfont.Font(root=root, family=fam, size=spec["chip"], weight="bold")
    F.note = tkfont.Font(root=root, family=fam, size=spec["label"], slant="italic")
    F.app_title = tkfont.Font(root=root, family=fam, size=spec["title"] + 3, weight="bold")

    root.configure(bg=C.bg)

    style = ttk.Style(root)
    style.theme_use("clam")

    for name, border in (("App.TCombobox", C.border_strong), ("Err.TCombobox", C.danger)):
        style.configure(name, fieldbackground=C.field, background=C.surface, foreground=C.text,
                        arrowcolor=C.muted, bordercolor=border, lightcolor=C.field, darkcolor=C.field,
                        padding=(px(spec["entry_bd"] + 1), px(spec["entry_bd"] - 1)), arrowsize=px(12))
        style.map(name,
                  fieldbackground=[("disabled", C.disabled), ("readonly", C.field)],
                  foreground=[("disabled", C.faint)],
                  arrowcolor=[("disabled", C.faint)],
                  background=[("disabled", C.disabled), ("active", C.surface)],
                  selectbackground=[("readonly", C.field)],
                  selectforeground=[("readonly", C.text)],
                  bordercolor=[("focus", C.accent if name == "App.TCombobox" else C.danger),
                               ("disabled", C.border)])

    style.configure("Card.TCheckbutton", background=C.surface, foreground=C.text, font=F.base,
                    focuscolor=C.surface)
    style.map("Card.TCheckbutton", background=[("active", C.surface)])

    style.configure("Vertical.TScrollbar", background=C.border_strong, troughcolor=C.bg, bordercolor=C.bg,
                    arrowcolor=C.muted, relief="flat")
    style.configure("Horizontal.TScrollbar", background=C.border_strong, troughcolor=C.bg, bordercolor=C.bg,
                    arrowcolor=C.muted, relief="flat")

    root.option_add("*TCombobox*Listbox.font", F.base)
    root.option_add("*TCombobox*Listbox.background", C.surface)
    root.option_add("*TCombobox*Listbox.foreground", C.text)
    root.option_add("*TCombobox*Listbox.selectBackground", C.accent)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    return density


def combo_values(combo):
    """ttk returns a tuple normally but a bare str in some cases; a str with
    spaces ("Conc Acid wash") must be parsed as a Tcl list, not split."""
    raw = combo.cget("values")
    if isinstance(raw, str):
        return list(combo.tk.splitlist(raw))
    return [str(v) for v in raw]


# ------------------------------------------------------------------ small widgets
_CHIP_KINDS = {
    "accent": (C.accent, C.accent_soft), "teal": (C.teal, C.teal_soft), "warn": (C.warn, C.warn_soft),
    "muted": (C.muted, C.disabled), "success": (C.success, C.success_soft), "danger": (C.danger, C.danger_soft),
}


def chip(master, text, kind="accent", bg=None):
    fg, soft = _CHIP_KINDS[kind]
    return tk.Label(master, text=text, font=F.chip, fg=fg, bg=soft, padx=px(6), pady=px(1))


def set_chip(label, text, kind):
    fg, soft = _CHIP_KINDS[kind]
    label.config(text=text, fg=fg, bg=soft)


def _hover(widget, normal, over):
    widget.bind("<Enter>", lambda e: widget.config(bg=over), add="+")
    widget.bind("<Leave>", lambda e: widget.config(bg=normal), add="+")


def primary_button(master, text, command):
    b = tk.Button(master, text=text, command=command, font=F.button, fg="#ffffff", bg=C.accent,
                  activebackground=C.accent_hover, activeforeground="#ffffff", relief="flat", bd=0,
                  highlightthickness=0, cursor="hand2", padx=px(14), pady=px(7))
    _hover(b, C.accent, C.accent_hover)
    return b


def ghost_button(master, text, command):
    b = tk.Button(master, text=text, command=command, font=F.button, fg=C.text, bg=C.surface,
                  activebackground=C.disabled, activeforeground=C.text, relief="flat", bd=0,
                  highlightthickness=1, highlightbackground=C.border_strong, highlightcolor=C.accent,
                  cursor="hand2", padx=px(10), pady=px(5))
    _hover(b, C.surface, C.disabled)
    return b


def icon_button(master, text, command, bg=C.surface):
    b = tk.Button(master, text=text, command=command, font=F.base_bold, fg=C.muted, bg=bg,
                  activebackground=C.accent_soft, activeforeground=C.accent, relief="flat", bd=0,
                  highlightthickness=0, cursor="hand2", padx=px(6), pady=px(1))
    b.bind("<Enter>", lambda e: b.config(fg=C.accent, bg=C.accent_soft), add="+")
    b.bind("<Leave>", lambda e: b.config(fg=C.muted, bg=bg), add="+")
    return b


def make_entry(master, textvariable=None):
    return tk.Entry(master, textvariable=textvariable, font=F.base, relief="flat", bd=px(D.entry_bd), bg=C.field,
                    fg=C.text, insertbackground=C.text, highlightthickness=1, highlightbackground=C.border_strong,
                    highlightcolor=C.accent, disabledbackground=C.disabled, disabledforeground=C.faint, width=1)


def make_combo(master, values, var=None):
    return ttk.Combobox(master, textvariable=var, values=values, state="readonly", style="App.TCombobox",
                        font=F.base, width=1)


# ------------------------------------------------------------------ cards and field grids
class Card(tk.Frame):
    """White panel with a title row (optional chip on the right) and a body."""

    def __init__(self, master, title, chip_text=None, chip_kind="accent", rule=None):
        super().__init__(master, bg=C.surface, highlightthickness=1, highlightbackground=C.border,
                         highlightcolor=C.border, bd=0)
        if rule:   # thin coloured top edge: indigo = Model 01 inputs, teal = Model 02 inputs
            tk.Frame(self, bg=rule, height=px(3)).pack(fill="x", side="top")
        self.head = tk.Frame(self, bg=C.surface)
        self.head.pack(fill="x", padx=px(D.pad), pady=(px(D.pad_top), px(5)))
        self.title_label = tk.Label(self.head, text=title, font=F.title, fg=C.text, bg=C.surface)
        self.title_label.pack(side="left")
        self.chip = None
        if chip_text:
            self.chip = chip(self.head, chip_text, chip_kind)
            self.chip.pack(side="right")
        self.body = tk.Frame(self, bg=C.surface)
        self.body.pack(fill="x", padx=px(D.pad), pady=(0, px(D.pad)))


class FieldGrid(tk.Frame):
    """A 6-column grid of stacked (label over input) fields. A numeric field
    spans 2 columns (3 per row), a dropdown 3 (2 per row) or more."""

    def __init__(self, master, ncols=6):
        super().__init__(master, bg=C.surface)
        self.ncols = ncols
        self.r = 0
        self.c = 0
        for i in range(ncols):
            self.grid_columnconfigure(i, weight=1, uniform="fg", minsize=px(D.col_min))

    def newline(self):
        if self.c:
            self.r += 1
            self.c = 0

    def cell(self, span):
        if self.c and self.c + span > self.ncols:
            self.newline()
        f = tk.Frame(self, bg=C.surface)
        last = self.c + span >= self.ncols
        f.grid(row=self.r, column=self.c, columnspan=span, sticky="ew",
               padx=(0, 0 if last else px(D.field_gap)), pady=(0, px(D.row_gap)))
        self.c += span
        if self.c >= self.ncols:
            self.r += 1
            self.c = 0
        return f

    @staticmethod
    def _label(cell, text):
        tk.Label(cell, text=text, font=F.label, fg=C.muted, bg=C.surface, anchor="w").pack(fill="x")

    def numeric(self, label, span=2, textvariable=None):
        cell = self.cell(span)
        self._label(cell, label)
        e = make_entry(cell, textvariable)
        e.pack(fill="x")
        return e

    def dropdown(self, label, values, var, span=3):
        cell = self.cell(span)
        self._label(cell, label)
        cb = make_combo(cell, values, var)
        cb.pack(fill="x")
        return cb

    def subheader(self, text, right_text=""):
        """Small caps row; the right label is where a live total goes."""
        self.newline()
        row = tk.Frame(self, bg=C.surface)
        row.grid(row=self.r, column=0, columnspan=self.ncols, sticky="ew", pady=(px(2), px(3)))
        left = tk.Label(row, text=text.upper(), font=F.sub, fg=C.muted, bg=C.surface)
        left.pack(side="left")
        right = tk.Label(row, text=right_text, font=F.label, fg=C.faint, bg=C.surface)
        right.pack(side="right")
        self.r += 1
        return left, right

    def full(self, pady=(0, 0)):
        """A frame spanning the whole row, for custom content."""
        self.newline()
        f = tk.Frame(self, bg=C.surface)
        f.grid(row=self.r, column=0, columnspan=self.ncols, sticky="ew", pady=pady)
        self.r += 1
        return f

    def note(self, text, wrap=None):
        f = self.full(pady=(0, px(D.row_gap)))
        lbl = tk.Label(f, text=text, font=F.note, fg=C.muted, bg=C.surface, justify="left", anchor="w",
                       wraplength=px(wrap) if wrap else 0)
        lbl.pack(fill="x")
        return lbl


# ------------------------------------------------------------------ navigation / status
class SegmentedControl(tk.Frame):
    def __init__(self, master, items, command):
        """items: [(key, number, text)]"""
        super().__init__(master, bg=C.disabled, padx=px(3), pady=px(3))
        self.command = command
        self.segments = {}
        self.current = None
        for key, number, text in items:
            lbl = tk.Label(self, text=f"{number}  {text}", font=F.base_bold, padx=px(14), pady=px(5), cursor="hand2",
                           fg=C.muted, bg=C.disabled)
            lbl.pack(side="left", padx=(0, px(2)))
            lbl.bind("<Button-1>", lambda e, k=key: self.select(k, fire=True))
            lbl.bind("<Enter>", lambda e, k=key: self._hover(k, True))
            lbl.bind("<Leave>", lambda e, k=key: self._hover(k, False))
            self.segments[key] = lbl

    def _hover(self, key, over):
        if key != self.current:
            self.segments[key].config(fg=C.text if over else C.muted)

    def select(self, key, fire=False):
        self.current = key
        for k, lbl in self.segments.items():
            if k == key:
                lbl.config(bg=C.surface, fg=C.accent)
            else:
                lbl.config(bg=C.disabled, fg=C.muted)
        if fire and self.command:
            self.command(key)


class StatusBar(tk.Frame):
    KINDS = {"info": (C.faint, C.muted), "ok": (C.success, C.text), "warn": (C.warn, C.text),
             "error": (C.danger, C.danger)}

    def __init__(self, master):
        super().__init__(master, bg=C.surface, highlightthickness=0)
        tk.Frame(self, bg=C.border, height=1).pack(fill="x", side="top")
        row = tk.Frame(self, bg=C.surface)
        row.pack(fill="x", padx=px(14), pady=px(4))
        self.dot = tk.Label(row, text="\u25cf", font=F.label, fg=C.faint, bg=C.surface)
        self.dot.pack(side="left")
        self.text_var = tk.StringVar(master=self, value="Ready.")
        self.label = tk.Label(row, textvariable=self.text_var, font=F.label, fg=C.muted, bg=C.surface, anchor="w",
                              justify="left")
        self.label.pack(side="left", padx=(px(6), 0), fill="x", expand=True)
        self.right = tk.Label(row, text="", font=F.label, fg=C.faint, bg=C.surface)
        self.right.pack(side="right")
        self.bind("<Configure>", lambda e: self.label.config(wraplength=max(px(300), e.width - px(60))))

    def set(self, kind, text):
        dot, fg = self.KINDS[kind]
        self.dot.config(fg=dot)
        self.label.config(fg=fg)
        self.text_var.set(text)


class AutoScrollFrame(tk.Frame):
    """Hosts a page. The page fills the window; scrollbars appear only when
    the content is genuinely larger than the window. `max_width` centres the
    content on very wide screens instead of stretching fields absurdly."""

    def __init__(self, master, max_width=None):
        super().__init__(master, bg=C.bg)
        self.max_width = px(max_width) if max_width else None
        self.canvas = tk.Canvas(self, bg=C.bg, highlightthickness=0, bd=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.inner = tk.Frame(self.canvas, bg=C.bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.needs_v = False
        self.needs_h = False
        self._pending = False
        self._last_sizes = None
        self.inner.bind("<Configure>", lambda e: self._schedule())
        self.canvas.bind("<Configure>", lambda e: self._schedule())
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)
        # The content's *requested* size can change without any <Configure> event (an
        # electrolyte row is added, a form is rebuilt) because the window item is sized
        # explicitly. A cheap poll keeps scrollbars and stretching correct in every case.
        self._poll_id = self.after(250, self._poll)
        self._idle_id = None
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _on_destroy(self, event):
        """Cancel pending timers so closing the window never prints 'invalid command name'."""
        if event.widget is not self:
            return
        for aid in (self._poll_id, self._idle_id):
            if aid:
                try:
                    self.after_cancel(aid)
                except tk.TclError:
                    pass

    def _poll(self):
        try:
            if not self.winfo_exists():
                return
            sizes = (self.canvas.winfo_width(), self.canvas.winfo_height(),
                     self.inner.winfo_reqwidth(), self.inner.winfo_reqheight())
            if sizes != self._last_sizes:
                self._reflow()
            self._poll_id = self.after(250, self._poll)
        except tk.TclError:
            pass

    def _schedule(self):
        if not self._pending:
            self._pending = True
            self._idle_id = self.after_idle(self._reflow)

    def refresh(self):
        """Measure and size the content right now. Needed because a page that was built while
        hidden gets no <Configure> event when it is shown (its canvas size did not change)."""
        self.update_idletasks()
        self._reflow()

    def _reflow(self):
        self._pending = False
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        rw, rh = self.inner.winfo_reqwidth(), self.inner.winfo_reqheight()
        self._last_sizes = (cw, ch, rw, rh)
        if cw <= 1 or ch <= 1:
            return
        w = max(cw, rw)
        if self.max_width and cw > self.max_width:
            w = max(self.max_width, rw)
        h = max(ch, rh)
        x = max(0, (cw - w) // 2)
        self.canvas.coords(self._win, x, 0)
        self.canvas.itemconfigure(self._win, width=w, height=h)
        self.canvas.configure(scrollregion=(0, 0, max(w, cw), h))

        need_v, need_h = rh > ch + 1, rw > cw + 1
        self.needs_v, self.needs_h = need_v, need_h
        if need_v and not self.vbar.winfo_ismapped():
            self.vbar.grid(row=0, column=1, sticky="ns")
        elif not need_v and self.vbar.winfo_ismapped():
            self.vbar.grid_remove()
            self.canvas.yview_moveto(0)
        if need_h and not self.hbar.winfo_ismapped():
            self.hbar.grid(row=1, column=0, sticky="ew")
        elif not need_h and self.hbar.winfo_ismapped():
            self.hbar.grid_remove()
            self.canvas.xview_moveto(0)

    def _bind_wheel(self, _e=None):
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _e=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event):
        if not self.needs_v:
            return
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(step * 3, "units")


# ------------------------------------------------------------------ results
class OutputTile(tk.Frame):
    """One prediction readout: coloured edge, small label, large value.
    `multi=True` shows several "rate: value" lines as a row of mini stats."""

    def __init__(self, master, key, label, color, var, multi=False, on_run=None, row=False):
        super().__init__(master, bg=C.surface, highlightthickness=1, highlightbackground=C.border,
                         highlightcolor=C.border)
        self.var = var
        self.multi = multi
        self.key = key
        tk.Frame(self, bg=color, width=px(4)).pack(side="left", fill="y")
        body = tk.Frame(self, bg=C.surface)
        body.pack(side="left", fill="both", expand=True, padx=(px(10), px(6)),
                  pady=px(max(3, D.tile_pad - 4)) if row else px(D.tile_pad))
        if row:    # compact one-line readout: label on the left, value on the right
            if on_run is not None:
                icon_button(body, "\u25b6", lambda k=key: on_run(k)).pack(side="right", padx=(px(4), 0))
            self.value_label = tk.Label(body, text="\u2014", font=F.value_small, fg=C.faint, bg=C.surface, anchor="e")
            self.value_label.pack(side="right")
            tk.Label(body, text=label, font=F.label, fg=C.muted, bg=C.surface, anchor="w").pack(side="left")
            self.mini = []
            self._row = True
            var.trace_add("write", lambda *a: self.refresh())
            self.refresh()
            return
        self._row = False

        top = tk.Frame(body, bg=C.surface)
        top.pack(fill="x")
        tk.Label(top, text=label, font=F.label, fg=C.muted, bg=C.surface, anchor="w").pack(side="left")
        if on_run is not None:
            icon_button(top, "\u25b6", lambda k=key: on_run(k)).pack(side="right")

        self.value_label = None
        self.mini = []
        if multi:
            row = tk.Frame(body, bg=C.surface)
            row.pack(fill="x")
            for _ in range(3):
                col = tk.Frame(row, bg=C.surface)
                col.pack(side="left", expand=True, fill="x")
                rate = tk.Label(col, text="", font=F.label, fg=C.faint, bg=C.surface)
                rate.pack(anchor="w")
                val = tk.Label(col, text="\u2014", font=F.value_small, fg=C.faint, bg=C.surface)
                val.pack(anchor="w")
                self.mini.append((rate, val))
        else:
            self.value_label = tk.Label(body, text="\u2014", font=F.value, fg=C.faint, bg=C.surface, anchor="w")
            self.value_label.pack(fill="x")

        var.trace_add("write", lambda *a: self.refresh())
        self.refresh()

    def refresh(self):
        text = self.var.get()
        if self.multi:
            lines = [l for l in text.split("\n") if l.strip()]
            for i, (rate, val) in enumerate(self.mini):
                if i < len(lines) and ":" in lines[i]:
                    r, v = [p.strip() for p in lines[i].split(":", 1)]
                    bad = v in ("Error", "N/A")
                    rate.config(text=r)
                    val.config(text=v, fg=C.danger if bad else C.text)
                else:
                    rate.config(text="")
                    val.config(text="\u2014", fg=C.faint)
            return
        big = F.value_small if self._row else F.value
        if text == "":
            self.value_label.config(text="\u2014", fg=C.faint, font=big)
        elif text in ("Error", "N/A"):
            self.value_label.config(text=text, fg=C.danger, font=F.value_small)
        else:
            self.value_label.config(text=text, fg=C.text, font=big)


class ResultsPanel(tk.Frame):
    """Right-hand column: title, action buttons, then the output tiles."""

    def __init__(self, master, title, subtitle, tiles, vars_, primary, secondary=(), on_tile_run=None,
                 footnote=None, width=250, row_tiles=False, legend=None):
        super().__init__(master, bg=C.surface, highlightthickness=1, highlightbackground=C.border,
                         highlightcolor=C.border)
        self.width_px = px(width)
        inner = tk.Frame(self, bg=C.surface)
        inner.pack(fill="both", expand=True, padx=px(D.pad), pady=px(D.pad_top))

        tk.Label(inner, text=title, font=F.title, fg=C.text, bg=C.surface, anchor="w").pack(fill="x")
        tk.Label(inner, text=subtitle, font=F.label, fg=C.muted, bg=C.surface, anchor="w", justify="left",
                 wraplength=self.width_px - 2 * px(D.pad)).pack(fill="x", pady=(0, px(8)))

        self.primary = primary_button(inner, primary[0], primary[1])
        self.primary.pack(fill="x")
        self.secondary_buttons = []
        # `secondary` is one row [(text, cmd), ...] or several rows [[...], [...]]
        rows = [list(secondary)] if secondary and isinstance(secondary[0], tuple) else [list(r) for r in secondary]
        for row_spec in rows:
            row = tk.Frame(inner, bg=C.surface)
            row.pack(fill="x", pady=(px(6), 0))
            for i, (text, cmd) in enumerate(row_spec):
                b = ghost_button(row, text, cmd)
                b.pack(side="left", fill="x", expand=True, padx=(0 if i == 0 else px(6), 0))
                self.secondary_buttons.append(b)

        tk.Frame(inner, bg=C.border, height=1).pack(fill="x", pady=px(10))

        self.tiles = {}
        for t in tiles:
            tile = OutputTile(inner, t["key"], t["label"], C.tile_colors[t["color"]], vars_[t["key"]],
                              multi=t.get("multi", False), on_run=on_tile_run, row=row_tiles)
            tile.pack(fill="x", pady=(0, px(D.gap - (5 if row_tiles else 2))))
            self.tiles[t["key"]] = tile
        if footnote:
            tk.Label(inner, text=footnote, font=F.note, fg=C.faint, bg=C.surface, anchor="w", justify="left",
                     wraplength=self.width_px - 2 * px(D.pad)).pack(fill="x", pady=(px(2), 0))
        if legend:
            row = tk.Frame(inner, bg=C.surface)
            row.pack(fill="x", pady=(px(8), 0))
            for color, text in legend:
                tk.Frame(row, bg=color, width=px(9), height=px(9)).pack(side="left")
                tk.Label(row, text=text, font=F.label, fg=C.muted, bg=C.surface).pack(side="left", padx=(px(4), px(10)))


# ------------------------------------------------------------------ layout helper
def partition(heights, k):
    """Split an ordered list of card heights into at most k contiguous groups
    minimising the tallest group (keeps reading order, balances columns).
    Returns a list of lists of indices."""
    n = len(heights)
    k = max(1, min(k, n))
    pre = [0]
    for h in heights:
        pre.append(pre[-1] + h)
    INF = float("inf")
    best = [[INF] * (k + 1) for _ in range(n + 1)]
    cut = [[0] * (k + 1) for _ in range(n + 1)]
    best[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            for m in range(j - 1, i):
                cost = max(best[m][j - 1], pre[i] - pre[m])
                if cost < best[i][j]:
                    best[i][j] = cost
                    cut[i][j] = m
    groups, i = [], n
    for j in range(k, 0, -1):
        m = cut[i][j]
        groups.append(list(range(m, i)))
        i = m
    groups.reverse()
    return [g for g in groups if g]

"""
form_base.py - behaviour shared by the Model 01 and Model 02 forms.

Validation problems are reported inline instead of in a modal dialog: the
offending field(s) get a red outline, the first one takes focus, and the
message goes to `notify("error", text)` (the app routes that to its status
bar). Forms used on their own, without an app, fall back to a message box.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from ui_kit import C


class FieldError(ValueError):
    """A validation error that knows which widgets caused it."""

    def __init__(self, message, widgets=()):
        super().__init__(message)
        self.widgets = tuple(widgets)


class FormBase:
    def __init__(self, master):
        self.master = master
        self.notify = self._default_notify
        self.last_validation_error = None
        self._flagged = []
        self._clear_bound = set()

    # ------------------------------------------------------------ notify
    @staticmethod
    def _default_notify(kind, text):
        if kind == "error":
            messagebox.showerror("Input Error", text)
        elif kind == "warn":
            messagebox.showwarning("Warning", text)

    # ------------------------------------------------------------ inline errors
    def flag(self, widget):
        if isinstance(widget, ttk.Combobox):
            widget.configure(style="Err.TCombobox")
            unflag_on = "<<ComboboxSelected>>"
        elif isinstance(widget, tk.Entry):
            widget.config(highlightbackground=C.danger, highlightcolor=C.danger)
            unflag_on = "<Key>"
        else:
            return
        if widget not in self._flagged:
            self._flagged.append(widget)
        if widget not in self._clear_bound:
            self._clear_bound.add(widget)
            widget.bind(unflag_on, lambda e, w=widget: self.unflag(w), add="+")

    def unflag(self, widget):
        if isinstance(widget, ttk.Combobox):
            widget.configure(style="App.TCombobox")
        elif isinstance(widget, tk.Entry):
            widget.config(highlightbackground=C.border_strong, highlightcolor=C.accent)
        if widget in self._flagged:
            self._flagged.remove(widget)

    def clear_flags(self):
        for w in list(self._flagged):
            try:
                self.unflag(w)
            except tk.TclError:
                self._flagged.remove(w)

    def fail(self, err):
        """Record and display a validation error. Returns False."""
        self.clear_flags()
        message = str(err)
        self.last_validation_error = message
        widgets = getattr(err, "widgets", ())
        for w in widgets:
            self.flag(w)
        if widgets:
            try:
                widgets[0].focus_set()
            except tk.TclError:
                pass
        self.notify("error", message)
        return False

    def ok(self):
        self.clear_flags()
        self.last_validation_error = None
        return True

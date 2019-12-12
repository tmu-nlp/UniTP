from tkinter import Tk, TOP, BOTH, X, Y, N, W, E, S, N, LEFT, RIGHT, END, YES, NO, SUNKEN, ALL, VERTICAL, HORIZONTAL, BOTTOM, CENTER
from tkinter import Text, Canvas, Listbox, Scale, Checkbutton, Label, Entry, Scrollbar, Frame, Button
from tkinter import IntVar, BooleanVar, DoubleVar, StringVar
from tkinter import Toplevel, TclError, filedialog, messagebox
from utils.param_ops import less_kwargs, more_kwargs
# from tkinter.ttk import Frame, Label, Entry, Button #- old fashion
from collections import namedtuple
from itertools import count

Checkboxes = namedtuple('Checkboxes', 'ckb, var')
Entries = namedtuple('Entries', 'lbl, etr, var, color')
CheckEntries = namedtuple('CheckEntries', 'ckb, etr, bvar, svar, color')

def __checkbox(panel, text, value, callback, gui_kwargs):
    var = BooleanVar(panel)
    gui = Checkbutton(panel, text = text, variable = var, command = callback)
    var.set(value)
    gui.var = var # delete?
    return gui, var

def make_checkbox(panel, text, value, callback, gui_kwargs, control = 1):
    gui, var = __checkbox(panel, text, value, callback, gui_kwargs)
    if control > 0: # 0 line for raw combination
        gui.pack(side = TOP, anchor = W, **gui_kwargs) # sticky = W is special for gird
    return Checkboxes(gui, var)

def __entry(panel, value, callback, gui_kwargs):
    char_width = less_kwargs(gui_kwargs, 'char_width', None)
    prompt_str = less_kwargs(gui_kwargs, 'prompt_str', str(value) + "  ")
    var = StringVar(panel)
    var.set(value) # even no initial value?
    gui = Entry(panel, textvariable = var, width = char_width, justify = CENTER)
    default_color = gui.cget('highlightbackground')
    gui.bind('<KeyRelease>', callback)
    # gui.var = var # delete ?
    def on_entry_click(event):
        if gui.get() == prompt_str:
            gui.delete(0, "end") # delete all the text in the entry
            gui.insert(0, '')    # insert blank for user input
            gui.config(fg = 'black')
    def on_focusout(event):
        if gui.get() == '':
            gui.insert(0, prompt_str)
            gui.config(fg = 'grey')
        else:
            callback(None)
    gui.bind('<FocusIn>', on_entry_click)
    gui.bind('<FocusOut>', on_focusout)
    if value:
        on_entry_click(None)
    else:
        on_focusout(None)
    return gui, var, default_color

def make_entry(panel, text, value, callback, gui_kwargs, control = 1):
    pnl = Frame(panel)
    if isinstance(control, dict):
        gui_kwargs.update(control)
        control = 1
    elif less_kwargs(gui_kwargs, 'char_width', None) is None:
        gui_kwargs['char_width'] = 4 if control == 1 else 20
    gui, var, clr = __entry(pnl, value, callback, gui_kwargs)
    lbl = Label(pnl, text = text)
    pnl.pack(side = TOP, fill = BOTH, **gui_kwargs)
    if control == 1:
        lbl.pack(side = LEFT,  anchor = W, fill = X, expand = YES)
        gui.pack(side = RIGHT, anchor = E)
    else: # 2 lines
        lbl.pack(side = TOP, anchor = W)
        gui.pack(side = TOP, anchor = E, expand = YES, fill = X)
    return Entries(lbl, gui, var, clr)

def make_checkbox_entry(panel, text, values, callbacks, gui_kwargs, control = 2):
    # e.g. (panel, 'curve', (True, 'x'), (func1, func2), {char_width:3, padx:...})
    ckb_value,    etr_value    = values
    ckb_callback, etr_callback = callbacks
    pnl = Frame(panel)
    pnl.pack(side = TOP, fill = X, anchor = W, **gui_kwargs)
    if less_kwargs(gui_kwargs, 'char_width', None) is None:
        gui_kwargs['char_width'] = 4 if control == 1 else 20
    etr, svar, clr = __entry(pnl,                  etr_value, etr_callback, gui_kwargs)
    ckb, bvar   = __checkbox(pnl, 'Apply ' + text, ckb_value, ckb_callback, gui_kwargs)
    if control == 1:
        wht = Label(pnl)
        ckb.pack(side = LEFT, anchor = W)
        wht.pack(side = LEFT,  fill = X, expand = YES)
        etr.pack(side = RIGHT, fill = X)
    else: # 2 lines
        ckb.pack(side = TOP, anchor = W)
        etr.pack(side = TOP, fill = X, anchor = E, expand = YES)
    return CheckEntries(ckb, etr, bvar, svar, clr)

def get_checkbox(ckbxes, ctype = 0):
    if ctype == 0:
        gen = (v.get() for _, v in ckbxes)
    else:
        gen = (v.get() for _, _, v, _, _ in ckbxes)
    return ckbxes.__class__(*gen)

def get_entry(entries, entry_dtypes, fallback_values, ctype = 0):
    gen = zip(entries, entry_dtypes, fallback_values)
    res = []
    if ctype == 0:
        for (l, g, v, c), d, f in gen:
            try:
                res.append(d(v.get()))
                g.config(highlightbackground = c)
            except Exception as e:
                print(l.cget('text'), e, 'use', f, 'instead')
                g.config(highlightbackground = 'pink')
                res.append(f)
    else:
        for (b, g, _, v, c), d, f in gen:
            try:
                t = d(v.get())
                if d is eval:
                    t(0.5)
                res.append(t)
                g.config(highlightbackground = c)
            except Exception as e:
                print(b.cget('text'), e, 'use', f, 'instead')
                g.config(highlightbackground = 'pink')
                res.append(f)
    if entries.__class__ is tuple:
        return tuple(res)
    return entries.__class__(*res)

def make_namedtuple_gui(make_func, panel, values, callback, control = None, **gui_kwargs):
    if control is None:
        return values.__class__(
            *(make_func(panel, n.replace('_', ' ').title(), v, callback, gui_kwargs.copy()) for n, v in zip(values._fields, values))
            )
    widgets = []
    for n, v, c in zip(values._fields, values, control):
        w = make_func(panel, n.replace('_', ' ').title(), v, callback, gui_kwargs.copy(), c)
        widgets.append(w)
    return values.__class__(*widgets)

    # demo_func = less_kwargs(gui_kwargs, 'demo_func', 'lambda x:x')
    # entry.pack(side = TOP, anchor = W, **gui_kwargs)
    # def on_entry_click(event):
    #     if entry.get() == demo_func:
    #         entry.delete(0, "end") # delete all the text in the entry
    #         entry.insert(0, 'x')    # Insert blank for user input
    #         entry.config(fg = 'black')

    # def on_focusout(event):
    #     if entry.get().strip() in ('', 'x'):
    #         entry.delete(0, "end")
    #         entry.insert(0, demo_func)
    #         entry.config(fg = 'grey')
    # entry.bind('<FocusIn>', on_entry_click)
    # entry.bind('<FocusOut>', on_focusout)
    # entry.config(fg = 'grey')
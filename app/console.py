import tkinter as tk


def log(text_widget, message, color=None):
  text_widget.insert(tk.END, message, color)
  text_widget.yview(tk.END)  # Auto-scroll to the end of the text widget
  text_widget.update_idletasks()

"""Replit entry point for the NanoSIMS converter GUI.

The actual application code is in convert.py. Replit's Run button starts this
file, which then launches the Tkinter desktop window.
"""

from convert import NanoSimsConverter


if __name__ == "__main__":
    app = NanoSimsConverter()
    app.mainloop()

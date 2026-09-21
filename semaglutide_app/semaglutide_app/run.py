"""Launcher: run this file from any Python editor (IDLE, VS Code, PyCharm, Spyder...).

Put run.py in the same folder as app.py, then press Run.
It installs missing packages (first time only) and starts the dashboard in your browser.
"""
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# 1) install packages if any are missing (first run only)
needed = {"streamlit": "streamlit", "plotly": "plotly", "scipy": "scipy", "pandas": "pandas",
          "numpy": "numpy", "pyteomics": "pyteomics", "psims": "psims", "lxml": "lxml", "openpyxl": "openpyxl"}
missing = [pkg for mod, pkg in needed.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

# 2) start the app (opens http://localhost:8501)
print("Starting dashboard... close this window or press Ctrl+C to stop.")
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

"""
Build AstroTransient.exe with PyInstaller
Run: python build_exe.py
Output: dist/AstroTransient.exe
"""
import os, sys, shutil, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

print("[1/3] Installing PyInstaller...")
subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller", "--quiet"], check=True)

print("[2/3] Building executable (5-10 minutes)...")

# Build command
sep = ";" if sys.platform == "win32" else ":"

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--name", "AstroTransient",
    "--add-data", f"models{os.pathsep}models",
    "--add-data", f"src{os.pathsep}src",
    "--add-data", f"data{os.pathsep}data",
    "--add-data", f"lightcurve_template.csv{os.pathsep}.",
    "--hidden-import", "sklearn",
    "--hidden-import", "sklearn.metrics",
    "--hidden-import", "scipy.stats",
    "--hidden-import", "PIL",
    "--hidden-import", "huggingface_hub",
    "--clean",
    "app.py",
]

subprocess.run(cmd, check=True)

# Done
exe = os.path.join(ROOT, "dist", "AstroTransient.exe")
if os.path.exists(exe):
    size = os.path.getsize(exe) / 1024 / 1024
    print(f"\n[3/3] Done! -> dist/AstroTransient.exe ({size:.0f} MB)")
    print("Double-click to launch. Share this single file.")
else:
    print("\n[ERROR] Build failed.")

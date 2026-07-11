"""Build GUI exe: AstroTransient_GUI.exe"""
import os, sys, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("[1/2] Installing PyInstaller...")
subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller", "--quiet"], check=True)

print("[2/2] Building GUI exe (5-10 min)...")
cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile", "--name", "AstroTransient_GUI",
    "--add-data", f"models{os.pathsep}models",
    "--add-data", f"src{os.pathsep}src",
    "--add-data", f"data{os.pathsep}data",
    "--add-data", f"lightcurve_template.csv{os.pathsep}.",
    "--hidden-import", "sklearn",
    "--hidden-import", "sklearn.metrics",
    "--hidden-import", "sklearn.metrics.cluster",
    "--hidden-import", "scipy.stats",
    "--hidden-import", "PIL",
    "--hidden-import", "huggingface_hub",
    "--collect-all", "gradio",
    "--collect-all", "gradio_client",
    "--clean",
    "app.py",
]
subprocess.run(cmd, check=True)

exe = os.path.join("dist", "AstroTransient_GUI.exe")
if os.path.exists(exe):
    print(f"Done: {exe} ({os.path.getsize(exe)/1024/1024:.0f} MB)")
else:
    print("FAILED - check errors above")

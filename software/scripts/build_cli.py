"""Build CLI exe: AstroTransient_CLI.exe"""
import os, sys, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("[1/2] Installing PyInstaller...")
subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller", "--quiet"], check=True)

print("[2/2] Building CLI exe (3-5 min)...")
cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile", "--name", "AstroTransient_CLI",
    "--add-data", f"models{os.pathsep}models",
    "--add-data", f"src{os.pathsep}src",
    "--add-data", f"data{os.pathsep}data",
    "--hidden-import", "sklearn",
    "--hidden-import", "sklearn.metrics",
    "--hidden-import", "scipy.stats",
    "--hidden-import", "huggingface_hub",
    "--clean",
    "predict.py",
]
subprocess.run(cmd, check=True)

exe = os.path.join("dist", "AstroTransient_CLI.exe")
if os.path.exists(exe):
    print(f"Done: {exe} ({os.path.getsize(exe)/1024/1024:.0f} MB)")
else:
    print("FAILED - check errors above")

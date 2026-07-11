#!/usr/bin/env python3
"""
AstroTransient Launcher - Cross-platform one-click startup.
Windows: double-click launcher.py
macOS/Linux: python3 launcher.py
"""
import subprocess, sys, os, platform, time, webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


def main():
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║                                              ║")
    print("  ║          AstroTransient  v1.0                ║")
    print("  ║     AI Astronomical Transient Classification  ║")
    print("  ║                                              ║")
    print("  ║     14-class light curve identification       ║")
    print("  ║     XGBoost model  ~82% accuracy             ║")
    print("  ║                                              ║")
    print("  ║     Daisy Shang, Nnll_Temp  ·  2026          ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    print(f"  Platform : {platform.system()} {platform.release()}")
    print(f"  Python   : {sys.version.split()[0]}")
    print()

    # Step 1: Dependencies
    print("  [1/5] Checking dependencies...")
    try:
        import gradio, xgboost, numpy, pandas, sklearn, scipy, matplotlib, PIL, joblib
        print("         OK")
    except ImportError:
        print("         Installing (first time, ~2 min)...")
        deps = ["numpy", "pandas", "scipy", "matplotlib", "scikit-learn",
                "xgboost", "joblib", "Pillow", "huggingface-hub", "astropy", "gradio"]
        for pkg in deps:
            r = subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                             capture_output=True)
            tag = "OK" if r.returncode == 0 else "FAIL"
            print(f"         {pkg:20s} [{tag}]")
        print("         Done")

    # Step 2: Model
    print("  [2/5] Checking model...")
    if not os.path.exists(os.path.join(ROOT, "models", "xgboost_final.pkl")):
        print("         FAIL - model not found")
        input("  Press Enter to exit...")
        return
    print("         OK")

    # Step 3: Data
    print("  [3/5] Checking demo data...")
    data_ok = os.path.exists(
        os.path.join(ROOT, "data", "raw", "plasticc", "data",
                     "train-00000-of-00001-1bb6aa51ef447f66.parquet"))
    if not data_ok:
        print()
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │ Demo data not found  (~35 MB).                     │")
        print("  │                                                     │")
        print("  │ With this data: Demo + Benchmark tabs enabled.      │")
        print("  │ Without it:      only Upload CSV tab works.          │")
        print("  │                                                     │")
        print("  │ A sample CSV file is included for testing:           │")
        print("  │   sample_lightcurve.csv                              │")
        print("  └─────────────────────────────────────────────────────┘")
        print()
        ans = input("  Download now? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            print("         Downloading (~35 MB, one time only)...")
            try:
                from src.data.download import download_plasticc
                download_plasticc(os.path.join(ROOT, "data", "raw", "plasticc"))
                print("         Done")
            except Exception as e:
                print(f"         Failed: {e}")
                print("         You can still use Upload CSV mode.")
        else:
            print("         Skipped. Upload CSV mode available.")
    else:
        print("         OK")

    # Step 4: Clean old instances
    print("  [4/5] Cleaning old instances...")
    killed = 0
    try:
        if platform.system() == "Windows":
            out = subprocess.run(
                'wmic process where "name=\'python.exe\'" get processid,commandline /format:csv',
                shell=True, capture_output=True, text=True, timeout=10
            )
            for line in out.stdout.split('\n'):
                if 'app.py' in line and 'AstroTransient' in line:
                    pid = [p for p in line.strip().split(',')[-1:] if p.strip().isdigit()]
                    for p in pid:
                        subprocess.run(f'taskkill /F /PID {p}', shell=True, capture_output=True)
                        killed += 1
        else:
            out = subprocess.run("ps aux | grep '[a]pp.py' | awk '{print $2}'",
                               shell=True, capture_output=True, text=True, timeout=10)
            for pid in out.stdout.strip().split('\n'):
                if pid.strip():
                    subprocess.run(f"kill -9 {pid.strip()}", shell=True)
                    killed += 1
    except Exception:
        pass
    print(f"         {'Stopped ' + str(killed) if killed else 'Clean'}")

    # Step 5: Start
    print("  [5/5] Starting server...")
    app_path = os.path.join(ROOT, "app.py")

    import socket
    port = None
    for p in range(38000, 38020):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                port = p
                break
    if port is None:
        print("         FAIL - no port available")
        input("  Press Enter to exit...")
        return
    print(f"         Port {port}")

    env = os.environ.copy()
    env["ASTROTRANSIENT_LAUNCHER"] = "1"

    server = subprocess.Popen(
        [sys.executable, app_path, "--port", str(port), "--no-browser"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
    )

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print(f"  ║  Open: http://localhost:{port:<21} ║")
    print("  ║  Close this window or Ctrl+C to stop         ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    # Wait for server to be ready, then open browser
    print("  Waiting for server...", end="", flush=True)
    for _ in range(30):
        time.sleep(1)
        print(".", end="", flush=True)
        try:
            import urllib.request
            urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=1)
            print(" ready!")
            break
        except Exception:
            continue
    else:
        print()

    webbrowser.open(f"http://localhost:{port}")

    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
        time.sleep(1)
        if server.poll() is None:
            server.kill()

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║           Server stopped. Goodbye!           ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    try:
        input("  Press Enter to close...")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted. Goodbye!")

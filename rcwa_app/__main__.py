# rcwa_app/__main__.py
def main():
    import subprocess, sys, pathlib
    app_path = pathlib.Path(__file__).resolve().parent.parent / "ui_streamlit" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)

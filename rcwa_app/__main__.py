from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """
    CLI entry.

    Keep this minimal: we do not import Streamlit here to avoid import-time
    side effects during packaging. Print a friendly hint.
    """
    repo_root = Path(__file__).resolve().parent.parent
    ui_script = repo_root / "ui_streamlit" / "app.py"
    msg = (
        "RCWA Emissivity — Modular UI\n"
        f"Project root: {repo_root}\n"
        f"Run the app with:\n\n"
        f"    streamlit run {ui_script}\n"
    )
    sys.stdout.write(msg)


if __name__ == "__main__":
    main()

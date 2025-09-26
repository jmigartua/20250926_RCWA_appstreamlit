# Project additions

This document provides two drop‑in files:

1) `LICENSE` — MIT license stub with your name and year
2) `.github/workflows/ci.yml` — minimal GitHub Actions workflow (lint, type‑check, tests)

Copy the code blocks to the indicated paths at the **repository root**.

---

## 1) `LICENSE` (MIT)

> **Path:** `LICENSE`

```text
MIT License

Copyright (c) 2025 Josu M. Igartua

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

> If you prefer a different holder name or year, edit the first two lines accordingly.

---

## 2) GitHub Actions CI

> **Path:** `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

# Cancel in‑flight runs on new commits to the same branch
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build-test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11"]

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: |
            pyproject.toml

      - name: Install project (editable) + dev extras
        run: |
          python -m pip install -U pip
          python -m pip install -e '.[dev]'

      - name: Lint with ruff
        run: ruff check .

      - name: Check formatting with black
        run: black --check .

      - name: Type-check with mypy
        run: mypy rcwa_app

      - name: Run tests (if present)
        run: |
          if [ -d tests ]; then
            pytest -q
          else
            echo "No tests/ directory; skipping pytest"
          fi

      # Optional: ensure the package builds
      - name: Build sdist/wheel
        run: python -m build
```

### Notes
- The workflow installs your package in **editable** mode with the **`dev`** extras defined in `pyproject.toml`.
- If you don’t want the optional build step, delete the last step.
- Add badges to your `README.md` once CI passes:
  ```markdown
  ![CI](https://github.com/<USER>/<REPO>/actions/workflows/ci.yml/badge.svg)
  ```

---

### Optional future enhancements
- **Coverage reporting** (e.g., Codecov): add a step to run `pytest --cov=rcwa_app` and upload the report.
- **Pre‑commit**: configure `pre-commit` to run ruff/black/mypy locally before commits.
- **Matrix for OS**: expand to `ubuntu-latest`, `macos-latest`, `windows-latest` if you want cross‑platform assurance.


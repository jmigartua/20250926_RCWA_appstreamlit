Below is a consolidated, “from-zero-to-green” checklist that captures every substantive fix we made on your repository to get CI consistently passing across Python 3.10 and 3.11. I group items by theme (packaging/build, lint/format/type-check, code/fixtures, CI workflow), note the symptom you saw, the concrete change we applied, and—where appropriate—cite the authoritative source.

# 1) Packaging & build-system hygiene

1. **Ensure `README.md` exists when referenced in `pyproject.toml`.**

   * **Symptom:** `OSError: Readme file does not exist: README.md`.
   * **Fix:** Add a minimal `README.md` at the repo root and keep `readme = "README.md"` in `[project]`. This aligns with PEP 621 metadata conventions for `pyproject.toml`. ([Guía de Usuario de Empaquetado de Python][1])

2. **Pin a recent Hatchling in `[build-system]`.**

   * **Symptom:** `AttributeError: module 'hatchling.build' has no attribute 'prepare_metadata_for_build_editable'`.
   * **Fix:** Require a newer Hatchling (e.g., `hatchling>=1.26`) in the `requires = [...]` list and keep `build-backend = "hatchling.build"`. See Hatch docs / history for ongoing backend capability changes. ([Hatch][2])

3. **Avoid source builds of heavy C++ dependencies where possible.**

   * **Symptom (local dev):** `pyarrow` wheel build fails with `cmake: No such file or directory`.
   * **Fix:** Prefer prebuilt wheels (standard `pip install pyarrow` on supported platforms) or install build tooling (`cmake`) only if building from source; Arrow’s build from source requires CMake. ([GitHub][3])

# 2) Linting, formatting, and type checking

4. **Ruff configuration migration.**

   * **Symptom:** Ruff printed: “Top-level linter settings are deprecated in favour of their counterparts in the `lint` section.”
   * **Fix:** Move `select`/`ignore` under `[tool.ruff.lint]`. This is the forward-compatible layout Ruff now expects. ([GitHub][4])

5. **Black vs. pre-commit: understand the “stashed changes conflicted” message.**

   * **Symptom:** Pre-commit reported “Stashed changes conflicted with hook auto-fixes… Rolling back fixes…”, while Black reformatted files.
   * **Fix:** Stage changes after running formatters locally (`black .` then `git add -A && git commit`). Pre-commit will re-run hooks; avoid concurrent unstaged edits during commit.

6. **mypy per-module ignores in `pyproject.toml`.**

   * **Symptom:** `plotly` lacks type stubs; mypy errors on imports.
   * **Fix:** Add a per-module block to `pyproject.toml`:

     ```toml
     [tool.mypy]
     # ...global options...

     [tool.mypy-plotly.*]
     ignore_missing_imports = true
     ```

     This is the documented way to silence missing-stub errors for a single library. ([mypy.readthedocs.io][5])

# 3) Domain models, dataclasses, and fixtures

7. **Use `default_factory` for mutable dataclass fields.**

   * **Symptom:** `ValueError: mutable default <class '...SurfaceConfig'> for field surface is not allowed: use default_factory`.
   * **Fix:** Replace `field: T = T()` with `field: T = field(default_factory=T)` in any `@dataclass`. This prevents shared mutable defaults. ([Python documentation][6])

8. **Pydantic v2 construction & copying.**

   * **Symptom:** Validation errors when constructing the root `ModelConfig`, and missing `.model_copy()` on nested models.
   * **Fixes:**

     * Instantiate the full, *typed* Pydantic model tree: explicitly build `GeometryConfig`, `IlluminationConfig`, `NumericsConfig`, then `ModelConfig(...)`.
     * Use v2 methods: `.model_copy(update=...)` and `.model_dump()` to update/serialize safely. ([docs.pydantic.dev][7])

9. **Align test fixtures with the public domain model.**

   * **Symptom:** `SweepRequest(config=cfg, ...)` rejected `cfg` when `cfg` wasn’t the exact root Pydantic model the tests expect.
   * **Fix:** Make `default_config()` return the actual `ModelConfig` instance (not a dataclass or partial dict).

10. **Respect required fields in models.**

    * **Symptom:** Errors that `TwoSinusoidSurface` is “missing required argument `duty`” or that `GeometryConfig.stack` is required.
    * **Fix:** Supply all required fields with sensible defaults in `default_config()` and, where tests construct surfaces directly, pass `duty=...`.

11. **Guard `Optional[float]` in tests.**

    * **Symptom:** mypy error: “Unsupported operand types for <= ('float' and 'None')”.
    * **Fix:**

      ```python
      assert surf.duty is not None
      duty: float = surf.duty
      assert 0.0 <= duty <= 1.0
      ```

12. **Mock solver ↔ presenter variable naming contract.**

    * **Symptom:** `KeyError: "No variable named 'eps'."` when presenter looked up `ds["eps"]`.
    * **Fix:** Emit the variable with the exact name the presenter expects (e.g., `emissivity` **or** change presenter to the produced name). We standardized on “emissivity” in the dataset and updated the presenter to read that key.

13. **Mock engine input contract.**

    * **Symptom:** `TypeError: 'SweepRequest' object is not subscriptable`.
    * **Fix:** Teach the mock engine to accept the actual `SweepRequest` model (unpack fields via attributes) *or* pass a plain mapping. We updated the engine to read attributes (e.g., `request.lambda_grid_um`) and to normalize grids to NumPy arrays.

# 4) Streamlit app minor correctness

14. **Literal-typed polarization.**

    * **Symptom:** mypy complained about `polarization` type.
    * **Fix:** Ensure the `pol` selected is one of `Literal["TE", "TM", "UNPOL"]` before passing to the updater function.

15. **Remove unnecessary `# type: ignore`.**

    * **Symptom:** “Unused type: ignore”.
    * **Fix:** Delete the comment where no longer needed.

# 5) CI workflow (GitHub Actions) robustness

16. **Make artifact names unique per job with Actions Artifacts v4.**

    * **Symptom:** `(409) Conflict: an artifact with this name already exists on the workflow run`.
    * **Cause:** With `upload-artifact@v4`, artifacts are immutable; you cannot upload to the same name more than once in a workflow (including a matrix).
    * **Fix (what we did):** Parameterize the name by Python version (e.g., `coverage-xml-py${{ matrix.python-version }}`), or upload once from a single job. When you need to aggregate later, use `download-artifact@v4` with `pattern:` and `merge-multiple:`. ([GitHub][3])

17. **Conditionally upload to Codecov without breaking workflow parsing.**

    * **Symptom:** “Invalid workflow file: Unrecognized named-value: ‘secrets’ …” when using `if: ${{ secrets.CODECOV_TOKEN != '' }}` at the wrong scope.
    * **Fix:** Move the check to a step-level `if:` (where contexts are defined) or define an environment variable in the job and test that. GitHub’s contexts (like `secrets`) are only available where expressions are supported. (We used a step-level `if:` and/or `env: CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}`.) ([The GitHub Blog][8])

18. **`upload-artifact@v4` overwrite semantics.**

    * **Symptom:** Adding `overwrite: true` did not resolve the 409.
    * **Explanation:** v4 artifacts are immutable per run; overwriting is not supported. Use unique names or a single upload. ([GitHub][3])

19. **Keep tooling versions explicit and consistent.**

    * **What we pinned/declared:**

      * `ruff` (and moved settings under `[tool.ruff.lint]`). ([GitHub][4])
      * `black` consistent with local runs.
      * `mypy` with strict-but-pragmatic flags; per-module ignore for Plotly. ([mypy.readthedocs.io][9])

---

## Final “Green CI” checklist (actionable)

1. **`pyproject.toml`**

   * Has valid `[build-system]` with modern Hatchling. ([Hatch][2])
   * `[project]` includes an existing `README.md`. ([Guía de Usuario de Empaquetado de Python][1])
   * Ruff config under `[tool.ruff.lint]`. ([GitHub][4])
   * mypy per-module ignore for Plotly under `[tool.mypy-plotly.*]`. ([mypy.readthedocs.io][5])

2. **Codebase**

   * Dataclasses use `field(default_factory=...)` for mutable defaults. ([Python documentation][6])
   * Pydantic v2 APIs used (`model_copy`, `model_dump`), and all required fields supplied. ([docs.pydantic.dev][7])
   * Presenter keys match mock-engine dataset variable names.
   * Tests/fixtures construct the *domain* root model `ModelConfig` that the rest of the code expects.
   * All functions (including tests) have explicit type hints; remove stray `type: ignore`.

3. **Local pre-commit discipline**

   * Run `ruff --fix` then `black .` before committing.
   * Stage and commit after formatters; avoid “stashed changes conflicted” messages.

4. **CI workflows**

   * Use unique artifact names per matrix job (`coverage-xml-py${{ matrix.python-version }}`), or upload once. ([GitHub][3])
   * Aggregate artifacts with `download-artifact@v4` + `pattern` + `merge-multiple` if needed. ([The GitHub Blog][8])
   * Place conditional `if:` checks for secrets at **step** scope (or use `env:` + truthy tests). ([The GitHub Blog][8])
   * Keep tool versions coherent with local dev (Ruff/Black/mypy).

---

### Why these points matter (brief rationale)

* **Reproducibility:** Modern `pyproject.toml` metadata + pinned tools give consistent local/CI behavior, following PEP 621 and current tool ecosystems. ([Python Enhancement Proposals (PEPs)][10])
* **Static guarantees:** mypy/ruff/black stabilize interfaces and formatting, catching drift early. ([mypy.readthedocs.io][9])
* **Runtime correctness:** Pydantic v2 models enforce schema at boundaries; constructing the full model graph avoids latent `ValidationError`s. ([docs.pydantic.dev][11])
* **CI determinism:** Artifacts v4 immutability and proper context usage in GitHub Actions remove a class of flaky failures and parsing errors. ([The GitHub Blog][8])

If you keep this checklist close to your repo (e.g., as `CONTRIBUTING.md`), future contributions should sail through CI with minimal friction.

[1]: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/?utm_source=chatgpt.com "Writing your pyproject.toml - Python Packaging User Guide"
[2]: https://hatch.pypa.io/1.9/history/hatchling/?utm_source=chatgpt.com "Hatchling"
[3]: https://github.com/actions/upload-artifact?utm_source=chatgpt.com "actions/upload-artifact"
[4]: https://github.com/astral-sh/ruff/discussions/9406?utm_source=chatgpt.com "top-level lint options grouped under [tool.ruff.lint] #9406"
[5]: https://mypy.readthedocs.io/en/stable/running_mypy.html?utm_source=chatgpt.com "Running mypy and managing imports"
[6]: https://docs.python.org/3/library/dataclasses.html?utm_source=chatgpt.com "dataclasses — Data Classes"
[7]: https://docs.pydantic.dev/latest/concepts/models/?utm_source=chatgpt.com "Models"
[8]: https://github.blog/changelog/2023-12-14-github-actions-artifacts-v4-is-now-generally-available/?utm_source=chatgpt.com "GitHub Actions - Artifacts v4 is now Generally Available"
[9]: https://mypy.readthedocs.io/en/stable/config_file.html?utm_source=chatgpt.com "The mypy configuration file - mypy 1.18.2 documentation"
[10]: https://peps.python.org/pep-0621/?utm_source=chatgpt.com "PEP 621 – Storing project metadata in pyproject.toml"
[11]: https://docs.pydantic.dev/latest/migration/?utm_source=chatgpt.com "Migration Guide"

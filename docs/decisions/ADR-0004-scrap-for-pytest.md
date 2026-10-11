# ADR-0004: Test-code quality with a pytest port of SCRAP

- **Status:** accepted
- **Date:** 2026-10-10
- **Issue:** tooling (branch `agent/tooling-scrap-pytest`)

## Context

The quality pipeline (`ruff`, `mypy`, `pytest`) checks production code and whether tests pass,
but not how well the tests themselves are structured. SCRAP
(<https://github.com/unclebob/scrap>, commit `f793b28`) is a structural quality analyzer for test
code in the spirit of CRAP: per-test complexity and smell penalties, fuzzy structural
duplication, extraction pressure, and refactoring guidance meant for AI assistants.

SCRAP cannot run here as-is:

- it analyzes only Clojure Speclj specs (`*_spec.clj`, `*_spec.cljc`); this project has none,
  so a Clojure step in CI would analyze zero files and always pass;
- its repository has no license, so its code cannot be copied or vendored.

## Decision

Reimplement SCRAP's documented metric independently for pytest, as a development tool in
`tools/scrap/` (not part of the installed `hyperdime` package), run with
`PYTHONPATH=tools python -m scrap tests`.

- **Mapping**: a Speclj `it` example is a `test_*` function; a `describe` block is a `Test*`
  class; helpers are other functions in the file (their lines are charged back to the tests
  that call them); setup forms are requested local fixtures, autouse fixtures, `setup_method`,
  and `with` blocks (except `with pytest.raises`, which is an assertion).
- **Assertions**: `assert`, `pytest.raises` / `warns` / `fail`, and any call whose name starts
  with `assert` (`np.testing.assert_allclose`, `mock.assert_called_once_with`, ...).
- **Redefinitions** (Speclj `with-redefs`): `monkeypatch.setattr` and friends, `mock.patch`, and
  mock factories. **Temp resources**: `tmp_path`-style fixtures and file, temp, subprocess, and
  thread calls. **Table-driven**: `@pytest.mark.parametrize` and loops over literal case tables.
- **Scores, thresholds, and weights** are SCRAP's (`tools/scrap/policy.py`): the saturating
  complexity score, the smell penalties, Jaccard duplication at 0.5, the extraction-pressure
  formula, file pressure levels, `STABLE` / `LOCAL` / `SPLIT`, and the AI actionability classes.
- **Structural checks** replace Speclj's nesting checks with the pytest mistakes that make tests
  silently not run: a test shadowed by a later test of the same name, a test nested inside
  another test, and parse errors. These are the only findings that fail the run (exit 1).
- **Pipeline**: CI runs the tool after `pytest` and writes its report to the job summary;
  `AGENTS.md` adds it to the pre-commit checks and the end-of-ticket report. Baselines go to
  `artifacts/scrap/` (gitignored).

## Evidence

Hand-computed tests in `tests/unit/test_scrap_analyze.py` and `tests/unit/test_scrap_summary.py`
cover the complexity curve, one fully worked test score, smells, normalization, structural
errors, duplication cost, pressure score, modes, comparison verdicts, and the CLI.

## Consequences

- Test-code structure becomes visible in every pull request without blocking on judgment calls;
  only tests that can never run fail CI.
- The port follows SCRAP's semantics where Python allows; differences (above) are part of this
  ADR, and SCRAP's thresholds were tuned on Clojure, so recommendations are advice to confirm
  against the code, never automatic refactors.
- `with pytest.raises` tests count as single-assertion tests, as `should-throw` does in SCRAP,
  so files made mostly of error-path tests can be reported as low-assertion.
- Changing the tool's policy values changes every score; baselines from before such a change
  must not be compared with reports after it.

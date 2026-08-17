# CI and quality gates

Every pull request runs the checks below. Two of them block a merge; the rest report
and let you decide. Actions in our workflow files are pinned to commit SHAs. CodeQL is
the exception, because it runs through GitHub's managed default setup and has no
workflow file to pin.

| Gate | Required? | What it enforces | Run locally |
|------|-----------|------------------|-------------|
| `pytest` | ✅ required | Full test suite (~1,500 tests) plus a coverage floor of **85%** (currently ~88%) | `poetry run python -m pytest tests/ --cov=src --cov-fail-under=85 -q` |
| `CodeQL` | ✅ required | Static security analysis, GitHub **default setup** (managed, no workflow file) | (runs on GitHub) |
| `ruff (lint + format)` | advisory | Import order plus pyflakes/pycodestyle (`E,F,I,W`), and formatting | `ruff check .` and `ruff format --check .` |
| `mypy (type check)` | advisory | Type-checks all of `src/`, including the bodies of unannotated functions (`check_untyped_defs`), with no per-module opt-outs. Lenient only about third-party imports (`ignore_missing_imports`), and CI runs deps-free so they resolve to `Any` and results stay deterministic | `poetry run mypy src/` |
| `pip-audit` | advisory | Known CVEs in the locked deps (OSV), with **zero** `--ignore-vuln` entries | `poetry run pip-audit --vulnerability-service osv` |
| `dependency-review` | advisory | Blocks PRs that add deps carrying `moderate`+ advisories | (PR-only, runs on GitHub) |
| Claude review | advisory | Automated PR review and `@claude` mentions (`claude.yml`, `claude-code-review.yml`, skipped until the app token is set) | (runs on GitHub) |

`ruff format` is enforced, not just `ruff check`. Running one without the other is the
most common way to get a red build here.

## Where the configuration lives

Lint and type settings sit in `pyproject.toml` under `[tool.ruff]` and `[tool.mypy]`.
The workflows are in `.github/workflows/`: `ci.yml`, `test-suite.yml`,
`dependency-review.yml`, `claude.yml` and `claude-code-review.yml`. CodeQL has no file,
because GitHub's default setup owns it. Most lint and format findings clear with
`ruff check --fix .` followed by `ruff format .`.

## Supply chain

Zero open Dependabot alerts, and a `pip-audit` with no ignore entries. Every advisory
that has reached this project was resolved by a constraint floor in `pyproject.toml`
rather than waived. The Qdrant server image is pinned by digest instead of `:latest`.

One pin is deliberate and awkward: `qdrant-client` is capped below 1.19, because that
release dropped a symbol `llama-index-vector-stores-qdrant` still imports.
[#106](https://github.com/fmasi/mailrag/issues/106) tracks lifting it. Every pin in the
tree carries a comment saying why it exists and what would let it go.

## Local pre-push routine

```bash
ruff check . && ruff format --check .
poetry run mypy src/
poetry run python -m pytest tests/ -q
```

Documentation-only changes skip the test steps. See the project `CLAUDE.md` for what
counts as documentation-only.

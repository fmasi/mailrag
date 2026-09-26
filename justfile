# Local CI mirror (the local-first standard, ~/.claude/skills/ci-guidelines).
# `just ci` runs what .github/workflows/ci.yml and test-suite.yml run, with the same commands,
# pins, flags and coverage floor. Run it before every push. The Makefile keeps demo/onboard/bench.
set shell := ["bash", "-euo", "pipefail", "-c"]

# Pinned exactly like ci.yml (and pyproject's dev group): bump all three together.
ruff_version := "0.15.20"
mypy_version := "2.1.0"

default: ci

# ci.yml (ruff, mypy, pip-audit), then test-suite.yml (pytest, 85% coverage floor)
ci: lint type audit test

# ci.yml `ruff`: lint + format check with the pinned standalone ruff, no project deps
lint:
    pipx run --spec ruff=={{ruff_version}} ruff check .
    pipx run --spec ruff=={{ruff_version}} ruff format --check .

# Deps-free on purpose, like CI: imports resolve to Any, so results don't depend on the venv.
# ci.yml `mypy`: the pinned mypy with NO project deps installed
type:
    pipx run --spec mypy=={{mypy_version}} mypy src/

# The locked dev env, exactly as both CI jobs install it
install:
    poetry install --with dev --all-extras --no-interaction --no-root

# ci.yml `pip-audit`: freshen the venv's pip (it's audited too), then audit against OSV
audit: install
    poetry run python -m pip install --upgrade pip pip-audit
    poetry run pip-audit --vulnerability-service osv

# CI apt-installs these (tesseract-ocr, poppler-utils); without them the OCR tests run degraded
ocr-deps:
    @command -v tesseract >/dev/null && command -v pdftoppm >/dev/null || { echo "Missing OCR binaries. Linux/devcontainer: sudo apt-get install -y tesseract-ocr poppler-utils. macOS: brew install tesseract poppler"; exit 1; }

# test-suite.yml `pytest`: the full suite with the 85% coverage floor
test: ocr-deps install
    poetry run python -m pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=85 -q

# The per-change run from CLAUDE.md (no coverage), via the Makefile
quick:
    make test

# Autofix lint and formatting with the pinned ruff
fmt:
    pipx run --spec ruff=={{ruff_version}} ruff check --fix .
    pipx run --spec ruff=={{ruff_version}} ruff format .

# Workflow parity in containers (arm64; add --container-architecture linux/amd64 for x64-only tools)
act:
    act pull_request -W .github/workflows/ci.yml
    act pull_request -W .github/workflows/test-suite.yml

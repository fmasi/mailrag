# The one task runner (CHEATSHEET R15, ~/.claude/skills/ci-guidelines).
# `just ci` runs what .github/workflows/ci.yml and test-suite.yml run, with the same commands,
# pins, flags and coverage floor (CI also scans the new commits with gitleaks). Run it before
# every push; the lefthook pre-push hook does. The public demo and benchmark live here too.
set shell := ["bash", "-euo", "pipefail", "-c"]

# Pinned exactly like ci.yml (and pyproject's dev group): bump all three together.
ruff_version := "0.15.20"
mypy_version := "2.1.0"

# pip-audit exceptions, identical to ci.yml's `pip-audit` job (and dependency-review.yml's
# `allow-ghsas`). Each one is an advisory with NO fixed release; the reason sits next to it in
# ci.yml and in pyproject's security-floor block.
# Drop an entry as soon as a fix ships (and raise the floor in pyproject.toml).
#   PYSEC-2026-3740 = GHSA-8mgp-746c-j5xp / CVE-2026-81726: nltk <= 3.10.3 (the latest).
audit_ignores := "--ignore-vuln PYSEC-2026-3740"

default: ci

# EXACTLY what CI runs: ci.yml `lint` + `pip-audit`, test-suite.yml `pytest`
ci: lint security workflows type test audit

# ci.yml `lint`: ruff lint + format check with the pinned standalone ruff, no project deps
lint:
    pipx run --spec ruff=={{ruff_version}} ruff check .
    pipx run --spec ruff=={{ruff_version}} ruff format --check .

# S101 = asserts (tests). S110/S112 = try/except/pass and /continue: bandit rates them low, they
# flag swallowed errors rather than a vulnerability, and the Claude review checks error handling.
# Per-path exceptions live in pyproject.toml; inline ones carry `# noqa: Sxxx - reason`.
# ci.yml `lint`: SAST with ruff's bandit rules (S)
security:
    pipx run --spec ruff=={{ruff_version}} ruff check --select S --ignore S101,S110,S112 .

# ci.yml `lint`: workflow lint, the same commands (brew install actionlint zizmor)
workflows:
    @for t in actionlint zizmor; do command -v "$t" >/dev/null || { echo "$t missing: brew install $t" >&2; exit 1; }; done
    actionlint
    zizmor --min-severity high .github/workflows

# Deps-free on purpose, like CI: imports resolve to Any, so results don't depend on the venv.
# ci.yml `lint`: the pinned mypy with NO project deps installed
type:
    pipx run --spec mypy=={{mypy_version}} mypy src/

# The locked dev env, exactly as both CI jobs install it
install:
    poetry install --with dev --all-extras --no-interaction --no-root

# ci.yml `pip-audit`: freshen the venv's pip (it's audited too), then audit against OSV
audit: install
    poetry run python -m pip install --upgrade pip pip-audit
    poetry run pip-audit --vulnerability-service osv {{audit_ignores}}

# CI apt-installs these (tesseract-ocr, poppler-utils); without them the OCR tests run degraded
ocr-deps:
    @command -v tesseract >/dev/null && command -v pdftoppm >/dev/null || { echo "Missing OCR binaries. Linux/devcontainer: sudo apt-get install -y tesseract-ocr poppler-utils. macOS: brew install tesseract poppler"; exit 1; }

# test-suite.yml `pytest`: the full suite with the 85% coverage floor
test: ocr-deps install
    poetry run python -m pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=85 -q

# The fast per-change run from AGENTS.md (no coverage), in the active environment
quick:
    python -m pytest tests/ -q

# Autofix lint and formatting with the pinned ruff
fmt:
    pipx run --spec ruff=={{ruff_version}} ruff check --fix .
    pipx run --spec ruff=={{ruff_version}} ruff format .

# Scan the staged changes for secrets (the pre-commit hook runs this)
secrets:
    gitleaks git --staged --redact --no-banner

# Workflow parity in containers (arm64; add --container-architecture linux/amd64 for x64-only tools)
act:
    act pull_request -W .github/workflows/ci.yml
    act pull_request -W .github/workflows/test-suite.yml

# Public demo (#125): Qdrant up, then plain vs contextual indexes over 1,200 public Enron emails
demo:
    bash scripts/quickstart.sh

# Guided onboarding; extra flags pass through, e.g. `just onboard --classic`
onboard *args:
    python -m src.cli onboard {{args}}

# Needs a running Qdrant (docker compose up -d) and downloads bge-m3 on first run. Both sizes
# discriminate between arms (see gen_public_benchset.py):
#   just bench        -> 2000 docs / 360 queries  (1.6 min MPS / 14.7 min CPU)
#   just bench large  -> 10000 docs / 360 queries (harder, ~4x the build)
# Public retrieval benchmark on Enron-QA (#97): `just bench [standard|large] [flags]`
bench size="standard" *args:
    python -m scripts.eval.bench_public --size {{size}} {{args}}

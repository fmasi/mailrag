# Poetry Migration - Transition Policy

## Status
Poetry has been successfully adopted in this repository as of February 2026.

## requirements.txt

`pyproject.toml` and `poetry.lock` are the only dependency sources. The old
hand-kept `requirements.txt` drifted from the lock and has been deleted. Do not
recreate it.

## Development Workflow

### For new developers:
1. Install Poetry with `pipx install poetry`, or into the conda env with
   `conda install -n mailrag -c conda-forge poetry`. Never with a host `pip install`.
2. With the `mailrag` conda env active, install dependencies:
   `poetry install --with dev --all-extras --no-root` (or `just install`)
3. Run scripts inside the env: `python -m scripts.demo` (or `just demo`)

### For CI/CD:
- `test-suite.yml` installs Poetry and runs
  `poetry install --with dev --all-extras --no-interaction --no-root`, the same
  command as `just install`
- Execute commands via `poetry run <command>`

## Benefits
- **Deterministic builds**: `poetry.lock` ensures identical dependency versions across environments
- **Better dependency management**: Poetry handles version conflicts automatically
- **Dependabot support**: GitHub Dependabot can now track and update Python dependencies
- **Development dependencies**: Dev dependencies (like pytest) are separate from production dependencies

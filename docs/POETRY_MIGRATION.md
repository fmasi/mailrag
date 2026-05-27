# Poetry Migration - Transition Policy

## Status
Poetry has been successfully adopted in this repository as of February 2026.

## Transition Policy for requirements.txt

We are following **Compat Mode** for backwards compatibility:

1. **Source of Truth**: `pyproject.toml` and `poetry.lock` are the primary dependency sources.
2. **requirements.txt Handling**: The existing `requirements.txt` is kept for compatibility with tools that don't support Poetry yet.
3. **Updating requirements.txt**: When dependencies change, `requirements.txt` can be regenerated using:
   ```bash
   poetry export -f requirements.txt --output requirements.txt --without-hashes
   ```
   Note: This requires the `poetry-plugin-export` plugin to be installed:
   ```bash
   poetry self add poetry-plugin-export
   ```

## Development Workflow

### For new developers:
1. Install Poetry: `pip install poetry`
2. Install dependencies: `poetry install`
3. Run scripts: `poetry run python main.py`

### For CI/CD:
- Install Poetry in the workflow
- Run `poetry install` to install dependencies
- Execute commands via `poetry run <command>`

## Benefits
- **Deterministic builds**: `poetry.lock` ensures identical dependency versions across environments
- **Better dependency management**: Poetry handles version conflicts automatically
- **Dependabot support**: GitHub Dependabot can now track and update Python dependencies
- **Development dependencies**: Dev dependencies (like pytest) are separate from production dependencies

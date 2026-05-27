# Project Rules

## After every code change

1. **Run the full test suite** — `python -m pytest tests/ -q` — and fix any failures before considering the task done.

2. **Write unit tests** for any new or modified logic. Follow the `unittest.TestCase` style in `tests/`. Cover the happy path, edge cases, and invalid inputs.

3. **Update documentation** when behaviour changes — docstrings on modified functions, inline comments for non-obvious logic, and any affected files under `docs/`.

## Documentation-only changes

No tests required. Skip steps 1 and 2 above.

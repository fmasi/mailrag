# Project Rules

## Planning & tracking work

Work is driven by discussion: talk through the change with the assistant, then
use the available tools (including plan mode) to execute it directly. There is
**no** plan-notebook workflow — do not author `.ipynb` plans under
`enhancement_plans/` or sync them to issues (that automation has been removed).

When a feature request, bug, or follow-up needs to be **persisted** for later,
create it directly as a **GitHub issue** (`gh issue create`), not as a notebook
or a doc.

## After every code change

1. **Run the full test suite** — `python -m pytest tests/ -q` — and fix any failures before considering the task done.

2. **Write unit tests** for any new or modified logic. Follow the `unittest.TestCase` style in `tests/`. Cover the happy path, edge cases, and invalid inputs.

3. **Update documentation** when behaviour changes — docstrings on modified functions, inline comments for non-obvious logic, and any affected files under `docs/`.

## Documentation-only changes

No tests required. Skip steps 1 and 2 above.

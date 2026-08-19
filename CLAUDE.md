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

## Know the audience before you write

Every published surface has a different reader. State which one you are writing
for — and for a blog post, which job it is doing — before drafting. The same
finding needs a different shape on each: a negative result is credibility on
GitHub, noise on a recruiter page, and a story hook on the blog.

| Surface | Reader |
|---------|--------|
| This repo — `README.md`, `docs/*.md` | Technical peers: solution architects, engineers. Often reached during an interview process or a community discussion. |
| `docs/index.html` → fmasi.eu | Recruiters first; hiring managers may get dedicated pages, and they need something different. |
| Blog (`fmasi.github.io/blog/…`), amplified on LinkedIn | Varies by post: demonstrate rigour and findings, promote the project, or show domain knowledge. Pick one. |

Within the repo, [`docs/INDEX.md`](docs/INDEX.md) is the canonical map and reading
order: README → INDEX → guides → deep dives → the evidence tier
(`CLAIMS` / `BENCHMARK` / `CASE_STUDY` / `EXPERIMENTS`). Route new material to the
file whose job it is rather than appending where it happens to fit — mechanics to
the subsystem doc, findings and dead ends to `EXPERIMENTS.md`, and **every
published number to `CLAIMS.md`** with its source, corpus and date.

Keep `README.md` short. It is the front door, not the manual: it points at
`docs/INDEX.md` instead of repeating it.

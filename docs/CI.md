# CI and quality gates

Every pull request runs the checks below once it is marked ready: draft PRs skip CI (see
[Local CI](#local-ci)). All the gates below block a merge through the `main-protection`
ruleset, which has no bypass, the owner included. Every third-party action in every workflow is
pinned to a commit SHA.

| Gate | Required? | What it enforces | Run locally |
|------|-----------|------------------|-------------|
| `pytest` | ✅ required | Full test suite plus a coverage floor of **85%** (currently ~90%) | `just test` |
| `pip-audit` | ✅ required | Known CVEs in the locked deps (OSV). The only `--ignore-vuln` entries are advisories with **no fixed release**, each with its reason in `ci.yml` (today: nltk GHSA-8mgp-746c-j5xp) | `just audit` |
| `CodeQL (python)` | ✅ required | Static security analysis of `src/`, `scripts/` and `tests/`, `default` query suite | (runs on GitHub) |
| `CodeQL (actions)` | ✅ required | Static analysis of the workflow files themselves | (runs on GitHub) |
| `review / review-gate` | ✅ required | Green only while the PR carries `claude-reviewed` and not `claude-blocked`: the one Claude review passed with no Critical finding | (see below) |
| `lint` | ✅ required | One job: ruff lint + format (`E,F,I,W`); SAST (ruff's bandit rules `S`, exceptions in `pyproject.toml`); mypy on `src/` (`check_untyped_defs`, run deps-free so third-party imports resolve to `Any` and results stay deterministic); actionlint + zizmor on the workflows; gitleaks on the new commits | `just lint security type workflows` |
| `dependency-review` | ✅ required | Blocks PRs that add deps carrying `moderate`+ advisories; `allow-ghsas` holds the same no-fix exceptions as pip-audit | (PR-only, runs on GitHub) |
| `review / claude-review` | — | The one Claude review per PR, when it is marked ready or labelled `ready-for-review` (`claude-review.yml`, rubric `.github/claude-review-prompt.md`). Re-review: remove and re-add `ready-for-review`. An `@claude` comment (`claude.yml`, owner/members/collaborators only) gets an answer, not a verdict | `/ci-review` in Claude Code |

`ruff format` is enforced, not just `ruff check`. Running one without the other is the
most common way to get a red build here.

## Where the configuration lives

Lint and type settings sit in `pyproject.toml` under `[tool.ruff]` and `[tool.mypy]`.
The workflows are in `.github/workflows/`: `ci.yml`, `test-suite.yml`, `codeql.yml`,
`dependency-review.yml`, `claude.yml` and `claude-review.yml`; zizmor's policy is
`.github/zizmor.yml`, and gitleaks' known false positives are in `.gitleaksignore`. Most lint
and format findings clear with `just fmt`.

### Why CodeQL has a workflow file

It used to run through GitHub's managed **default setup**, which had two costs. Its
actions could not be pinned like everything else here, and it only analysed pull
requests targeting `main`. That second one quietly breaks **stacked pull requests**: a PR
based on another branch never receives the required CodeQL status, and nothing you can do
to that PR will trigger it, so it stays blocked until it is retargeted *and* given a new
commit.

`codeql.yml` uses an unfiltered `pull_request:` trigger, so every PR is analysed whatever
it is based on. It runs the same `default` query suite the managed setup ran, so the swap
changed how CodeQL is invoked rather than which alerts it raises.

## Supply chain

`pip-audit` is a required check. Every advisory with a fixed release is cleared by a
constraint floor in `pyproject.toml` (the "Transitive security floors" block names each
advisory) rather than waived. An advisory with no fixed release is the one exception: it is
ignored by ID in `ci.yml` and the justfile's `audit_ignores`, with its reason, until a fix
ships. GitHub keeps the matching Dependabot alert open meanwhile. The Qdrant server image is
pinned by digest instead of `:latest`.

One pin is deliberate and awkward: `qdrant-client` is capped below 1.19, because that
release dropped a symbol `llama-index-vector-stores-qdrant` still imports.
[#106](https://github.com/fmasi/mailrag/issues/106) tracks lifting it. Every pin in the
tree carries a comment saying why it exists and what would let it go.

## Local CI

CI runs once per PR, when it is marked ready, so run the same checks on your machine first.
The `justfile` mirrors `ci.yml` and `test-suite.yml` command for command, with the same pins
and the same coverage floor. It needs `just`, `pipx`, Poetry, the tesseract and poppler
binaries, and `gitleaks`, `actionlint` and `zizmor` (`brew install just pipx poetry tesseract
poppler gitleaks actionlint zizmor`):

```bash
lefthook install      # once per clone: pre-commit ruff/gitleaks/shellcheck/actionlint, pre-push `just ci`
just ci               # lint, SAST, workflow lint, mypy (deps-free, like CI), pytest with the 85% floor, pip-audit
/ci-review            # in Claude Code: the same review rubric CI uses
gh pr create --draft  # CI skips drafts
gh pr ready           # runs CI and the Claude review once
```

`just quick` is the fast per-change run (`python -m pytest tests/ -q`, no coverage).
Documentation-only changes skip the test steps. `AGENTS.md` has the full workflow.

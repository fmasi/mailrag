Review the PR for mailrag (a local-first email RAG system).
Be concise and high-signal — only flag things that matter. Focus on:
- Correctness bugs, edge cases, and broken error handling.
- Test coverage: per the repo's TDD rule, new or changed logic should
  have unit tests (unittest.TestCase style) covering happy path, edge
  cases, and invalid inputs. Flag untested new behaviour.
- Security: leaked secrets/keys, unsafe deserialization, injection.
- Clarity and consistency with surrounding code and docstrings.
- Whether behaviour changes are reflected in docs (docstrings, docs/).

Do not approve or block; just review.

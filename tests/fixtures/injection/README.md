# Prompt-injection fixtures

Hostile email bodies for measuring how often the grounded-answer path follows
instructions planted in retrieved mail (#138). Each case carries the technique
it represents, a question, and substrings the answer must and must not contain.

Two are controls and matter as much as the attacks. `benign-control` is
ordinary mail. `benign-lookalike` is legitimate mail containing the exact
phrase a naive detector would fire on ("please ignore my previous email") —
it is there to keep anyone from "fixing" injection with a keyword filter and
calling it done.

These measure prompt *structure*, not a detector: there is no injection
detection in mailrag by design. A pass rate here is a property of the model
you run, not of the codebase, so re-measure when the model changes and record
the number in `docs/CLAIMS.md` with the model that produced it.

Run: `python scripts/eval/injection_probe.py --model <id>`

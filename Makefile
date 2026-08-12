.PHONY: demo test onboard bench
demo:
	bash scripts/quickstart.sh
test:
	python -m pytest tests/ -q
onboard:
	python -m src.cli onboard $(ARGS)

# Public, reproducible retrieval benchmark on Enron-QA (issue #97).
#   make bench             -> 2000 docs / 360 queries  (1.6 min MPS / 14.7 min CPU)
#   make bench SIZE=large  -> 10000 docs / 360 queries (harder, ~4x the build)
# Both tiers are sized to discriminate between arms; see gen_public_benchset.py.
# Needs a running Qdrant (docker compose up -d) and downloads bge-m3 on first run.
SIZE ?= standard
bench:
	python -m scripts.eval.bench_public --size $(SIZE) $(ARGS)

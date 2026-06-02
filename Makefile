.PHONY: demo test
demo:
	bash scripts/quickstart.sh
test:
	python -m pytest tests/ -q

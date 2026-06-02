.PHONY: demo test onboard
demo:
	bash scripts/quickstart.sh
test:
	python -m pytest tests/ -q
onboard:
	python -m src.cli onboard $(ARGS)

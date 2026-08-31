PYTHON ?= python
CONFIG ?= configs/qwen3_30b_a3b_edge12.yaml

.PHONY: install test estimate smoke doctor cuda clean

install:
	$(PYTHON) -m pip install -e .

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

estimate:
	PYTHONPATH=src $(PYTHON) -m nhdf_edge.cli estimate --config $(CONFIG)

smoke:
	PYTHONPATH=src $(PYTHON) -m nhdf_edge.cli smoke --output metrics/smoke_test.json

doctor:
	PYTHONPATH=src $(PYTHON) -m nhdf_edge.cli doctor --config $(CONFIG) || true

cuda:
	$(PYTHON) setup_cuda.py build_ext --inplace

clean:
	rm -rf build dist *.egg-info src/*.egg-info __pycache__

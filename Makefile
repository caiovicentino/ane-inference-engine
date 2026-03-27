.PHONY: setup download draft test test-draft test-gpu test-pipeline test-opt \
       bench bench-pytorch bench-e2e bench-compare clean

# --- setup ---
setup:
	pip3 install -r requirements.txt

download:
	python3 tools/download_model.py qwen2.5-0.5b

# --- conversion ---
draft:
	python3 draft/convert.py

# --- tests ---
test:
	python3 -m pytest tests/ -v

test-draft:
	python3 -m pytest tests/test_draft.py -v

test-gpu:
	python3 -m pytest tests/test_gpu.py -v

test-pipeline:
	python3 -m pytest tests/test_speculative.py tests/test_pipeline.py -v

test-opt:
	python3 -m pytest tests/test_optimization.py -v

# --- benchmarks ---
bench:
	python3 draft/benchmark.py

bench-pytorch:
	python3 draft/benchmark.py --pytorch

bench-e2e:
	python3 benchmarks/end_to_end.py

bench-compare:
	python3 benchmarks/compare.py

# --- clean ---
clean:
	rm -rf __pycache__ */__pycache__ *.pyc draft_model.mlpackage

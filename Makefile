.PHONY: setup draft bench test clean

setup:
	pip3 install -r requirements.txt

draft:
	python3 draft/convert.py

bench:
	python3 benchmarks/end_to_end.py

test:
	python3 -m pytest tests/

clean:
	rm -rf __pycache__ */__pycache__ *.pyc

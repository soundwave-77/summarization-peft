.PHONY: fix_style check_style train

lint_fix:
	ruff check . --fix

lint_check:
	ruff check .

train:
	python3 -m src.train
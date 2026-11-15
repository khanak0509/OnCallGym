.PHONY: setup lint test shell

IMAGE := oncallgym-dev

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

# run everything inside the dev container, never directly on the host
shell:
	docker build -t $(IMAGE) .devcontainer
	docker run --rm -it --memory=6g --pids-limit=2048 --cpus=6 \
		-v $(CURDIR):/workspace -v oncallgym-venv:/opt/venv $(IMAGE) bash

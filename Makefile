.PHONY: setup lint test test-all shell image up probe parallel clean

IMAGE := oncallgym-dev
DOCKER_RUN := docker run --rm --memory=6g --pids-limit=2048 --cpus=6 \
	-v $(CURDIR):/home/dev/OnCallGym -v oncallgym-venv:/opt/venv $(IMAGE)

# Northwind must never run directly on a macOS host: on Darwin the stack targets
# re-run themselves inside the dev container.
ifeq ($(shell uname -s),Darwin)
IN_CONTAINER = $(MAKE) image >/dev/null && $(DOCKER_RUN) sh -c "uv sync -q && make $@"
else
IN_CONTAINER =
endif

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q -m "not stack"

# includes integration tests that start real stacks
test-all:
ifeq ($(shell uname -s),Darwin)
	$(MAKE) image >/dev/null && $(DOCKER_RUN) sh -c "uv sync -q && \
		uv pip install -q -e environments/oncallgym_env && uv run --no-sync pytest -q"
else
	uv pip install -q -e environments/oncallgym_env && uv run --no-sync pytest -q
endif

image:
	docker build -q -t $(IMAGE) .devcontainer

shell: image
	docker run --rm -it --memory=6g --pids-limit=2048 --cpus=6 \
		-v $(CURDIR):/home/dev/OnCallGym -v oncallgym-venv:/opt/venv $(IMAGE) bash

up:
ifeq ($(shell uname -s),Darwin)
	$(MAKE) image >/dev/null && docker run --rm -it --memory=6g --pids-limit=2048 --cpus=6 \
		-v $(CURDIR):/home/dev/OnCallGym -v oncallgym-venv:/opt/venv $(IMAGE) \
		sh -c "uv sync -q && uv run python -m oncallgym.stack up"
else
	uv run python -m oncallgym.stack up
endif

probe:
ifeq ($(shell uname -s),Darwin)
	$(IN_CONTAINER)
else
	uv run python -m oncallgym.stack probe
endif

parallel:
ifeq ($(shell uname -s),Darwin)
	$(IN_CONTAINER)
else
	uv run python -m oncallgym.stack parallel 6
endif

clean:
ifeq ($(shell uname -s),Darwin)
	$(IN_CONTAINER)
else
	uv run python -m oncallgym.stack clean
endif

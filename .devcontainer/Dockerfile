FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git make curl procps sqlite3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

RUN useradd -m dev && mkdir -p /opt/venv && chown dev /opt/venv
USER dev
WORKDIR /workspace

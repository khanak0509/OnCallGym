FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git make curl procps sqlite3 openssl sudo \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

# dev runs the stacks; agent runs the AI agent's shell commands (no write access to data)
RUN useradd -m dev && useradd --no-create-home --shell /bin/sh agent \
    && mkdir -p /opt/venv && chown dev /opt/venv \
    && echo "dev ALL=(agent) NOPASSWD: ALL" > /etc/sudoers.d/agent && chmod 0440 /etc/sudoers.d/agent
USER dev
WORKDIR /workspace

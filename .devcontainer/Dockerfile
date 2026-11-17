FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git make curl procps sqlite3 openssl sudo \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

# dev runs the stacks; agent runs the AI agent's shell commands (no write access to data)
RUN useradd -m dev && mkdir -p /opt/venv && chown dev /opt/venv \
    && users="" && for i in $(seq 0 15); do useradd --no-create-home --shell /bin/sh agent$i; \
       users="$users,agent$i"; done \
    && echo "dev ALL=(${users#,}) NOPASSWD: ALL" > /etc/sudoers.d/agent \
    && chmod 0440 /etc/sudoers.d/agent
# the repo is mounted under dev's home, which agent users cannot enter
RUN chmod 700 /home/dev
USER dev
WORKDIR /home/dev/OnCallGym

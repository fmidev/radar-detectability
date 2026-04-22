# Stage 1: install into a venv (git available for hatch-vcs versioning)
FROM python:3.14-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgdal-dev libproj-dev libgeos-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /src
COPY .git .git
COPY pyproject.toml README.md LICENSE.txt ./
COPY src/ src/

RUN pip install --no-cache-dir .


# Stage 2: runtime — copy venv, no build tools
FROM python:3.14-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libexpat1 libproj25 libgeos-c1v5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

# No fixed entrypoint — Airflow @task.docker supplies the command.
# For manual use:  podman run ... python -c 'from detectability import process; ...'

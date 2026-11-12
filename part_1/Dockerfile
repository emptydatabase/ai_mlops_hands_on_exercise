# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS builder

# Pin uv so local and image builds use the same resolver.
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

WORKDIR /app/part_1

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1

# Install dependencies first so source changes do not invalidate this layer.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy only what the dashboard needs at runtime.
COPY src ./src
COPY scripts ./scripts
COPY run_dashboard.py ./
COPY data ./data
COPY dataset/census_income.csv /app/datasets/census_income.csv

# Generate the cleaned CSV inside the image; the project is not installed yet.
RUN PYTHONPATH=/app/part_1/src uv run --frozen --no-sync python scripts/clean_dataset.py
RUN uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8050 \
    DASH_DEBUG=false \
    CENSUS_DATA_PATH=/app/part_1/dataset/census_clean.csv \
    PREDICTIONS_DIR=/app/part_1/data \
    PATH="/app/part_1/.venv/bin:$PATH"

# Install only the account tools, then remove the apt cache.
RUN apt-get update \
    && apt-get install -y --no-install-recommends passwd \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app/part_1

COPY --from=builder /app /app

EXPOSE 8050

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/health', timeout=3)"

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--threads", "4", "task2.app:server"]

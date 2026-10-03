# Reproducible container for the Confforge benchmark.
#   docker build -t confforge .
#   docker run --rm -v "$PWD/results:/app/results" confforge

FROM python:3.13-slim AS base

# Keep the image small and the build quiet.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so the layer caches across source edits. Installed from the
# LOCK file (D4) so the image reproduces the exact combination CI verifies.
COPY requirements.lock.txt ./
RUN pip install --upgrade pip && pip install -r requirements.lock.txt

COPY pyproject.toml README.md LICENSE ./
COPY core/ ./core/
COPY data/ ./data/
COPY methods/ ./methods/
COPY eval/ ./eval/
COPY pipeline/ ./pipeline/
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY cli.py ./

# Fail the build if the package cannot even be imported.
RUN pip install -e . --no-deps
# Fail the build if the package cannot be imported or the backend contract broke.
RUN python -c "import cli, core.config, data.synthetic, methods.methods, eval.metrics" \
 && python cli.py selftest

# Default: run the full benchmark. Override to run anything else, e.g.
#   docker run --rm confforge python cli.py gates
CMD ["python", "examples/run_demo.py"]

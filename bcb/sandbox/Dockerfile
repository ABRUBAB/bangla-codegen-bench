# Execution sandbox for LLM-generated code (BanglaCodeBench).
# Build : docker build -t bcb-sandbox bcb/sandbox
# Run   : see bcb/execute.py  (always with --network none, read-only root, dropped caps)
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1

# Libraries that *secure* and *insecure* solutions commonly import. Having them installed
# prevents import-errors from being (mis)counted as functional failures of secure code.
RUN apt-get update && apt-get install -y --no-install-recommends gzip tar zip unzip coreutils \
    && rm -rf /var/lib/apt/lists/* \
    && pip install pytest==8.3.5 pyyaml defusedxml lxml bcrypt cryptography argon2-cffi passlib \
       numpy requests markupsafe jinja2 bleach

RUN useradd -m -u 1001 runner
COPY runner.py /opt/runner.py
USER runner
WORKDIR /home/runner
ENTRYPOINT ["python", "/opt/runner.py"]

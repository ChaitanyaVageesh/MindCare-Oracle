# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Heavy deps in a separate layer — only re-runs when requirements.txt changes.
# BuildKit cache mount keeps the pip download cache across rebuilds.
COPY api/requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY api/main.py main.py
COPY src/ src/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Bus Headways Anomaly Detection — Dockerfile
# Modern multi-service image (dashboard + live collectors) on Python 3.11 slim.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Copenhagen

WORKDIR /app

# System deps (sqlite3 comes with Python; tzdata for correct local time)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

# Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (bind-mounted at runtime for live reload)
COPY . .

# Non-root execution matching host UID (overridable via compose)
USER appuser

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8050/ || exit 1

CMD ["python", "Dashboard/index.py"]

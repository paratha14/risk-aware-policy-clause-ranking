# Production container for the Flask application in app2.py
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first to preserve Docker layer caching.
COPY requirements2.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements2.txt \
    && pip install --no-cache-dir gunicorn

COPY . .

# Run as a non-root user.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# app2.py must expose the Flask application as `app`.
CMD ["sh", "-c", "gunicorn --preload --bind 0.0.0.0:${PORT:-8000} --workers 2 --threads 4 --timeout 120 app2:app"]

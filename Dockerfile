# Slim, single-stage image. The app is pure Python with light deps, so a
# multi-stage build would add complexity for little gain (New-Thing checklist:
# not worth it at this size).
FROM python:3.11-slim

# Don't buffer stdout (so logs appear immediately) and don't write .pyc files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy requirements first so Docker caches the pip layer when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as a non-root user — a basic container security best practice.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Default command runs the offline mock stack (no API key required).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

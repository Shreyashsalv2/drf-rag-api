FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt gunicorn==27.0.1

COPY . .
RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app

ENV DATA_DIR=/data CHROMA_PATH=/data/chroma
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=10 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health/').status == 200 else 1)"
# One worker with threads: the persistent ChromaDB index is written by a single process.
CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py load_samples && gunicorn config.wsgi -b 0.0.0.0:8000 --workers 1 --threads 4 --timeout 60"]

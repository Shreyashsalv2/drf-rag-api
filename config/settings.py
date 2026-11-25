"""Django settings for the DRF RAG API.

Everything that changes between laptop, Docker, CI and the Vercel demo comes from environment variables.
"""

import os
from pathlib import Path

import dj_database_url

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).lower() in {"1", "true", "yes"}


# On Vercel the only writable place is /tmp.
ON_VERCEL = bool(os.environ.get("VERCEL"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/drf-rag" if ON_VERCEL else BASE_DIR / "var"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = env_bool("DJANGO_DEBUG", False)
ALLOWED_HOSTS = [
    h for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,.vercel.app,testserver").split(",") if h
]

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "rest_framework",
    "rest_framework.authtoken",
    "rag",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    },
]

# PostgreSQL when DATABASE_URL is set (Docker Compose, CI); SQLite otherwise (local dev, Vercel demo).
DATABASES = {
    "default": dj_database_url.config(
        default=f"sqlite:///{DATA_DIR / 'db.sqlite3'}",
        conn_max_age=60,
    )
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_TZ = True

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": ["rest_framework.authentication.TokenAuthentication"],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_THROTTLE_RATES": {"uploads": os.environ.get("UPLOAD_RATE", "10/hour")},
    "EXCEPTION_HANDLER": "rag.errors.json_exception_handler",
    "UNAUTHENTICATED_USER": None,
}

CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}

# RAG settings
RAG = {
    "CHROMA_PATH": os.environ.get("CHROMA_PATH", ""),  # empty = in-memory index (rebuilt from the database)
    "COLLECTION": os.environ.get("CHROMA_COLLECTION", "chunks"),
    "CHUNK_SIZE": int(os.environ.get("CHUNK_SIZE", "1200")),  # best recall@5 and MRR in results/eval.json
    "CHUNK_OVERLAP": int(os.environ.get("CHUNK_OVERLAP", "180")),
    "MAX_UPLOAD_MB": float(os.environ.get("MAX_UPLOAD_MB", "4")),  # Vercel caps request bodies at 4.5 MB
    "LLM_TIMEOUT": float(os.environ.get("LLM_TIMEOUT", "20")),
}
DATA_UPLOAD_MAX_MEMORY_SIZE = int(RAG["MAX_UPLOAD_MB"] * 1024 * 1024) + 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = DATA_UPLOAD_MAX_MEMORY_SIZE

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}

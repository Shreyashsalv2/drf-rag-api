"""One JSON error shape for every failure: {"error": {"code", "message", "details"}}."""

import logging

from rest_framework import exceptions, status
from rest_framework.response import Response
from rest_framework.views import exception_handler

from rag.services.extract import ExtractionError

log = logging.getLogger(__name__)


def json_exception_handler(exc, context):
    if isinstance(exc, ExtractionError):
        return Response({"error": {"code": "unreadable_file", "message": str(exc), "details": None}}, status=400)
    response = exception_handler(exc, context)
    if response is None:
        log.exception("Unhandled error", exc_info=exc)
        return Response(
            {"error": {"code": "internal_error", "message": "Something went wrong.", "details": None}},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    if isinstance(exc, exceptions.ValidationError):
        code, message, details = "invalid_request", "Some fields are invalid.", response.data
    else:
        code = getattr(exc, "default_code", "error")
        message = str(getattr(exc, "detail", exc))
        details = None
    response.data = {"error": {"code": code, "message": message, "details": details}}
    return response

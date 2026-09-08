"""Request/response logging middleware with latency tracking."""
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

logger = logging.getLogger("financeai.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "%s %s %d %.2fms %s",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            request.client.host if request.client else "-",
        )
        response.headers["X-Response-Time"] = f"{latency_ms}ms"
        return response

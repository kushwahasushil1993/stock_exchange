"""Rate limiting middleware using token bucket algorithm."""
import time
from collections import defaultdict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls  = calls
        self.period = period
        self._store: dict = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.period

        # Purge old timestamps
        self._store[client_ip] = [
            t for t in self._store[client_ip] if t > window_start
        ]

        if len(self._store[client_ip]) >= self.calls:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(self.period)},
            )

        self._store[client_ip].append(now)
        return await call_next(request)

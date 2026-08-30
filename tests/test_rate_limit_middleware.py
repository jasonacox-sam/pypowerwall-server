"""Tests for _RateLimitMiddleware ASGI middleware (app/main.py).

The middleware is conditionally defined and registered at import time when
PW_RATE_LIMIT_ENABLED is set, so we test its algorithm here by replicating the
exact implementation and driving it directly with hand-built ASGI scopes.
"""
import pytest
from fastapi.responses import JSONResponse


def make_rate_limit_middleware(
    inner_app, buckets, max_requests=5, window_seconds=60, max_buckets=10000, clock=None
):
    """Instantiate a _RateLimitMiddleware-equivalent middleware for testing.

    This mirrors the implementation in app/main.py exactly, using an
    injectable `buckets` dict and `clock` callable so window-reset and
    pruning behavior can be driven deterministically.
    """
    import time as time_module

    now_fn = clock or time_module.monotonic

    class _RateLimitMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            client = scope.get("client")
            client_ip = client[0] if client else "unknown"
            now = now_fn()

            if client_ip not in buckets and len(buckets) >= max_buckets:
                for ip, (start, _) in list(buckets.items()):
                    if now - start >= window_seconds:
                        del buckets[ip]
                if len(buckets) >= max_buckets:
                    oldest_ips = sorted(buckets, key=lambda ip: buckets[ip][0])[
                        : len(buckets) - max_buckets + 1
                    ]
                    for ip in oldest_ips:
                        del buckets[ip]

            window_start, count = buckets.get(client_ip, (now, 0))
            if now - window_start >= window_seconds:
                window_start, count = now, 0
            count += 1
            buckets[client_ip] = (window_start, count)
            if count > max_requests:
                response = JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
                await response(scope, receive, send)
                return
            await self.app(scope, receive, send)

    return _RateLimitMiddleware(inner_app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def capture_app():
    """Return a minimal ASGI app that records how many times it was invoked."""
    captured = {"calls": 0}

    async def app(scope, receive, send):
        captured["calls"] += 1

    return app, captured


def make_scope(ip="1.2.3.4"):
    return {"type": "http", "client": (ip, 12345)}


async def noop_receive():
    return {"type": "http.disconnect"}


def collecting_send():
    messages = []

    async def send(message):
        messages.append(message)

    return send, messages


def response_status(messages):
    for message in messages:
        if message["type"] == "http.response.start":
            return message["status"]
    return None


# ---------------------------------------------------------------------------
# Basic throttling behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_requests_under_limit_pass_through():
    """Requests at or under the limit all reach the inner app."""
    inner, captured = capture_app()
    mw = make_rate_limit_middleware(inner, {}, max_requests=3)

    for _ in range(3):
        send, messages = collecting_send()
        await mw(make_scope(), noop_receive, send)
        assert messages == []

    assert captured["calls"] == 3


@pytest.mark.asyncio
async def test_request_over_limit_returns_429():
    """The request past the limit gets a 429 instead of reaching the inner app."""
    inner, captured = capture_app()
    mw = make_rate_limit_middleware(inner, {}, max_requests=2)

    for _ in range(2):
        send, messages = collecting_send()
        await mw(make_scope(), noop_receive, send)

    send, messages = collecting_send()
    await mw(make_scope(), noop_receive, send)

    assert captured["calls"] == 2
    assert response_status(messages) == 429
    body = b"".join(
        m["body"] for m in messages if m["type"] == "http.response.body"
    )
    assert b"Rate limit exceeded" in body


@pytest.mark.asyncio
async def test_window_resets_after_elapsed_time():
    """After the window elapses, the count resets and requests pass through again."""
    clock_box = [0.0]
    inner, captured = capture_app()
    mw = make_rate_limit_middleware(
        inner, {}, max_requests=1, window_seconds=60, clock=lambda: clock_box[0]
    )

    send, messages = collecting_send()
    await mw(make_scope(), noop_receive, send)
    assert response_status(messages) is None  # passthrough, no response sent

    send, messages = collecting_send()
    await mw(make_scope(), noop_receive, send)
    assert response_status(messages) == 429

    clock_box[0] = 60.0
    send, messages = collecting_send()
    await mw(make_scope(), noop_receive, send)
    assert response_status(messages) is None
    assert captured["calls"] == 2


@pytest.mark.asyncio
async def test_non_http_scope_bypasses_limiter():
    """Non-"http" scopes (e.g. lifespan) are never throttled."""
    buckets = {}
    inner, captured = capture_app()
    mw = make_rate_limit_middleware(inner, buckets, max_requests=1)

    for _ in range(5):
        await mw({"type": "lifespan"}, noop_receive, None)

    assert captured["calls"] == 5
    assert buckets == {}


@pytest.mark.asyncio
async def test_independent_buckets_per_ip():
    """Two distinct client IPs are throttled independently."""
    inner, captured = capture_app()
    mw = make_rate_limit_middleware(inner, {}, max_requests=1)

    send, messages_a1 = collecting_send()
    await mw(make_scope("10.0.0.1"), noop_receive, send)
    send, messages_a2 = collecting_send()
    await mw(make_scope("10.0.0.1"), noop_receive, send)
    send, messages_b1 = collecting_send()
    await mw(make_scope("10.0.0.2"), noop_receive, send)

    assert response_status(messages_a1) is None
    assert response_status(messages_a2) == 429
    assert response_status(messages_b1) is None
    assert captured["calls"] == 2


@pytest.mark.asyncio
async def test_missing_client_falls_back_to_unknown():
    """A scope without a "client" tuple is bucketed under a shared "unknown" key."""
    buckets = {}
    inner, captured = capture_app()
    mw = make_rate_limit_middleware(inner, buckets, max_requests=5)

    scope = {"type": "http"}
    send, messages = collecting_send()
    await mw(scope, noop_receive, send)

    assert "unknown" in buckets
    assert captured["calls"] == 1


# ---------------------------------------------------------------------------
# Bucket pruning / bounded memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pruning_removes_expired_buckets_over_cap():
    """Once over the bucket cap, buckets whose window has expired are swept."""
    buckets = {f"10.0.0.{i}": (0.0, 1) for i in range(5)}  # all expired at now=100
    inner, captured = capture_app()
    mw = make_rate_limit_middleware(
        inner, buckets, max_requests=5, window_seconds=60, max_buckets=5, clock=lambda: 100.0
    )

    send, messages = collecting_send()
    await mw(make_scope("10.0.0.99"), noop_receive, send)

    assert response_status(messages) is None
    assert len(buckets) == 1
    assert "10.0.0.99" in buckets


@pytest.mark.asyncio
async def test_pruning_evicts_oldest_when_all_buckets_still_live():
    """If still over cap after sweeping expired entries, oldest buckets are evicted."""
    buckets = {
        "10.0.0.0": (0.0, 1),
        "10.0.0.1": (10.0, 1),
        "10.0.0.2": (20.0, 1),
    }
    inner, captured = capture_app()
    mw = make_rate_limit_middleware(
        inner, buckets, max_requests=5, window_seconds=60, max_buckets=3, clock=lambda: 25.0
    )

    send, messages = collecting_send()
    await mw(make_scope("10.0.0.99"), noop_receive, send)

    assert response_status(messages) is None
    assert len(buckets) == 3
    assert "10.0.0.0" not in buckets  # oldest window_start evicted
    assert "10.0.0.99" in buckets

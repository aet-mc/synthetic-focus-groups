from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from email.utils import format_datetime

from discussion.llm_client import LLMClient, MockLLMClient


class _FailingLimiter:
    async def acquire(self) -> None:
        raise AssertionError("rate limiter should be bypassed for MockLLMClient")


class _FailingBreaker:
    def allow_request(self) -> bool:
        raise AssertionError("circuit breaker should be bypassed for MockLLMClient")


class _FailingHTTPClient:
    async def post(self, *args, **kwargs):
        raise AssertionError("http client should be bypassed for MockLLMClient")


class _HeaderResponse:
    def __init__(self, retry_after: str | None):
        self.headers = {}
        if retry_after is not None:
            self.headers["retry-after"] = retry_after


def test_mock_client_explicitly_bypasses_limiter_breaker_and_http() -> None:
    mock = MockLLMClient()

    # Inject exploding objects; if any transport path is touched this test fails.
    mock._rate_limiter = _FailingLimiter()
    mock._circuit_breaker = _FailingBreaker()
    mock._shared_client = _FailingHTTPClient()

    result = asyncio.run(mock.complete(system_prompt="sys", user_prompt="usr"))

    assert isinstance(result, str)
    assert result


def test_mock_client_context_manager_and_aclose_are_noop_safe() -> None:
    async def _run() -> None:
        async with MockLLMClient() as client:
            assert isinstance(client, MockLLMClient)
        await client.aclose()

    asyncio.run(_run())


def test_retry_after_parses_seconds_and_units() -> None:
    numeric = _HeaderResponse("2.5")
    millis = _HeaderResponse("1500ms")

    assert LLMClient._retry_after_seconds(numeric, attempt=0) == 2.5
    assert abs(LLMClient._retry_after_seconds(millis, attempt=0) - 1.5) < 1e-9


def test_retry_after_parses_http_date() -> None:
    future = time.time() + 4.0
    retry_after = format_datetime(datetime.fromtimestamp(future, tz=UTC), usegmt=True)
    response = _HeaderResponse(retry_after)

    delay = LLMClient._retry_after_seconds(response, attempt=0)
    assert 0.0 <= delay <= 5.0


def test_retry_after_invalid_header_falls_back_to_backoff() -> None:
    invalid = _HeaderResponse("definitely-not-a-date")
    expected = LLMClient._compute_backoff(2)

    assert LLMClient._retry_after_seconds(invalid, attempt=2) == expected


def test_retry_after_missing_header_falls_back_to_backoff() -> None:
    missing = _HeaderResponse(None)
    expected = LLMClient._compute_backoff(1)

    assert LLMClient._retry_after_seconds(missing, attempt=1) == expected

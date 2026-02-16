from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from datetime import UTC
from email.utils import parsedate_to_datetime

# ---------------------------------------------------------------------------
# Provider registry — maps short names to (base_url, env_var, model_id)
# ---------------------------------------------------------------------------
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "default_model": "moonshotai/kimi-k2.5",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "mistralai/mistral-nemo",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
    "moonshotai": {
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-32k",
    },
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response that may contain markdown fences or preamble."""
    # Try markdown code block first
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try to find the first [ or { and match to the end
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx = text.find(start_char)
        if idx != -1:
            ridx = text.rfind(end_char)
            if ridx > idx:
                return text[idx : ridx + 1]
    return text.strip()


class LLMClient:
    """Async wrapper for LLM API calls via OpenAI-compatible endpoints."""

    _call_count: int = 0

    def __init__(
        self,
        provider: str = "groq",
        model: str | None = None,
        api_key: str | None = None,
        http_client=None,
        requests_per_second: int = 4,
        rate_limit_period: float = 1.0,
        circuit_failure_threshold: int = 5,
        circuit_recovery_seconds: float = 30.0,
    ) -> None:
        import httpx

        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")

        info = PROVIDERS[provider]
        self.provider = provider
        self.base_url = info["base_url"]
        self.model = model or info["default_model"]
        self._api_key = api_key or os.getenv(info["env_key"])

        if not self._api_key:
            # Try loading from OpenClaw config as fallback
            self._api_key = self._load_key_from_config(provider)

        if not self._api_key:
            raise RuntimeError(
                f"No API key for provider '{provider}'. "
                f"Set {info['env_key']} or pass api_key="
            )

        self._timeout = 60.0 if self.provider == "deepseek" else 30.0
        self._shared_client: httpx.AsyncClient | None = http_client
        self._owns_client = http_client is None
        self._rate_limiter = (
            _RateLimiter(max_calls=requests_per_second, period_seconds=rate_limit_period)
            if requests_per_second > 0
            else None
        )
        self._circuit_breaker = (
            _CircuitBreaker(
                failure_threshold=max(1, circuit_failure_threshold),
                recovery_timeout_seconds=max(1.0, circuit_recovery_seconds),
            )
            if circuit_failure_threshold > 0 and circuit_recovery_seconds > 0
            else None
        )
        self._last_complete_metrics: dict[str, int | float | bool | None] = {}

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Rough fallback for providers that do not return usage.
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def _load_key_from_config(provider: str) -> str | None:
        """Try to read API key from OpenClaw config file."""
        config_path = os.path.expanduser("~/.openclaw/openclaw.json")
        if not os.path.isfile(config_path):
            return None
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            return (
                cfg.get("models", {})
                .get("providers", {})
                .get(provider, {})
                .get("apiKey")
            )
        except Exception:
            return None

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 500,
    ) -> str:
        """Send a chat completion request with automatic retry on rate limits."""
        import httpx

        started_at = time.monotonic()
        fell_back_to_mock = self._bypass_transport_stack()
        input_tokens = self._estimate_tokens(system_prompt) + self._estimate_tokens(user_prompt)
        output_tokens = 0
        http_status_code: int | None = None
        retries_used = 0

        if fell_back_to_mock:
            print("Falling back to mock LLM client.")
            content = await self._complete_with_bypass(system_prompt, user_prompt, temperature, max_tokens)
            output_tokens = self._estimate_tokens(content)
            latency_ms = (time.monotonic() - started_at) * 1000.0
            self._last_complete_metrics = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "http_status_code": None,
                "retries": 0,
                "fell_back_to_mock": True,
                "latency_ms": latency_ms,
            }
            logger.debug(
                "llm.complete provider=%s model=%s input_tokens=%s output_tokens=%s latency_ms=%.2f http_status_code=%s retries=%s fell_back_to_mock=%s",
                self.provider,
                self.model,
                input_tokens,
                output_tokens,
                latency_ms,
                None,
                0,
                True,
            )
            return content

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        max_retries = 5
        last_error: Exception | None = None
        response: httpx.Response | None = None

        if self._circuit_breaker is not None and not self._circuit_breaker.allow_request():
            retry_in = self._circuit_breaker.seconds_until_retry()
            raise RuntimeError(
                f"Circuit breaker is open for provider '{self.provider}'. "
                f"Retry in {retry_in:.1f}s."
            )

        for attempt in range(max_retries):
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire()
            retries_used = attempt

            try:
                client = await self._get_http_client()
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                if attempt < max_retries - 1:
                    await asyncio.sleep(self._compute_backoff(attempt))
                    continue
                raise

            http_status_code = response.status_code
            if response.status_code == 429:
                # Rate limited — extract retry-after or use exponential backoff
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                if attempt == max_retries - 1:
                    response.raise_for_status()
                await asyncio.sleep(self._retry_after_seconds(response, attempt))
                continue

            if response.status_code >= 500:
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                if attempt == max_retries - 1:
                    response.raise_for_status()
                await asyncio.sleep(self._compute_backoff(attempt))
                continue

            if response.status_code >= 400:
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                response.raise_for_status()

            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {}) if isinstance(data, dict) else {}
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    if isinstance(prompt_tokens, int):
                        input_tokens = prompt_tokens
                    if isinstance(completion_tokens, int):
                        output_tokens = completion_tokens
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
                raise RuntimeError("Invalid completion payload from LLM provider.") from exc

            if output_tokens <= 0:
                output_tokens = self._estimate_tokens(str(content))
            if self._circuit_breaker is not None:
                self._circuit_breaker.record_success()
            latency_ms = (time.monotonic() - started_at) * 1000.0
            self._last_complete_metrics = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "http_status_code": http_status_code,
                "retries": retries_used,
                "fell_back_to_mock": False,
                "latency_ms": latency_ms,
            }
            logger.debug(
                "llm.complete provider=%s model=%s input_tokens=%s output_tokens=%s latency_ms=%.2f http_status_code=%s retries=%s fell_back_to_mock=%s",
                self.provider,
                self.model,
                input_tokens,
                output_tokens,
                latency_ms,
                http_status_code,
                retries_used,
                False,
            )
            return str(content).strip()

        # Final attempt failed
        latency_ms = (time.monotonic() - started_at) * 1000.0
        self._last_complete_metrics = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "http_status_code": http_status_code,
            "retries": retries_used,
            "fell_back_to_mock": False,
            "latency_ms": latency_ms,
        }
        logger.debug(
            "llm.complete provider=%s model=%s input_tokens=%s output_tokens=%s latency_ms=%.2f http_status_code=%s retries=%s fell_back_to_mock=%s",
            self.provider,
            self.model,
            input_tokens,
            output_tokens,
            latency_ms,
            http_status_code,
            retries_used,
            False,
        )
        if response is not None:
            response.raise_for_status()
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM completion failed after retries.")

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 2000,
        retries: int = 2,
    ) -> str:
        """Like complete(), but extracts JSON from response with retries.

        Returns the extracted JSON string. Caller should json.loads() it.
        Retries on JSON parse failure with a nudge to return valid JSON.
        """
        started_at = time.monotonic()
        total_input_tokens = 0
        total_output_tokens = 0
        retries_used = 0
        http_status_code: int | None = None
        fell_back_to_mock = False

        for attempt in range(retries + 1):
            retries_used = attempt
            raw = await self.complete(system_prompt, user_prompt, temperature, max_tokens)
            metrics = self._last_complete_metrics
            total_input_tokens += int(metrics.get("input_tokens", 0) or 0)
            total_output_tokens += int(metrics.get("output_tokens", 0) or 0)
            status = metrics.get("http_status_code")
            if isinstance(status, int):
                http_status_code = status
            fell_back_to_mock = fell_back_to_mock or bool(metrics.get("fell_back_to_mock", False))
            extracted = _extract_json(raw)
            try:
                json.loads(extracted)
                latency_ms = (time.monotonic() - started_at) * 1000.0
                logger.debug(
                    "llm.complete_json provider=%s model=%s input_tokens=%s output_tokens=%s latency_ms=%.2f http_status_code=%s retries=%s fell_back_to_mock=%s",
                    self.provider,
                    self.model,
                    total_input_tokens,
                    total_output_tokens,
                    latency_ms,
                    http_status_code,
                    retries_used,
                    fell_back_to_mock,
                )
                return extracted
            except json.JSONDecodeError:
                if attempt < retries:
                    # Retry with explicit nudge
                    user_prompt = (
                        f"{user_prompt}\n\n"
                        "IMPORTANT: Return ONLY valid JSON, no explanatory text. "
                        "Your previous response was not valid JSON."
                    )
                else:
                    # Return raw on final attempt — caller handles error
                    latency_ms = (time.monotonic() - started_at) * 1000.0
                    logger.debug(
                        "llm.complete_json provider=%s model=%s input_tokens=%s output_tokens=%s latency_ms=%.2f http_status_code=%s retries=%s fell_back_to_mock=%s",
                        self.provider,
                        self.model,
                        total_input_tokens,
                        total_output_tokens,
                        latency_ms,
                        http_status_code,
                        retries_used,
                        fell_back_to_mock,
                    )
                    return extracted

    async def _get_http_client(self):
        import httpx

        if self._shared_client is None:
            self._shared_client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True
        return self._shared_client

    @staticmethod
    def _compute_backoff(attempt: int, cap: float = 60.0) -> float:
        base = min(2 ** attempt * 1.5, cap)
        # deterministic jitter by attempt index to avoid lockstep retries
        jitter = 0.2 * (attempt + 1)
        return min(base + jitter, cap)

    @staticmethod
    def _retry_after_seconds(response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after is None:
            return LLMClient._compute_backoff(attempt)
        parsed_delay = LLMClient._parse_retry_after_header(retry_after)
        if parsed_delay is None:
            return LLMClient._compute_backoff(attempt)
        return parsed_delay

    @staticmethod
    def _parse_retry_after_header(raw_value: str) -> float | None:
        value = str(raw_value).strip()
        if not value:
            return None

        unit_match = re.fullmatch(
            r"([+-]?\d+(?:\.\d+)?)\s*(ms|s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)?",
            value,
            flags=re.IGNORECASE,
        )
        if unit_match:
            amount = float(unit_match.group(1))
            if amount < 0:
                return 0.0
            unit = (unit_match.group(2) or "s").lower()
            if unit == "ms":
                return amount / 1000.0
            if unit in {"m", "min", "mins", "minute", "minutes"}:
                return amount * 60.0
            if unit in {"h", "hr", "hrs", "hour", "hours"}:
                return amount * 3600.0
            return amount

        try:
            dt = parsedate_to_datetime(value)
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return max(0.0, dt.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None

    def _bypass_transport_stack(self) -> bool:
        return False

    async def _complete_with_bypass(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise RuntimeError("Transport bypass requested but not implemented by this client.")

    async def aclose(self) -> None:
        if self._shared_client is not None and self._owns_client:
            await self._shared_client.aclose()
            self._shared_client = None

    async def __aenter__(self):
        await self._get_http_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


# ---------------------------------------------------------------------------
# Mock client for tests / offline demo
# ---------------------------------------------------------------------------
class MockLLMClient(LLMClient):
    """Deterministic mock LLM for tests and offline simulation."""

    def __init__(self, model: str = "mock/model", **kwargs):
        # Skip parent __init__ entirely — mock needs no API key/client state.
        self.model = model
        self.provider = "mock"
        self.base_url = ""
        self._api_key = "mock"
        self._timeout = 0.0
        self._shared_client = None
        self._owns_client = False
        self._rate_limiter = None
        self._circuit_breaker = None
        self._transport_bypass_enabled = True

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 300,
    ) -> str:
        return await self._complete_with_bypass(system_prompt, user_prompt, temperature, max_tokens)

    def _bypass_transport_stack(self) -> bool:
        return bool(getattr(self, "_transport_bypass_enabled", False))

    async def _complete_with_bypass(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        del temperature, max_tokens
        digest = hashlib.sha256(f"{system_prompt}||{user_prompt}".encode()).hexdigest()
        idx = int(digest[:8], 16)

        if "moderating a market research focus group" in user_prompt:
            return self._mock_question(user_prompt, idx)
        if "Determine whether this participant shifted their opinion" in user_prompt:
            return self._mock_shift(user_prompt)
        if "classify opinion shifts" in system_prompt.lower():
            return self._mock_shift(user_prompt)
        return self._mock_response(system_prompt, user_prompt, idx)

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 2000,
        retries: int = 2,
    ) -> str:
        """Mock always returns valid JSON via complete()."""
        raw = await self.complete(system_prompt, user_prompt, temperature, max_tokens)
        extracted = _extract_json(raw)
        return extracted

    def _mock_question(self, prompt: str, idx: int) -> str:
        phase = "general"
        for candidate in ("warmup", "exploration", "deep_dive", "reaction", "synthesis"):
            if f"Current phase: {candidate}" in prompt:
                phase = candidate
                break

        quiet_names = ""
        marker = "Quiet participants to draw out:"
        if marker in prompt:
            quiet_names = prompt.split(marker, 1)[1].splitlines()[0].strip()

        phase_templates = {
            "warmup": [
                "To get started, what does buying in this category usually look like for you?",
                "Before we dive in, tell us about your recent experience with products like this.",
            ],
            "exploration": [
                "What is the first thing that comes to mind when you hear this concept?",
                "What expectations or concerns show up for you when you hear this idea?",
            ],
            "deep_dive": [
                "Which feature, pricing detail, or practical concern would make or break this for you?",
                "What would you need to trust this enough to choose it over alternatives?",
            ],
            "reaction": [
                "After seeing this, what is your immediate reaction and what would you do next?",
                "Based on this material, would you seriously consider trying it? Why or why not?",
            ],
            "synthesis": [
                "If this were available tomorrow, would you buy it? What is the main reason?",
                "Final take: who is this for, and would you personally purchase it?",
            ],
        }

        question = phase_templates.get(phase, phase_templates["exploration"])[idx % 2]
        if quiet_names and quiet_names != "None":
            first_name = quiet_names.split(",")[0].strip()
            if first_name:
                question = f"{question} {first_name}, I especially want your perspective."
        return question

    def _mock_response(self, system_prompt: str, user_prompt: str, idx: int) -> str:
        response_variants = [
            "I can see the upside, but I would still compare options before deciding.",
            "This sounds useful in theory, though I would need clearer proof it works in real life.",
            "I like the direction, especially if pricing stays reasonable and setup is simple.",
            "I am not fully convinced yet; reliability and value would decide it for me.",
            "I would consider trying it once, but long-term use depends on consistency.",
            "Part of me likes it, but I also worry about hidden tradeoffs after purchase.",
        ]
        base = response_variants[idx % len(response_variants)]

        if "Discussion context:" in user_prompt and "Moderator:" in user_prompt:
            if idx % 3 == 0:
                return f"I agree with what was said earlier. {base}"
            if idx % 3 == 1:
                return f"I see it differently from some of the group. {base}"
        return base

    def _mock_shift(self, prompt: str) -> str:
        lowered = prompt.lower()
        if any(token in lowered for token in ["now i would buy", "changed my mind", "more positive now"]):
            return '{"reasoning": "Participant shifted positively", "changed_mind": true, "shift_magnitude": "moderate", "new_valence": 0.4}'
        if any(token in lowered for token in ["now i would avoid", "less interested now", "more negative now"]):
            return '{"reasoning": "Participant shifted negatively", "changed_mind": true, "shift_magnitude": "moderate", "new_valence": -0.4}'
        return '{"reasoning": "No shift detected", "changed_mind": false, "shift_magnitude": "none", "new_valence": 0.0}'

    async def aclose(self) -> None:
        # Mock never allocates network resources.
        return None

    async def _get_http_client(self):
        # Mock never allocates/uses a real HTTP client.
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _RateLimiter:
    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period_seconds:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return

                wait_for = self.period_seconds - (now - self._calls[0])

            await asyncio.sleep(max(wait_for, 0.0))


class _CircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_timeout_seconds: float) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._state = "closed"  # closed | open | half_open

    def allow_request(self) -> bool:
        if self._state == "open":
            if self._opened_at is None:
                self._opened_at = time.monotonic()
                return False
            if time.monotonic() - self._opened_at >= self.recovery_timeout_seconds:
                self._state = "half_open"
                return True
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._state = "closed"

    def record_failure(self) -> None:
        self._failures += 1
        if self._state == "half_open" or self._failures >= self.failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()

    def seconds_until_retry(self) -> float:
        if self._state != "open" or self._opened_at is None:
            return 0.0
        elapsed = time.monotonic() - self._opened_at
        return max(0.0, self.recovery_timeout_seconds - elapsed)

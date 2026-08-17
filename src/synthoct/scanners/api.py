from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import requests


@dataclass(frozen=True)
class ApiRenderRequest:
    """Prepared hosted scanner request metadata with secret-safe representation."""

    phantom_path: Path
    config_path: Path
    endpoint: str
    api_key: str = field(repr=False)

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    @property
    def redacted_headers(self) -> dict[str, str]:
        return {"X-API-Key": "<redacted>"}

    @property
    def file_fields(self) -> tuple[str, str]:
        return ("config_file", "scatter_file")


DEFAULT_API_KEY_ENVS = ("SYNTHOCT_API_KEY", "SYNTHOCT_CHALLENGE_API_KEY")
DEFAULT_POST_TIMEOUT_SECONDS = (15, 180)
DEFAULT_GET_TIMEOUT_SECONDS = (15, 120)


def resolve_api_key(
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
    env_names: Sequence[str] = DEFAULT_API_KEY_ENVS,
) -> str:
    """Resolve the hosted scanner API key without persisting it in artifacts."""
    if api_key:
        return api_key.strip()
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value.strip()
    if api_key_file is not None:
        path = Path(api_key_file).expanduser()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                _name, value = line.split("=", 1)
                return value.strip().strip('"').strip("'")
            return line
    raise RuntimeError(f"Set one of {', '.join(env_names)} or pass --api-key-file.")


def prepare_api_render_request(
    phantom_path: str | Path,
    config_path: str | Path,
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
    endpoint: str = "https://synthoct.com/process_oct",
) -> ApiRenderRequest:
    """Prepare the hosted API request object without exposing the key in repr/logs."""
    return ApiRenderRequest(
        phantom_path=Path(phantom_path),
        config_path=Path(config_path),
        endpoint=endpoint,
        api_key=resolve_api_key(api_key=api_key, api_key_file=api_key_file),
    )


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _retry_delay(response: requests.Response, attempt: int) -> float:
    retry_after = getattr(response, "headers", {}).get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass
    return 2.0 * attempt


def _request_with_retries(method: str, url: str, attempts: int = 3, **kwargs) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < attempts:
                time.sleep(_retry_delay(response, attempt))
                continue
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(2.0 * attempt)
    raise last_error or RuntimeError(f"Request failed: {method} {url}")


def submit_api_render(
    phantom_path: str | Path,
    config_path: str | Path,
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
    endpoint: str = "https://synthoct.com/process_oct",
) -> tuple[str, float]:
    """Submit one hosted scanner job and return its request id and submit time."""
    request = prepare_api_render_request(
        phantom_path,
        config_path,
        api_key=api_key,
        api_key_file=api_key_file,
        endpoint=endpoint,
    )

    start = time.perf_counter()
    with request.config_path.open("rb") as config_f, request.phantom_path.open("rb") as phantom_f:
        response = _request_with_retries(
            "POST",
            request.endpoint,
            headers=request.headers,
            files={
                "config_file": ("Configuration.ini", config_f, "text/plain"),
                "scatter_file": ("Scatterers.txt", phantom_f, "text/plain"),
            },
            timeout=DEFAULT_POST_TIMEOUT_SECONDS,
        )
    response.raise_for_status()
    payload = response.json()
    request_id = payload.get("request_id") or payload.get("id")
    if not request_id:
        raise RuntimeError(f"API response did not include request_id: {payload}")
    return str(request_id), time.perf_counter() - start


def poll_api_result(
    request_id: str,
    out_png: str | Path,
    result_base_url: str = "https://synthoct.com/results",
    poll_interval_seconds: float = 10.0,
    max_polls: int = 60,
    initial_delay_seconds: float = 2.0,
) -> tuple[Path, float, int]:
    """Poll a submitted hosted scanner job until the PNG result is available."""
    out_png = Path(out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    result_url = f"{result_base_url.rstrip('/')}/result_{request_id}.png"
    for poll_idx in range(1, max_polls + 1):
        time.sleep(poll_interval_seconds if poll_idx > 1 else initial_delay_seconds)
        result = _request_with_retries("GET", result_url, timeout=DEFAULT_GET_TIMEOUT_SECONDS)
        if result.status_code == 200 and result.content.startswith(b"\x89PNG"):
            out_png.write_bytes(result.content)
            return out_png, time.perf_counter() - start, poll_idx
        if result.status_code not in {202, 404}:
            result.raise_for_status()

    raise TimeoutError(f"API result was not ready after {max_polls} polls: {result_url}")


def render_with_api(
    phantom_path: str | Path,
    config_path: str | Path,
    out_png: str | Path,
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
    endpoint: str = "https://synthoct.com/process_oct",
    result_base_url: str = "https://synthoct.com/results",
    poll_interval_seconds: float = 10.0,
    max_polls: int = 60,
) -> tuple[str, Path, float, int]:
    start = time.perf_counter()
    request_id, _submit_seconds = submit_api_render(
        phantom_path,
        config_path,
        api_key=api_key,
        api_key_file=api_key_file,
        endpoint=endpoint,
    )
    try:
        rendered_path, _poll_seconds, poll_count = poll_api_result(
            request_id,
            out_png,
            result_base_url=result_base_url,
            poll_interval_seconds=poll_interval_seconds,
            max_polls=max_polls,
        )
    except Exception as exc:
        setattr(exc, "request_id", request_id)
        raise
    return request_id, rendered_path, time.perf_counter() - start, poll_count

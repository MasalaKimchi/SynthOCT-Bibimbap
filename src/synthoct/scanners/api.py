from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import requests


@dataclass(frozen=True)
class ApiRenderResult:
    request_id: str
    synthetic_png: Path
    synthetic_gray_png: Path
    reference_png: Path
    elapsed_seconds: float
    poll_count: int


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


def _request_with_retries(method: str, url: str, attempts: int = 3, **kwargs) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return requests.request(method, url, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(2.0 * attempt)
    raise last_error or RuntimeError(f"Request failed: {method} {url}")


class HostedApiScanner:
    """Hosted SynthOCT scanner backend, the default renderer for macOS workflows."""

    def __init__(
        self,
        api_key: str | None = None,
        api_key_file: str | Path | None = None,
        endpoint: str = "https://synthoct.com/process_oct",
        result_base_url: str = "https://synthoct.com/results",
        poll_interval_seconds: float = 10.0,
        max_polls: int = 60,
    ) -> None:
        self.api_key = api_key
        self.api_key_file = api_key_file
        self.endpoint = endpoint
        self.result_base_url = result_base_url
        self.poll_interval_seconds = poll_interval_seconds
        self.max_polls = max_polls

    def render(self, phantom_path: str | Path, config_path: str | Path, output_png: str | Path) -> Path:
        _request_id, out_png, _elapsed, _polls = render_with_api(
            phantom_path,
            config_path,
            output_png,
            api_key=self.api_key,
            api_key_file=self.api_key_file,
            endpoint=self.endpoint,
            result_base_url=self.result_base_url,
            poll_interval_seconds=self.poll_interval_seconds,
            max_polls=self.max_polls,
        )
        return out_png


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
    request = prepare_api_render_request(
        phantom_path,
        config_path,
        api_key=api_key,
        api_key_file=api_key_file,
        endpoint=endpoint,
    )

    out_png = Path(out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

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
            timeout=180,
        )
    response.raise_for_status()
    payload = response.json()
    request_id = payload.get("request_id") or payload.get("id")
    if not request_id:
        raise RuntimeError(f"API response did not include request_id: {payload}")

    result_url = f"{result_base_url.rstrip('/')}/result_{request_id}.png"
    for poll_idx in range(1, max_polls + 1):
        time.sleep(poll_interval_seconds if poll_idx > 1 else 2.0)
        result = _request_with_retries("GET", result_url, timeout=120)
        if result.status_code == 200 and result.content.startswith(b"\x89PNG"):
            out_png.write_bytes(result.content)
            elapsed = time.perf_counter() - start
            return request_id, out_png, elapsed, poll_idx
        if result.status_code not in {202, 404}:
            result.raise_for_status()

    raise TimeoutError(f"API result was not ready after {max_polls} polls: {result_url}")

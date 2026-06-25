"""Scanner backend interfaces for rendering digital phantoms into OCT PNGs."""

from .api import (
    ApiRenderRequest,
    ApiRenderResult,
    HostedApiScanner,
    prepare_api_render_request,
    render_with_api,
    resolve_api_key,
)
from .config import write_api_config, write_scanner_config
from .core import ScannerBackend, ScannerError, render_phantom
from .windows import WindowsExecutableScanner

__all__ = [
    "ApiRenderRequest",
    "ApiRenderResult",
    "HostedApiScanner",
    "ScannerBackend",
    "ScannerError",
    "WindowsExecutableScanner",
    "prepare_api_render_request",
    "render_phantom",
    "render_with_api",
    "resolve_api_key",
    "write_api_config",
    "write_scanner_config",
]

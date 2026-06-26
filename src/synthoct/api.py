"""Backward-compatible hosted API and preliminary submission facade.

New code should import scanner rendering from :mod:`synthoct.scanners` and portal
PNG-pair helpers from :mod:`synthoct.submission`.
"""

from .scanners import (
    ApiRenderRequest,
    ApiRenderResult,
    HostedApiScanner,
    prepare_api_render_request,
    render_with_api,
    resolve_api_key,
    write_api_config,
)
from .submission import benchmark_submission_api, prepare_preliminary_png_pairs, to_gray_png, write_preliminary_upload_plan

_to_gray_png = to_gray_png

__all__ = [
    "ApiRenderRequest",
    "ApiRenderResult",
    "HostedApiScanner",
    "_to_gray_png",
    "benchmark_submission_api",
    "prepare_api_render_request",
    "prepare_preliminary_png_pairs",
    "render_with_api",
    "resolve_api_key",
    "to_gray_png",
    "write_api_config",
    "write_preliminary_upload_plan",
]

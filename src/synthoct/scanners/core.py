from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ScannerError(RuntimeError):
    pass


class ScannerBackend(Protocol):
    """Backend contract for challenge scanner renderers."""

    def render(self, phantom_path: str | Path, config_path: str | Path, output_png: str | Path) -> Path:
        """Render a four-column phantom scatterer file into a synthetic OCT PNG."""


def render_phantom(
    phantom_path: str | Path,
    config_path: str | Path,
    output_png: str | Path,
    backend: str | ScannerBackend = "api",
    **backend_kwargs,
) -> Path:
    """Render a digital phantom using either the hosted API or Windows scanner backend.

    macOS users should keep the default ``backend="api"`` because
    ``Part2_Scanner.exe`` is a Windows executable.
    """
    if isinstance(backend, str):
        if backend == "api":
            from .api import HostedApiScanner

            renderer: ScannerBackend = HostedApiScanner(**backend_kwargs)
        elif backend in {"windows", "exe"}:
            from .windows import WindowsExecutableScanner

            renderer = WindowsExecutableScanner(**backend_kwargs)
        else:
            raise ScannerError(f"Unknown scanner backend: {backend}")
    else:
        if backend_kwargs:
            raise ScannerError("backend_kwargs cannot be used with a scanner instance.")
        renderer = backend
    return renderer.render(phantom_path, config_path, output_png)

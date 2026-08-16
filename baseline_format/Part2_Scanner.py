"""Part2 (Scanner) — the FIXED imaging device, wired to the hosted true scanner.

Baseline role (``Part2_Scanner.exe``): a compiled, physics-based coherent
scanner.  Phantom ``.txt`` -> raw OCT B-scan ``.png`` (51 dB), driven as
``Part2_Scanner.exe <Configuration.ini> <Scatterers.txt> <Output.png>``.

You are not meant to modify Part2.  The organizers' binary is Windows-only, so
this submission renders through the **hosted true scanner** at synthoct.com
(the same engine, exposed as an HTTP endpoint).  This file is a thin façade over
``synthoct.scanners`` that keeps the fixed-scanner contract: it writes the
scanner ``Configuration.ini`` and posts the phantom, returning the rendered PNG.

Authentication: the API key is read from ``$SYNTHOCT_API_KEY`` /
``$SYNTHOCT_CHALLENGE_API_KEY`` or an ``--api-key-file``.  It is never written
into any output artifact.

Note: ``synthoct`` also ships a local implementation of the published forward
model (``synthoct.scanners.render_reference_scanner``) for development.  It
agrees with the hosted scanner at MS-SSIM ~0.9997 but is not the organizers'
binary, so it is intentionally not the default here.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (must precede any `synthoct` import; sets sys.path)

import argparse
from collections import namedtuple
from pathlib import Path

from synthoct.scanners import render_with_api, write_api_config

__all__ = ["ScanResult", "scan", "write_config", "SCANNER_ENDPOINT"]

SCANNER_ENDPOINT = "https://synthoct.com/process_oct"
RESULT_BASE_URL = "https://synthoct.com/results"

ScanResult = namedtuple(
    "ScanResult",
    ["png", "request_id", "elapsed_seconds", "poll_count", "config"],
)


def write_config(config_path: str | Path, *, scatterers_count: int = 300_000) -> Path:
    """Write the hosted-scanner ``Configuration.ini`` for a phantom render."""
    return write_api_config(config_path, scatterers_count=scatterers_count)


def scan(
    phantom_path: str | Path,
    output_png: str | Path,
    *,
    api_key_file: str | Path | None = None,
    scatterers_count: int = 300_000,
    config_path: str | Path | None = None,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 60,
    endpoint: str = SCANNER_ENDPOINT,
    result_base_url: str = RESULT_BASE_URL,
) -> ScanResult:
    """Render ``phantom_path`` through the hosted true scanner.

    Mirrors ``Part2_Scanner.exe <Configuration.ini> <Scatterers.txt> <Output.png>``:
    a ``Configuration.ini`` is written (next to the output unless ``config_path``
    is given), the phantom is posted, and the resulting raw B-scan PNG is saved
    to ``output_png``.  Returns the output path plus hosted-request provenance.
    """
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if config_path is None:
        config_path = output_png.with_name(f"{output_png.stem}_Configuration.ini")
    config = write_config(config_path, scatterers_count=scatterers_count)

    request_id, rendered, elapsed, polls = render_with_api(
        phantom_path,
        config,
        output_png,
        api_key_file=api_key_file,
        endpoint=endpoint,
        result_base_url=result_base_url,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
    )
    return ScanResult(
        png=Path(rendered),
        request_id=request_id,
        elapsed_seconds=elapsed,
        poll_count=polls,
        config=config,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="Part2_Scanner",
        description="Render a digital phantom through the hosted SynthOCT true scanner.",
    )
    parser.add_argument("phantom", type=Path, help="Digital phantom .txt (X Y Z Energy%%).")
    parser.add_argument("out", type=Path, help="Output raw B-scan PNG path.")
    parser.add_argument("--api-key-file", type=Path, help="File containing the SynthOCT API key.")
    parser.add_argument("--scatterers-count", type=int, default=300_000)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--max-polls", type=int, default=60)
    args = parser.parse_args(argv)

    result = scan(
        args.phantom,
        args.out,
        api_key_file=args.api_key_file,
        scatterers_count=args.scatterers_count,
        poll_interval_seconds=args.poll_interval_seconds,
        max_polls=args.max_polls,
    )
    print(f"[Scanner] request_id={result.request_id} polls={result.poll_count} "
          f"elapsed={result.elapsed_seconds:.1f}s")
    print(result.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

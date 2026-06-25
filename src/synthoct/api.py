from __future__ import annotations

import csv
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import requests
from skimage import io

from .dataset import load_scan_from_zip
from .metrics import calculate_metrics
from .phantom import ExperimentConfig


@dataclass(frozen=True)
class ApiRenderResult:
    request_id: str
    synthetic_png: Path
    synthetic_gray_png: Path
    reference_png: Path
    elapsed_seconds: float
    poll_count: int


DEFAULT_API_KEY_ENVS = ("SYNTHOCT_API_KEY", "SYNTHOCT_CHALLENGE_API_KEY")


def resolve_api_key(
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
    env_names: Sequence[str] = DEFAULT_API_KEY_ENVS,
) -> str:
    """Resolve the hosted scanner API key without ever persisting it in artifacts."""
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


def write_api_config(path: str | Path, scatterers_count: int = 300_000) -> Path:
    config = ExperimentConfig(scatterers_count=scatterers_count)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "[Parameters]",
                "scan filename = Scan_Raw.bin",
                "scatterers coordinates file = Scatterers.txt",
                f"a-scan pixel numbers = {config.n_depth}",
                f"vertical pixel size mcm = {config.pixel_size_z}",
                f"central wavelength mcm = {config.wavelength}",
                f"number of a-scans in b-scan = {config.n_lateral}",
                f"xmax mcm = {config.x_max}",
                f"number of b-scans = {config.b_scans_count}",
                "ymax mcm = 0.0",
                f"beam radius mcm = {config.beam_radius}",
                f"number of scatterers in b-scan = {scatterers_count}",
                "output filename = Output.png",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _to_gray_png(src: Path, dst: Path) -> Path:
    img = io.imread(src)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(dst, np.clip(img, 0, 255).astype(np.uint8))
    return dst


def _write_reference(zip_path: str | Path, archive_path: str, out_path: Path) -> Path:
    arr = load_scan_from_zip(zip_path, archive_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if arr.max(initial=0) <= 1.0:
        arr = arr * 255.0
    io.imsave(out_path, np.clip(arr, 0, 255).astype(np.uint8))
    return out_path


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
    api_key = resolve_api_key(api_key=api_key, api_key_file=api_key_file)

    phantom_path = Path(phantom_path)
    config_path = Path(config_path)
    out_png = Path(out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    with config_path.open("rb") as config_f, phantom_path.open("rb") as phantom_f:
        response = _request_with_retries(
            "POST",
            endpoint,
            headers={"X-API-Key": api_key},
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
            out_png.parent.mkdir(parents=True, exist_ok=True)
            out_png.write_bytes(result.content)
            elapsed = time.perf_counter() - start
            return request_id, out_png, elapsed, poll_idx
        if result.status_code not in {202, 404}:
            result.raise_for_status()

    raise TimeoutError(f"API result was not ready after {max_polls} polls: {result_url}")


def _iter_manifest_rows(manifest_path: Path, limit: int | None = None) -> Iterable[dict[str, str]]:
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            yield row


def benchmark_submission_api(
    zip_path: str | Path,
    submission_dir: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    scatterers_count: int = 300_000,
    include_lpips: bool = False,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 60,
    skip_existing: bool = True,
    progress: bool = False,
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
) -> tuple[Path, Path]:
    submission_dir = Path(submission_dir).resolve()
    out_dir = Path(out_dir).resolve()
    manifest_path = submission_dir / "submission_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    resolved_api_key: str | None = api_key

    config_path = write_api_config(out_dir / "Configuration_api.ini", scatterers_count=scatterers_count)
    results_csv = out_dir / "api_metrics.csv"
    fieldnames = [
        "source_archive_path",
        "phantom_path",
        "request_id",
        "reference_png",
        "synthetic_png",
        "synthetic_gray_png",
        "elapsed_seconds",
        "poll_count",
        "MSE",
        "PSNR",
        "SSIM",
        "MS-SSIM",
        "VIF",
        "LPIPS",
        "LPIPS_PROXY",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for idx, row in enumerate(_iter_manifest_rows(manifest_path, limit=limit)):
            source_archive_path = row["source_archive_path"]
            phantom_path = submission_dir / row["phantom_path"]
            stem = f"{idx:04d}_{Path(source_archive_path).stem}"
            reference_png = out_dir / "references" / f"{stem}_reference.png"
            synthetic_png = out_dir / "synthetic" / f"{stem}_synthetic.png"
            synthetic_gray_png = out_dir / "synthetic_gray" / f"{stem}_synthetic_gray.png"
            if progress:
                print(f"[{idx + 1}] rendering {source_archive_path}", file=sys.stderr, flush=True)

            _write_reference(zip_path, source_archive_path, reference_png)
            if skip_existing and synthetic_png.exists() and synthetic_gray_png.exists():
                request_id = "existing"
                elapsed_seconds = 0.0
                poll_count = 0
            else:
                if resolved_api_key is None:
                    resolved_api_key = resolve_api_key(api_key_file=api_key_file)
                request_id, synthetic_png, elapsed_seconds, poll_count = render_with_api(
                    phantom_path,
                    config_path,
                    synthetic_png,
                    api_key=resolved_api_key,
                    api_key_file=api_key_file,
                    poll_interval_seconds=poll_interval_seconds,
                    max_polls=max_polls,
                )
                _to_gray_png(synthetic_png, synthetic_gray_png)

            metrics = calculate_metrics(reference_png, synthetic_gray_png, include_lpips=include_lpips)
            if progress:
                print(
                    f"[{idx + 1}] request={request_id} elapsed={elapsed_seconds:.1f}s "
                    f"ms_ssim={metrics['MS-SSIM']:.4f} ssim={metrics['SSIM']:.4f}",
                    file=sys.stderr,
                    flush=True,
                )
            writer.writerow(
                {
                    "source_archive_path": source_archive_path,
                    "phantom_path": str(phantom_path),
                    "request_id": request_id,
                    "reference_png": str(reference_png),
                    "synthetic_png": str(synthetic_png),
                    "synthetic_gray_png": str(synthetic_gray_png),
                    "elapsed_seconds": elapsed_seconds,
                    "poll_count": poll_count,
                    **metrics,
                }
            )
            f.flush()
    write_preliminary_upload_plan(results_csv, out_dir / "preliminary_upload_plan.csv")
    return results_csv, config_path


def _score_for_upload_plan(row: dict[str, str]) -> float:
    for key in ("MS-SSIM", "SSIM"):
        try:
            value = float(row.get(key) or 0.0)
        except ValueError:
            continue
        if math.isfinite(value):
            return value
    return 0.0


def write_preliminary_upload_plan(results_csv: str | Path, out_csv: str | Path, limit: int | None = None) -> Path:
    results_csv = Path(results_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=_score_for_upload_plan, reverse=True)
    if limit is not None:
        rows = rows[:limit]
    fieldnames = [
        "upload_order",
        "source_archive_path",
        "reference_png",
        "synthetic_png",
        "synthetic_gray_png",
        "phantom_path",
        "request_id",
        "MS-SSIM",
        "SSIM",
        "LPIPS",
        "LPIPS_PROXY",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow({key: idx if key == "upload_order" else row.get(key, "") for key in fieldnames})
    return out_csv


def prepare_preliminary_png_pairs(upload_plan_csv: str | Path, out_dir: str | Path, limit: int | None = None) -> Path:
    """Copy rendered PNG pairs into a clean folder for the preliminary web form."""
    upload_plan_csv = Path(upload_plan_csv)
    out_dir = Path(out_dir)
    synthetic_dir = out_dir / "synthetic_scans"
    reference_dir = out_dir / "real_reference_scans"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    with upload_plan_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]

    manifest_path = out_dir / "png_pair_manifest.csv"
    fieldnames = [
        "upload_order",
        "source_archive_path",
        "synthetic_scan_png",
        "real_reference_scan_png",
        "source_synthetic_png",
        "source_reference_png",
        "MS-SSIM",
        "SSIM",
        "LPIPS_PROXY",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            order = int(row["upload_order"])
            stem = Path(row["source_archive_path"]).stem
            synthetic_src = Path(row["synthetic_png"])
            reference_src = Path(row["reference_png"])
            synthetic_dst = synthetic_dir / f"{order:03d}_{stem}_synthetic.png"
            reference_dst = reference_dir / f"{order:03d}_{stem}_reference.png"
            shutil.copyfile(synthetic_src, synthetic_dst)
            shutil.copyfile(reference_src, reference_dst)
            writer.writerow(
                {
                    "upload_order": order,
                    "source_archive_path": row["source_archive_path"],
                    "synthetic_scan_png": str(synthetic_dst),
                    "real_reference_scan_png": str(reference_dst),
                    "source_synthetic_png": str(synthetic_src),
                    "source_reference_png": str(reference_src),
                    "MS-SSIM": row.get("MS-SSIM", ""),
                    "SSIM": row.get("SSIM", ""),
                    "LPIPS_PROXY": row.get("LPIPS_PROXY", ""),
                }
            )
    return manifest_path

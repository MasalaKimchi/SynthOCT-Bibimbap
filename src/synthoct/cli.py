from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from skimage import io

from . import __version__
from .benchmark import run_local_benchmark
from .evaluation import (
    calculate_metrics,
    competition_formula_estimate,
    evaluate_feature_map_metrics,
    profile_scores,
)
from .evidence import build_evidence_manifest, verify_evidence_manifest
from .features import ORGANIZER_MAP_MODE, SCIENTIFIC_MAP_MODE, load_scan
from .holographic_inverse import HolographicInverseConfig, holographic_inverse_phantom
from .provenance import dependency_versions, git_provenance
from .scanners import render_reference_scanner, render_with_api, write_api_config
from .submission import (
    build_preliminary_submission,
    generate_phantom_batch,
    verify_submission_archive,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthoct",
        description="Coherent holographic inversion for SynthOCT digital phantoms.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser("baseline", help="Generate a scanner-compatible phantom.")
    baseline_sub = baseline.add_subparsers(dest="baseline_command", required=True)
    inverse = baseline_sub.add_parser("holographic-inverse", help="Invert a reference B-scan into a scanner phantom.")
    inverse.add_argument("--input", required=True, type=Path)
    inverse.add_argument("--out", required=True, type=Path)
    inverse.add_argument("--diagnostics", type=Path)
    inverse.add_argument("--seed", type=int, default=7)
    inverse.add_argument("--scatterers-count", type=int, default=300_000)
    inverse.add_argument("--axial-regularization", type=float, default=0.02)
    inverse.add_argument("--lateral-regularization", type=float, default=0.05)
    inverse.add_argument("--max-reflection-amplitude", type=float, default=0.001)
    inverse.add_argument("--dynamic-range-db", type=float, default=51.0)
    inverse.add_argument("--phase-iterations", type=int, default=200)
    inverse.add_argument("--phase-momentum", type=float, default=1.0)
    inverse.add_argument(
        "--phase-encoding",
        choices=("single", "dispersion-canceling-pair"),
        default="dispersion-canceling-pair",
    )

    scan = commands.add_parser("scan", help="Render a phantom through the hosted SynthOCT API.")
    scan.add_argument("--phantom", required=True, type=Path)
    scan.add_argument("--out", required=True, type=Path)
    scan.add_argument("--api-key-file", type=Path)
    scan.add_argument("--poll-interval-seconds", type=float, default=10.0)
    scan.add_argument("--max-polls", type=int, default=60)
    scan.add_argument("--scatterers-count", type=int, default=300_000)
    scan.add_argument("--manifest", type=Path)
    scan.add_argument("--reference", type=Path)
    scan.add_argument("--generator-diagnostics", type=Path)

    reference = commands.add_parser(
        "render-reference",
        help="Render with the local implementation of the published forward model.",
    )
    reference.add_argument("--phantom", required=True, type=Path)
    reference.add_argument("--out", required=True, type=Path)

    evaluate = commands.add_parser("evaluate", help="Compute full-frame image and feature-map metrics.")
    evaluate.add_argument("--ref", required=True, type=Path)
    evaluate.add_argument("--pred", required=True, type=Path)
    evaluate.add_argument("--out-csv", required=True, type=Path)
    evaluate.add_argument("--maps", action="store_true")
    evaluate.add_argument("--include-lpips", action="store_true")
    evaluate.add_argument(
        "--map-mode",
        choices=(ORGANIZER_MAP_MODE, SCIENTIFIC_MAP_MODE),
        default=ORGANIZER_MAP_MODE,
        help="Published competition encoding or calibrated float/masked audit maps.",
    )

    benchmark = commands.add_parser(
        "benchmark-local",
        help=(
            "Run a fixed inverse over a reference directory with the local "
            "implementation of the published forward model."
        ),
    )
    benchmark.add_argument("--input-root", required=True, type=Path)
    benchmark.add_argument("--out-dir", required=True, type=Path)
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--include-lpips", action="store_true")
    benchmark.add_argument("--no-maps", action="store_true")
    benchmark.add_argument(
        "--map-mode",
        choices=(ORGANIZER_MAP_MODE, SCIENTIFIC_MAP_MODE),
        default=ORGANIZER_MAP_MODE,
    )
    benchmark.add_argument("--scatterers-count", type=int, default=300_000)
    benchmark.add_argument("--axial-regularization", type=float, default=0.02)
    benchmark.add_argument("--lateral-regularization", type=float, default=0.05)
    benchmark.add_argument("--phase-iterations", type=int, default=200)
    benchmark.add_argument("--phase-momentum", type=float, default=1.0)
    benchmark.add_argument(
        "--phase-encoding",
        choices=("single", "dispersion-canceling-pair"),
        default="dispersion-canceling-pair",
    )

    generate_batch = commands.add_parser(
        "generate-batch",
        help="Invert every reference B-scan into a retained, submission-named phantom directory.",
    )
    generate_batch.add_argument("--reference-root", required=True, type=Path)
    generate_batch.add_argument("--out-dir", required=True, type=Path)
    generate_batch.add_argument("--limit", type=int)
    generate_batch.add_argument("--seed", type=int, default=7)
    generate_batch.add_argument("--scatterers-count", type=int, default=300_000)
    generate_batch.add_argument("--no-diagnostics", action="store_true")
    generate_batch.add_argument("--axial-regularization", type=float, default=0.02)
    generate_batch.add_argument("--lateral-regularization", type=float, default=0.05)
    generate_batch.add_argument("--max-reflection-amplitude", type=float, default=0.001)
    generate_batch.add_argument("--dynamic-range-db", type=float, default=51.0)
    generate_batch.add_argument("--phase-iterations", type=int, default=200)
    generate_batch.add_argument("--phase-momentum", type=float, default=1.0)
    generate_batch.add_argument(
        "--phase-encoding",
        choices=("single", "dispersion-canceling-pair"),
        default="dispersion-canceling-pair",
    )

    evidence = commands.add_parser(
        "evidence-manifest",
        help="Build or verify a complete SHA-256 manifest for a retained evidence root.",
    )
    evidence.add_argument("--root", required=True, type=Path)
    evidence.add_argument("--hosted-request-id")
    evidence.add_argument("--phantom", default="phantom.txt", type=Path)
    evidence.add_argument(
        "--claim",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="Repeatable typed claim, for example public120_paired_wins=120.",
    )
    evidence.add_argument("--verify", action="store_true")

    prepare_submission = commands.add_parser(
        "prepare-submission",
        help="Build a deterministic, validated root-flat ZIP of phantom text files.",
    )
    prepare_submission.add_argument("--reference-root", required=True, type=Path)
    prepare_submission.add_argument("--phantom-dir", required=True, type=Path)
    prepare_submission.add_argument("--out", required=True, type=Path)
    prepare_submission.add_argument("--count", type=int, default=60)
    prepare_submission.add_argument("--seed", type=int, default=2026)
    prepare_submission.add_argument("--manifest", type=Path)

    verify_submission = commands.add_parser(
        "verify-submission",
        help="Fail-closed validate an existing phantom submission ZIP.",
    )
    verify_submission.add_argument("--archive", required=True, type=Path)
    verify_submission.add_argument("--expected-count", type=int, default=60)
    verify_submission.add_argument("--manifest", type=Path)
    verify_submission.add_argument("--reference-root", type=Path)
    return parser


def _parse_claims(items: list[str]) -> dict[str, object]:
    claims: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"claim must be KEY=JSON_VALUE: {item}")
        key, value = item.split("=", 1)
        if not key or key in claims:
            raise ValueError(f"claim key is empty or duplicated: {key!r}")
        try:
            claims[key] = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"claim value must be valid JSON: {item}") from error
    return claims


def _write_grayscale(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = np.rint(load_scan(source) * 255.0).astype(np.uint8)
    io.imsave(destination, image, check_contrast=False)
    return destination


def _write_metrics(path: Path, row: dict[str, float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, str | int]:
    resolved = path.resolve()
    root = Path(__file__).resolve().parents[2]
    try:
        display_path = resolved.relative_to(root)
    except ValueError:
        display_path = resolved
    return {
        "path": str(display_path),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _write_api_manifest(
    path: Path,
    *,
    phantom: Path,
    config: Path,
    raw_png: Path,
    gray_png: Path,
    request_id: str,
    elapsed_seconds: float,
    poll_count: int,
    scatterers_count: int,
    reference: Path | None = None,
    generator_diagnostics: Path | None = None,
) -> Path:
    inputs = {
        "phantom": _file_record(phantom),
        "scanner_config": _file_record(config),
    }
    if reference is not None:
        inputs["evaluation_reference"] = _file_record(reference)
    if generator_diagnostics is not None:
        inputs["generator_diagnostics"] = _file_record(generator_diagnostics)
    payload = {
        "schema_version": 3,
        "command": "synthoct-hosted-api-render-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "request": {
            "request_id": request_id,
            "elapsed_seconds": elapsed_seconds,
            "poll_count": poll_count,
        },
        "inputs": inputs,
        "outputs": {
            "raw_scanner_png": _file_record(raw_png),
            "derived_gray_png": _file_record(gray_png),
        },
        "parameters": {"backend": "hosted_api", "scatterers_count": scatterers_count},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
            "git": git_provenance(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "baseline":
        if args.baseline_command != "holographic-inverse":
            raise ValueError(f"Unknown baseline command: {args.baseline_command}")
        result = holographic_inverse_phantom(
            args.input,
            args.out,
            seed=args.seed,
            scatterers_count=args.scatterers_count,
            inverse=HolographicInverseConfig(
                axial_regularization=args.axial_regularization,
                lateral_regularization=args.lateral_regularization,
                max_reflection_amplitude=args.max_reflection_amplitude,
                dynamic_range_db=args.dynamic_range_db,
                phase_iterations=args.phase_iterations,
                phase_momentum=args.phase_momentum,
                phase_encoding=args.phase_encoding,
            ),
            diagnostics_path=args.diagnostics,
        )
        print(result)
        return 0

    if args.command == "scan":
        config = write_api_config(
            args.out.with_name(f"{args.out.stem}_Configuration.ini"),
            scatterers_count=args.scatterers_count,
        )
        request_id, rendered, elapsed, polls = render_with_api(
            args.phantom,
            config,
            args.out,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
        )
        gray = _write_grayscale(rendered, args.out.with_name(f"{args.out.stem}_gray.png"))
        if args.manifest:
            print(
                _write_api_manifest(
                    args.manifest,
                    phantom=args.phantom,
                    config=config,
                    raw_png=rendered,
                    gray_png=gray,
                    request_id=request_id,
                    elapsed_seconds=elapsed,
                    poll_count=polls,
                    scatterers_count=args.scatterers_count,
                    reference=args.reference,
                    generator_diagnostics=args.generator_diagnostics,
                )
            )
        print(rendered)
        print(gray)
        return 0

    if args.command == "render-reference":
        print(render_reference_scanner(args.phantom, args.out))
        return 0

    if args.command == "evaluate":
        struct_metrics = calculate_metrics(args.ref, args.pred, include_lpips=args.include_lpips)
        row = (
            {f"Struct_{key}": value for key, value in struct_metrics.items()}
            if args.maps
            else struct_metrics
        )
        row.update(profile_scores(args.ref, args.pred, map_mode=args.map_mode))
        if args.maps:
            map_root = args.out_csv.parent / f"{args.out_csv.stem}_maps"
            row.update(
                evaluate_feature_map_metrics(
                    args.ref,
                    args.pred,
                    map_root / "reference",
                    map_root / "prediction",
                    include_lpips=args.include_lpips,
                    map_mode=args.map_mode,
                )
            )
            if args.include_lpips:
                score, medians = competition_formula_estimate(
                    row,
                    map_mode=args.map_mode,
                )
                row.update(medians)
                row["competition_formula_estimate"] = score
        print(_write_metrics(args.out_csv, row))
        return 0

    if args.command == "benchmark-local":
        detail, summary = run_local_benchmark(
            args.input_root,
            args.out_dir,
            inverse=HolographicInverseConfig(
                axial_regularization=args.axial_regularization,
                lateral_regularization=args.lateral_regularization,
                phase_iterations=args.phase_iterations,
                phase_momentum=args.phase_momentum,
                phase_encoding=args.phase_encoding,
            ),
            scatterers_count=args.scatterers_count,
            include_maps=not args.no_maps,
            include_lpips=args.include_lpips,
            limit=args.limit,
            map_mode=args.map_mode,
        )
        print(detail)
        print(summary)
        return 0

    if args.command == "evidence-manifest":
        if args.verify:
            verify_evidence_manifest(args.root)
            print(args.root / "evidence_manifest.json")
            return 0
        if not args.hosted_request_id:
            raise ValueError("--hosted-request-id is required when building a manifest")
        print(
            build_evidence_manifest(
                args.root,
                hosted_request_id=args.hosted_request_id,
                claims=_parse_claims(args.claim),
                phantom=args.phantom,
            )
        )
        return 0

    if args.command == "generate-batch":
        output_dir, summary = generate_phantom_batch(
            args.reference_root,
            args.out_dir,
            inverse=HolographicInverseConfig(
                axial_regularization=args.axial_regularization,
                lateral_regularization=args.lateral_regularization,
                max_reflection_amplitude=args.max_reflection_amplitude,
                dynamic_range_db=args.dynamic_range_db,
                phase_iterations=args.phase_iterations,
                phase_momentum=args.phase_momentum,
                phase_encoding=args.phase_encoding,
            ),
            seed=args.seed,
            scatterers_count=args.scatterers_count,
            limit=args.limit,
            retain_diagnostics=not args.no_diagnostics,
        )
        print(output_dir)
        print(summary)
        return 0

    if args.command == "prepare-submission":
        archive, manifest = build_preliminary_submission(
            args.reference_root,
            args.phantom_dir,
            args.out,
            count=args.count,
            seed=args.seed,
            manifest_path=args.manifest,
        )
        print(archive)
        print(manifest)
        return 0

    if args.command == "verify-submission":
        report = verify_submission_archive(
            args.archive,
            expected_count=args.expected_count,
            manifest_path=args.manifest,
            reference_root=args.reference_root,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

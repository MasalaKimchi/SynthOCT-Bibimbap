from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from skimage import io

from . import __version__
from .evaluation import calculate_metrics, evaluate_feature_map_metrics, profile_scores
from .features import load_scan
from .holographic_inverse import HolographicInverseConfig, holographic_inverse_phantom
from .scanners import render_reference_scanner, render_with_api, write_api_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthoct",
        description="Coherent holographic inversion for SynthOCT digital phantoms.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser("baseline", help="Generate the winning phantom.")
    baseline_sub = baseline.add_subparsers(dest="baseline_command", required=True)
    inverse = baseline_sub.add_parser("holographic-inverse", help="Invert a reference B-scan into a scanner phantom.")
    inverse.add_argument("--input", required=True, type=Path)
    inverse.add_argument("--out", required=True, type=Path)
    inverse.add_argument("--diagnostics", type=Path)
    inverse.add_argument("--seed", type=int, default=7)
    inverse.add_argument("--scatterers-count", type=int, default=300_000)
    inverse.add_argument("--axial-regularization", type=float, default=0.03)
    inverse.add_argument("--lateral-regularization", type=float, default=0.20)
    inverse.add_argument("--max-reflection-amplitude", type=float, default=0.001)
    inverse.add_argument("--dynamic-range-db", type=float, default=51.0)

    scan = commands.add_parser("scan", help="Render a phantom through the hosted SynthOCT API.")
    scan.add_argument("--phantom", required=True, type=Path)
    scan.add_argument("--out", required=True, type=Path)
    scan.add_argument("--api-key-file", type=Path)
    scan.add_argument("--poll-interval-seconds", type=float, default=10.0)
    scan.add_argument("--max-polls", type=int, default=60)
    scan.add_argument("--scatterers-count", type=int, default=300_000)

    reference = commands.add_parser("render-reference", help="Render with the recovered exact scanner locally.")
    reference.add_argument("--phantom", required=True, type=Path)
    reference.add_argument("--out", required=True, type=Path)

    evaluate = commands.add_parser("evaluate", help="Compute full-frame image and feature-map metrics.")
    evaluate.add_argument("--ref", required=True, type=Path)
    evaluate.add_argument("--pred", required=True, type=Path)
    evaluate.add_argument("--out-csv", required=True, type=Path)
    evaluate.add_argument("--maps", action="store_true")
    evaluate.add_argument("--include-lpips", action="store_true")
    return parser


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
        _request_id, rendered, _elapsed, _polls = render_with_api(
            args.phantom,
            config,
            args.out,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
        )
        gray = _write_grayscale(rendered, args.out.with_name(f"{args.out.stem}_gray.png"))
        print(rendered)
        print(gray)
        return 0

    if args.command == "render-reference":
        print(render_reference_scanner(args.phantom, args.out))
        return 0

    if args.command == "evaluate":
        row = calculate_metrics(args.ref, args.pred, include_lpips=args.include_lpips)
        row.update(profile_scores(args.ref, args.pred))
        if args.maps:
            map_root = args.out_csv.parent / f"{args.out_csv.stem}_maps"
            row.update(
                evaluate_feature_map_metrics(
                    args.ref,
                    args.pred,
                    map_root / "reference",
                    map_root / "prediction",
                    include_lpips=args.include_lpips,
                )
            )
        print(_write_metrics(args.out_csv, row))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

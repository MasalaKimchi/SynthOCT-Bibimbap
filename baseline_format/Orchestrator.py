"""Orchestrator — the manager, reconfigured for Challenge mode (real vs synthetic).

Baseline role (``Orchestrator.py``): ``import Part1_Generator as Generator`` and
drive generate -> scan -> maps -> metrics, writing ``Final_Metrics_Report.csv``.
The shipped baseline runs a *self-consistency* check; its README says "Please
reconfigure for real/synthetic comparison (Challenge mode)."  This Orchestrator
defaults to that Challenge mode and keeps the self-consistency check as an option.

Pipeline (Challenge mode), per reference OCT B-scan:

    Part1  generate_from_reference  ->  digital phantom .txt   (phase-pair inverse)
    Part2  scan (organizer-hosted service) ->  synthetic B-scan .png
    Part3  generate_maps            ->  Struct / OAC / SC / RSC  (both ref & synth)
    metrics: synthetic vs real, per map  (the ``synthoct`` evaluation stack)

The metric rows reproduce ``synthoct``'s ``benchmark.py`` schema exactly (Struct
+ profile + OAC/SC/RSC), and, when ``--include-lpips`` is set with the default
organizer maps, the published eight-median competition-formula estimate.

Scores written here are **local / hosted development estimates, explicitly not an
organizer-issued leaderboard or hidden-test score** (``official_or_hidden_score``
is recorded as ``false``).  Rendering requires a SynthOCT API key
(``$SYNTHOCT_API_KEY`` or ``--api-key-file``) and network access.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (must precede any `synthoct` import; sets sys.path)

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import Part1_Generator as Generator
import Part2_Scanner as Scanner
import Part3_Processor as Processor
from synthoct.evaluation import (
    calculate_metrics,
    competition_formula_estimate,
    evaluate_feature_map_metrics,
    profile_scores,
)
from synthoct.features import ORGANIZER_MAP_MODE, SCIENTIFIC_MAP_MODE

try:  # provenance is best-effort; never block a run on it
    from synthoct.provenance import dependency_versions, git_provenance
except Exception:  # pragma: no cover

    def dependency_versions() -> dict:
        return {}

    def git_provenance() -> dict:
        return {}


MAP_TYPES = ("Struct", "OAC", "SC", "RSC")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _method_name() -> str:
    inv = Generator.FIXED_INVERSE
    if inv.phase_encoding == "dispersion-canceling-pair":
        return "holographic-inverse-v3-phase-pair"
    if inv.phase_iterations:
        return "holographic-inverse-v2-phase-retrieval"
    return "holographic-inverse-v1"


def _fmt(val) -> str:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return "N/A"
    return f"{f:.4f}" if np.isfinite(f) else "N/A"


def discover_references(root: Path, limit: int | None) -> list[Path]:
    """Reference B-scans below ``root`` (excludes derived _OAC/_SC/_RSC maps)."""
    references = [
        path
        for path in sorted(root.rglob("*.png"))
        if not path.stem.endswith(("_OAC", "_SC", "_RSC"))
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        references = references[:limit]
    return references


def flatten_name(reference: Path, root: Path) -> str:
    """Collision-safe stem from a reference path relative to ``root``."""
    try:
        rel = reference.resolve().relative_to(root.resolve())
    except ValueError:
        rel = Path(reference.name)
    return "_".join(rel.with_suffix("").parts)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_metric_table(row: dict, keyed_by_map: bool) -> None:
    for m in MAP_TYPES:
        prefix = f"{m}_"
        if keyed_by_map and f"{prefix}MS-SSIM" not in row:
            continue
        print(
            f"   [{m:6}] SSIM: {_fmt(row.get(prefix + 'SSIM'))} | "
            f"MS-SSIM: {_fmt(row.get(prefix + 'MS-SSIM'))} | "
            f"PSNR: {_fmt(row.get(prefix + 'PSNR'))} | "
            f"MSE: {_fmt(row.get(prefix + 'MSE'))} | "
            f"VIF: {_fmt(row.get(prefix + 'VIF'))} | "
            f"LPIPS: {_fmt(row.get(prefix + 'LPIPS'))}"
        )


def evaluate_pair(
    reference_png: Path,
    synthetic_png: Path,
    map_root: Path,
    *,
    include_lpips: bool,
    map_mode: str,
) -> dict:
    """Build one metric row (synthetic vs real), matching benchmark.py's schema."""
    struct = calculate_metrics(reference_png, synthetic_png, include_lpips=include_lpips)
    row: dict = {f"Struct_{key}": value for key, value in struct.items()}
    row.update(profile_scores(reference_png, synthetic_png, map_mode=map_mode))
    row.update(
        evaluate_feature_map_metrics(
            reference_png,
            synthetic_png,
            map_root / "reference",
            map_root / "prediction",
            include_lpips=include_lpips,
            map_mode=map_mode,
        )
    )
    return row


def _environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        "git": git_provenance(),
    }


# --------------------------------------------------------------------------- #
# Challenge mode: synthetic (from inverse phantom) vs real reference
# --------------------------------------------------------------------------- #
def run_challenge(args: argparse.Namespace) -> int:
    root = args.reference_root
    references = discover_references(root, args.limit)
    if not references:
        raise ValueError(f"no reference PNGs found below {root}")

    out_dir = args.out_dir
    phantoms_dir = out_dir / "phantoms"
    renders_dir = out_dir / "renders"
    maps_dir = out_dir / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = Generator.ScattererGenerator(
        Generator.ExperimentConfig(scatterers_count=args.scatterers_count)
    )
    method = _method_name()

    print("==================================================")
    print("   SynthOCT Orchestrator - Challenge Mode (hosted)")
    print("==================================================")
    print(f"[Setup] method={method}  references={len(references)}  root={root}")

    rows: list[dict] = []
    hosted_requests: list[dict] = []
    detail_path = out_dir / "Final_Metrics_Report.csv"
    partial_path = out_dir / "Final_Metrics_Report.partial.csv"
    started = time.perf_counter()

    for index, reference in enumerate(references, start=1):
        name = flatten_name(reference, root)
        rel = str(reference.resolve().relative_to(root.resolve())) if reference.is_absolute() else str(reference)
        print(f"\n--- [{index}/{len(references)}] {rel} ---")

        phantom = phantoms_dir / f"{name}.txt"
        diagnostics = phantoms_dir / f"{name}.diagnostics.json"
        print("Generating phantom (Part1: phase-pair inverse)...")
        generator.generate_from_reference(
            reference, phantom, seed=args.seed, diagnostics_path=diagnostics
        )

        render = renders_dir / f"{name}.png"
        print("Scanning phantom (Part2: organizer-hosted service)...")
        scan_result = Scanner.scan(
            phantom,
            render,
            api_key_file=args.api_key_file,
            scatterers_count=args.scatterers_count,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
        )

        print("Generating maps + metrics (Part3 + evaluation)...")
        row = evaluate_pair(
            reference,
            scan_result.png,
            maps_dir / name,
            include_lpips=args.include_lpips,
            map_mode=args.map_mode,
        )
        row = {
            "index": index,
            "reference": rel,
            "method": method,
            "evidence_source": "organizer_hosted_challenge_service",
            "request_id": scan_result.request_id,
            **row,
        }
        rows.append(row)
        hosted_requests.append(
            {
                "reference": rel,
                "request_id": scan_result.request_id,
                "elapsed_seconds": scan_result.elapsed_seconds,
                "poll_count": scan_result.poll_count,
            }
        )
        _write_rows(partial_path, rows)
        _print_metric_table(row, keyed_by_map=(args.map_mode == ORGANIZER_MAP_MODE))

    _write_rows(detail_path, rows)
    partial_path.unlink(missing_ok=True)

    summary: dict = {
        "mode": "challenge",
        "method": method,
        "evidence_source": "organizer_hosted_challenge_service",
        "official_or_hidden_score": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(rows),
        "reference_root": str(root),
        "map_mode": args.map_mode,
        "include_lpips": args.include_lpips,
        "parameters": asdict(Generator.FIXED_INVERSE),
        "scanner": asdict(Generator.ExperimentConfig(scatterers_count=args.scatterers_count).to_synthoct()),
        "hosted_requests": hosted_requests,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": _environment(),
    }
    if args.include_lpips and args.map_mode == ORGANIZER_MAP_MODE:
        score, medians = competition_formula_estimate(rows, map_mode=args.map_mode)
        summary["competition_formula_estimate"] = score
        summary["competition_medians"] = medians
        print(f"\n[Estimate] competition_formula_estimate (local, non-official) = {score:.6f}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n=== DONE. Report: {detail_path} | Summary: {summary_path} ===")
    return 0


# --------------------------------------------------------------------------- #
# Self-consistency mode: faithful port of the shipped baseline check (hosted)
# --------------------------------------------------------------------------- #
def _scan_and_map(generator_data, name: str, out_dir: Path, args: argparse.Namespace):
    """Save a phantom, hosted-scan it, and generate its maps."""
    phantom = out_dir / f"Scatterers_{name}.txt"
    Generator.ScattererGenerator(
        Generator.ExperimentConfig(scatterers_count=args.scatterers_count)
    ).save_to_file(generator_data, phantom)
    scan = out_dir / f"Scan_{name}.png"
    print(f"Scanning {name} (organizer-hosted service)...")
    result = Scanner.scan(
        phantom,
        scan,
        api_key_file=args.api_key_file,
        scatterers_count=args.scatterers_count,
        poll_interval_seconds=args.poll_interval_seconds,
        max_polls=args.max_polls,
    )
    maps = Processor.generate_maps(str(result.png))
    return result, maps


def run_self_consistency(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = Generator.ScattererGenerator(
        Generator.ExperimentConfig(scatterers_count=args.scatterers_count)
    )

    print("==================================================")
    print("   SynthOCT Orchestrator - Self-Consistency (hosted)")
    print("==================================================")

    experiments = [
        ("Exp1_Uniform", lambda seed: gen.generate_uniform(seed=seed, amp=1.0), 42, 101),
        (
            "Exp2_Layers",
            lambda seed: gen.generate_two_layers(
                seed=seed, boundary_z_mcm=400.0, amp_top=0.01, amp_bottom=2.5
            ),
            42,
            101,
        ),
    ]
    all_rows: list[dict] = []
    map_bank: dict[str, dict] = {}

    for exp_name, gen_func, seed_a, seed_b in experiments:
        print(f"\n--- Running Experiment: {exp_name} ---")
        _, maps_a = _scan_and_map(gen_func(seed_a), f"{exp_name}_A", out_dir, args)
        _, maps_b = _scan_and_map(gen_func(seed_b), f"{exp_name}_B", out_dir, args)
        map_bank[exp_name] = maps_a
        row: dict = {"Experiment": exp_name}
        for m in MAP_TYPES:
            metrics = calculate_metrics(maps_a[m], maps_b[m], include_lpips=args.include_lpips)
            for key, value in metrics.items():
                row[f"{m}_{key}"] = value
        all_rows.append(row)
        _print_metric_table(row, keyed_by_map=True)

    if "Exp1_Uniform" in map_bank and "Exp2_Layers" in map_bank:
        print("\n--- Cross-Comparison: Uniform vs Layers ---")
        cross: dict = {"Experiment": "Cross_Uniform_vs_Layers"}
        for m in MAP_TYPES:
            metrics = calculate_metrics(
                map_bank["Exp1_Uniform"][m],
                map_bank["Exp2_Layers"][m],
                include_lpips=args.include_lpips,
            )
            for key, value in metrics.items():
                cross[f"{m}_{key}"] = value
        all_rows.append(cross)
        _print_metric_table(cross, keyed_by_map=True)

    report = out_dir / "Final_Metrics_Report.csv"
    _write_rows(report, all_rows)
    print(f"\n=== DONE. Results saved to {report} ===")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Orchestrator",
        description="Drive Part1 -> Part2 (hosted) -> Part3 -> metrics for SynthOCT.",
    )
    parser.add_argument(
        "--mode",
        choices=("challenge", "self-consistency"),
        default="challenge",
        help="challenge = synthetic vs real references; self-consistency = baseline A/B check.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="Directory of real reference OCT PNGs (challenge mode).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/baseline_format_run"),
        help="Where phantoms, renders, maps, and reports are written.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N references.")
    parser.add_argument("--include-lpips", action="store_true", help="Compute real AlexNet LPIPS + competition estimate.")
    parser.add_argument(
        "--map-mode",
        choices=(ORGANIZER_MAP_MODE, SCIENTIFIC_MAP_MODE),
        default=ORGANIZER_MAP_MODE,
    )
    parser.add_argument("--api-key-file", type=Path, help="File containing the SynthOCT API key.")
    parser.add_argument("--seed", type=int, default=Generator.DEFAULT_SEED)
    parser.add_argument("--scatterers-count", type=int, default=300_000)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--max-polls", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "challenge":
        if args.reference_root is None:
            raise SystemExit("challenge mode requires --reference-root")
        return run_challenge(args)
    return run_self_consistency(args)


if __name__ == "__main__":
    raise SystemExit(main())

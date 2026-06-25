from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from . import __version__
from .api import benchmark_submission_api, prepare_preliminary_png_pairs, write_preliminary_upload_plan
from .baselines import (
    FINAL_CONFIG_NAME,
    HYPOTHESIS_CONFIGS,
    final_baseline,
    heuristic_baseline,
    hypothesis_baseline,
    official_baseline,
    physics_guided_baseline,
)
from .dataset import iter_records, prepare_dataset
from .metrics import calculate_metrics
from .optimizer import run_candidate_search
from .processor import generate_maps
from .scanner import run_scanner
from .submission import prepare_submission_bundle
from .validation import METHOD_WAVES, METHODS, plot_hypothesis_progress, resolve_method_wave, run_internal_validation


def _add_common_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="Output phantom .txt path.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scatterers-count", type=int, default=300_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthoct", description="SynthOCT challenge baselines.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="Dataset utilities.")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_prepare = data_sub.add_parser("prepare", help="Extract/list dataset and write manifest.")
    data_prepare.add_argument("--zip", required=True, dest="zip_path")
    data_prepare.add_argument("--out", required=True)
    data_prepare.add_argument("--no-extract", action="store_true")
    data_list = data_sub.add_parser("list", help="Print dataset records as JSON lines.")
    data_list.add_argument("--zip", required=True, dest="zip_path")

    baseline = sub.add_parser("baseline", help="Generate phantom baselines.")
    base_sub = baseline.add_subparsers(dest="baseline_command", required=True)
    official = base_sub.add_parser("official")
    _add_common_baseline_args(official)
    official.add_argument("--method", choices=["uniform", "two-layer"], default="two-layer")
    heuristic = base_sub.add_parser("heuristic")
    heuristic.add_argument("--input", required=True)
    _add_common_baseline_args(heuristic)
    physics = base_sub.add_parser("physics-guided")
    physics.add_argument("--input", required=True)
    physics.add_argument("--lateral-bins", type=int, default=64)
    physics.add_argument("--depth-bins", type=int, default=64)
    _add_common_baseline_args(physics)
    hypothesis = base_sub.add_parser("hypothesis")
    hypothesis.add_argument("--input", required=True)
    hypothesis.add_argument("--name", choices=tuple(HYPOTHESIS_CONFIGS.keys()), default=FINAL_CONFIG_NAME)
    _add_common_baseline_args(hypothesis)
    final = base_sub.add_parser("final")
    final.add_argument("--input", required=True)
    _add_common_baseline_args(final)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--ref", required=True)
    evaluate.add_argument("--pred", required=True)
    evaluate.add_argument("--maps", action="store_true")
    evaluate.add_argument("--metrics", action="store_true")
    evaluate.add_argument("--out-csv")
    evaluate.add_argument("--no-lpips", action="store_true")

    scan = sub.add_parser("scan")
    scan.add_argument("--phantom", required=True)
    scan.add_argument("--out", required=True)
    scan.add_argument("--mode", choices=["real", "precomputed"], default="real")
    scan.add_argument("--scanner-exe", default="Part2_Scanner.exe")
    scan.add_argument("--precomputed")

    bench = sub.add_parser("benchmark")
    bench.add_argument("--command", dest="bench_command", nargs=argparse.REMAINDER, required=True)
    bench.add_argument("--limit-seconds", type=float, default=600.0)

    validate = sub.add_parser("validate-internal", help="Hosted-API validation against Zenodo reference scans.")
    validate.add_argument("--zip", required=True, dest="zip_path")
    validate.add_argument("--out", default="outputs/internal_validation")
    validate.add_argument("--methods", nargs="+", choices=METHODS)
    validate.add_argument("--wave", choices=METHOD_WAVES, help="Named method set for staged hypothesis triage.")
    validate.add_argument("--folds", type=int, default=3)
    validate.add_argument("--max-per-fold", type=int, default=4)
    validate.add_argument("--scatterers-count", type=int, default=300_000)
    validate.add_argument("--no-maps", action="store_true")
    validate.add_argument("--include-lpips", action="store_true")
    validate.add_argument("--seed", type=int, default=7)
    validate.add_argument("--plot", help="Optional path for hypothesis progression figure.")
    validate.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    validate.add_argument("--poll-interval-seconds", type=float, default=10.0)
    validate.add_argument("--max-polls", type=int, default=60)
    validate.add_argument("--rerun-existing", action="store_true")

    optimize = sub.add_parser("optimize-physics", help="Hosted-API candidate search; expensive because each candidate is rendered by the challenge scanner.")
    optimize.add_argument("--zip", required=True, dest="zip_path")
    optimize.add_argument("--out", default="outputs/optimizer")
    optimize.add_argument("--folds", type=int, default=3)
    optimize.add_argument("--max-per-fold", type=int, default=1)
    optimize.add_argument("--scatterers-count", type=int, default=300_000)
    optimize.add_argument("--random-count", type=int, default=48)
    optimize.add_argument("--seed", type=int, default=23)
    optimize.add_argument("--no-maps", action="store_true")
    optimize.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    optimize.add_argument("--poll-interval-seconds", type=float, default=10.0)
    optimize.add_argument("--max-polls", type=int, default=60)
    optimize.add_argument("--rerun-existing", action="store_true")

    submit = sub.add_parser("prepare-submission")
    submit.add_argument("--zip", required=True, dest="zip_path")
    submit.add_argument("--out", default="outputs/submission_ready")
    submit.add_argument("--scatterers-count", type=int, default=300_000)
    submit.add_argument("--limit", type=int, help="Limit number of PNG B-scans for smoke packaging.")
    submit.add_argument("--seed", type=int, default=7)
    submit.add_argument("--method", choices=tuple(HYPOTHESIS_CONFIGS.keys()), help="Hypothesis method to package.")

    api_eval = sub.add_parser("api-evaluate-submission", help="Render submission phantoms with the hosted SynthOCT API.")
    api_eval.add_argument("--zip", required=True, dest="zip_path")
    api_eval.add_argument("--submission-dir", required=True)
    api_eval.add_argument("--out", default="outputs/api_preliminary")
    api_eval.add_argument("--limit", type=int, help="Limit number of manifest rows to render.")
    api_eval.add_argument("--scatterers-count", type=int, default=300_000)
    api_eval.add_argument("--include-lpips", action="store_true")
    api_eval.add_argument("--poll-interval-seconds", type=float, default=10.0)
    api_eval.add_argument("--max-polls", type=int, default=60)
    api_eval.add_argument("--rerun-existing", action="store_true")
    api_eval.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")

    upload_plan = sub.add_parser("prepare-upload-plan", help="Rank rendered PNG pairs for preliminary portal upload.")
    upload_plan.add_argument("--api-results", required=True, help="Path to api_metrics.csv from api-evaluate-submission.")
    upload_plan.add_argument("--out", required=True, help="Output upload plan CSV.")
    upload_plan.add_argument("--limit", type=int, help="Keep only the top N ranked pairs.")

    png_pairs = sub.add_parser("prepare-png-pairs", help="Copy rendered synthetic/reference PNGs into a clean preliminary-upload folder.")
    png_pairs.add_argument("--upload-plan", required=True, help="Path to preliminary_upload_plan.csv.")
    png_pairs.add_argument("--out", required=True, help="Output directory with synthetic_scans/ and real_reference_scans/.")
    png_pairs.add_argument("--limit", type=int, help="Copy only the top N pairs from the upload plan.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "data":
        if args.data_command == "prepare":
            manifest = prepare_dataset(args.zip_path, args.out, extract=not args.no_extract)
            print(manifest)
            return 0
        for record in iter_records(args.zip_path):
            print(json.dumps(record.__dict__, sort_keys=True))
        return 0

    if args.command == "baseline":
        if args.baseline_command == "official":
            path = official_baseline(args.out, method=args.method, seed=args.seed, scatterers_count=args.scatterers_count)
        elif args.baseline_command == "heuristic":
            path = heuristic_baseline(args.input, args.out, seed=args.seed, scatterers_count=args.scatterers_count)
        elif args.baseline_command == "physics-guided":
            path = physics_guided_baseline(
                args.input,
                args.out,
                seed=args.seed,
                scatterers_count=args.scatterers_count,
                lateral_bins=args.lateral_bins,
                depth_bins=args.depth_bins,
            )
        elif args.baseline_command == "hypothesis":
            path = hypothesis_baseline(args.input, args.out, args.name, seed=args.seed, scatterers_count=args.scatterers_count)
        elif args.baseline_command == "final":
            path = final_baseline(args.input, args.out, seed=args.seed, scatterers_count=args.scatterers_count)
        else:
            raise ValueError(f"Unknown baseline command: {args.baseline_command}")
        print(path)
        return 0

    if args.command == "evaluate":
        rows = []
        pairs = [("Struct", Path(args.ref), Path(args.pred))]
        if args.maps:
            ref_maps = generate_maps(args.ref, output_dir=Path(args.ref).parent / "maps")
            pred_maps = generate_maps(args.pred, output_dir=Path(args.pred).parent / "maps")
            pairs.extend((name, ref_maps[name], pred_maps[name]) for name in ["OAC", "SC", "RSC"])
        if args.metrics or not args.maps:
            for name, ref, pred in pairs:
                row = {"Map_Type": name, **calculate_metrics(ref, pred, include_lpips=not args.no_lpips)}
                rows.append(row)
                print(json.dumps(row, sort_keys=True))
        if args.out_csv and rows:
            with Path(args.out_csv).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        return 0

    if args.command == "scan":
        path = run_scanner(args.phantom, args.out, mode=args.mode, scanner_exe=args.scanner_exe, precomputed_path=args.precomputed)
        print(path)
        return 0

    if args.command == "benchmark":
        import subprocess

        start = time.perf_counter()
        result = subprocess.run(args.bench_command, check=False)
        elapsed = time.perf_counter() - start
        print(json.dumps({"elapsed_seconds": elapsed, "limit_seconds": args.limit_seconds, "within_limit": elapsed <= args.limit_seconds}))
        return result.returncode

    if args.command == "validate-internal":
        methods = resolve_method_wave(args.wave, args.methods)
        detail, summary = run_internal_validation(
            args.zip_path,
            args.out,
            methods=methods,
            folds=args.folds,
            max_per_fold=args.max_per_fold,
            scatterers_count=args.scatterers_count,
            include_maps=not args.no_maps,
            include_lpips=args.include_lpips,
            seed=args.seed,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            skip_existing=not args.rerun_existing,
        )
        result = {
            "challenge_metrics": str(Path(args.out).resolve() / "challenge_metrics_summary.csv"),
            "detail": str(detail),
            "summary": str(summary),
            "wins": str(Path(args.out).resolve() / "hypothesis_wins.csv"),
        }
        if args.plot:
            result["plot"] = str(plot_hypothesis_progress(summary, args.plot))
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "optimize-physics":
        detail, summary, best_config = run_candidate_search(
            args.zip_path,
            args.out,
            folds=args.folds,
            max_per_fold=args.max_per_fold,
            scatterers_count=args.scatterers_count,
            random_count=args.random_count,
            seed=args.seed,
            include_maps=not args.no_maps,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            skip_existing=not args.rerun_existing,
        )
        print(json.dumps({"detail": str(detail), "summary": str(summary), "best_config": str(best_config)}, sort_keys=True))
        return 0

    if args.command == "prepare-submission":
        bundle = prepare_submission_bundle(
            args.zip_path,
            Path.cwd(),
            args.out,
            scatterers_count=args.scatterers_count,
            limit=args.limit,
            seed=args.seed,
            method=args.method or FINAL_CONFIG_NAME,
        )
        print(json.dumps({k: str(v) for k, v in bundle.__dict__.items()}, sort_keys=True))
        return 0

    if args.command == "api-evaluate-submission":
        results_csv, config_path = benchmark_submission_api(
            args.zip_path,
            args.submission_dir,
            args.out,
            limit=args.limit,
            scatterers_count=args.scatterers_count,
            include_lpips=args.include_lpips,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            skip_existing=not args.rerun_existing,
            progress=True,
            api_key_file=args.api_key_file,
        )
        upload_plan = Path(args.out).resolve() / "preliminary_upload_plan.csv"
        print(json.dumps({"results_csv": str(results_csv), "config": str(config_path), "upload_plan": str(upload_plan)}, sort_keys=True))
        return 0

    if args.command == "prepare-upload-plan":
        path = write_preliminary_upload_plan(args.api_results, args.out, limit=args.limit)
        print(path)
        return 0

    if args.command == "prepare-png-pairs":
        manifest = prepare_preliminary_png_pairs(args.upload_plan, args.out, limit=args.limit)
        print(manifest)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

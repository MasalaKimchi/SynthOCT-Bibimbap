from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from . import __version__
from .baselines import heuristic_baseline, official_baseline, parameter_search_baseline, pretrained_cnn_baseline
from .dataset import iter_records, prepare_dataset
from .metrics import calculate_metrics
from .processor import generate_maps
from .scanner import run_scanner
from .train import train_hybrid


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
    search = base_sub.add_parser("parameter-search")
    search.add_argument("--input", required=True)
    _add_common_baseline_args(search)
    cnn = base_sub.add_parser("pretrained-cnn")
    cnn.add_argument("--input", required=True)
    cnn.add_argument("--backbone", choices=["resnet50", "efficientnet_b0", "convnext_tiny"], default="resnet50")
    cnn.add_argument("--no-pretrained", action="store_true", help="Avoid downloading ImageNet weights.")
    _add_common_baseline_args(cnn)

    train = sub.add_parser("train")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    hybrid = train_sub.add_parser("hybrid")
    hybrid.add_argument("--config", default="configs/hybrid.yaml")

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
    scan.add_argument("--mode", choices=["stub", "real", "precomputed"], default="stub")
    scan.add_argument("--scanner-exe", default="Part2_Scanner.exe")
    scan.add_argument("--precomputed")

    bench = sub.add_parser("benchmark")
    bench.add_argument("--command", dest="bench_command", nargs=argparse.REMAINDER, required=True)
    bench.add_argument("--limit-seconds", type=float, default=600.0)
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
        elif args.baseline_command == "parameter-search":
            path = parameter_search_baseline(args.input, args.out, seed=args.seed, scatterers_count=args.scatterers_count)
        else:
            path = pretrained_cnn_baseline(
                args.input,
                args.out,
                backbone=args.backbone,
                seed=args.seed,
                scatterers_count=args.scatterers_count,
                pretrained=not args.no_pretrained,
            )
        print(path)
        return 0

    if args.command == "train":
        print(train_hybrid(args.config))
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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

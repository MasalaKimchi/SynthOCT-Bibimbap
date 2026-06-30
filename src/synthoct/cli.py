from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

from . import __version__
from .adaptive_flow_batch import (
    calibrate_energy_followup_gate,
    enrich_residual_probe_energy_maps,
    run_adaptive_flow_strength_batch,
    run_residual_selector_flow_batch,
)
from .api_recovery import recover_api_results
from .candidate_rendering import render_candidate_queue
from .correction_refinement import (
    DEFAULT_CORRECTION_EXPONENTS,
    DEFAULT_RATIO_HIGHS,
    run_density_correction_refinement,
)
from .direct_lattice import run_direct_lattice_refinement
from .energy_ratio_refinement import DEFAULT_ENERGY_RATIO_EXPONENTS, run_energy_ratio_refinement
from .empirical_basis import run_empirical_basis_refinement
from .evaluation import audit_challenge_evidence, calculate_metrics, challenge_readiness_report, decide_candidate_promotion, select_best_candidate
from .features import generate_maps
from .flow_energy_batch import run_flow_energy_rank_batch
from .flow_refinement import DEFAULT_FLOW_VARIANTS, run_flow_refinement
from .generators import (
    FINAL_CONFIG_NAME,
    HYPOTHESIS_CONFIGS,
    PROMISING_PIPELINE_CONFIGS,
    VISUAL_PIPELINE_CONFIGS,
    final_phantom,
    heuristic_layer_phantom,
    hybrid_neural_prior_phantom,
    hypothesis_phantom,
    learned_prior_phantom,
    official_baseline_phantom,
    pipeline_phantom,
    physics_guided_phantom,
)
from .learned_prior import LEARNED_PRIOR_CONFIGS, train_empirical_phantom_prior
from .learned_surrogate import run_learned_surrogate_refinement
from .neural_prior import HYBRID_PRIOR_CONFIGS, neural_prior_phantom, train_neural_phantom_prior
from .patch_basis import run_patch_basis_refinement
from .dataset import iter_records, prepare_dataset
from .optimizer import run_candidate_search
from .residual_selector import (
    enrich_residual_teacher_maps,
    plan_residual_api_budget,
    plan_residual_selector_queue,
    train_residual_parameter_selector,
)
from .scanners import render_phantom, write_api_config
from .seed_search import run_api_seed_search
from .submission import (
    benchmark_submission_api,
    prepare_preliminary_png_pairs,
    prepare_submission_bundle,
    write_preliminary_upload_plan,
)
from .texture_refinement import DEFAULT_TEXTURE_VARIANTS, run_texture_refinement
from .transfer_refinement import DEFAULT_TRANSFER_EXPONENTS, run_selective_transfer_refinement
from .validation import METHOD_CHOICES, METHOD_WAVES, METHODS, plot_hypothesis_progress, resolve_method_wave, run_internal_validation


def _add_common_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="Output phantom .txt path.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scatterers-count", type=int, default=300_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthoct", description="SynthOCT inverse-physics digital phantom toolkit.")
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

    baseline = sub.add_parser("baseline", help="Generate digital phantoms from real OCT reference scans.")
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
    pipeline = base_sub.add_parser("pipeline")
    pipeline.add_argument("--input", required=True)
    pipeline.add_argument(
        "--name",
        choices=tuple(PROMISING_PIPELINE_CONFIGS.keys()) + tuple(VISUAL_PIPELINE_CONFIGS.keys()),
        default="P06_visual_surface_dark_body",
    )
    _add_common_baseline_args(pipeline)
    learned_prior = base_sub.add_parser("learned-prior")
    learned_prior.add_argument("--input", required=True)
    learned_prior.add_argument("--artifact", required=True, help="Empirical phantom prior .npz from train-phantom-prior.")
    learned_prior.add_argument("--target-blend", type=float, default=0.62)
    learned_prior.add_argument("--prior-blend", type=float, default=0.38)
    learned_prior.add_argument("--texture-weight", type=float, default=0.24)
    _add_common_baseline_args(learned_prior)
    neural_prior = base_sub.add_parser("neural-prior")
    neural_prior.add_argument("--input", required=True)
    neural_prior.add_argument("--artifact", required=True, help="Neural phantom prior .pt from train-neural-phantom-prior.")
    neural_prior.add_argument("--density-power", type=float, default=1.0)
    neural_prior.add_argument("--target-detail-blend", type=float, default=0.10)
    neural_prior.add_argument("--energy-floor", type=float, default=0.002)
    neural_prior.add_argument("--energy-ceiling", type=float, default=0.075)
    _add_common_baseline_args(neural_prior)
    hybrid_prior = base_sub.add_parser("hybrid-neural-prior")
    hybrid_prior.add_argument("--input", required=True)
    hybrid_prior.add_argument("--learned-artifact", required=True, help="Empirical learned-prior .npz artifact.")
    hybrid_prior.add_argument("--neural-artifact", required=True, help="Neural phantom prior .pt artifact.")
    hybrid_prior.add_argument("--neural-density-blend", type=float, default=0.10)
    hybrid_prior.add_argument("--neural-energy-blend", type=float, default=0.0)
    _add_common_baseline_args(hybrid_prior)
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

    scan = sub.add_parser("scan", help="Render a generated phantom into a synthetic OCT PNG.")
    scan.add_argument("--phantom", required=True)
    scan.add_argument("--out", required=True)
    scan.add_argument("--mode", choices=["api", "windows", "precomputed"], default="api")
    scan.add_argument("--config", help="Scanner Configuration.ini path. API mode writes one if missing.")
    scan.add_argument("--scanner-exe", default="Part2_Scanner.exe")
    scan.add_argument("--precomputed")
    scan.add_argument("--scatterers-count", type=int, default=300_000)
    scan.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    scan.add_argument("--poll-interval-seconds", type=float, default=10.0)
    scan.add_argument("--max-polls", type=int, default=60)

    bench = sub.add_parser("benchmark")
    bench.add_argument("--command", dest="bench_command", nargs=argparse.REMAINDER, required=True)
    bench.add_argument("--limit-seconds", type=float, default=600.0)

    validate = sub.add_parser("validate-internal", help="Hosted-API validation against Zenodo reference scans.")
    validate.add_argument("--zip", required=True, dest="zip_path")
    validate.add_argument("--out", default="outputs/internal_validation")
    validate.add_argument("--methods", nargs="+", choices=METHOD_CHOICES)
    validate.add_argument("--wave", choices=METHOD_WAVES, help="Named method set for staged hypothesis triage.")
    validate.add_argument("--learned-prior-artifact", help="Required when validating method learned-prior.")
    validate.add_argument("--neural-prior-artifact", help="Required when validating method neural-prior.")
    validate.add_argument("--folds", type=int, default=3)
    validate.add_argument("--max-per-fold", type=int, default=4)
    validate.add_argument("--sample-offset", type=int, default=0, help="Skip this many sorted records per fold before validation.")
    validate.add_argument("--scatterers-count", type=int, default=300_000)
    validate.add_argument("--no-maps", action="store_true")
    validate.add_argument("--include-lpips", action="store_true")
    validate.add_argument("--seed", type=int, default=7)
    validate.add_argument("--plot", help="Optional path for hypothesis progression figure.")
    validate.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    validate.add_argument("--poll-interval-seconds", type=float, default=10.0)
    validate.add_argument("--max-polls", type=int, default=60)
    validate.add_argument("--rerun-existing", action="store_true")

    evidence = sub.add_parser("audit-evidence", help="Classify whether a metrics CSV is fair challenge evidence.")
    evidence.add_argument("--metrics", required=True, help="Metrics CSV to audit.")
    evidence.add_argument("--min-samples-per-method", type=int, default=2)
    evidence.add_argument("--require-real-lpips", action="store_true")
    evidence.add_argument("--max-generation-seconds", type=float, default=600.0)
    evidence.add_argument("--strict", action="store_true", help="Exit nonzero unless the CSV is promotion-ready.")

    promote = sub.add_parser("decide-promotion", help="Compare a candidate against a baseline under fair challenge evidence.")
    promote.add_argument("--metrics", required=True, help="Challenge metrics summary CSV to compare.")
    promote.add_argument("--candidate", required=True, help="Candidate method name.")
    promote.add_argument("--baseline", required=True, help="Baseline/current-final method name.")
    promote.add_argument("--min-samples-per-method", type=int, default=2)
    promote.add_argument("--require-real-lpips", action="store_true")
    promote.add_argument("--min-ms-ssim-delta", type=float, default=0.0)
    promote.add_argument("--max-lpips-delta", type=float, default=0.0)
    promote.add_argument("--max-generation-seconds", type=float, default=600.0)
    promote.add_argument("--max-guardrail-regression", type=float, default=0.0)
    promote.add_argument("--strict", action="store_true", help="Exit nonzero unless the candidate is promoted.")

    select_best = sub.add_parser("select-best", help="Select the best method that beats a baseline under fair challenge evidence.")
    select_best.add_argument("--metrics", required=True, help="Challenge metrics summary CSV to rank.")
    select_best.add_argument("--baseline", required=True, help="Baseline/current-final method name.")
    select_best.add_argument("--min-samples-per-method", type=int, default=2)
    select_best.add_argument("--require-real-lpips", action="store_true")
    select_best.add_argument("--min-ms-ssim-delta", type=float, default=0.0)
    select_best.add_argument("--max-lpips-delta", type=float, default=0.0)
    select_best.add_argument("--max-generation-seconds", type=float, default=600.0)
    select_best.add_argument("--max-guardrail-regression", type=float, default=0.0)
    select_best.add_argument("--strict", action="store_true", help="Exit nonzero unless a candidate is selected.")

    readiness = sub.add_parser("challenge-readiness", help="Summarize whether local evidence supports the current challenge candidate.")
    readiness.add_argument("--metrics", required=True, help="Challenge metrics summary CSV to audit and rank.")
    readiness.add_argument("--baseline", required=True, help="Baseline/current-final method name.")
    readiness.add_argument("--method", help="Candidate/submission method expected to be selected.")
    readiness.add_argument("--submission-dir", help="Optional prepared submission directory to check.")
    readiness.add_argument("--min-samples-per-method", type=int, default=2)
    readiness.add_argument("--require-real-lpips", action="store_true")
    readiness.add_argument("--min-ms-ssim-delta", type=float, default=0.0)
    readiness.add_argument("--max-lpips-delta", type=float, default=0.0)
    readiness.add_argument("--max-generation-seconds", type=float, default=600.0)
    readiness.add_argument("--max-guardrail-regression", type=float, default=0.0)
    readiness.add_argument("--strict", action="store_true", help="Exit nonzero unless local candidate readiness passes.")

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

    seed_search = sub.add_parser("optimize-seeds", help="Hosted-API seed search for one named method on one reference scan.")
    seed_search.add_argument("--input", required=True, help="Reference B-scan PNG/NPY.")
    seed_search.add_argument("--out", default="outputs/seed_search")
    seed_search.add_argument("--method", choices=METHODS, required=True)
    seed_search.add_argument("--seeds", nargs="+", type=int, required=True)
    seed_search.add_argument("--scatterers-count", type=int, default=300_000)
    seed_search.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    seed_search.add_argument("--poll-interval-seconds", type=float, default=10.0)
    seed_search.add_argument("--max-polls", type=int, default=60)
    seed_search.add_argument("--rerun-existing", action="store_true")

    correction = sub.add_parser(
        "optimize-correction",
        help="Hosted-API scanner-in-loop density correction for one reference scan.",
    )
    correction.add_argument("--input", required=True, help="Reference B-scan PNG/NPY.")
    correction.add_argument("--out", default="outputs/correction_refinement")
    correction.add_argument("--scatterers-count", type=int, default=900_000)
    correction.add_argument("--seed", type=int, default=7)
    correction.add_argument("--exponents", nargs="+", type=float, default=list(DEFAULT_CORRECTION_EXPONENTS))
    correction.add_argument("--ratio-highs", nargs="+", type=float, default=list(DEFAULT_RATIO_HIGHS))
    correction.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    correction.add_argument("--poll-interval-seconds", type=float, default=10.0)
    correction.add_argument("--max-polls", type=int, default=90)
    correction.add_argument("--rerun-existing", action="store_true")

    transfer = sub.add_parser(
        "optimize-transfer",
        help="Hosted-API selective energy transfer refinement from an existing phantom/render pair.",
    )
    transfer.add_argument("--ref", required=True, help="Reference B-scan PNG/NPY.")
    transfer.add_argument("--phantom", required=True, help="Base phantom scatterer file.")
    transfer.add_argument("--rendered-gray", required=True, help="Hosted-rendered grayscale PNG for the base phantom.")
    transfer.add_argument("--out", default="outputs/transfer_refinement")
    transfer.add_argument("--exponents", nargs="+", type=float, default=list(DEFAULT_TRANSFER_EXPONENTS))
    transfer.add_argument("--percentile", type=float, default=85.0)
    transfer.add_argument("--floor", type=float, default=0.025)
    transfer.add_argument("--cap-high", type=float, default=1.25)
    transfer.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    transfer.add_argument("--poll-interval-seconds", type=float, default=10.0)
    transfer.add_argument("--max-polls", type=int, default=90)
    transfer.add_argument("--rerun-existing", action="store_true")

    energy_ratio = sub.add_parser(
        "optimize-energy-ratio",
        help="Hosted-API coordinate-preserving energy-ratio refinement from an existing phantom/render pair.",
    )
    energy_ratio.add_argument("--ref", required=True, help="Reference B-scan PNG/NPY.")
    energy_ratio.add_argument("--phantom", required=True, help="Base phantom scatterer file.")
    energy_ratio.add_argument("--rendered-gray", required=True, help="Hosted-rendered grayscale PNG for the base phantom.")
    energy_ratio.add_argument("--out", default="outputs/energy_ratio_refinement")
    energy_ratio.add_argument("--exponents", nargs="+", type=float, default=list(DEFAULT_ENERGY_RATIO_EXPONENTS))
    energy_ratio.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    energy_ratio.add_argument("--poll-interval-seconds", type=float, default=10.0)
    energy_ratio.add_argument("--max-polls", type=int, default=90)
    energy_ratio.add_argument("--rerun-existing", action="store_true")
    energy_ratio.add_argument("--stabilizer", type=float, default=0.025)
    energy_ratio.add_argument("--sigma", type=float, default=2.0)
    energy_ratio.add_argument("--ratio-low", type=float, default=0.65)
    energy_ratio.add_argument("--ratio-high", type=float, default=1.32)
    energy_ratio.add_argument("--clip-low", type=float, default=0.78)
    energy_ratio.add_argument("--clip-high", type=float, default=1.22)
    energy_ratio.add_argument("--air-boundary", type=float, default=28.0)
    energy_ratio.add_argument("--no-match-total-energy", action="store_true")

    texture = sub.add_parser(
        "optimize-texture",
        help="Hosted-API local-variance texture refinement from an existing phantom/render pair.",
    )
    texture.add_argument("--ref", required=True, help="Reference B-scan PNG/NPY.")
    texture.add_argument("--phantom", required=True, help="Base phantom scatterer file.")
    texture.add_argument("--rendered-gray", required=True, help="Hosted-rendered grayscale PNG for the base phantom.")
    texture.add_argument("--out", default="outputs/texture_refinement")
    texture.add_argument(
        "--variants",
        nargs="+",
        default=[],
        help="Optional mean:texture:deep exponent triples, e.g. 0.10:0.6:1.8.",
    )
    texture.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    texture.add_argument("--poll-interval-seconds", type=float, default=10.0)
    texture.add_argument("--max-polls", type=int, default=90)
    texture.add_argument("--rerun-existing", action="store_true")

    flow = sub.add_parser(
        "optimize-flow",
        help="Hosted-API optical-flow coordinate transport from an existing phantom/render pair.",
    )
    flow.add_argument("--ref", required=True, help="Reference B-scan PNG/NPY.")
    flow.add_argument("--phantom", required=True, help="Base phantom scatterer file.")
    flow.add_argument("--rendered-gray", required=True, help="Hosted-rendered grayscale PNG for the base phantom.")
    flow.add_argument("--out", default="outputs/flow_refinement")
    flow.add_argument(
        "--variants",
        nargs="+",
        default=[],
        help="Optional strength:sigma:attachment triples, e.g. 0.20:1.0:5.0.",
    )
    flow.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    flow.add_argument("--poll-interval-seconds", type=float, default=10.0)
    flow.add_argument("--max-polls", type=int, default=90)
    flow.add_argument("--rerun-existing", action="store_true")

    learned = sub.add_parser(
        "optimize-learned-surrogate",
        help="Train a local CNN scanner surrogate from hosted renders and emit inverse candidates.",
    )
    learned.add_argument("--ref", required=True, help="Reference B-scan PNG/NPY.")
    learned.add_argument("--out", default="outputs/learned_surrogate_refinement")
    learned.add_argument("--outputs-dir", default="outputs")
    learned.add_argument("--base-phantom", help="Optional base phantom to initialize inverse optimization.")
    learned.add_argument("--base-rendered-gray", help="Optional base rendered gray PNG for documentation/continuity.")
    learned.add_argument("--shape", nargs=2, type=int, default=[128, 256], metavar=("ROWS", "COLS"))
    learned.add_argument("--train-limit", type=int, default=64)
    learned.add_argument("--epochs", type=int, default=160)
    learned.add_argument("--optimize-steps", type=int, default=220)
    learned.add_argument("--scatterers-count", type=int, default=900_000)
    learned.add_argument("--seed", type=int, default=23)
    learned.add_argument("--energy-ratios", nargs="+", type=float, default=[0.78, 0.88, 1.0])
    learned.add_argument("--texture-strengths", nargs="+", type=float, default=[0.0, 0.45])
    learned.add_argument("--holdout-fraction", type=float, default=0.2)
    learned.add_argument("--anchored-residual", action="store_true", help="Optimize smoothed residuals around the base phantom fields instead of free fields.")
    learned.add_argument("--density-residual-scale", type=float, default=0.18)
    learned.add_argument("--energy-residual-scale", type=float, default=0.22)
    learned.add_argument("--anchor-weight", type=float, default=0.35)
    learned.add_argument("--residual-kernel", type=int, default=15)

    prior = sub.add_parser("train-phantom-prior", help="Distill true-scanner phantom/render pairs into a reusable empirical prior artifact.")
    prior.add_argument("--outputs-dir", default="outputs", help="Directory containing hosted-scanner metrics and artifacts.")
    prior.add_argument("--out", required=True, help="Output .npz prior artifact.")
    prior.add_argument("--shape", nargs=2, type=int, default=[128, 256], metavar=("ROWS", "COLS"))
    prior.add_argument("--pair-limit", type=int, default=64)
    prior.add_argument("--temperature", type=float, default=0.08)

    neural_train = sub.add_parser(
        "train-neural-phantom-prior",
        help="Train a scanner-compatible CNN phantom-field generator from true-scanner validation rows.",
    )
    neural_train.add_argument("--outputs-dir", default="outputs", help="Directory containing hosted-scanner metrics and artifacts.")
    neural_train.add_argument("--out", required=True, help="Output .pt neural prior artifact.")
    neural_train.add_argument("--shape", nargs=2, type=int, default=[128, 256], metavar=("ROWS", "COLS"))
    neural_train.add_argument("--pair-limit", type=int, default=128)
    neural_train.add_argument("--epochs", type=int, default=80)
    neural_train.add_argument("--learning-rate", type=float, default=2e-3)
    neural_train.add_argument("--seed", type=int, default=37)

    empirical = sub.add_parser(
        "optimize-empirical-basis",
        help="Use existing hosted scanner renders as a nonnegative empirical inverse basis.",
    )
    empirical.add_argument("--ref", required=True, help="Reference B-scan PNG/NPY.")
    empirical.add_argument("--out", default="outputs/empirical_basis_refinement")
    empirical.add_argument("--outputs-dir", default="outputs")
    empirical.add_argument("--shape", nargs=2, type=int, default=[128, 256], metavar=("ROWS", "COLS"))
    empirical.add_argument("--pair-limit", type=int, default=64)
    empirical.add_argument("--basis-count", type=int, default=24)
    empirical.add_argument("--scatterers-count", type=int, default=900_000)
    empirical.add_argument("--seed", type=int, default=71)
    empirical.add_argument("--residual-exponents", nargs="+", type=float, default=[0.0, 0.18, 0.36])
    empirical.add_argument("--energy-ratios", nargs="+", type=float, default=[0.82, 0.94, 1.06])
    empirical.add_argument("--texture-strengths", nargs="+", type=float, default=[0.0, 0.25])

    patch_basis = sub.add_parser(
        "optimize-patch-basis",
        help="Use local patchwise nonnegative mixtures of hosted scanner renders to emit inverse candidates.",
    )
    patch_basis.add_argument("--ref", required=True, help="Reference B-scan PNG/NPY.")
    patch_basis.add_argument("--out", default="outputs/patch_basis_refinement")
    patch_basis.add_argument("--outputs-dir", default="outputs")
    patch_basis.add_argument("--shape", nargs=2, type=int, default=[128, 256], metavar=("ROWS", "COLS"))
    patch_basis.add_argument("--basis-count", type=int, default=32)
    patch_basis.add_argument("--tile-shape", nargs=2, type=int, default=[24, 32], metavar=("ROWS", "COLS"))
    patch_basis.add_argument("--scatterers-count", type=int, default=900_000)
    patch_basis.add_argument("--seed", type=int, default=101)
    patch_basis.add_argument("--residual-exponents", nargs="+", type=float, default=[0.0, 0.18])
    patch_basis.add_argument("--texture-strengths", nargs="+", type=float, default=[0.0, 0.25])
    patch_basis.add_argument("--energy-ratios", nargs="+", type=float, default=[0.82, 0.94, 1.06])

    lattice = sub.add_parser(
        "optimize-direct-lattice",
        help="Generate deterministic target-locked lattice phantoms from the reference B-scan.",
    )
    lattice.add_argument("--ref", required=True, help="Reference B-scan PNG/NPY.")
    lattice.add_argument("--out", default="outputs/direct_lattice_refinement")
    lattice.add_argument("--scatterers-count", type=int, default=900_000)
    lattice.add_argument("--seed", type=int, default=131)
    lattice.add_argument("--recipes", nargs="+", default=["sqrt_attn", "surface_locked", "speckle_microgrid"])
    lattice.add_argument("--energy-scales", nargs="+", type=float, default=[0.026, 0.034, 0.044])

    render_queue = sub.add_parser(
        "render-candidate-queue",
        help="Render a ranked phantom candidate queue through the hosted scanner.",
    )
    render_queue.add_argument("--queue", required=True, help="CSV with method and phantom_path columns.")
    render_queue.add_argument("--ref", required=True, help="Reference B-scan PNG/NPY.")
    render_queue.add_argument("--out", default="outputs/candidate_queue_render")
    render_queue.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    render_queue.add_argument("--poll-interval-seconds", type=float, default=10.0)
    render_queue.add_argument("--max-polls", type=int, default=90)
    render_queue.add_argument("--max-candidates", type=int)
    render_queue.add_argument("--api-concurrency", type=int, default=1, help="Number of hosted API render jobs to run at once.")
    render_queue.add_argument("--rerun-existing", action="store_true")

    flow_energy_batch = sub.add_parser(
        "flow-energy-rank-batch",
        help="Run fixed flow+energy API rescue over a base metrics MS-SSIM rank window.",
    )
    flow_energy_batch.add_argument("--base-api-metrics", required=True, help="api_metrics.csv from a base hosted validation run.")
    flow_energy_batch.add_argument("--out", required=True, help="Output directory for queues, phantoms, renders, and metrics.")
    flow_energy_batch.add_argument("--rank-start", type=int, required=True, help="1-based low MS-SSIM rank to include.")
    flow_energy_batch.add_argument("--rank-end", type=int, required=True, help="1-based low MS-SSIM rank to include.")
    flow_energy_batch.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    flow_energy_batch.add_argument("--api-concurrency", type=int, default=2, help="Number of hosted API render jobs to run at once.")
    flow_energy_batch.add_argument("--poll-interval-seconds", type=float, default=3.0)
    flow_energy_batch.add_argument("--max-polls", type=int, default=80)
    flow_energy_batch.add_argument("--flow-strength", type=float, default=0.25)
    flow_energy_batch.add_argument("--flow-smooth-sigma", type=float, default=1.2)
    flow_energy_batch.add_argument("--flow-attachment", type=float, default=6.0)
    flow_energy_batch.add_argument("--energy-exponent", type=float, default=0.8)

    adaptive_flow = sub.add_parser(
        "adaptive-flow-strength-batch",
        help="Sweep flow strengths plus energy follow-ups over weak current API rows.",
    )
    adaptive_flow.add_argument("--current-api-metrics", required=True, help="api_metrics.csv for the current best hosted run.")
    adaptive_flow.add_argument("--out", required=True, help="Output directory for queues, phantoms, renders, and metrics.")
    adaptive_flow.add_argument("--rank-start", type=int, required=True, help="1-based low MS-SSIM rank to include.")
    adaptive_flow.add_argument("--rank-end", type=int, required=True, help="1-based low MS-SSIM rank to include.")
    adaptive_flow.add_argument("--strengths", nargs="+", type=float, default=[0.12, 0.38])
    adaptive_flow.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    adaptive_flow.add_argument("--api-concurrency", type=int, default=2, help="Number of hosted API render jobs to run at once.")
    adaptive_flow.add_argument("--poll-interval-seconds", type=float, default=3.0)
    adaptive_flow.add_argument("--max-polls", type=int, default=80)
    adaptive_flow.add_argument("--flow-smooth-sigma", type=float, default=1.2)
    adaptive_flow.add_argument("--flow-attachment", type=float, default=6.0)
    adaptive_flow.add_argument("--energy-exponent", type=float, default=0.8)

    residual_selector = sub.add_parser(
        "train-residual-selector",
        help="Distill public flow+energy rescue rows into a conservative residual-parameter selector.",
    )
    residual_selector.add_argument("--base-api-metrics", required=True, help="Current/base api_metrics.csv.")
    residual_selector.add_argument("--teacher-metrics", nargs="+", required=True, help="Flow/adaptive teacher metrics CSVs.")
    residual_selector.add_argument("--out", required=True, help="Output selector artifact JSON.")
    residual_selector.add_argument("--default-strength", type=float, default=0.25)
    residual_selector.add_argument("--holdout-fraction", type=float, default=0.25)
    residual_selector.add_argument("--min-delta", type=float, default=0.0)
    residual_selector.add_argument("--uncertainty-z", type=float, default=1.0)
    residual_selector.add_argument("--neighbor-count", type=int, default=12)
    residual_selector.add_argument("--exploration-weight", type=float, default=0.25)
    residual_selector.add_argument("--surrogate-calibration-metrics", nargs="*", default=[])
    residual_selector.add_argument("--min-surrogate-ms-ssim", type=float, default=0.90)
    residual_selector.add_argument("--max-surrogate-lpips-proxy", type=float, default=0.03)
    residual_selector.add_argument(
        "--holdout-group-fields",
        nargs="*",
        default=["sex", "age_band", "body_site"],
        help="Fields used to keep anatomical cohorts together in selector holdout validation.",
    )

    residual_enrich = sub.add_parser(
        "enrich-residual-teacher-maps",
        help="Compute Struct/OAC/SC/RSC metrics for residual teacher CSV rows.",
    )
    residual_enrich.add_argument("--base-api-metrics", required=True, help="Base/current api_metrics.csv for path joins.")
    residual_enrich.add_argument("--teacher-metrics", required=True, help="Teacher metrics CSV to enrich.")
    residual_enrich.add_argument("--out", required=True, help="Output enriched teacher metrics CSV.")
    residual_enrich.add_argument("--maps-dir", help="Directory for generated map PNGs.")
    residual_enrich.add_argument("--include-lpips", action="store_true")
    residual_enrich.add_argument("--max-rows", type=int)

    residual_queue = sub.add_parser(
        "plan-residual-selector-queue",
        help="Write row/strength probe recommendations from a residual selector artifact.",
    )
    residual_queue.add_argument("--base-api-metrics", required=True, help="Current/base api_metrics.csv.")
    residual_queue.add_argument("--artifact", required=True, help="Selector artifact JSON from train-residual-selector.")
    residual_queue.add_argument("--out", required=True, help="Output recommendation CSV.")
    residual_queue.add_argument("--limit", type=int)
    residual_queue.add_argument("--min-expected-delta-lcb", type=float, default=0.0)
    residual_queue.add_argument("--row-wise-strengths", action="store_true", help="Choose a residual strength per row from local teacher neighbors.")
    residual_queue.add_argument(
        "--min-map-delta-lcb",
        type=float,
        default=-0.0005,
        help="Minimum lower-confidence bound for each Struct/OAC/SC/RSC delta when map-safety is required.",
    )
    residual_queue.add_argument(
        "--min-map-safe-win-rate",
        type=float,
        default=0.5,
        help="Minimum neighbor win rate with all Struct/OAC/SC/RSC deltas above the map-safety threshold.",
    )
    residual_queue.add_argument("--require-map-safe", action="store_true", help="Reject rows that fail uncertainty-gated map safety.")

    residual_budget = sub.add_parser(
        "plan-residual-api-budget",
        help="Expand selector recommendations into a bounded two-stage API probe queue.",
    )
    residual_budget.add_argument("--selector-queue", required=True, help="CSV from plan-residual-selector-queue.")
    residual_budget.add_argument("--out", required=True, help="Output budgeted selector queue CSV.")
    residual_budget.add_argument("--total-api-calls", type=int, default=120)
    residual_budget.add_argument("--energy-followup-fraction", type=float, default=1.0 / 3.0)
    residual_budget.add_argument("--strength-multipliers", nargs="+", type=float, default=[1.0, 0.72, 1.38])
    residual_budget.add_argument("--energy-exponents", nargs="+", type=float, default=[0.80, 0.72, 0.92])
    residual_budget.add_argument("--min-strength", type=float, default=0.05)
    residual_budget.add_argument("--max-strength", type=float, default=0.55)
    residual_budget.add_argument(
        "--diversity-fields",
        nargs="*",
        default=[],
        help="Optional active-learning strata such as sex age_band body_site rank_bucket.",
    )
    residual_budget.add_argument("--max-per-stratum", type=int, help="Soft cap on planned flow probes per diversity stratum.")
    residual_budget.add_argument(
        "--probe-feedback-metrics",
        nargs="*",
        default=[],
        help="Optional residual-selector true-scanner probe metrics used as cautious active-learning feedback.",
    )
    residual_budget.add_argument(
        "--feedback-group-fields",
        nargs="*",
        default=["sex", "age_band", "body_site"],
        help="Fields used to share true-probe feedback across similar queued candidates.",
    )
    residual_budget.add_argument(
        "--energy-gate-summary",
        help="Optional calibrated energy-followup gate JSON from calibrate-energy-followup-gate.",
    )
    residual_budget.add_argument(
        "--exclude-exact-probed",
        action="store_true",
        help="Plan only rows without exact-source true-probe feedback, for fresh active-learning batches.",
    )

    residual_batch = sub.add_parser(
        "residual-selector-flow-batch",
        help="Run selector-recommended residual flow+energy probes through the hosted scanner.",
    )
    residual_batch.add_argument("--selector-queue", required=True, help="CSV from plan-residual-selector-queue.")
    residual_batch.add_argument("--out", required=True, help="Output directory for queues, phantoms, renders, and metrics.")
    residual_batch.add_argument("--max-candidates", type=int, help="Limit recommendations consumed from the selector queue.")
    residual_batch.add_argument("--max-energy-followups", type=int, help="Limit stage-2 energy API calls after flow renders.")
    residual_batch.add_argument(
        "--min-energy-flow-delta",
        type=float,
        help="Skip stage-2 energy followups when flow MS-SSIM delta falls below this threshold.",
    )
    residual_batch.add_argument("--api-key-file", help="Optional untracked file containing the hosted scanner API key.")
    residual_batch.add_argument("--api-concurrency", type=int, default=2, help="Number of hosted API render jobs to run at once.")
    residual_batch.add_argument("--poll-interval-seconds", type=float, default=3.0)
    residual_batch.add_argument("--max-polls", type=int, default=80)
    residual_batch.add_argument("--flow-smooth-sigma", type=float, default=1.2)
    residual_batch.add_argument("--flow-attachment", type=float, default=6.0)
    residual_batch.add_argument("--energy-exponent", type=float, default=0.8)

    energy_gate = sub.add_parser(
        "calibrate-energy-followup-gate",
        help="Learn a flow-delta threshold for stage-2 energy followups from true-scanner probe metrics.",
    )
    energy_gate.add_argument("--probe-metrics", nargs="+", required=True, help="Residual selector probe metrics CSVs.")
    energy_gate.add_argument("--out", required=True, help="Output calibration summary JSON.")
    energy_gate.add_argument("--min-retained-delta-fraction", type=float, default=0.995)
    energy_gate.add_argument("--min-retained-map-delta-fraction", type=float)
    energy_gate.add_argument("--min-energy-calls", type=int, default=1)
    energy_gate.add_argument("--max-negative-energy-calls", type=int, default=0)
    energy_gate.add_argument("--max-negative-map-calls", type=int)

    energy_maps = sub.add_parser(
        "enrich-residual-probe-energy-maps",
        help="Backfill flow-energy map objective scores from rendered residual probe PNGs.",
    )
    energy_maps.add_argument("--probe-metrics", nargs="+", required=True, help="Residual selector probe metrics CSVs.")
    energy_maps.add_argument("--out", required=True, help="Output enriched probe metrics CSV.")
    energy_maps.add_argument("--maps-dir", help="Optional directory for generated feature maps.")

    recover_api = sub.add_parser(
        "recover-api-results",
        help="Recover late hosted-scanner PNGs from failed metrics rows and recompute real metrics.",
    )
    recover_api.add_argument("--metrics", required=True, help="Metrics CSV containing result_<id>.png errors.")
    recover_api.add_argument("--ref", required=True, help="Reference B-scan PNG/NPY.")

    submit = sub.add_parser("prepare-submission")
    submit.add_argument("--zip", required=True, dest="zip_path")
    submit.add_argument("--out", default="outputs/submission_ready")
    submit.add_argument("--scatterers-count", type=int, default=300_000)
    submit.add_argument("--limit", type=int, help="Limit number of PNG B-scans for smoke packaging.")
    submit.add_argument("--seed", type=int, default=7)
    submit.add_argument(
        "--method",
        choices=tuple(HYPOTHESIS_CONFIGS.keys())
        + tuple(PROMISING_PIPELINE_CONFIGS.keys())
        + tuple(VISUAL_PIPELINE_CONFIGS.keys())
        + ("learned-prior",)
        + tuple(LEARNED_PRIOR_CONFIGS.keys())
        + ("neural-prior",)
        + tuple(HYBRID_PRIOR_CONFIGS.keys()),
        help="Named H- or P-series method to package.",
    )
    submit.add_argument("--learned-prior-artifact", help="Required when preparing --method learned-prior.")
    submit.add_argument("--neural-prior-artifact", help="Required when preparing --method neural-prior.")
    submit.add_argument("--evidence-metrics", help="Optional challenge_metrics_summary.csv used to write submission_readiness_report.json.")
    submit.add_argument("--baseline", help="Baseline/current-final method for evidence-backed readiness checks.")
    submit.add_argument("--min-samples-per-method", type=int, default=2)
    submit.add_argument("--require-real-lpips", action="store_true")
    submit.add_argument("--max-generation-seconds", type=float, default=600.0)
    submit.add_argument("--max-guardrail-regression", type=float, default=0.0)
    submit.add_argument("--strict-evidence", action="store_true", help="Fail packaging unless the method is the selected promoted candidate.")

    api_eval = sub.add_parser("api-evaluate-submission", help="Render submission phantoms with the hosted SynthOCT API.")
    api_eval.add_argument("--zip", required=True, dest="zip_path")
    api_eval.add_argument("--submission-dir", required=True)
    api_eval.add_argument("--out", default="outputs/api_preliminary")
    api_eval.add_argument("--limit", type=int, help="Limit number of manifest rows to render.")
    api_eval.add_argument("--scatterers-count", type=int, default=300_000)
    api_eval.add_argument("--include-lpips", action="store_true")
    api_eval.add_argument("--poll-interval-seconds", type=float, default=10.0)
    api_eval.add_argument("--max-polls", type=int, default=60)
    api_eval.add_argument("--api-concurrency", type=int, default=1, help="Number of hosted API render jobs to run in parallel.")
    api_eval.add_argument("--rerun-existing", action="store_true")
    api_eval.add_argument("--progress", action="store_true", help="Print hosted render and metric progress to stderr.")
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
            path = official_baseline_phantom(args.out, method=args.method, seed=args.seed, scatterers_count=args.scatterers_count)
        elif args.baseline_command == "heuristic":
            path = heuristic_layer_phantom(args.input, args.out, seed=args.seed, scatterers_count=args.scatterers_count)
        elif args.baseline_command == "physics-guided":
            path = physics_guided_phantom(
                args.input,
                args.out,
                seed=args.seed,
                scatterers_count=args.scatterers_count,
                lateral_bins=args.lateral_bins,
                depth_bins=args.depth_bins,
            )
        elif args.baseline_command == "hypothesis":
            path = hypothesis_phantom(args.input, args.out, args.name, seed=args.seed, scatterers_count=args.scatterers_count)
        elif args.baseline_command == "pipeline":
            path = pipeline_phantom(args.input, args.out, args.name, seed=args.seed, scatterers_count=args.scatterers_count)
        elif args.baseline_command == "learned-prior":
            path = learned_prior_phantom(
                args.input,
                args.out,
                args.artifact,
                seed=args.seed,
                scatterers_count=args.scatterers_count,
                target_blend=args.target_blend,
                prior_blend=args.prior_blend,
                texture_weight=args.texture_weight,
            )
        elif args.baseline_command == "neural-prior":
            path = neural_prior_phantom(
                args.input,
                args.out,
                args.artifact,
                seed=args.seed,
                scatterers_count=args.scatterers_count,
                density_power=args.density_power,
                target_detail_blend=args.target_detail_blend,
                energy_floor=args.energy_floor,
                energy_ceiling=args.energy_ceiling,
            )
        elif args.baseline_command == "hybrid-neural-prior":
            path = hybrid_neural_prior_phantom(
                args.input,
                args.out,
                args.learned_artifact,
                args.neural_artifact,
                seed=args.seed,
                scatterers_count=args.scatterers_count,
                neural_density_blend=args.neural_density_blend,
                neural_energy_blend=args.neural_energy_blend,
            )
        elif args.baseline_command == "final":
            path = final_phantom(args.input, args.out, seed=args.seed, scatterers_count=args.scatterers_count)
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
        if args.mode == "precomputed":
            if args.precomputed is None:
                raise ValueError("--precomputed is required when mode=precomputed.")
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.precomputed, path)
        else:
            config_path = Path(args.config) if args.config else Path(args.out).with_suffix(".ini")
            if args.mode == "api":
                if not config_path.exists():
                    write_api_config(config_path, scatterers_count=args.scatterers_count)
                path = render_phantom(
                    args.phantom,
                    config_path,
                    args.out,
                    backend="api",
                    api_key_file=args.api_key_file,
                    poll_interval_seconds=args.poll_interval_seconds,
                    max_polls=args.max_polls,
                )
            else:
                path = render_phantom(args.phantom, config_path, args.out, backend="windows", scanner_exe=args.scanner_exe)
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
            sample_offset=args.sample_offset,
            scatterers_count=args.scatterers_count,
            include_maps=not args.no_maps,
            include_lpips=args.include_lpips,
            seed=args.seed,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            skip_existing=not args.rerun_existing,
            learned_prior_artifact=args.learned_prior_artifact,
            neural_prior_artifact=args.neural_prior_artifact,
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

    if args.command == "audit-evidence":
        report = audit_challenge_evidence(
            args.metrics,
            min_samples_per_method=args.min_samples_per_method,
            require_real_lpips=args.require_real_lpips,
            max_generation_seconds=args.max_generation_seconds,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if (not args.strict or report["promotion_ready"]) else 2

    if args.command == "decide-promotion":
        report = decide_candidate_promotion(
            args.metrics,
            candidate=args.candidate,
            baseline=args.baseline,
            min_samples_per_method=args.min_samples_per_method,
            require_real_lpips=args.require_real_lpips,
            min_ms_ssim_delta=args.min_ms_ssim_delta,
            max_lpips_delta=args.max_lpips_delta,
            max_generation_seconds=args.max_generation_seconds,
            max_guardrail_regression=args.max_guardrail_regression,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if (not args.strict or report["promote"]) else 2

    if args.command == "select-best":
        report = select_best_candidate(
            args.metrics,
            baseline=args.baseline,
            min_samples_per_method=args.min_samples_per_method,
            require_real_lpips=args.require_real_lpips,
            min_ms_ssim_delta=args.min_ms_ssim_delta,
            max_lpips_delta=args.max_lpips_delta,
            max_generation_seconds=args.max_generation_seconds,
            max_guardrail_regression=args.max_guardrail_regression,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if (not args.strict or report["promote"]) else 2

    if args.command == "challenge-readiness":
        report = challenge_readiness_report(
            args.metrics,
            baseline=args.baseline,
            method=args.method,
            submission_dir=args.submission_dir,
            min_samples_per_method=args.min_samples_per_method,
            require_real_lpips=args.require_real_lpips,
            min_ms_ssim_delta=args.min_ms_ssim_delta,
            max_lpips_delta=args.max_lpips_delta,
            max_generation_seconds=args.max_generation_seconds,
            max_guardrail_regression=args.max_guardrail_regression,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if (not args.strict or report["local_candidate_ready"]) else 2

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

    if args.command == "optimize-seeds":
        metrics_path = run_api_seed_search(
            args.input,
            args.out,
            method=args.method,
            seeds=args.seeds,
            scatterers_count=args.scatterers_count,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            skip_existing=not args.rerun_existing,
        )
        print(metrics_path)
        return 0

    if args.command == "optimize-correction":
        metrics_path = run_density_correction_refinement(
            args.input,
            args.out,
            scatterers_count=args.scatterers_count,
            seed=args.seed,
            correction_exponents=args.exponents,
            ratio_highs=args.ratio_highs,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            skip_existing=not args.rerun_existing,
        )
        print(metrics_path)
        return 0

    if args.command == "optimize-transfer":
        metrics_path = run_selective_transfer_refinement(
            args.ref,
            args.phantom,
            args.rendered_gray,
            args.out,
            exponents=args.exponents,
            percentile=args.percentile,
            floor=args.floor,
            cap_high=args.cap_high,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            skip_existing=not args.rerun_existing,
        )
        print(metrics_path)
        return 0

    if args.command == "optimize-energy-ratio":
        metrics_path = run_energy_ratio_refinement(
            args.ref,
            args.phantom,
            args.rendered_gray,
            args.out,
            exponents=args.exponents,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            skip_existing=not args.rerun_existing,
            stabilizer=args.stabilizer,
            sigma=args.sigma,
            ratio_low=args.ratio_low,
            ratio_high=args.ratio_high,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            air_boundary=args.air_boundary,
            match_total_energy=not args.no_match_total_energy,
        )
        print(metrics_path)
        return 0

    if args.command == "optimize-texture":
        variants = None
        if args.variants:
            variants = []
            for value in args.variants:
                parts = value.split(":")
                if len(parts) != 3:
                    raise ValueError(f"Texture variants must be mean:texture:deep triples, got {value!r}.")
                variants.append(tuple(float(part) for part in parts))
        metrics_path = run_texture_refinement(
            args.ref,
            args.phantom,
            args.rendered_gray,
            args.out,
            variants=variants if variants is not None else DEFAULT_TEXTURE_VARIANTS,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            skip_existing=not args.rerun_existing,
        )
        print(metrics_path)
        return 0

    if args.command == "optimize-flow":
        variants = None
        if args.variants:
            variants = []
            for value in args.variants:
                parts = value.split(":")
                if len(parts) != 3:
                    raise ValueError(f"Flow variants must be strength:sigma:attachment triples, got {value!r}.")
                variants.append(tuple(float(part) for part in parts))
        metrics_path = run_flow_refinement(
            args.ref,
            args.phantom,
            args.rendered_gray,
            args.out,
            variants=variants if variants is not None else DEFAULT_FLOW_VARIANTS,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            skip_existing=not args.rerun_existing,
        )
        print(metrics_path)
        return 0

    if args.command == "optimize-learned-surrogate":
        metrics_path = run_learned_surrogate_refinement(
            args.ref,
            args.out,
            outputs_dir=args.outputs_dir,
            base_phantom_path=args.base_phantom,
            base_rendered_gray_path=args.base_rendered_gray,
            shape=tuple(args.shape),
            train_limit=args.train_limit,
            epochs=args.epochs,
            optimize_steps=args.optimize_steps,
            seed=args.seed,
            scatterers_count=args.scatterers_count,
            energy_ratios=args.energy_ratios,
            texture_strengths=args.texture_strengths,
            holdout_fraction=args.holdout_fraction,
            anchored_residual=args.anchored_residual,
            density_residual_scale=args.density_residual_scale,
            energy_residual_scale=args.energy_residual_scale,
            anchor_weight=args.anchor_weight,
            residual_kernel=args.residual_kernel,
        )
        print(metrics_path)
        return 0

    if args.command == "train-phantom-prior":
        path = train_empirical_phantom_prior(
            args.outputs_dir,
            args.out,
            shape=tuple(args.shape),
            pair_limit=args.pair_limit,
            temperature=args.temperature,
        )
        print(path)
        return 0

    if args.command == "train-neural-phantom-prior":
        path = train_neural_phantom_prior(
            args.outputs_dir,
            args.out,
            shape=tuple(args.shape),
            pair_limit=args.pair_limit,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        print(path)
        return 0

    if args.command == "optimize-empirical-basis":
        metrics_path = run_empirical_basis_refinement(
            args.ref,
            args.out,
            outputs_dir=args.outputs_dir,
            shape=tuple(args.shape),
            pair_limit=args.pair_limit,
            basis_count=args.basis_count,
            scatterers_count=args.scatterers_count,
            seed=args.seed,
            residual_exponents=args.residual_exponents,
            energy_ratios=args.energy_ratios,
            texture_strengths=args.texture_strengths,
        )
        print(metrics_path)
        return 0

    if args.command == "optimize-patch-basis":
        metrics_path = run_patch_basis_refinement(
            args.ref,
            args.out,
            outputs_dir=args.outputs_dir,
            shape=tuple(args.shape),
            basis_count=args.basis_count,
            tile_shape=tuple(args.tile_shape),
            scatterers_count=args.scatterers_count,
            seed=args.seed,
            residual_exponents=args.residual_exponents,
            texture_strengths=args.texture_strengths,
            energy_ratios=args.energy_ratios,
        )
        print(metrics_path)
        return 0

    if args.command == "optimize-direct-lattice":
        metrics_path = run_direct_lattice_refinement(
            args.ref,
            args.out,
            scatterers_count=args.scatterers_count,
            seed=args.seed,
            recipes=args.recipes,
            energy_scales=args.energy_scales,
        )
        print(metrics_path)
        return 0

    if args.command == "render-candidate-queue":
        metrics_path = render_candidate_queue(
            args.queue,
            args.ref,
            args.out,
            api_key_file=args.api_key_file,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            max_candidates=args.max_candidates,
            skip_existing=not args.rerun_existing,
            api_concurrency=args.api_concurrency,
        )
        print(metrics_path)
        return 0

    if args.command == "flow-energy-rank-batch":
        metrics_path = run_flow_energy_rank_batch(
            args.base_api_metrics,
            args.out,
            rank_start=args.rank_start,
            rank_end=args.rank_end,
            api_key_file=args.api_key_file,
            api_concurrency=args.api_concurrency,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            flow_strength=args.flow_strength,
            flow_smooth_sigma=args.flow_smooth_sigma,
            flow_attachment=args.flow_attachment,
            energy_exponent=args.energy_exponent,
        )
        print(metrics_path)
        return 0

    if args.command == "adaptive-flow-strength-batch":
        metrics_path = run_adaptive_flow_strength_batch(
            args.current_api_metrics,
            args.out,
            rank_start=args.rank_start,
            rank_end=args.rank_end,
            strengths=tuple(args.strengths),
            api_key_file=args.api_key_file,
            api_concurrency=args.api_concurrency,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            flow_smooth_sigma=args.flow_smooth_sigma,
            flow_attachment=args.flow_attachment,
            energy_exponent=args.energy_exponent,
        )
        print(metrics_path)
        return 0

    if args.command == "train-residual-selector":
        artifact_path = train_residual_parameter_selector(
            args.base_api_metrics,
            list(args.teacher_metrics),
            args.out,
            default_strength=args.default_strength,
            holdout_fraction=args.holdout_fraction,
            min_delta=args.min_delta,
            uncertainty_z=args.uncertainty_z,
            neighbor_count=args.neighbor_count,
            exploration_weight=args.exploration_weight,
            surrogate_calibration_metrics=list(args.surrogate_calibration_metrics),
            min_surrogate_ms_ssim=args.min_surrogate_ms_ssim,
            max_surrogate_lpips_proxy=args.max_surrogate_lpips_proxy,
            holdout_group_fields=tuple(args.holdout_group_fields),
        )
        print(artifact_path)
        return 0

    if args.command == "enrich-residual-teacher-maps":
        enriched_path = enrich_residual_teacher_maps(
            args.base_api_metrics,
            args.teacher_metrics,
            args.out,
            maps_dir=args.maps_dir,
            include_lpips=args.include_lpips,
            max_rows=args.max_rows,
        )
        print(enriched_path)
        return 0

    if args.command == "plan-residual-selector-queue":
        queue_path = plan_residual_selector_queue(
            args.base_api_metrics,
            args.artifact,
            args.out,
            limit=args.limit,
            min_expected_delta_lcb=args.min_expected_delta_lcb,
            row_wise_strengths=args.row_wise_strengths,
            min_map_delta_lcb=args.min_map_delta_lcb,
            min_map_safe_win_rate=args.min_map_safe_win_rate,
            require_map_safe=args.require_map_safe,
        )
        print(queue_path)
        return 0

    if args.command == "plan-residual-api-budget":
        queue_path = plan_residual_api_budget(
            args.selector_queue,
            args.out,
            total_api_calls=args.total_api_calls,
            energy_followup_fraction=args.energy_followup_fraction,
            strength_multipliers=tuple(args.strength_multipliers),
            energy_exponents=tuple(args.energy_exponents),
            min_strength=args.min_strength,
            max_strength=args.max_strength,
            diversity_fields=tuple(args.diversity_fields),
            max_per_stratum=args.max_per_stratum,
            probe_feedback_metrics=tuple(args.probe_feedback_metrics),
            feedback_group_fields=tuple(args.feedback_group_fields),
            energy_gate_summary=args.energy_gate_summary,
            exclude_exact_probed=args.exclude_exact_probed,
        )
        print(queue_path)
        return 0

    if args.command == "residual-selector-flow-batch":
        metrics_path = run_residual_selector_flow_batch(
            args.selector_queue,
            args.out,
            max_candidates=args.max_candidates,
            max_energy_followups=args.max_energy_followups,
            min_energy_flow_delta=args.min_energy_flow_delta,
            api_key_file=args.api_key_file,
            api_concurrency=args.api_concurrency,
            poll_interval_seconds=args.poll_interval_seconds,
            max_polls=args.max_polls,
            flow_smooth_sigma=args.flow_smooth_sigma,
            flow_attachment=args.flow_attachment,
            energy_exponent=args.energy_exponent,
        )
        print(metrics_path)
        return 0

    if args.command == "calibrate-energy-followup-gate":
        summary_path = calibrate_energy_followup_gate(
            tuple(args.probe_metrics),
            args.out,
            min_retained_delta_fraction=args.min_retained_delta_fraction,
            min_retained_map_delta_fraction=args.min_retained_map_delta_fraction,
            min_energy_calls=args.min_energy_calls,
            max_negative_energy_calls=args.max_negative_energy_calls,
            max_negative_map_calls=args.max_negative_map_calls,
        )
        print(summary_path)
        return 0

    if args.command == "enrich-residual-probe-energy-maps":
        enriched_path = enrich_residual_probe_energy_maps(
            tuple(args.probe_metrics),
            args.out,
            maps_dir=args.maps_dir,
        )
        print(enriched_path)
        return 0

    if args.command == "recover-api-results":
        metrics_path = recover_api_results(args.metrics, args.ref)
        print(metrics_path)
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
            learned_prior_artifact=args.learned_prior_artifact,
            neural_prior_artifact=args.neural_prior_artifact,
            evidence_metrics=args.evidence_metrics,
            baseline=args.baseline,
            min_samples_per_method=args.min_samples_per_method,
            require_real_lpips=args.require_real_lpips,
            max_generation_seconds=args.max_generation_seconds,
            max_guardrail_regression=args.max_guardrail_regression,
            strict_evidence=args.strict_evidence,
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
            progress=args.progress,
            api_key_file=args.api_key_file,
            api_concurrency=args.api_concurrency,
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

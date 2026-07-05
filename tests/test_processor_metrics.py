from __future__ import annotations

import csv
import json

import numpy as np
from skimage import io

from synthoct.evaluation import calculate_metrics
import synthoct.evaluation.maps as eval_maps
from synthoct.evaluation.summaries import summarize_challenge_metrics, write_ms_ssim_pair_comparison
from synthoct.features import calculate_oac, calculate_speckle_contrast_map, generate_maps


def test_processor_maps_and_metrics(tmp_path):
    image = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    ref = tmp_path / "ref.png"
    pred = tmp_path / "pred.png"
    io.imsave(ref, image)
    io.imsave(pred, image)

    maps = generate_maps(ref, output_dir=tmp_path / "maps")
    assert {"Struct", "OAC", "SC", "RSC"} == set(maps)
    assert maps["OAC"].exists()

    metrics = calculate_metrics(ref, pred, include_lpips=False)
    assert metrics["MSE"] == 0.0
    assert metrics["SSIM"] > 0.99


def test_oac_and_speckle_shapes():
    arr = np.ones((32, 48), dtype=np.float32)
    assert calculate_oac(arr).shape == arr.shape
    assert calculate_speckle_contrast_map(arr, window_size=6).shape == arr.shape


def test_feature_map_metrics_can_include_real_lpips(tmp_path, monkeypatch):
    image = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    ref = tmp_path / "ref.png"
    pred = tmp_path / "pred.png"
    io.imsave(ref, image)
    io.imsave(pred, image)
    include_lpips_calls = []

    def fake_calculate_metrics(ref_path, pred_path, include_lpips=True):
        include_lpips_calls.append(include_lpips)
        return {
            "MSE": 0.0,
            "PSNR": 99.0,
            "SSIM": 1.0,
            "MS-SSIM": 1.0,
            "VIF": 1.0,
            "LPIPS": 0.12 if include_lpips else float("nan"),
            "LPIPS_PROXY": 0.01,
        }

    monkeypatch.setattr(eval_maps, "calculate_metrics", fake_calculate_metrics)
    row = eval_maps.evaluate_feature_map_metrics(ref, pred, tmp_path / "ref_maps", tmp_path / "pred_maps", include_lpips=True)

    assert include_lpips_calls == [True, True, True]
    assert row["OAC_LPIPS"] == 0.12
    assert row["SC_LPIPS"] == 0.12
    assert row["RSC_LPIPS"] == 0.12


def test_challenge_summary_reports_official_median_score():
    rows = [
        {
            "fold": 0,
            "sample": 0,
            "archive_path": "a.png",
            "method": "candidate",
            "evidence_source": "hosted_api_true_scanner",
            "evidence_scope": "grouped_validation_2fold_1perfold",
            "Struct_evaluation_region": "full_frame",
            "generation_seconds": 1.0,
            "Struct_MS-SSIM": 0.6,
            "Struct_LPIPS": 0.2,
            "OAC_MS-SSIM": 0.5,
            "OAC_LPIPS": 0.3,
            "SC_MS-SSIM": 0.4,
            "SC_LPIPS": 0.4,
            "RSC_MS-SSIM": 0.7,
            "RSC_LPIPS": 0.1,
        },
        {
            "fold": 1,
            "sample": 0,
            "archive_path": "b.png",
            "method": "candidate",
            "evidence_source": "hosted_api_true_scanner",
            "evidence_scope": "grouped_validation_2fold_1perfold",
            "Struct_evaluation_region": "full_frame",
            "generation_seconds": 1.5,
            "Struct_MS-SSIM": 0.8,
            "Struct_LPIPS": 0.4,
            "OAC_MS-SSIM": 0.7,
            "OAC_LPIPS": 0.5,
            "SC_MS-SSIM": 0.6,
            "SC_LPIPS": 0.2,
            "RSC_MS-SSIM": 0.9,
            "RSC_LPIPS": 0.3,
        },
    ]

    summary = summarize_challenge_metrics(rows)
    assert summary[0]["official_metric_complete"] == 1
    assert summary[0]["official_missing_metrics"] == ""
    assert summary[0]["Struct_MS-SSIM_median"] == 0.7
    assert summary[0]["Struct_LPIPS_median"] == 0.30000000000000004
    assert np.isclose(summary[0]["official_score"], 0.675)


def test_write_ms_ssim_pair_comparison_outputs_rows_and_summary(tmp_path):
    base = tmp_path / "base.csv"
    candidate = tmp_path / "candidate.csv"
    queue = tmp_path / "queue.csv"
    for path, rows in [
        (
            base,
            [
                {"source_archive_path": "a.png", "MS-SSIM": "0.5", "LPIPS": "0.6"},
                {"source_archive_path": "b.png", "MS-SSIM": "0.7", "LPIPS": "0.4"},
            ],
        ),
        (
            candidate,
            [
                {"source_archive_path": "a.png", "MS-SSIM": "0.55", "LPIPS": "0.58"},
                {"source_archive_path": "b.png", "MS-SSIM": "0.7", "LPIPS": "0.4"},
            ],
        ),
    ]:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_archive_path", "MS-SSIM", "LPIPS"])
            writer.writeheader()
            writer.writerows(rows)
    with queue.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_archive_path", "expected_delta_lcb", "selected_strength", "residual_control_policy", "map_safety_status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_archive_path": "a.png",
                "expected_delta_lcb": "0.01",
                "selected_strength": "0.2",
                "residual_control_policy": "topology_residual_control_model_v1",
                "map_safety_status": "map_safety_not_trained",
            }
        )

    out = write_ms_ssim_pair_comparison(
        base,
        candidate,
        tmp_path / "comparison.csv",
        base_method="base",
        candidate_method="candidate",
        model_queue=queue,
    )
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    assert rows[0]["winner"] == "candidate"
    assert rows[0]["model_residual_policy"] == "topology_residual_control_model_v1"
    summary = json.loads(out.with_name("comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 2
    assert summary["candidate_wins"] == 1
    assert summary["ties"] == 1

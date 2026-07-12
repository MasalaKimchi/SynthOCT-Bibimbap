from __future__ import annotations

import json
import zipfile
from collections import Counter

import numpy as np
import pytest
from skimage import io

from synthoct.phantom import ExperimentConfig, save_phantom
from synthoct.submission import (
    build_preliminary_submission,
    flatten_reference,
    generate_phantom_batch,
    select_balanced_references,
    sha256_file,
    verify_submission_archive,
)


SEXES = ("Female", "Male")
AGES = ("1950-1960", "1990-2000")
SITES = ("Cheek", "Eye_corner")


def _references(root, per_stratum):
    paths = []
    for sex in SEXES:
        for age in AGES:
            for site in SITES:
                directory = root / sex / age / site
                directory.mkdir(parents=True, exist_ok=True)
                for index in range(per_stratum):
                    path = directory / f"scan_{index:02d}.png"
                    io.imsave(path, np.zeros((4, 4), dtype=np.uint8), check_contrast=False)
                    paths.append(path)
    return paths


def test_balanced_60_selection_has_exact_sex_age_and_site_marginals(tmp_path):
    root = tmp_path / "references"
    _references(root, per_stratum=15)
    selected = select_balanced_references(root, count=60, seed=2026)
    strata = Counter(path.relative_to(root).parts[:3] for path in selected)
    assert len(selected) == 60
    assert sorted(strata.values()) == [7, 7, 7, 7, 8, 8, 8, 8]
    for axis in range(3):
        marginals = Counter(stratum[axis] for stratum in (path.relative_to(root).parts[:3] for path in selected))
        assert set(marginals.values()) == {30}


def test_submission_zip_is_root_flat_reproducible_and_verified(tmp_path):
    reference_root = tmp_path / "references"
    references = _references(reference_root, per_stratum=1)
    phantom_dir = tmp_path / "phantoms"
    phantom_dir.mkdir()
    config = ExperimentConfig(scatterers_count=4)
    phantom = np.asarray(
        [
            [-1.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0, 0.001],
            [1.0, 1.0, 3.0, 0.01],
            [2.0, 2.0, 4.0, 0.1],
        ],
        dtype=float,
    )
    for reference in references:
        save_phantom(
            phantom,
            phantom_dir / flatten_reference(reference, reference_root),
            config=config,
        )

    first, manifest = build_preliminary_submission(
        reference_root,
        phantom_dir,
        tmp_path / "first.zip",
        count=8,
        seed=7,
        config=config,
    )
    second, _ = build_preliminary_submission(
        reference_root,
        phantom_dir,
        tmp_path / "second.zip",
        count=8,
        seed=7,
        config=config,
    )
    assert sha256_file(first) == sha256_file(second)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "validated"
    assert payload["archive"]["members"] == 8
    assert payload["scanner_contract"]["rows_per_phantom"] == 4
    with zipfile.ZipFile(first) as handle:
        assert all("/" not in name for name in handle.namelist())
        assert all(name.endswith(".txt") for name in handle.namelist())
        assert {info.date_time for info in handle.infolist()} == {(1980, 1, 1, 0, 0, 0)}
    report = verify_submission_archive(first, expected_count=8, config=config)
    assert report["status"] == "contract_validated"
    assert report["members"] == 8
    report = verify_submission_archive(
        first,
        expected_count=8,
        config=config,
        manifest_path=manifest,
        reference_root=reference_root,
    )
    assert report["status"] == "provenance_validated"


def test_submission_builder_rejects_wrong_row_count(tmp_path):
    reference_root = tmp_path / "references"
    references = _references(reference_root, per_stratum=1)
    phantom_dir = tmp_path / "phantoms"
    phantom_dir.mkdir()
    for reference in references:
        np.savetxt(
            phantom_dir / flatten_reference(reference, reference_root),
            np.zeros((3, 4)),
        )
    with pytest.raises(ValueError, match="exactly 4 scatterers"):
        build_preliminary_submission(
            reference_root,
            phantom_dir,
            tmp_path / "bad.zip",
            count=8,
            config=ExperimentConfig(scatterers_count=4),
        )


def test_submission_builder_rejects_archive_manifest_path_collision(tmp_path):
    reference_root = tmp_path / "references"
    _references(reference_root, per_stratum=1)
    phantom_dir = tmp_path / "phantoms"
    phantom_dir.mkdir()
    collision = tmp_path / "submission.zip"
    with pytest.raises(ValueError, match="paths must differ"):
        build_preliminary_submission(
            reference_root,
            phantom_dir,
            collision,
            count=8,
            config=ExperimentConfig(scatterers_count=4),
            manifest_path=collision,
        )


def test_generate_batch_retains_named_phantoms_and_feeds_submission(tmp_path, monkeypatch):
    reference_root = tmp_path / "references"
    references = _references(reference_root, per_stratum=1)
    config = ExperimentConfig(scatterers_count=4)

    def fake_generator(_reference, output, *, scatterers_count, **_kwargs):
        data = np.asarray(
            [
                [-1.0, -1.0, 1.0, 0.0],
                [0.0, 0.0, 2.0, 0.001],
                [1.0, 1.0, 3.0, 0.01],
                [2.0, 2.0, 4.0, 0.1],
            ],
            dtype=float,
        )
        return save_phantom(data, output, config=ExperimentConfig(scatterers_count=scatterers_count))

    monkeypatch.setattr(
        "synthoct.holographic_inverse.holographic_inverse_phantom",
        fake_generator,
    )

    output_dir, summary_path = generate_phantom_batch(
        reference_root,
        tmp_path / "phantoms",
        scatterers_count=4,
        scanner=config,
        retain_diagnostics=False,
    )

    expected_names = {flatten_reference(ref, reference_root) for ref in references}
    produced = {path.name for path in output_dir.glob("*.txt")}
    assert produced == expected_names

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "generation_complete"
    assert summary["n"] == len(references)
    assert summary["method"] == "holographic-inverse-v3-phase-pair"
    assert summary["scanner_contract"]["rows_per_phantom"] == 4
    assert {record["archive_name"] for record in summary["files"]} == expected_names
    assert all(record["rows"] == 4 for record in summary["files"])
    assert all(record["sha256"] for record in summary["files"])
    assert not (output_dir / "generation_summary.partial.json").exists()

    # The retained directory must be a drop-in --phantom-dir for the ZIP builder.
    archive, manifest = build_preliminary_submission(
        reference_root,
        output_dir,
        tmp_path / "batch.zip",
        count=8,
        seed=7,
        config=config,
    )
    report = verify_submission_archive(
        archive,
        expected_count=8,
        config=config,
        manifest_path=manifest,
        reference_root=reference_root,
    )
    assert report["status"] == "provenance_validated"


def test_generate_batch_limit_must_be_positive(tmp_path):
    reference_root = tmp_path / "references"
    _references(reference_root, per_stratum=1)
    with pytest.raises(ValueError, match="limit must be positive"):
        generate_phantom_batch(
            reference_root,
            tmp_path / "phantoms",
            scatterers_count=4,
            limit=0,
        )


def test_verifier_rejects_windows_style_member_paths(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("folder\\phantom.txt", "0 0 0 0\n")
    with pytest.raises(ValueError, match="unsafe submission member"):
        verify_submission_archive(
            archive,
            expected_count=1,
            config=ExperimentConfig(scatterers_count=1),
        )

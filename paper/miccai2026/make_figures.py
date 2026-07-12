from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from skimage import io


ROOT = Path(__file__).resolve().parents[2]
FIGURES = Path(__file__).resolve().parent / "figures"
HOSTED = ROOT / "outputs/hosted_api_public120_v3_200"
REFERENCE_ROOT = ROOT / "tmp/dataset/extracted/DATASET_PNG"
LOCAL_V1 = (
    ROOT
    / "outputs/holographic_inverse_v1_current_public120"
    / "detail.csv"
)
LOCAL_V3 = (
    ROOT
    / "outputs/holographic_inverse_v3_phase_pair_iter200_contract300k"
    / "public120_structural/detail.csv"
)


def _scan(path: Path) -> np.ndarray:
    image = np.asarray(io.imread(path, as_gray=True), dtype=np.float64)
    if image.max(initial=0.0) > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def _method_figure() -> None:
    fig = plt.figure(figsize=(7.2, 3.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(0.9, 1.5))
    pipeline = fig.add_subplot(grid[0])
    pipeline.set_axis_off()
    labels = [
        "51 dB target\nmagnitude",
        "Momentum alternating\nprojections",
        "Tikhonov scanner\npseudoinverse",
        "Carrier-equivalent\nscatterer pairs",
        "Virtual scanner\nrender",
    ]
    colors = ["#e8f1f8", "#d9eaf4", "#c8e1ee", "#f7e6c4", "#e6efdc"]
    xs = np.linspace(0.01, 0.81, len(labels))
    for index, (x, label, color) in enumerate(zip(xs, labels, colors, strict=True)):
        box = FancyBboxPatch(
            (x, 0.22),
            0.17,
            0.56,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=0.8,
            edgecolor="#334155",
            facecolor=color,
            transform=pipeline.transAxes,
        )
        pipeline.add_patch(box)
        pipeline.text(
            x + 0.085,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            transform=pipeline.transAxes,
        )
        if index < len(labels) - 1:
            pipeline.add_patch(
                FancyArrowPatch(
                    (x + 0.17, 0.5),
                    (xs[index + 1], 0.5),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=0.8,
                    color="#475569",
                    transform=pipeline.transAxes,
                )
            )

    lower = grid[1].subgridspec(1, 2, width_ratios=(1.0, 1.15), wspace=0.28)
    geometry = fig.add_subplot(lower[0])
    geometry.axhline(0.0, color="#64748b", lw=0.8)
    phi = 0.7 * np.pi
    wavelength = 1.0
    d0 = -phi * wavelength / (4.0 * np.pi)
    d1 = d0 + wavelength / 2.0
    w0 = -d1 / (d0 - d1)
    w1 = d0 / (d0 - d1)
    geometry.scatter([d0, d1], [0.0, 0.0], s=[180 * w0 + 20, 180 * w1 + 20], c=["#2563eb", "#d97706"], zorder=3)
    geometry.vlines([d0, d1], 0.0, [0.55, 0.35], colors=["#2563eb", "#d97706"], lw=1.2)
    geometry.text(d0, 0.62, r"$w_0 a$", ha="center", fontsize=8)
    geometry.text(d1, 0.42, r"$w_1 a$", ha="center", fontsize=8)
    geometry.annotate(
        r"$\lambda/2$",
        xy=(d0, -0.13),
        xytext=(d1, -0.13),
        arrowprops={"arrowstyle": "<->", "lw": 0.8},
        ha="center",
        va="top",
        fontsize=8,
    )
    geometry.set_xlim(min(d0, d1) - 0.2, max(d0, d1) + 0.2)
    geometry.set_ylim(-0.32, 0.82)
    geometry.set_yticks([])
    geometry.set_xlabel("depth offset (normalized by wavelength)", fontsize=8)
    geometry.set_title(r"Pair geometry: $w_0d_0+w_1d_1=0$", fontsize=9)
    geometry.spines[["left", "right", "top"]].set_visible(False)
    geometry.tick_params(axis="x", labelsize=7)

    response = fig.add_subplot(lower[1])
    relative_k = np.linspace(-0.1, 0.1, 401)
    k0 = 2.0 * np.pi / wavelength
    k = k0 * (1.0 + relative_k)
    target = np.exp(1j * phi)
    single = np.exp(-2j * k * d0)
    pair = w0 * np.exp(-2j * k * d0) + w1 * np.exp(-2j * k * d1)
    response.plot(relative_k * 100.0, np.abs(single - target), color="#64748b", lw=1.4, label="single")
    response.plot(relative_k * 100.0, np.abs(pair - target), color="#0f766e", lw=1.7, label="phase pair")
    response.axvline(0.0, color="#94a3b8", lw=0.7, ls="--")
    response.set_xlabel(r"wavenumber offset $(k-k_0)/k_0$ (\%)", fontsize=8)
    response.set_ylabel("complex coefficient error", fontsize=8)
    response.set_title("Analytic carrier response", fontsize=9)
    response.tick_params(labelsize=7)
    response.grid(alpha=0.2, lw=0.5)
    response.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(FIGURES / "method_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def _qualitative_figure(hosted: pd.DataFrame) -> dict[str, object]:
    median_value = float(hosted["Struct_MS-SSIM"].median())
    median_index = (hosted["Struct_MS-SSIM"] - median_value).abs().idxmin()
    selections = [
        ("Median-nearest", hosted.loc[median_index]),
        ("Lowest", hosted.loc[hosted["Struct_MS-SSIM"].idxmin()]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.05), constrained_layout=True)
    details: dict[str, object] = {}
    for row_index, (selection_name, row) in enumerate(selections):
        reference = _scan(REFERENCE_ROOT / str(row["reference"]))
        prediction = _scan(HOSTED / "gray" / f"{row['case_id']}.png")
        if reference.shape != prediction.shape:
            raise ValueError(f"shape mismatch for {row['case_id']}")
        residual = np.abs(prediction - reference)
        images = (reference, prediction, residual)
        for column, image in enumerate(images):
            axes[row_index, column].imshow(
                image,
                cmap="gray" if column < 2 else "inferno",
                vmin=0.0,
                vmax=1.0 if column < 2 else 0.25,
                aspect="auto",
            )
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        axes[row_index, 0].set_ylabel(
            f"{selection_name}\nMS-SSIM {row['Struct_MS-SSIM']:.4f}\nLPIPS {row['Struct_LPIPS']:.4f}",
            fontsize=7.4,
        )
        details[selection_name] = {
            "case_id": str(row["case_id"]),
            "reference": str(row["reference"]),
            "struct_ms_ssim": float(row["Struct_MS-SSIM"]),
            "struct_lpips": float(row["Struct_LPIPS"]),
        }
    for axis, title in zip(axes[0], ("Reference", "Hosted render", r"$|$residual$|$ (0--0.25)"), strict=True):
        axis.set_title(title, fontsize=8.5)
    fig.savefig(FIGURES / "qualitative.pdf", bbox_inches="tight")
    plt.close(fig)
    return details


def _cluster_statistics() -> dict[str, object]:
    v1 = pd.read_csv(LOCAL_V1)
    v3 = pd.read_csv(LOCAL_V3)
    if not v1["reference"].equals(v3["reference"]):
        raise ValueError("local result rows are not paired")
    delta = v3["Struct_MS-SSIM"].to_numpy() - v1["Struct_MS-SSIM"].to_numpy()
    series = v1["reference"].map(
        lambda value: re.sub(r"_frame(?:50|250|450)\.png$", "", value)
    )
    cluster_means = (
        pd.DataFrame({"series": series, "delta": delta})
        .groupby("series", sort=True)["delta"]
        .mean()
        .to_numpy()
    )
    if len(cluster_means) != 40:
        raise ValueError(f"expected 40 filename-defined series, found {len(cluster_means)}")
    rng = np.random.default_rng(20260711)
    bootstrap = np.empty(200_000, dtype=np.float64)
    for start in range(0, len(bootstrap), 10_000):
        sample = rng.choice(cluster_means, size=(10_000, len(cluster_means)), replace=True)
        bootstrap[start : start + 10_000] = sample.mean(axis=1)
    ci = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "series_count": int(len(cluster_means)),
        "series_wins": int(np.count_nonzero(cluster_means > 0.0)),
        "mean_image_delta": float(delta.mean()),
        "mean_cluster_delta": float(cluster_means.mean()),
        "cluster_bootstrap_seed": 20260711,
        "cluster_bootstrap_replicates": 200_000,
        "cluster_bootstrap_ci95": [float(ci[0]), float(ci[1])],
        "one_sided_exact_wilcoxon_p": float(2.0**-40),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    hosted = pd.read_csv(HOSTED / "detail.csv")
    if len(hosted) != 120:
        raise ValueError(f"expected 120 hosted rows, found {len(hosted)}")
    _method_figure()
    selections = _qualitative_figure(hosted)
    derived = {
        "hosted_rows": int(len(hosted)),
        "qualitative_selection": selections,
        "cluster_statistics": _cluster_statistics(),
    }
    (Path(__file__).resolve().parent / "derived_results.json").write_text(
        json.dumps(derived, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

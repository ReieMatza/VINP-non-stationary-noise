"""
Comprehensive evaluation script for VINP dereverberation results.

Compares multiple conditions (e.g. clean, white noise, non-stationary noise)
using matched speech/RIR pairs.

Metrics computed:
  Speech quality:  SI-SDR, ESTOI, PESQ (wideband)
  RIR estimation:  RT60 error (absolute), DRR error (absolute)

Outputs:
  - Per-sample results tables (printed + CSV)
  - Summary table comparing all conditions (printed + CSV)
  - Bar plots, per-sample line plots, RT60 scatter

Usage:
    python evaluate.py \
        --datasets clean=test_data \
                   white=test_data_white \
                   nonstat=test_data_nonstat \
        --output_dir eval_results

Directory structure expected under each dataset dir:
    <dataset_dir>/
        clean/                  # ground-truth direct-path speech
        rir/                    # ground-truth RIRs
        output_oSpatialNet/     # (or custom --model_subdir)
            normed/             # enhanced (dereverberated) speech
            rir/                # estimated RIRs
        metadata.json           # contains GT RT60 per sample
"""

import argparse
import os
import glob
import json
import csv

import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

SR = 16000

# Consistent colors for up to 6 conditions
COLORS = ["#4C9BE8", "#E8764C", "#5EBA7D", "#9B59B6", "#F1C40F", "#E74C3C"]


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_wav(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav[:1]
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0)


# ---------------------------------------------------------------------------
# RIR analysis (RT60 / DRR)
# ---------------------------------------------------------------------------

def estimate_rt60(rir: np.ndarray, sr: int = SR) -> float:
    import scipy.stats
    power = np.abs(rir) ** 2
    edc_flip = np.cumsum(power[::-1])
    edc = edc_flip[::-1]
    edc_db = 10 * np.log10(edc / edc.max() + 1e-12)

    win = int(0.005 * sr)
    if win > 1:
        kernel = np.ones(win) / win
        edc_db = np.convolve(edc_db, kernel, mode="same")

    edc_db = edc_db - edc_db[0]
    taxis = np.arange(len(edc_db)) / sr

    search_beg = 0
    while search_beg < len(edc_db) - 1 and edc_db[search_beg] >= -5:
        search_beg += 1

    best_r, best_k = float("inf"), -60
    lo = min(search_beg, int(sr * 0.05))
    hi = max(search_beg, int(sr * 0.05)) + 1
    for beg in range(lo, hi):
        end = beg
        while end < len(edc_db) - 1 and edc_db[end] >= edc_db[beg] - 5:
            end += 1
        if end <= beg + 2:
            continue
        k, b, r, _, _ = scipy.stats.linregress(
            taxis[beg:end], edc_db[beg:end], alternative="less"
        )
        if r < best_r:
            best_r, best_k = r, k

    return -60.0 / best_k if best_k < 0 else float("nan")


def estimate_drr(rir: np.ndarray, sr: int = SR) -> float:
    peak = np.argmax(np.abs(rir))
    dp_start = int(max(peak - 0.0025 * sr, 0))
    dp_stop = int(peak + 0.0025 * sr)
    dp_power = np.sum(rir[dp_start:dp_stop] ** 2)
    total_power = np.sum(rir ** 2)
    reverb_power = total_power - dp_power
    if reverb_power <= 0:
        return float("inf")
    return 10 * np.log10(dp_power / reverb_power)


# ---------------------------------------------------------------------------
# Evaluate one dataset
# ---------------------------------------------------------------------------

def evaluate_dataset(dataset_dir: str, model_subdir: str = "output_oSpatialNet"):
    clean_dir = os.path.join(dataset_dir, "clean")
    gt_rir_dir = os.path.join(dataset_dir, "rir")
    est_speech_dir = os.path.join(dataset_dir, model_subdir, "normed")
    est_rir_dir = os.path.join(dataset_dir, model_subdir, "rir")
    meta_path = os.path.join(dataset_dir, "metadata.json")

    with open(meta_path) as f:
        metadata = json.load(f)

    samples = sorted(glob.glob(os.path.join(est_speech_dir, "*.wav")))
    assert len(samples) > 0, f"No samples found in {est_speech_dir}"

    si_sdr_metric = ScaleInvariantSignalDistortionRatio()
    estoi_metric = ShortTimeObjectiveIntelligibility(SR, extended=True)
    pesq_metric = PerceptualEvaluationSpeechQuality(SR, "wb")

    results = []
    for est_path in samples:
        basename = os.path.basename(est_path)
        clean_path = os.path.join(clean_dir, basename)
        gt_rir_path = os.path.join(gt_rir_dir, basename)
        est_rir_path = os.path.join(est_rir_dir, basename)

        if not os.path.exists(clean_path):
            print(f"  Warning: missing reference {clean_path}, skipping")
            continue

        est_speech = load_wav(est_path)
        ref_speech = load_wav(clean_path)
        min_len = min(est_speech.shape[0], ref_speech.shape[0])
        est_speech = est_speech[:min_len]
        ref_speech = ref_speech[:min_len]

        si_sdr_val = si_sdr_metric(est_speech, ref_speech).item()
        estoi_val = estoi_metric(est_speech.unsqueeze(0), ref_speech.unsqueeze(0)).item()
        try:
            pesq_val = pesq_metric(est_speech.unsqueeze(0), ref_speech.unsqueeze(0)).item()
        except Exception:
            pesq_val = float("nan")

        gt_rt60 = metadata.get(basename, {}).get("RT60", float("nan"))
        est_rt60 = est_drr = gt_drr = float("nan")
        if os.path.exists(est_rir_path) and os.path.exists(gt_rir_path):
            est_rir_wav = load_wav(est_rir_path).numpy()
            gt_rir_wav = load_wav(gt_rir_path).numpy()
            est_rt60 = estimate_rt60(est_rir_wav, SR)
            est_drr = estimate_drr(est_rir_wav, SR)
            gt_drr = estimate_drr(gt_rir_wav, SR)

        results.append({
            "sample": basename,
            "SI-SDR": round(si_sdr_val, 2),
            "ESTOI": round(estoi_val, 4),
            "PESQ": round(pesq_val, 3),
            "GT_RT60": round(gt_rt60, 3),
            "Est_RT60": round(est_rt60, 3),
            "RT60_err": round(abs(est_rt60 - gt_rt60), 3),
            "GT_DRR": round(gt_drr, 2),
            "Est_DRR": round(est_drr, 2),
            "DRR_err": round(abs(est_drr - gt_drr), 2),
        })
        si_sdr_metric.reset()
        estoi_metric.reset()
        pesq_metric.reset()

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

METRIC_KEYS = ["SI-SDR", "ESTOI", "PESQ", "RT60_err", "DRR_err"]


def compute_summary(results: list, label: str) -> dict:
    summary = {"condition": label}
    for m in METRIC_KEYS:
        vals = [r[m] for r in results if not (isinstance(r[m], float) and np.isnan(r[m]))]
        summary[f"{m}_mean"] = round(np.mean(vals), 3) if vals else float("nan")
        summary[f"{m}_std"] = round(np.std(vals), 3) if vals else float("nan")
    return summary


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_table(results: list, title: str):
    if not results:
        return
    keys = list(results[0].keys())
    col_widths = {k: max(len(k), max(len(str(r[k])) for r in results)) for k in keys}
    header = " | ".join(k.rjust(col_widths[k]) for k in keys)
    sep = "-+-".join("-" * col_widths[k] for k in keys)
    print(f"\n{'='*len(header)}")
    print(f"  {title}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for r in results:
        print(" | ".join(str(r[k]).rjust(col_widths[k]) for k in keys))
    print()


def save_csv(results: list, path: str):
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# Plots (N conditions)
# ---------------------------------------------------------------------------

METRICS_INFO = {
    "SI-SDR":   {"unit": "dB", "higher_better": True},
    "ESTOI":    {"unit": "",   "higher_better": True},
    "PESQ":     {"unit": "",   "higher_better": True},
    "RT60_err": {"unit": "s",  "higher_better": False},
    "DRR_err":  {"unit": "dB", "higher_better": False},
}


def plot_comparison(all_results: dict, output_dir: str):
    """all_results: {label: [per-sample dicts]}"""
    os.makedirs(output_dir, exist_ok=True)
    labels = list(all_results.keys())
    n_cond = len(labels)
    colors = COLORS[:n_cond]

    # --- Compute means and stds ---
    means = {m: [] for m in METRICS_INFO}
    stds = {m: [] for m in METRICS_INFO}
    for label in labels:
        for m in METRICS_INFO:
            vals = [r[m] for r in all_results[label] if not np.isnan(r[m])]
            means[m].append(np.mean(vals) if vals else 0)
            stds[m].append(np.std(vals) if vals else 0)

    # --- Speech quality bar chart ---
    speech_metrics = ["SI-SDR", "ESTOI", "PESQ"]
    fig, axes = plt.subplots(1, len(speech_metrics), figsize=(4.5 * len(speech_metrics), 5.5))
    for ax, m in zip(axes, speech_metrics):
        x = np.arange(n_cond)
        bars = ax.bar(x, means[m], yerr=stds[m], color=colors, capsize=5, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
        unit = f" ({METRICS_INFO[m]['unit']})" if METRICS_INFO[m]["unit"] else ""
        ax.set_ylabel(f"{m}{unit}", fontsize=12)
        ax.set_title(m, fontsize=13, fontweight="bold")
        for bar, v in zip(bars, means[m]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Speech Quality Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "speech_quality_comparison.png"), dpi=150)
    plt.close(fig)

    # --- RIR estimation bar chart ---
    rir_metrics = ["RT60_err", "DRR_err"]
    fig, axes = plt.subplots(1, len(rir_metrics), figsize=(4.5 * len(rir_metrics), 5.5))
    for ax, m in zip(axes, rir_metrics):
        x = np.arange(n_cond)
        bars = ax.bar(x, means[m], yerr=stds[m], color=colors, capsize=5, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
        unit = f" ({METRICS_INFO[m]['unit']})" if METRICS_INFO[m]["unit"] else ""
        ax.set_ylabel(f"{m}{unit}", fontsize=12)
        ax.set_title(m, fontsize=13, fontweight="bold")
        for bar, v in zip(bars, means[m]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("RIR Estimation Errors", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rir_estimation_comparison.png"), dpi=150)
    plt.close(fig)

    # --- Per-sample line plots ---
    all_metrics = list(METRICS_INFO.keys())
    n_metrics = len(all_metrics)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes_flat = axes.flatten() if n_metrics > 1 else [axes]

    for idx, m in enumerate(all_metrics):
        ax = axes_flat[idx]
        for ci, label in enumerate(labels):
            vals = [r[m] for r in all_results[label]]
            ax.plot(range(len(vals)), vals, "o-", color=colors[ci],
                    label=label, markersize=4, linewidth=1.2)
        ax.set_xlabel("Sample index", fontsize=10)
        unit = f" ({METRICS_INFO[m]['unit']})" if METRICS_INFO[m]["unit"] else ""
        ax.set_ylabel(f"{m}{unit}", fontsize=10)
        ax.set_title(m, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for idx in range(len(all_metrics), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Per-Sample Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_sample_comparison.png"), dpi=150)
    plt.close(fig)

    # --- RT60 scatter: estimated vs GT (one subplot per condition) ---
    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 5))
    if n_cond == 1:
        axes = [axes]
    for ax, label, color in zip(axes, labels, colors):
        gt = [r["GT_RT60"] for r in all_results[label]]
        est = [r["Est_RT60"] for r in all_results[label]]
        ax.scatter(gt, est, c=color, s=50, edgecolors="black", linewidth=0.5, zorder=3)
        lims = [0, max(max(gt), max(est)) * 1.1 + 0.1]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Ideal")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Ground Truth RT60 (s)", fontsize=10)
        ax.set_ylabel("Estimated RT60 (s)", fontsize=10)
        ax.set_title(f"RT60: {label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rt60_scatter.png"), dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_dataset_arg(s: str):
    """Parse 'label=path' into (label, path)."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"Expected format 'label=path', got '{s}'"
        )
    label, path = s.split("=", 1)
    return label, path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VINP dereverberation results across multiple conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python evaluate.py \\\n"
               "    --datasets clean=test_data white=test_data_white nonstat=test_data_nonstat \\\n"
               "    --output_dir eval_results"
    )
    parser.add_argument("--datasets", type=parse_dataset_arg, nargs="+", required=True,
                        help="One or more label=path pairs, e.g. clean=test_data white=test_data_white")
    parser.add_argument("--model_subdir", type=str, default="output_oSpatialNet",
                        help="Subdirectory containing model outputs")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Directory to save evaluation outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    summaries = []

    for label, path in args.datasets:
        print(f"\nEvaluating [{label}] from {path} ...")
        results = evaluate_dataset(path, args.model_subdir)
        print(f"  {len(results)} samples evaluated")
        all_results[label] = results

        print_table(results, f"Per-Sample Results: {label}")
        save_csv(results, os.path.join(args.output_dir, f"{label}_per_sample.csv"))
        summaries.append(compute_summary(results, label))

    # --- Summary table ---
    print_table(summaries, "Summary: Mean ± Std across conditions")
    save_csv(summaries, os.path.join(args.output_dir, "summary.csv"))
    print(f"  CSVs saved to {args.output_dir}/")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_comparison(all_results, args.output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

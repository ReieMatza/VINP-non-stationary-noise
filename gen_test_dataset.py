"""
Generate a small test dataset of reverberant speech for VINP evaluation.

Takes clean speech from VCTK and convolves with simulated RIRs to produce:
  - reverberant/  : reverberant speech (input to enhance_rir_avg.py)
  - clean/        : direct-path convolved speech (reference for metrics)
  - rir/          : ground-truth RIRs

Supports three noise modes via --noise_type:
  1. none             -- reverb only (default)
  2. white            -- reverb + white Gaussian noise
  3. non-stationary   -- reverb + real non-stationary noise (requires --noise_dir)

All modes use the same seed-based speech/RIR pairing, so running with
different --noise_type produces matched pairs for fair comparison.

Usage (clean):
    python gen_test_dataset.py \
        --speech_dir .../wav48 --rir_dir .../test \
        --output_dir ./test_data --num_samples 20

Usage (white noise, SNR=10 dB):
    python gen_test_dataset.py \
        --speech_dir .../wav48 --rir_dir .../test \
        --noise_type white --snr 10 \
        --output_dir ./test_data_white --num_samples 20

Usage (non-stationary noise, SNR=10 dB):
    python gen_test_dataset.py \
        --speech_dir .../wav48 --rir_dir .../test \
        --noise_type non-stationary --noise_dir .../non-stationary --snr 10 \
        --output_dir ./test_data_nonstat --num_samples 20
"""

import argparse
import os
import glob
import json
import random

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import soundfile as sf


SR = 16000


def load_wav_16k(path: str) -> torch.Tensor:
    """Load a wav file and ensure it is 16kHz mono."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav[:1]
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0)


# ---------------------------------------------------------------------------
# Non-stationarity analysis
# ---------------------------------------------------------------------------

def compute_frame_energy_db(wav: torch.Tensor, frame_len: int = 800) -> torch.Tensor:
    n_frames = wav.shape[0] // frame_len
    frames = wav[:n_frames * frame_len].reshape(n_frames, frame_len)
    energy = (frames ** 2).mean(dim=1)
    return 10 * torch.log10(energy + 1e-10)


def score_non_stationarity(energy_db: torch.Tensor) -> float:
    std = energy_db.std().item()
    peak_to_mean = (energy_db.max() - energy_db.mean()).item()
    return std + 0.5 * peak_to_mean


def find_non_stationary_segments(
    wav: torch.Tensor, seg_duration: float, num_segments: int,
    frame_len: int = 800, hop_ratio: float = 0.25, min_energy_db: float = -65.0,
) -> list:
    seg_samples = int(seg_duration * SR)
    seg_frames = int(seg_duration / (frame_len / SR))
    hop_frames = max(1, int(seg_frames * hop_ratio))

    energy_db = compute_frame_energy_db(wav, frame_len)
    n_frames = energy_db.shape[0]

    candidates = []
    for start_f in range(0, n_frames - seg_frames, hop_frames):
        window = energy_db[start_f : start_f + seg_frames]
        if window.mean().item() < min_energy_db:
            continue
        score = score_non_stationarity(window)
        start_sample = start_f * frame_len
        end_sample = start_sample + seg_samples
        candidates.append((score, start_sample, end_sample))

    candidates.sort(reverse=True, key=lambda x: x[0])

    selected, used_ranges = [], []
    for score, s, e in candidates:
        if any(s < ue and e > us for us, ue in used_ranges):
            continue
        selected.append((score, s, e))
        used_ranges.append((s, e))
        if len(selected) >= num_segments:
            break
    return selected


def load_noise_segments(noise_dir: str, num_needed: int, max_seg_duration: float):
    noise_files = sorted(
        glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
        + glob.glob(os.path.join(noise_dir, "**", "*.flac"), recursive=True)
    )
    seen_dirs = set()
    unique_files = []
    for f in noise_files:
        d = os.path.dirname(f)
        if d not in seen_dirs:
            seen_dirs.add(d)
            unique_files.append(f)

    print(f"  Found {len(unique_files)} unique noise source(s)")
    per_file = max(1, (num_needed + len(unique_files) - 1) // len(unique_files))

    all_segments = []
    for nf in unique_files:
        src_name = os.path.basename(os.path.dirname(nf)) + "/" + os.path.basename(nf)
        print(f"  Analyzing {src_name} for non-stationary segments...")
        wav = load_wav_16k(nf)
        segs = find_non_stationary_segments(wav, max_seg_duration, per_file * 2)
        for score, s, e in segs:
            all_segments.append((score, wav[s:e], src_name))
        print(f"    Found {len(segs)} candidate segments (top score={segs[0][0]:.2f})")

    all_segments.sort(reverse=True, key=lambda x: x[0])
    return [(seg, name, sc) for sc, seg, name in all_segments[:num_needed]]


def mix_noise_at_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    sig_len = signal.shape[0]
    if noise.shape[0] >= sig_len:
        noise = noise[:sig_len]
    else:
        repeats = (sig_len // noise.shape[0]) + 1
        noise = noise.repeat(repeats)[:sig_len]

    sig_power = (signal ** 2).mean() + 1e-10
    noi_power = (noise ** 2).mean() + 1e-10
    gain = (sig_power / noi_power * 10 ** (-snr_db / 10)).sqrt()
    return signal + gain * noise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate small reverberant test set")
    parser.add_argument("--speech_dir", type=str, required=True,
                        help="Directory with clean speech .wav files (searched recursively)")
    parser.add_argument("--rir_dir", type=str, required=True,
                        help="Directory with .npz RIR files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the test dataset")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of reverberant samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--min_duration", type=float, default=2.0,
                        help="Minimum speech duration in seconds")
    parser.add_argument("--max_duration", type=float, default=8.0,
                        help="Maximum speech duration in seconds")

    # Noise options
    parser.add_argument("--noise_type", type=str, default="none",
                        choices=["none", "white", "non-stationary"],
                        help="Type of noise to add: none, white, non-stationary")
    parser.add_argument("--noise_dir", type=str, default=None,
                        help="Directory with non-stationary noise recordings "
                             "(required when --noise_type=non-stationary)")
    parser.add_argument("--snr", type=float, default=10.0,
                        help="Fixed SNR in dB for noise mixing (default: 10)")

    args = parser.parse_args()

    if args.noise_type == "non-stationary" and args.noise_dir is None:
        parser.error("--noise_dir is required when --noise_type=non-stationary")

    # --- Deterministic seeding (same speech/RIR pairs for all noise modes) ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Discover speech & RIR files ---
    speech_files = sorted(
        glob.glob(os.path.join(args.speech_dir, "**", "*.wav"), recursive=True)
        + glob.glob(os.path.join(args.speech_dir, "**", "*.flac"), recursive=True)
    )
    rir_files = sorted(glob.glob(os.path.join(args.rir_dir, "*.npz")))

    print(f"Found {len(speech_files)} speech files, {len(rir_files)} RIR files")
    assert len(speech_files) >= args.num_samples, \
        f"Need at least {args.num_samples} speech files, found {len(speech_files)}"
    assert len(rir_files) >= args.num_samples, \
        f"Need at least {args.num_samples} RIR files, found {len(rir_files)}"

    # --- Sample a subset (deterministic via seed -- identical across noise modes) ---
    speech_subset = random.sample(speech_files, args.num_samples)
    rir_subset = random.sample(rir_files, args.num_samples)

    # --- Load non-stationary noise segments if needed ---
    noise_segments = []
    if args.noise_type == "non-stationary":
        noise_rng_state = random.getstate()
        print(f"\nLoading non-stationary noise from: {args.noise_dir}")
        noise_segments = load_noise_segments(
            args.noise_dir, args.num_samples, args.max_duration
        )
        print(f"  Selected {len(noise_segments)} noise segments\n")
        random.setstate(noise_rng_state)

    # --- Create output dirs ---
    out_reverb = os.path.join(args.output_dir, "reverberant")
    out_clean = os.path.join(args.output_dir, "clean")
    out_rir = os.path.join(args.output_dir, "rir")
    os.makedirs(out_reverb, exist_ok=True)
    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_rir, exist_ok=True)

    # Separate RNG for white noise generation (does not affect pairing)
    white_rng = torch.Generator().manual_seed(args.seed + 1000)

    metadata = {}

    for i, (speech_path, rir_path) in enumerate(zip(speech_subset, rir_subset)):
        print(f"[{i+1}/{args.num_samples}] {os.path.basename(speech_path)} + {os.path.basename(rir_path)}")

        speech = load_wav_16k(speech_path)

        if speech.shape[0] / SR < args.min_duration:
            print(f"  Skipping (too short: {speech.shape[0]/SR:.1f}s)")
            continue

        max_samples = int(args.max_duration * SR)
        if speech.shape[0] > max_samples:
            speech = speech[:max_samples]

        if speech.abs().max() > 0:
            speech = speech / speech.abs().max()

        # Load RIR
        rir_data = np.load(rir_path, allow_pickle=True)
        rir_full = torch.from_numpy(rir_data["rir"][0, 0].copy()).float()
        rir_dp = torch.from_numpy(rir_data["rir_dp"][0, 0].copy()).float()
        rt60 = float(rir_data["RT60"])

        rir_scale = rir_full.abs().max()
        rir_full = rir_full / rir_scale
        rir_dp = rir_dp / rir_scale

        reverb = F.fftconvolve(speech, rir_full, mode="full")
        clean = F.fftconvolve(speech, rir_dp, mode="full")

        out_len = speech.shape[0]
        reverb = reverb[:out_len]
        clean = clean[:out_len]

        # --- Add noise ---
        noise_info = {}
        if args.noise_type == "white":
            white_noise = torch.randn(reverb.shape[0], generator=white_rng)
            reverb = mix_noise_at_snr(reverb, white_noise, args.snr)
            noise_info = {"noise_type": "white", "SNR_dB": args.snr}
            print(f"  + white noise (SNR={args.snr:.1f}dB)")

        elif args.noise_type == "non-stationary" and i < len(noise_segments):
            noise_seg, noise_source, noise_score = noise_segments[i]
            reverb = mix_noise_at_snr(reverb, noise_seg, args.snr)
            noise_info = {
                "noise_type": "non-stationary",
                "noise_source": noise_source,
                "noise_non_stationarity_score": round(noise_score, 2),
                "SNR_dB": args.snr,
            }
            print(f"  + noise: {noise_source} (score={noise_score:.2f}, SNR={args.snr:.1f}dB)")

        # Normalize
        scale = reverb.abs().max() + 1e-32
        reverb = reverb / scale
        clean = clean / scale

        basename = f"sample_{i:03d}.wav"

        sf.write(os.path.join(out_reverb, basename), reverb.numpy(), SR, subtype="FLOAT")
        sf.write(os.path.join(out_clean, basename), clean.numpy(), SR, subtype="FLOAT")
        sf.write(os.path.join(out_rir, basename),
                 (rir_full / rir_full.abs().max()).numpy(), SR, subtype="FLOAT")

        sample_meta = {
            "speech_source": os.path.basename(speech_path),
            "rir_source": os.path.basename(rir_path),
            "RT60": rt60,
            "speech_duration_s": speech.shape[0] / SR,
        }
        sample_meta.update(noise_info)
        metadata[basename] = sample_meta
        print(f"  RT60={rt60:.2f}s, duration={speech.shape[0]/SR:.2f}s")

    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    noise_label = {
        "none": "no noise",
        "white": f"white noise at SNR={args.snr}dB",
        "non-stationary": f"non-stationary noise at SNR={args.snr}dB",
    }[args.noise_type]

    print(f"\nDone! Generated {len(metadata)} samples in {args.output_dir}")
    print(f"  reverberant/ : input for enhance_rir_avg.py (-i)")
    print(f"  clean/       : reference for evaluation")
    print(f"  rir/         : ground-truth RIRs")
    print(f"  Noise mode: {noise_label}")


if __name__ == "__main__":
    main()

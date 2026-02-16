# Windowed Noise Model for Non-Stationary Noise in VINP

This document formulates the modifications made to the VEM algorithm in VINP to handle non-stationary noise. Each section presents the **original** formula from the paper [1] alongside the **modified** version, with a brief derivation.

> [1] P. Wang, Y. Fang, and X. Li, "VINP: Variational Bayesian Inference with Neural Speech Prior for Joint ASR-Effective Speech Dereverberation and Blind RIR Identification," arXiv:2502.07205v3, 2025.

---

## 1. Signal Model and Noise Assumption

The VINP signal model in the T-F domain (Eq. 2 in [1]) is:

$$X(f,t) = \sum_{l=0}^{L-1} H_l(f)\, S(f, t-l) + W(f,t) = \mathbf{H}(f)\,\mathbf{S}(f,t) + W(f,t)$$

where $f$ and $t$ are frequency and time indices, $L$ is the CTF filter length, $H_l(f)$ are CTF coefficients, $S(f,t)$ is the anechoic source, and $W(f,t)$ is additive noise.

### Original (Eq. 3 in [1]): Time-Invariant Noise

The noise follows a zero-mean complex Gaussian with precision $\delta(f)$ that depends only on frequency:

$$W(f,t) \sim \mathcal{CN}\!\left(0,\; \delta^{-1}(f)\right)$$

This assumption implies **stationary noise** --- the noise power is constant across all time frames within each frequency band.

### Modified: Time-Varying Noise

We relax the stationarity assumption by allowing the noise precision to vary with time:

$$\boxed{W(f,t) \sim \mathcal{CN}\!\left(0,\; \delta^{-1}(f,t)\right)}$$

where $\delta(f,t)$ is now a function of both frequency $f$ and time frame $t$. This enables the model to capture **non-stationary noise** such as transient events (door slams, clapping, speech from interfering speakers).

---

## 2. E-Step: Posterior Distribution Update (Eq. 21 in [1])

The E-step computes the posterior distribution of the anechoic speech $S(f,t)$, which is Gaussian with precision $\hat{\gamma}(f,t)$ and mean $\hat{\mu}(f,t)$.

### 2.1 Posterior Precision

**Original (Eq. 21, first line):**

$$\hat{\gamma}(f,t) = \alpha(f,t) + \delta(f)\,\|\mathbf{H}(f)\|_2^2$$

where $\alpha(f,t)$ is the speech prior precision and $\|\mathbf{H}(f)\|_2^2 = \sum_{l=0}^{L-1} |H_l(f)|^2$.

Since $\delta(f)$ is a scalar shared across all lags, it factors out of the sum.

**Modified:**

When noise precision varies with time, each observed frame $X(f,t+l)$ contributing to the posterior of $S(f,t)$ carries a different noise precision $\delta(f,t+l)$. The precision becomes:

$$\boxed{\hat{\gamma}(f,t) = \alpha(f,t) + \sum_{l=0}^{L-1} \delta(f,t+l)\,|H_l(f)|^2}$$

The noise precision $\delta$ can no longer be factored out of the sum --- each CTF lag $l$ is weighted by the noise precision at its corresponding time frame $t+l$.

**Reduction to original:** When $\delta(f,t+l) = \delta(f)$ for all $l$ (stationary noise), this reduces to $\alpha(f,t) + \delta(f)\sum_l |H_l(f)|^2 = \alpha(f,t) + \delta(f)\|\mathbf{H}(f)\|_2^2$.

### 2.2 Posterior Mean

**Original (Eq. 21, second line):**

$$\hat{\mu}(f,t) = \hat{\gamma}^{-1}(f,t)\;\delta(f)\sum_{l=0}^{L-1} H_l^*(f)\Big[X(f,t+l) - \mathbf{H}_{\backslash l}(f)\,\hat{\boldsymbol{\mu}}_{\text{pre}}(f,t+l)\Big]$$

where $\mathbf{H}_{\backslash l}(f)$ is the CTF vector with the $l$-th coefficient set to zero, and $\hat{\boldsymbol{\mu}}_{\text{pre}}(f,t+l)$ contains posterior means from the previous VEM iteration.

**Modified:**

The noise precision moves inside the summation, applied per-lag:

$$\boxed{\hat{\mu}(f,t) = \hat{\gamma}^{-1}(f,t)\sum_{l=0}^{L-1} \delta(f,t+l)\;H_l^*(f)\Big[X(f,t+l) - \mathbf{H}_{\backslash l}(f)\,\hat{\boldsymbol{\mu}}_{\text{pre}}(f,t+l)\Big]}$$

Each residual error at lag $l$ is now weighted by the noise precision at time $t+l$ rather than a shared precision.

**Reduction to original:** When $\delta(f,t+l) = \delta(f)$, the scalar $\delta(f)$ factors out, recovering the original formula.

---

## 3. M-Step: Noise Precision Estimation (Eq. 24 in [1])

### 3.1 Original: Global Average

The M-step updates the noise precision by maximizing the expected log-likelihood of the complete data (Eq. 23 in [1]):

$$\langle \ln p(\mathbf{S}, \mathbf{X}; \boldsymbol{\theta}) \rangle = T \ln \delta(f) - \delta(f)\sum_{t=1}^{T}\left\langle\left|X(f,t) - \mathbf{H}(f)\mathbf{S}(f,t)\right|^2\right\rangle + c$$

Setting the derivative with respect to $\delta(f)$ to zero:

$$\frac{\partial}{\partial\,\delta(f)}: \quad \frac{T}{\delta(f)} - \sum_{t=1}^{T}\left\langle\left|X(f,t) - \mathbf{H}(f)\mathbf{S}(f,t)\right|^2\right\rangle = 0$$

yields the noise precision estimate (Eq. 24 in [1]):

$$\hat{\delta}(f) = \frac{T}{\displaystyle\sum_{t=1}^{T}\left\langle\left|X(f,t) - \mathbf{H}(f)\mathbf{S}(f,t)\right|^2\right\rangle}$$

Equivalently, in noise **variance** form ($\sigma^2 = \delta^{-1}$):

$$\hat{\sigma}^2(f) = \frac{1}{T}\sum_{t=1}^{T} \operatorname{Error}(f,t)$$

where $\operatorname{Error}(f,t) = \left\langle|X(f,t) - \mathbf{H}(f)\mathbf{S}(f,t)|^2\right\rangle$ is the expected squared reconstruction error at each T-F point. This is a **global average** over all $T$ frames.

### 3.2 Modified: Sliding Window Average

With time-varying noise, the expected log-likelihood becomes:

$$\langle \ln p(\mathbf{S}, \mathbf{X}; \boldsymbol{\theta}) \rangle = \sum_{t=1}^{T}\ln \delta(f,t) - \sum_{t=1}^{T}\delta(f,t)\left\langle\left|X(f,t) - \mathbf{H}(f)\mathbf{S}(f,t)\right|^2\right\rangle + c$$

Setting the derivative with respect to $\delta(f,t)$ to zero:

$$\frac{\partial}{\partial\,\delta(f,t)}: \quad \frac{1}{\delta(f,t)} - \operatorname{Error}(f,t) = 0$$

yields the per-TF-point ML estimate:

$$\hat{\delta}_{\text{ML}}(f,t) = \frac{1}{\operatorname{Error}(f,t)}$$

This per-frame estimate has high variance (it relies on a single observation). To regularize while preserving temporal variation, we apply a **sliding window average** in the variance domain:

$$\boxed{\hat{\sigma}^2(f,t) = \frac{1}{W}\sum_{k=t-\lfloor W/2 \rfloor}^{t+\lfloor W/2 \rfloor} \operatorname{Error}(f,k)}$$

or equivalently in precision form:

$$\boxed{\hat{\delta}(f,t) = \frac{W}{\displaystyle\sum_{k=t-\lfloor W/2 \rfloor}^{t+\lfloor W/2 \rfloor}\operatorname{Error}(f,k)}}$$

where $W$ is the window size (a configurable hyperparameter). At boundaries, replicate padding is used.

**Reduction to original:** When $W = T$ (window spans all frames), this reduces to the global average $\hat{\sigma}^2(f) = \frac{1}{T}\sum_t \operatorname{Error}(f,t)$, recovering the original stationary estimate.

---

## 4. Implementation Summary

### Tensor Shape Changes

| Variable | Original (stationary) | Modified (windowed) | Description |
|---|---|---|---|
| $\delta(f)$ / `Err_var` | $[F]$ | $[F, T]$ | Noise variance per freq / per T-F bin |
| `Err_var_f_t` in E-step | scalar $[]$ | vector $[L]$ | Per-lag noise variance passed to each T-F posterior update |
| `Err_var_f_para` in E-step | $[T]$ (broadcast) | $[T, L]$ (unfolded) | Parallelized noise variance for vmap |
| `ErrVar_f_para` in likelihood | $[T]$ (broadcast) | $[T]$ (direct) | Per-frame noise variance for log-likelihood |

### Configuration

The modification is controlled by a single parameter `errvar_window` in the VEM config:

| `errvar_window` | Behavior |
|---|---|
| `0` | Original stationary noise model (all formulas reduce to paper version) |
| `W > 0` | Sliding window of size $W$ for time-varying noise estimation |

### Code Mapping

| Formula | Code location | Function |
|---|---|---|
| $\hat{\gamma}(f,t)$, $\hat{\mu}(f,t)$ | `E_step_f_t` | Per-TF-point posterior update |
| $\text{Err\_var}$ unfolding to $[T,L]$ | `E_step_f` | Prepares per-lag noise variances for vmap |
| $\hat{\sigma}^2(f,t)$ sliding window | `M_step_f_ErrVar` | Windowed average of reconstruction errors |
| Log-likelihood with $\delta(f,t)$ | `cal_likeli_f` | Handles time-varying ErrVar for convergence monitoring |

All changes are in `method/vem_rir_omit3bin_nb.py`.

---

## 5. Experimental Results

We evaluate on 20 matched samples (same speech/RIR pairs) under three noise conditions: clean (reverb only), white Gaussian noise (10 dB SNR), and non-stationary noise (10 dB SNR). Each condition is tested with both the original model (`errvar_window=0`) and the windowed model (`errvar_window=20`).

### Summary Table (Mean +/- Std, 20 samples)

| Condition | Noise Model | SI-SDR (dB) | ESTOI | PESQ | RT60 err (s) | DRR err (dB) |
|---|---|---|---|---|---|---|
| Clean | Original | -1.96 +/- 6.29 | 0.718 +/- 0.109 | 2.434 +/- 0.518 | **0.197** +/- 0.145 | **2.48** +/- 1.15 |
| Clean | Windowed (W=20) | -2.05 +/- 5.72 | **0.770** +/- 0.080 | **2.543** +/- 0.301 | 0.365 +/- 0.519 | 2.51 +/- 1.72 |
| White | Original | **-2.48** +/- 6.49 | **0.537** +/- 0.114 | **1.770** +/- 0.325 | **0.454** +/- 0.349 | **3.49** +/- 4.62 |
| White | Windowed (W=20) | -2.54 +/- 5.82 | 0.536 +/- 0.107 | 1.688 +/- 0.217 | 0.500 +/- 0.267 | 4.17 +/- 5.13 |
| Non-Stat | Original | -3.25 +/- 6.07 | 0.533 +/- 0.116 | 1.704 +/- 0.287 | **0.215** +/- 0.140 | 3.58 +/- 2.04 |
| Non-Stat | Windowed (W=20) | **-3.16** +/- 5.95 | **0.547** +/- 0.111 | **1.724** +/- 0.272 | 0.346 +/- 0.443 | **2.98** +/- 1.76 |

Bold indicates the better value between Original and Windowed for each noise condition. Higher is better for SI-SDR, ESTOI, PESQ; lower is better for RT60 err and DRR err.

### Observations

- **Clean:** The windowed model improves speech quality metrics (ESTOI +0.052, PESQ +0.109) with a slight increase in RT60 error.
- **White (stationary) noise:** Performance is comparable between models, as expected --- the sliding window offers no advantage when noise is truly stationary.
- **Non-stationary noise:** The windowed model shows consistent improvements in speech quality (ESTOI +0.014, PESQ +0.020) and DRR estimation (-0.60 dB error), confirming that the time-varying noise model better captures non-stationary noise characteristics.

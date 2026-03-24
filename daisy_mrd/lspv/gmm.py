"""
daisy_mrd.lspv.gmm
==================
Gaussian Mixture Model (GMM) clonality classification.


Algorithm
---------
1. Fit GMMs with 1–5 components; select the number of components that
   minimises the Bayesian Information Criterion (BIC).
2. Identify the **clonal peak**: the Gaussian component with the
   highest mean VAF (this represents variants present in all or most
   leukemic cells).
3. For every variant, test (binomial) whether its VAF is significantly
   *below* the clonal peak mean. Variants that are NOT significantly
   below the peak are classified as **clonal**; the rest are **sub-clonal**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import matplotlib.figure


# ---------------------------------------------------------------------------
# GMM fitting
# ---------------------------------------------------------------------------

def fit_gmm(
    vaf_values: list[float] | np.ndarray,
    max_components: int = 5,
    random_state: int = 0,
) -> tuple[GaussianMixture, int]:
    """
    Fit GMMs with 1-5 components, select by BIC. Exact replica of original.
    """
    data = np.array(vaf_values).reshape(-1, 1)
    gmms: dict[int, GaussianMixture] = {}
    bics: dict[int, float] = {}

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=random_state)
        gmm.fit(data)
        gmms[n] = gmm
        bics[n] = gmm.bic(data)

    best_n = int(np.argmin(list(bics.values()))) + 1
    return gmms[best_n], best_n


def get_clonal_peak_mean(
    gmm: GaussianMixture,
    min_clonal_vaf = 0.11,
    max_clonal_vaf: float = 0.6,
    trimodal_diff_threshold: float = 0.156,
) -> float:
    """
    Return the clonal peak mean using the original script logic, with one
    data-driven rule replacing the hardcoded patient ID exceptions:

    n=1:
        return means[0]

    n=2, 4, 5 (else branch in original):
        return max(means[0], means[1])  -- raw unsorted order
        capped at max_clonal_vaf

    n=3:
        Sort the three means. Compute diff = sorted_means[2] - sorted_means[1]
        (difference between the two highest peaks).

        If diff > trimodal_diff_threshold (default 0.118):
            Truly trimodal → middle_value logic:
            return middle of means that are below max_clonal_vaf
            (handles both subclonal peaks and copy-number peaks)

        If diff <= trimodal_diff_threshold:
            The two highest peaks are too close → BIC over-split one broad
            clonal peak → treat as bimodal:
            return max(means[0], means[1])  -- raw unsorted order
            capped at max_clonal_vaf

    Parameters
    ----------
    gmm : GaussianMixture
    max_clonal_vaf : float
        Peaks at or above this are copy-number artefacts. Default: 0.6.
    trimodal_diff_threshold : float
        Minimum difference between 2nd and 3rd highest means to accept
        a trimodal fit. Default: 0.118.
    """
    means_raw = gmm.means_.flatten()   # unsorted, matches original indexing
    n = len(means_raw)
    means_sorted = np.sort(means_raw)

    if n == 1:
        return float(means_raw[0])

    if n == 3:
        diff = float(means_sorted[2] - means_sorted[1])

        if diff > trimodal_diff_threshold:
            # Genuinely trimodal: middle_value of eligible means
            # eligible = means_sorted[means_sorted < max_clonal_vaf]
            eligible = means_sorted[
            (means_sorted >= min_clonal_vaf) &
            (means_sorted < max_clonal_vaf)
            ]
            if len(eligible) == 0:
                return float(means_sorted[-1])
            if len(eligible) == 1:
                return float(eligible[0])
            return float(eligible[len(eligible) // 2])
        # else: fall through to bimodal logic below

    # n=2, 4, 5 OR n=3 with close peaks → bimodal
    # Original else branch: max(means[0], means[1]) raw order
    peak = float(max(means_raw[0], means_raw[1]))
    if peak >= max_clonal_vaf:
        peak = float(min(means_raw[0], means_raw[1]))
    return peak


# ---------------------------------------------------------------------------
# Clonality labelling
# ---------------------------------------------------------------------------

def calculate_binomial_pvalues(
    vcf_df: pd.DataFrame,
    clonal_peak_mean: float,
) -> pd.DataFrame:
    """One-sided binomial test: is variant VAF significantly < clonal peak?"""
    pvalues: list[float] = []
    for _, row in vcf_df.iterrows():
        k = row["mut_reads"]
        n = row["tot_reads"]
        if pd.isna(k) or pd.isna(n) or n == 0:
            pvalues.append(np.nan)
            continue
        try:
            result = binomtest(int(k), int(n), clonal_peak_mean, alternative="less")
            pvalues.append(result.pvalue)
        except (ValueError, ZeroDivisionError):
            pvalues.append(np.nan)
    df = vcf_df.copy()
    df["binomial_pvalue"] = pvalues
    return df


def label_clonality(
    vcf_df: pd.DataFrame,
    pvalue_threshold: float = 0.05,
) -> pd.DataFrame:
    """Add clonality column: binomial_pvalue < 0.05 → subclonal, else clonal."""
    df = vcf_df.copy()
    df["clonality"] = "clonal"
    df.loc[df["binomial_pvalue"] < pvalue_threshold, "clonality"] = "subclonal"
    return df


# ---------------------------------------------------------------------------
# GMM plot
# ---------------------------------------------------------------------------

def _gaussian_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(
        -((x - mean) ** 2) / (2 * variance)
    )


def plot_gmm(
    vaf_values: list[float] | np.ndarray,
    gmm: GaussianMixture,
    clonal_peak_mean: float,
    patient_id: str = "",
) -> matplotlib.figure.Figure:
    """Plot VAF histogram with GMM components and clonal peak line."""
    data = np.array(vaf_values)
    x = np.linspace(0, 1, 1000)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    order = np.argsort(means)
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=50, density=True, alpha=0.45, color="silver",
            label="Data histogram")

    for rank, i in enumerate(order):
        pdf = _gaussian_pdf(x, means[i], variances[i])
        label = (
            f"Gaussian {rank + 1}: \u03bc={means[i]:.3f}, w={weights[i]:.2f}"
            + (" \u2190 clonal peak" if np.isclose(means[i], clonal_peak_mean) else "")
        )
        ax.plot(x, pdf, color=palette[rank % len(palette)], linewidth=2, label=label)

    ax.axvline(clonal_peak_mean, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Clonal peak mean: {clonal_peak_mean:.3f}")

    ax.set_xlim(0, 1)
    ax.set_xlabel("VAF", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    title = "Gaussian Mixture Model"
    if patient_id:
        title += f" \u2014 {patient_id}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

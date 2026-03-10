"""
daisy_mrd.lspv.gmm
==================
Gaussian Mixture Model (GMM) clonality classification.

Each somatic variant is classified as **clonal** (pan-clonal) or
**sub-clonal** based on its VAF relative to the dominant clonal peak.

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
from scipy.optimize import brentq
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
    Fit a Gaussian Mixture Model to VAF values, selecting the optimal
    number of components by minimising BIC.

    Parameters
    ----------
    vaf_values : array-like of float
        VAF values (should be in [0, 1]).
    max_components : int
        Maximum number of GMM components to try.
    random_state : int

    Returns
    -------
    (GaussianMixture, int)
        The fitted GMM and the chosen number of components.
    """
    data = np.array(vaf_values).reshape(-1, 1)
    best_gmm: GaussianMixture | None = None
    best_bic = np.inf
    best_n = 1

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=random_state)
        gmm.fit(data)
        bic = gmm.bic(data)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n

    assert best_gmm is not None
    return best_gmm, best_n


def get_clonal_peak_mean(gmm: GaussianMixture) -> float:
    """
    Return the mean of the highest-VAF Gaussian component.

    This component represents the pan-clonal (clonal) variant population —
    variants present in all or nearly all leukemic cells.

    Parameters
    ----------
    gmm : GaussianMixture
        A fitted :class:`sklearn.mixture.GaussianMixture`.

    Returns
    -------
    float
        Mean VAF of the clonal peak.
    """
    means = gmm.means_.flatten()
    return float(np.max(means))


# ---------------------------------------------------------------------------
# Clonality labelling
# ---------------------------------------------------------------------------

def calculate_binomial_pvalues(
    vcf_df: pd.DataFrame,
    clonal_peak_mean: float,
) -> pd.DataFrame:
    """
    For each variant, test whether its VAF is significantly *less than*
    the clonal peak mean (one-sided binomial test, ``alternative="less"``).

    A variant is sub-clonal if ``p-value < 0.05`` — i.e. its VAF is
    significantly below the dominant clonal peak.

    Parameters
    ----------
    vcf_df : pd.DataFrame
        Must have ``mut_reads`` and ``tot_reads`` columns.
    clonal_peak_mean : float
        Mean VAF of the clonal component (probability parameter for the
        binomial test).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a ``binomial_pvalue`` column added.
    """
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
    """
    Add a ``clonality`` column: ``"clonal"`` or ``"subclonal"``.

    Variants with ``binomial_pvalue < pvalue_threshold`` are sub-clonal
    (their VAF is significantly below the clonal peak); all others are
    clonal.

    Parameters
    ----------
    vcf_df : pd.DataFrame
        Must have ``binomial_pvalue`` column (from
        :func:`calculate_binomial_pvalues`).
    pvalue_threshold : float

    Returns
    -------
    pd.DataFrame
    """
    df = vcf_df.copy()
    df["clonality"] = "clonal"
    df.loc[df["binomial_pvalue"] < pvalue_threshold, "clonality"] = "subclonal"
    return df


# ---------------------------------------------------------------------------
# GMM plot
# ---------------------------------------------------------------------------

def _gaussian_pdf(
    x: np.ndarray,
    mean: float,
    variance: float,
) -> np.ndarray:
    return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(
        -((x - mean) ** 2) / (2 * variance)
    )


def plot_gmm(
    vaf_values: list[float] | np.ndarray,
    gmm: GaussianMixture,
    clonal_peak_mean: float,
    patient_id: str = "",
) -> matplotlib.figure.Figure:
    """
    Plot the VAF histogram with overlaid GMM component curves and a
    vertical line marking the clonal peak mean.

    Parameters
    ----------
    vaf_values : array-like of float
    gmm : GaussianMixture
        Fitted GMM.
    clonal_peak_mean : float
        Mean of the clonal component (shown as a dashed vertical line).
    patient_id : str
        Used in the plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = np.array(vaf_values)
    x = np.linspace(0, 1, 1000)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    # Sort components by mean for consistent colouring
    order = np.argsort(means)
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=50, density=True, alpha=0.45, color="silver",
            label="Data histogram")

    for rank, i in enumerate(order):
        pdf = _gaussian_pdf(x, means[i], variances[i])
        label = (
            f"Gaussian {rank + 1}: μ={means[i]:.3f}, w={weights[i]:.2f}"
            + (" ← clonal peak" if np.isclose(means[i], clonal_peak_mean) else "")
        )
        color = palette[rank % len(palette)]
        ax.plot(x, pdf, color=color, linewidth=2, label=label)

    ax.axvline(
        clonal_peak_mean,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Clonal peak mean: {clonal_peak_mean:.3f}",
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("VAF", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    title = f"Gaussian Mixture Model"
    if patient_id:
        title += f" — {patient_id}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8)
    fig.tight_layout()

    return fig

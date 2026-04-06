"""
daisy_mrd.lspv.pipeline
=======================
End-to-end LSPV identification pipeline.

This module provides :func:`run_lspv_pipeline`, the single entry-point
for Step 1 of DAISY-MRD.

Input
-----
* A diagnosis VCF file (unfiltered or PASS-filtered output from a
  variant caller such as DeepVariant, GATK HaplotypeCaller, etc.).
  The VCF should contain somatic variants called against a matched
  germline control (T-cell DNA).

Output  (:class:`LspvResult`)
------------------------------
* ``lspvs``         — DataFrame of leukemia-specific passenger variants
* ``all_variants``  — DataFrame of all variants after hard filtering
                      (before LSPV extraction), with ``clonality`` labels
* ``summary``       — One-row summary DataFrame (counts)
* ``fig_gmm``       — GMM plot (matplotlib Figure)
* ``fig_pie``       — Clonality pie chart (matplotlib Figure)
* ``clonal_peak_mean`` — Mean VAF of the clonal peak

All figures and the LSPV CSV are saved to ``output_dir``.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.figure
import pandas as pd

from daisy_mrd.lspv.filter import apply_hard_filters
from daisy_mrd.lspv.gmm import (
    calculate_binomial_pvalues,
    fit_gmm,
    get_clonal_peak_mean,
    label_clonality,
    plot_gmm,
)
from daisy_mrd.lspv.identify import extract_lspvs, lspv_summary, plot_clonality_pie
from daisy_mrd.lspv.pon import filter_pon, load_pon
from daisy_mrd.lspv.reads import extract_info, get_reads, get_vaf
from daisy_mrd.utils import ensure_output_dir, read_vcf, resolve_pon_path
from daisy_mrd.lspv.annotations import verify_vcf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LspvResult:
    """
    Container for all outputs of :func:`run_lspv_pipeline`.

    Attributes
    ----------
    lspvs : pd.DataFrame
        Final LSPV table (clonal, non-coding variants).
    all_variants : pd.DataFrame
        All variants after hard filtering with clonality labels.
    summary : pd.DataFrame
        One-row summary of variant and LSPV counts.
    fig_gmm : matplotlib.figure.Figure
        Gaussian Mixture Model plot.
    fig_pie : matplotlib.figure.Figure
        Clonal vs sub-clonal pie chart.
    clonal_peak_mean : float
        Mean VAF of the identified clonal peak.
    output_dir : Path
        Directory where output files were written.
    """

    lspvs: pd.DataFrame
    all_variants: pd.DataFrame
    summary: pd.DataFrame
    fig_gmm: matplotlib.figure.Figure
    fig_pie: matplotlib.figure.Figure
    clonal_peak_mean: float
    output_dir: Path

@dataclass
class InputFiles:
    vcf_file: Path
    germline_file: Path
    pon_file: Path

def ensure_input_files(vcf_path: str, germline_vcf_file: str, pon_path: str) -> InputFiles:
    """
    Ensure that all input files exist and that output folder structure exists
    Returns InputFiles
    -------
    """
    vcf_path = Path(vcf_path)
    vcf_is_good = verify_vcf(vcf_path)
    for vcf_feature, is_present in vcf_is_good.items():
        if not is_present:
            raise ValueError(
                f"VCF lacks {vcf_feature}. See README.md for instructions for instructions on how to construct an appropriate VCF file.")
    if not os.path.exists(germline_vcf_file) or not germline_vcf_file.endswith(".vcf.gz") or not os.path.exists(germline_vcf_file + ".tbi"):
        raise RuntimeError("Germline VCF file not found, or it's index file was not found, or it is not properly formatted (.vcf.gz")
    pon_path = resolve_pon_path(pon_path)
    return InputFiles(vcf_path, Path(germline_vcf_file), pon_path)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_lspv_pipeline(
    vcf_path: Union[str, Path],
    output_dir: Union[str, Path],
    germline_vcf_file: str,
    patient_id: str = "",
    # --- PoN options ---
    pon_path: Union[str, Path] = None,
    pon_pvalue_threshold: float = 0.05,
    # --- GMM options ---
    gmm_max_components: int = 5,
    clonality_pvalue_threshold: float = 0.05,
) -> LspvResult:
    """
    Run the full LSPV identification pipeline.

    Parameters
    ----------
    vcf_path : str or Path
        Path to the diagnosis VCF (plain ``.vcf`` or ``.vcf.gz``).
        Can be unfiltered or pre-filtered to PASS variants.
    output_dir : str or Path
        Directory where output files (LSPV CSV, figures) are saved.
        Created automatically if it does not exist.
    patient_id : str
        Optional sample / patient label used in plot titles and filenames.
    pon_path : str, Path, or None
        Path to a custom Panel of Normals CSV. Pass ``None`` (default) to
        use the built-in PoN bundled with the package.
    pon_pvalue_threshold : float
        Binomial p-value threshold for PoN filtering. Default: 0.05.
    gmm_max_components : int
        Maximum number of Gaussian components to evaluate. Default: 5.
    clonality_pvalue_threshold : float
        Binomial p-value threshold for sub-clonal classification.
        Variants with p < threshold are labelled sub-clonal. Default: 0.05.

    Returns
    -------
    LspvResult
        Named container with all outputs. See :class:`LspvResult`.

    Examples
    --------
    Minimal usage with default built-in PoN and no VEP:

    >>> from daisy_mrd import run_lspv_pipeline
    >>> result = run_lspv_pipeline(
    ...     vcf_path="patient_001_diagnosis.vcf",
    ...     output_dir="results/patient_001/",
    ...     patient_id="001",
    ... )
    >>> print(result.summary)
    >>> result.lspvs.to_csv("lspvs_001.csv", index=False)

    With a custom PoN

    >>> result = run_lspv_pipeline(
    ...     vcf_path="patient_001_diagnosis.vcf",
    ...     output_dir="results/patient_001/",
    ...     patient_id="001",
    ...     pon_path="/data/my_pon.csv",
    ... )
    """

    # ------------------------------------------------------------------
    # Step 3: Verify input and output files exist and are appropriate for Daisy
    # ------------------------------------------------------------------

    input_files = ensure_input_files(vcf_path, germline_vcf_file, pon_path)
    vcf_path = input_files.vcf_file
    germline_vcf_file = input_files.germline_file
    pon_path = input_files.pon_file
    out_dir = ensure_output_dir(output_dir)
    label = patient_id or vcf_path.stem

    log.info("[%s] Starting LSPV pipeline", label)



    # ------------------------------------------------------------------
    # Step 4: Load into DataFrame and apply hard filters
    # ------------------------------------------------------------------
    log.info("[%s] Loading annotated VCF into DataFrame", label)
    df = read_vcf(vcf_path)

    # Always extract hgvsp_short and so_term from the INFO field.
    log.info("[%s] Extracting VEP INFO fields (hgvsp_short, so_term)", label)
    df = extract_info(df)

    log.info("[%s] Applying hard filters (germline, rs, indel)", label)
    df = apply_hard_filters(df, germline_vcf_file)

    # ------------------------------------------------------------------
    # Step 5: Extract reads (depth + alt counts)
    # ------------------------------------------------------------------
    log.info("[%s] Extracting read depth and alt counts", label)
    df = get_reads(df)

    # ------------------------------------------------------------------
    # Step 6: Panel of Normals filter
    # (must happen BEFORE get_vaf, matching original script order:
    #  get_reads -> filter_pon -> get_vaf -> GMM)
    # ------------------------------------------------------------------
    resolved_pon = resolve_pon_path(pon_path)
    log.info("[%s] Loading PoN from %s", label, resolved_pon)
    pon = load_pon(resolved_pon)

    log.info("[%s] Applying PoN filter (p <= %.3f)", label, pon_pvalue_threshold)
    df = filter_pon(df, pon, pvalue_threshold=pon_pvalue_threshold)

    # Extract VAF AFTER PoN filtering - matches original process_file() order
    df, vaf_values = get_vaf(df)

    # ------------------------------------------------------------------
    # Step 7: GMM clonality classification
    # ------------------------------------------------------------------
    log.info("[%s] Fitting GMM (max_components=%d)", label, gmm_max_components)
    gmm, n_components = fit_gmm(vaf_values, max_components=gmm_max_components)
    clonal_peak_mean = get_clonal_peak_mean(gmm)
    log.info(
        "[%s] GMM selected %d component(s); clonal peak mean = %.4f",
        label, n_components, clonal_peak_mean,
    )

    df = calculate_binomial_pvalues(df, clonal_peak_mean)
    df = label_clonality(df, pvalue_threshold=clonality_pvalue_threshold)

    counts = df["clonality"].value_counts().to_dict()
    log.info("[%s] Clonality counts: %s", label, counts)

    # ------------------------------------------------------------------
    # Step 8: Extract LSPVs
    # ------------------------------------------------------------------
    lspv_df = extract_lspvs(df)
    log.info("[%s] %d LSPVs identified", label, len(lspv_df))

    # ------------------------------------------------------------------
    # Step 9: Summary and figures
    # ------------------------------------------------------------------
    summary = lspv_summary(df, lspv_df, patient_id=label)
    fig_gmm = plot_gmm(vaf_values, gmm, clonal_peak_mean, patient_id=label)
    fig_pie = plot_clonality_pie(df, patient_id=label)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    lspv_csv = out_dir / f"{label}_lspvs.csv"
    lspv_df.to_csv(lspv_csv, index=False)
    log.info("[%s] LSPVs saved → %s", label, lspv_csv)

    gmm_pdf = out_dir / f"{label}_gmm.pdf"
    fig_gmm.savefig(gmm_pdf, dpi=150, bbox_inches="tight")
    log.info("[%s] GMM plot saved → %s", label, gmm_pdf)

    pie_pdf = out_dir / f"{label}_clonality_pie.pdf"
    fig_pie.savefig(pie_pdf, dpi=150, bbox_inches="tight")
    log.info("[%s] Pie chart saved → %s", label, pie_pdf)

    summary_csv = out_dir / f"{label}_summary.csv"
    summary.to_csv(summary_csv, index=False)

    log.info("[%s] Pipeline complete. Outputs in %s", label, out_dir)

    return LspvResult(
        lspvs=lspv_df,
        all_variants=df,
        summary=summary,
        fig_gmm=fig_gmm,
        fig_pie=fig_pie,
        clonal_peak_mean=clonal_peak_mean,
        output_dir=out_dir,
    )

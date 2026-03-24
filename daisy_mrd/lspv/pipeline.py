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

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.figure
import pandas as pd

from daisy_mrd.lspv.annotate import annotate_vcf, run_vep
from daisy_mrd.lspv.filter import apply_hard_filters, filter_vcf_pass
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_lspv_pipeline(
    vcf_path: str | Path,
    output_dir: str | Path,
    patient_id: str = "",
    # --- Annotation options ---
    run_vep_annotation: bool = False,
    vep_cache_dir: str | Path = "~/.vep",
    vep_assembly: str = "GRCh38",
    vep_use_cache: bool = False,
    gnomad_path: str | Path | None = None,
    gnomad_use_api: bool = True,
    # --- PoN options ---
    pon_path: str | Path | None = None,
    pon_pvalue_threshold: float = 0.05,
    # --- GMM options ---
    gmm_max_components: int = 5,
    clonality_pvalue_threshold: float = 0.05,
    # --- VCF already PASS-filtered? ---
    vcf_is_filtered: bool = False,
    # --- VCF already gnomAD-annotated? ---
    vcf_is_gnomad_annotated: bool = False,
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
    run_vep_annotation : bool
        If ``True``, annotate the VCF with VEP via Docker before
        processing. Requires Docker and the ``ensemblorg/ensembl-vep``
        image. Set ``False`` (default) if the VCF is already annotated
        or if you prefer to annotate externally.
    vep_cache_dir : str or Path
        Local VEP cache directory (used only when ``run_vep_annotation=True``
        and ``vep_use_cache=True``).
    vep_assembly : str
        Genome assembly for VEP (``"GRCh38"`` or ``"GRCh37"``).
    vep_use_cache : bool
        Use local VEP cache (offline mode). Requires a pre-downloaded cache.
    gnomad_path : str, Path, or None
        Directory containing pre-built gnomAD position index files
        (``chrom_pos.txt``). Pass ``None`` to skip local lookup.
    gnomad_use_api : bool
        Fall back to the gnomAD API for positions not in the local index.
        Disable to avoid network calls (not recommended unless a complete
        local index is provided).
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
    vcf_is_filtered : bool
        Set ``True`` if the input VCF is already filtered to PASS variants,
        to skip the PASS-filtering step.
    vcf_is_gnomad_annotated : bool
        Set ``True`` if the input VCF already has gnomAD annotation
        (``GNOMAD`` and ``gnomad_AF`` INFO fields present), to skip the
        gnomAD annotation step entirely.

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

    With a custom PoN and VEP annotation:

    >>> result = run_lspv_pipeline(
    ...     vcf_path="patient_001_diagnosis.vcf",
    ...     output_dir="results/patient_001/",
    ...     patient_id="001",
    ...     run_vep_annotation=True,
    ...     vep_assembly="GRCh38",
    ...     pon_path="/data/my_pon.csv",
    ... )
    """
    vcf_path = Path(vcf_path)
    out_dir = ensure_output_dir(output_dir)
    label = patient_id or vcf_path.stem

    log.info("[%s] Starting LSPV pipeline", label)

    # ------------------------------------------------------------------
    # Step 1: PASS filter
    # ------------------------------------------------------------------
    if vcf_is_filtered:
        pass_vcf = vcf_path
        log.info("[%s] Skipping PASS filter (vcf_is_filtered=True)", label)
    else:
        pass_vcf = out_dir / f"{label}_pass.vcf"
        log.info("[%s] Filtering to PASS variants → %s", label, pass_vcf)
        filter_vcf_pass(vcf_path, pass_vcf)

    # ------------------------------------------------------------------
    # Step 2: VEP annotation (optional)
    # ------------------------------------------------------------------
    if run_vep_annotation:
        vep_vcf = out_dir / f"{label}_pass_vep.vcf"
        log.info("[%s] Running VEP annotation → %s", label, vep_vcf)
        run_vep(
            pass_vcf,
            vep_vcf,
            cache_dir=vep_cache_dir,
            assembly=vep_assembly,
            use_cache=vep_use_cache,
        )
        annotated_vcf = vep_vcf
    else:
        log.info("[%s] Skipping VEP (run_vep_annotation=False)", label)
        annotated_vcf = pass_vcf

    # ------------------------------------------------------------------
    # Step 3: gnomAD annotation
    # ------------------------------------------------------------------
    if vcf_is_gnomad_annotated:
        log.info("[%s] Skipping gnomAD annotation (vcf_is_gnomad_annotated=True)", label)
        gnomad_vcf = annotated_vcf
    else:
        gnomad_vcf = out_dir / f"{label}_gnomad.vcf"
        log.info("[%s] Running gnomAD annotation → %s", label, gnomad_vcf)
        annotate_vcf(
            annotated_vcf,
            gnomad_vcf,
            gnomad_path=gnomad_path,
            use_api=gnomad_use_api,
        )

    # ------------------------------------------------------------------
    # Step 4: Load into DataFrame and apply hard filters
    # ------------------------------------------------------------------
    log.info("[%s] Loading annotated VCF into DataFrame", label)
    df = read_vcf(gnomad_vcf)

    # Always extract hgvsp_short and so_term from the INFO field.
    # Even when run_vep_annotation=False, the VCF is already VEP-annotated
    # so extract_info() can parse coding variants correctly.
    # Without this, hgvsp_short is set to None for all variants and
    # coding variants are never filtered out of LSPVs.
    log.info("[%s] Extracting VEP INFO fields (hgvsp_short, so_term)", label)
    df = extract_info(df)

    log.info("[%s] Applying hard filters (germline, rs, indel)", label)
    df = apply_hard_filters(df)

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

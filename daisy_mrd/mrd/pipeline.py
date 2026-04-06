"""
daisy_mrd.mrd.pipeline
======================
End-to-end DAISY-MRD scoring pipeline (Step 2).

Takes the LSPV list produced by Step 1 and one or more remission
BAM/CRAM files, runs ``samtools mpileup`` over LSPV positions, applies
multi-layer noise filtering, and computes the DAISY-MRD score.

Two entry-points are provided:

* :func:`run_mrd_single`   — one patient, one remission BAM/CRAM
* :func:`run_mrd_cohort`   — multiple patients, each with their own
                             LSPV list and remission BAM/CRAM

Single-patient example
----------------------
>>> from daisy_mrd.mrd.pipeline import run_mrd_single
>>> result = run_mrd_single(
...     lspv_csv="results/patient_001/001_lspvs.csv",
...     remission_bam="data/patient_001_remission.cram",
...     reference="GRCh38.fa",
...     output_dir="results/patient_001/mrd/",
...     patient_id="001",
... )
>>> print(result.score)

Multi-patient example
---------------------
>>> from daisy_mrd.mrd.pipeline import run_mrd_cohort
>>> cohort = [
...     dict(patient_id="001",
...          lspv_csv="results/001/001_lspvs.csv",
...          remission_bam="data/001_remission.cram"),
...     dict(patient_id="002",
...          lspv_csv="results/002/002_lspvs.csv",
...          remission_bam="data/002_remission.cram"),
... ]
>>> scores_df = run_mrd_cohort(
...     patients=cohort,
...     reference="GRCh38.fa",
...     output_dir="results/cohort_mrd/",
... )
>>> scores_df.to_csv("daisy_mrd_scores.csv", index=False)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.figure
import pandas as pd

from daisy_mrd.lspv.pon import load_pon
from daisy_mrd.mrd.filters import apply_all_filters
from daisy_mrd.mrd.pileup import merge_lspv_alts, pileup_to_df, run_mpileup
from daisy_mrd.mrd.readcount import apply_read_counts
from daisy_mrd.mrd.score import (
    MrdScore,
    compute_mrd_score,
    plot_noise_distributions,
)
from daisy_mrd.utils import ensure_output_dir, resolve_pon_path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MrdResult:
    """
    All outputs from :func:`run_mrd_single`.

    Attributes
    ----------
    patient_id : str
    score : MrdScore
        The final DAISY-MRD score.
    filter_layers : dict[str, pd.DataFrame]
        Mapping of filter-layer name → filtered DataFrame.
        Includes all intermediate layers so users can inspect each step.
    filter_summary : pd.DataFrame
        One-row DataFrame with LSPV count at each filter layer.
    fig_noise : matplotlib.figure.Figure or None
        Noise distribution plot. ``None`` if fewer than 2 patients
        were available for cross-comparison.
    output_dir : Path
    """

    patient_id: str
    score: MrdScore
    filter_layers: dict[str, pd.DataFrame]
    filter_summary: pd.DataFrame
    fig_noise: matplotlib.figure.Figure | None
    output_dir: Path


# ---------------------------------------------------------------------------
# Single-patient pipeline
# ---------------------------------------------------------------------------

def run_mrd_single(
    lspv_csv: str | Path,
    remission_bam: str | Path,
    reference: str | Path,
    output_dir: str | Path,
    patient_id: str = "",
    # samtools options
    samtools_path: str = "samtools",
    extra_mpileup_flags: list[str] | None = None,
    # filter options
    pon_path: str | Path | None = None,
    twobit_path: str | Path | None = None,
    max_depth: int = 200,
    germline_pvalue_threshold: float = 0.01,
    pon_pvalue_threshold: float = 0.05,
    max_vaf: float = 0.05,
    use_noise_pvalue_filter: bool = True,
    # skip mpileup if pileup file already exists
    pileup_file: str | Path | None = None,
) -> MrdResult:
    """
    Run the DAISY-MRD scoring pipeline for a single patient.

    Parameters
    ----------
    lspv_csv : str or Path
        Path to the LSPV CSV produced by Step 1
        (:func:`~daisy_mrd.lspv.pipeline.run_lspv_pipeline`).
    remission_bam : str or Path
        Path to the remission BAM or CRAM file.
    reference : str or Path
        Path to the reference FASTA (required for CRAM decoding).
    output_dir : str or Path
        Directory for all output files. Created if it does not exist.
    patient_id : str
        Label used in filenames and plot titles.
    samtools_path : str
        Path to the ``samtools`` binary. Default: ``"samtools"``
        (assumes it is on ``$PATH``).
    extra_mpileup_flags : list[str] or None
        Extra flags forwarded to ``samtools mpileup``.
    pon_path : str, Path, or None
        Custom PoN CSV. ``None`` uses the built-in PoN.
    twobit_path : str, Path, or None
        Path to a ``.2bit`` genome file for context filtering.
        If ``None``, the C>TG/CG>A context filter is skipped.
    max_depth : int
        Read-depth ceiling filter. Default: 200.
    germline_pvalue_threshold : float
        Binomial p-value threshold for germline filtering. Default: 0.01.
    pon_pvalue_threshold : float
        Binomial p-value threshold for PoN filtering. Default: 0.05.
    max_vaf : float
        Maximum VAF allowed in the final filter layer. Default: 0.05.
    use_noise_pvalue_filter : bool
        If ``True``, restrict the MRD numerator to positions with
        ``noise_pvalue < 0.05`` (requires PoN filter to have run).
    pileup_file : str, Path, or None
        If a pre-computed ``.pileup`` file is provided, the
        ``samtools mpileup`` step is skipped. Useful for re-running
        downstream steps without re-running samtools.

    Returns
    -------
    MrdResult
    """
    lspv_csv = Path(lspv_csv)
    out_dir = ensure_output_dir(output_dir)
    label = patient_id or lspv_csv.stem

    log.info("[%s] Starting DAISY-MRD scoring pipeline", label)

    # ------------------------------------------------------------------
    # Load LSPVs
    # ------------------------------------------------------------------
    lspv_df = pd.read_csv(lspv_csv)
    log.info("[%s] Loaded %d LSPVs from %s", label, len(lspv_df), lspv_csv)

    # ------------------------------------------------------------------
    # Run samtools mpileup (or load pre-computed pileup)
    # ------------------------------------------------------------------
    if pileup_file is not None:
        pileup_path = Path(pileup_file)
        log.info("[%s] Using pre-computed pileup: %s", label, pileup_path)
    else:
        pileup_path = out_dir / f"{label}_remission.pileup"
        log.info("[%s] Running samtools mpileup → %s", label, pileup_path)
        run_mpileup(
            cram_or_bam=remission_bam,
            reference=reference,
            lspv_df=lspv_df,
            output_pileup=pileup_path,
            samtools_path=samtools_path,
            extra_flags=extra_mpileup_flags,
        )

    # ------------------------------------------------------------------
    # Parse pileup
    # ------------------------------------------------------------------
    pileup_csv = out_dir / f"{label}_pileup.csv"
    log.info("[%s] Parsing pileup → %s", label, pileup_csv)
    pileup_df = pileup_to_df(pileup_path, output_csv=pileup_csv)

    # ------------------------------------------------------------------
    # Attach original ALT alleles
    # ------------------------------------------------------------------
    pileup_df = merge_lspv_alts(pileup_df, lspv_df)
    log.info("[%s] Merged LSPV ALT alleles (%d positions)", label, len(pileup_df))

    # ------------------------------------------------------------------
    # Count ALT-supporting reads
    # ------------------------------------------------------------------
    log.info("[%s] Counting ALT-supporting reads", label)
    pileup_df = apply_read_counts(pileup_df)

    read_count_csv = out_dir / f"{label}_read_counts.csv"
    pileup_df.to_csv(read_count_csv, index=False)

    # ------------------------------------------------------------------
    # Load PoN
    # ------------------------------------------------------------------
    pon: pd.DataFrame | None = None
    resolved_pon = resolve_pon_path(pon_path)
    pon = load_pon(resolved_pon)

    # ------------------------------------------------------------------
    # Apply filter layers
    # ------------------------------------------------------------------
    filter_dir = out_dir / "filter_layers"
    log.info("[%s] Applying filter layers → %s", label, filter_dir)
    layers, filter_summary = apply_all_filters(
        df=pileup_df,
        output_dir=filter_dir,
        file_name=label,
        pon=pon,
        twobit_path=twobit_path,
        max_depth=max_depth,
        germline_pvalue_threshold=germline_pvalue_threshold,
        pon_pvalue_threshold=pon_pvalue_threshold,
        max_vaf=max_vaf,
    )

    filter_summary_csv = out_dir / f"{label}_filter_summary.csv"
    filter_summary.to_csv(filter_summary_csv, index=False)
    log.info("[%s] Filter summary:\n%s", label, filter_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # Compute DAISY-MRD score from the final filter layer
    # ------------------------------------------------------------------
    final_layer = layers.get(
        "no_xy_no_germline_no_pon_no_hVAF",
        layers.get("no_xy_no_germline_no_pon", pileup_df),
    )
    mrd_score = compute_mrd_score(
        final_layer,
        patient_id=label,
        use_noise_pvalue_filter=use_noise_pvalue_filter,
    )
    log.info(
        "[%s] DAISY-MRD score = %.6f  (%d alt reads / %d total reads, %d positions)",
        label,
        mrd_score.score,
        mrd_score.total_alt_reads,
        mrd_score.total_read_depth,
        mrd_score.n_lspv_positions,
    )

    # Save score
    score_csv = out_dir / f"{label}_daisy_mrd_score.csv"
    pd.DataFrame(
        {
            "patient_id": [label],
            "daisy_mrd_score": [mrd_score.score],
            "total_alt_reads": [mrd_score.total_alt_reads],
            "total_read_depth": [mrd_score.total_read_depth],
            "n_lspv_positions": [mrd_score.n_lspv_positions],
        }
    ).to_csv(score_csv, index=False)

    log.info("[%s] Pipeline complete. Outputs in %s", label, out_dir)

    return MrdResult(
        patient_id=label,
        score=mrd_score,
        filter_layers=layers,
        filter_summary=filter_summary,
        fig_noise=None,  # populated by run_mrd_cohort if cross-comparisons exist
        output_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# Multi-patient pipeline
# ---------------------------------------------------------------------------

def run_mrd_cohort(
    patients: list[dict],
    reference: str | Path,
    output_dir: str | Path,
    samtools_path: str = "samtools",
    pon_path: str | Path | None = None,
    twobit_path: str | Path | None = None,
    max_depth: int = 200,
    germline_pvalue_threshold: float = 0.01,
    pon_pvalue_threshold: float = 0.05,
    max_vaf: float = 0.05,
    use_noise_pvalue_filter: bool = True,
    plot_noise: bool = True,
) -> pd.DataFrame:
    """
    Run the DAISY-MRD scoring pipeline for a cohort of patients.

    Each patient is processed independently by :func:`run_mrd_single`.
    After all patients are scored, a summary CSV and (optionally) noise
    distribution plots are produced.

    Parameters
    ----------
    patients : list[dict]
        Each element is a dict with keys:

        * ``patient_id`` (str) — patient label
        * ``lspv_csv`` (str or Path) — LSPV CSV from Step 1
        * ``remission_bam`` (str or Path) — remission BAM/CRAM
        * ``pileup_file`` (str or Path, optional) — pre-computed pileup

        Any key accepted by :func:`run_mrd_single` can also be included
        to override cohort-level defaults for individual patients.

    reference : str or Path
        Reference FASTA (shared across all patients).
    output_dir : str or Path
        Root output directory. Per-patient outputs go into
        ``output_dir/{patient_id}/``.
    samtools_path : str
    pon_path : str, Path, or None
    twobit_path : str, Path, or None
    max_depth : int
    germline_pvalue_threshold : float
    pon_pvalue_threshold : float
    max_vaf : float
    use_noise_pvalue_filter : bool
    plot_noise : bool
        If ``True``, generate noise distribution figures for each patient
        (requires scores from all patients).

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with one row per patient: ``patient_id`` and
        ``daisy_mrd_score``.

    Examples
    --------
    >>> cohort = [
    ...     {"patient_id": "001", "lspv_csv": "001_lspvs.csv",
    ...      "remission_bam": "001_remission.cram"},
    ...     {"patient_id": "002", "lspv_csv": "002_lspvs.csv",
    ...      "remission_bam": "002_remission.cram"},
    ... ]
    >>> scores = run_mrd_cohort(cohort, reference="hg38.fa",
    ...                         output_dir="results/cohort/")
    """
    out_dir = ensure_output_dir(output_dir)
    all_results: list[MrdResult] = []

    # Shared defaults that can be overridden per-patient
    cohort_defaults = dict(
        reference=reference,
        samtools_path=samtools_path,
        pon_path=pon_path,
        twobit_path=twobit_path,
        max_depth=max_depth,
        germline_pvalue_threshold=germline_pvalue_threshold,
        pon_pvalue_threshold=pon_pvalue_threshold,
        max_vaf=max_vaf,
        use_noise_pvalue_filter=use_noise_pvalue_filter,
    )

    for patient_dict in patients:
        patient_id = patient_dict.get("patient_id", "unknown")
        patient_out = out_dir / patient_id
        log.info("=== Cohort: processing patient %s ===", patient_id)

        # Build kwargs: cohort defaults, then per-patient overrides
        kwargs = {**cohort_defaults}
        for k, v in patient_dict.items():
            if k not in ("patient_id",):
                kwargs[k] = v

        try:
            result = run_mrd_single(
                output_dir=patient_out,
                patient_id=patient_id,
                **kwargs,
            )
            all_results.append(result)
        except Exception as e:
            log.error("Patient %s failed: %s", patient_id, e)
            continue

    # ------------------------------------------------------------------
    # Build cohort summary
    # ------------------------------------------------------------------
    summary_rows = [
        {
            "patient_id": r.patient_id,
            "daisy_mrd_score": r.score.score,
            "total_alt_reads": r.score.total_alt_reads,
            "total_read_depth": r.score.total_read_depth,
            "n_lspv_positions": r.score.n_lspv_positions,
        }
        for r in all_results
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "cohort_daisy_mrd_scores.csv"
    summary_df.to_csv(summary_csv, index=False)
    log.info("Cohort summary saved → %s", summary_csv)

    # ------------------------------------------------------------------
    # Noise distribution plots (requires cross-comparison matrix)
    # ------------------------------------------------------------------
    if plot_noise and len(all_results) > 1:
        scores_dict = {r.patient_id: r.score.score for r in all_results}
        scores_df = pd.DataFrame(
            [
                {"Patient": pid, **{f"Compared_to_{pid}": score}}
                for pid, score in scores_dict.items()
            ]
        )
        fig_dir = out_dir / "mrd_figures"
        for result in all_results:
            try:
                fig = plot_noise_distributions(
                    scores_df,
                    patient_id=result.patient_id,
                    output_dir=fig_dir,
                )
                result.fig_noise = fig
            except Exception as e:
                log.warning("Could not plot noise for patient %s: %s",
                            result.patient_id, e)

    return summary_df

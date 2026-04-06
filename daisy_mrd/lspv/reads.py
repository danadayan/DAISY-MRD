"""
daisy_mrd.lspv.reads
====================
Extract read-level information from the VCF FORMAT / sample columns.

The VCF FORMAT column defines a colon-separated list of field keys
(e.g. ``GT:DP:AD:VAF``). The adjacent sample column holds the
corresponding colon-separated values for that sample.

Functions
---------
detect_sample_column : Auto-detect the sample column name
extract_info         : Parse VEP INFO field → SO term, gene, HGVSp
get_reads            : Add ``tot_reads`` (DP) and ``mut_reads`` (ALT AD)
get_vaf              : Add ``VAF`` column
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sample column detection
# ---------------------------------------------------------------------------

def detect_sample_column(vcf_df: pd.DataFrame) -> str:
    """
    Return the name of the sample genotype column.

    Standard VCF columns are::

        CHROM POS ID REF ALT QUAL FILTER INFO FORMAT

    Everything after ``FORMAT`` (column index 9 onward) is a sample column.
    This function returns the **first** sample column name.

    Parameters
    ----------
    vcf_df : pd.DataFrame
        DataFrame produced by :func:`~daisy_mrd.utils.read_vcf`.

    Returns
    -------
    str
        Name of the first sample column.

    Raises
    ------
    ValueError
        If the DataFrame has fewer than 10 columns (no sample column).
    """
    if vcf_df.shape[1] < 10:
        raise ValueError(
            "VCF DataFrame has fewer than 10 columns — no sample column found. "
            "Make sure the VCF contains at least one sample."
        )
    return vcf_df.columns[9]


# ---------------------------------------------------------------------------
# VEP INFO field parser
# ---------------------------------------------------------------------------

def extract_info(vcf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse VEP annotation from the INFO field.

    Adds two columns:

    * ``so_term``      — Sequence Ontology consequence term (e.g.
      ``"missense_variant"``, ``"intron_variant"``).
    * ``hgvsp_short``  — ``"GENE p.AminoAcidChange"`` string, or ``None``
      for non-coding variants.

    The function searches for the *most severe* impact level in order:
    ``HIGH → MODERATE → LOW → MODIFIER``.

    Parameters
    ----------
    vcf_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``so_term`` and ``hgvsp_short`` added.
    """
    so_terms: list[str] = []
    gene_hgvsps: list[str] = []

    impact_patterns = [
        re.compile(r"\|([^|]+)\|HIGH"),
        re.compile(r"\|([^|]+)\|MODERATE"),
        re.compile(r"\|([^|]+)\|LOW"),
        re.compile(r"\|([^|]+)\|MODIFIER"),
    ]
    gene_re = re.compile(r"\|([^|]+)\|ENSG\d+")
    hgvsp_re = re.compile(r"p\.\w+\d+\w+")

    for info_string in vcf_df["INFO"]:
        so_term = None
        for pat in impact_patterns:
            m = pat.search(str(info_string))
            if m:
                so_term = m.group(1)
                break
        # if so_term is None:
        #     raise ValueError("Given VCF is not VEP annotated. Instructions for adding VEP annotation can be found in the README.md")

        gene_m = gene_re.search(str(info_string))
        gene = gene_m.group(1) if gene_m else None

        hgvsp_m = hgvsp_re.search(str(info_string))
        hgvsp = hgvsp_m.group() if hgvsp_m else None

        gene_hgvsp = f"{gene} {hgvsp}" if (gene and hgvsp) else None

        so_terms.append(so_term)
        gene_hgvsps.append(gene_hgvsp)

    df = vcf_df.copy()
    df["so_term"] = so_terms
    df["hgvsp_short"] = gene_hgvsps
    return df


# ---------------------------------------------------------------------------
# Read depth extraction
# ---------------------------------------------------------------------------

def get_reads(
    vcf_df: pd.DataFrame,
    sample_col: str | None = None,
) -> pd.DataFrame:
    """
    Extract total read depth (``DP``) and alt-allele read count (``AD``)
    from the FORMAT / sample columns.

    Adds two columns:

    * ``tot_reads`` — total read depth at the variant position (float)
    * ``mut_reads`` — number of reads supporting the ALT allele (float)

    Rows where ``DP`` or ``AD`` cannot be parsed are filled with ``NaN``.

    Parameters
    ----------
    vcf_df : pd.DataFrame
    sample_col : str or None
        Name of the sample column. If ``None``, auto-detected via
        :func:`detect_sample_column`.

    Returns
    -------
    pd.DataFrame
    """
    if sample_col is None:
        sample_col = detect_sample_column(vcf_df)

    tot_reads: list[float | None] = []
    mut_reads: list[float | None] = []

    for _, row in vcf_df.iterrows():
        fmt_fields = str(row["FORMAT"]).split(":")
        sample_values = str(row[sample_col]).split(":")

        # --- DP ---
        dp: float | None = None
        if "DP" in fmt_fields:
            idx = fmt_fields.index("DP")
            try:
                dp = float(str(sample_values[idx]).split(",")[0])
            except (IndexError, ValueError):
                dp = None
        tot_reads.append(dp)

        # --- AD (second value = ALT allele) ---
        ad: float | None = None
        if "AD" in fmt_fields:
            idx = fmt_fields.index("AD")
            try:
                ad_vals = str(sample_values[idx]).split(",")
                if len(ad_vals) >= 2:
                    ad = float(ad_vals[1])
            except (IndexError, ValueError):
                ad = None
        mut_reads.append(ad)

    df = vcf_df.copy()
    df["tot_reads"] = tot_reads
    df["mut_reads"] = mut_reads
    return df


# ---------------------------------------------------------------------------
# VAF extraction
# ---------------------------------------------------------------------------

def get_vaf(
    vcf_df: pd.DataFrame,
    sample_col: str | None = None,
) -> tuple[pd.DataFrame, list[float]]:
    """
    Extract the variant allele frequency (``VAF``) from the FORMAT /
    sample columns.

    Adds a ``VAF`` column to the DataFrame. If the FORMAT field does not
    contain a ``VAF`` key, VAF is computed from ``AD`` and ``DP`` columns
    (must have been added by :func:`get_reads` first).

    Parameters
    ----------
    vcf_df : pd.DataFrame
    sample_col : str or None

    Returns
    -------
    (pd.DataFrame, list[float])
        Updated DataFrame and a plain list of VAF values (for downstream
        use in GMM fitting).
    """
    if sample_col is None:
        sample_col = detect_sample_column(vcf_df)

    df = vcf_df.copy()
    vaf_values: list[float] = []

    if "FORMAT" in df.columns and df["FORMAT"].str.contains("VAF").any():
        # Extract VAF directly from FORMAT
        raw: list[float | None] = []
        for _, row in df.iterrows():
            fmt_fields = str(row["FORMAT"]).split(":")
            sample_values = str(row[sample_col]).split(":")
            val: float | None = None
            if "VAF" in fmt_fields:
                idx = fmt_fields.index("VAF")
                try:
                    val = float(str(sample_values[idx]).split(",")[0])
                except (IndexError, ValueError):
                    val = None
            raw.append(val)
        df["VAF"] = raw
    elif "mut_reads" in df.columns and "tot_reads" in df.columns:
        # Fallback: compute VAF = AD / DP
        df["VAF"] = df["mut_reads"] / df["tot_reads"].replace(0, np.nan)
    else:
        raise ValueError(
            "Cannot extract VAF: FORMAT column has no VAF field and "
            "get_reads() has not been called yet."
        )

    vaf_values = df["VAF"].dropna().tolist()
    return df, vaf_values

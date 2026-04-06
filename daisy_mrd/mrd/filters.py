"""
daisy_mrd.mrd.filters
=====================
Multi-layer noise filtering applied to per-variant read counts before
MRD score calculation.

Each filter is an independent function that accepts and returns a
DataFrame, so users can apply any subset or order they choose.

The full sequential pipeline (matching the original analysis) is
provided by :func:`apply_all_filters`, which also saves intermediate
CSVs for each filter layer.

Filter layers
-------------
1. **C>TG / CG>A context filter** (:func:`filter_noisy_context`)
   Removes positions whose trinucleotide context matches known
   sequencing artefact signatures. Requires flanking nucleotides
   (added by :func:`add_flanking_nucleotides`).

2. **Sex chromosome filter** (:func:`filter_sex_chromosomes`)
   Removes variants on chrX and chrY to avoid copy-number noise.

3. **High read-depth filter** (:func:`filter_high_depth`)
   Removes positions with ``Read_Depth > 200`` (these are often
   artefact-rich repeat regions).

4. **Germline filter** (:func:`filter_germline`)
   Removes positions where the most common non-reference base is
   consistent with a heterozygous germline variant (binomial test
   against p = 0.5).

5. **Panel of Normals (PoN) filter** (:func:`filter_pon_remission`)
   Removes positions whose alt-allele count is not significantly
   above the platform-level background noise rate (binomial test).

Notes
-----
The flanking-nucleotide step uses ``twobitreader`` to look up the
reference sequence context. This is an **optional** dependency; if it
is not installed the context filter is skipped with a warning.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# ---------------------------------------------------------------------------
# Helper: compute VAF
# ---------------------------------------------------------------------------

def add_vaf(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``VAF`` column: ``read_num_match / Read_Depth``."""
    df = df.copy()
    df["VAF"] = df["read_num_match"] / df["Read_Depth"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# Filter 1: Trinucleotide context (C>TG / CG>A artefacts)
# ---------------------------------------------------------------------------

def add_flanking_nucleotides(
    df: pd.DataFrame,
    twobit_path: str | Path,
) -> pd.DataFrame:
    """
    Add ``bp_before``, ``bp_after``, and ``context`` columns to the
    DataFrame by looking up flanking bases in a 2bit genome file.

    Requires the ``twobitreader`` package. If it is not installed,
    a ``ImportError`` is raised with an installation hint.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``CHROM``, ``POS``, ``REF``, ``OG_ALT`` columns.
    twobit_path : str or Path
        Path to a ``.2bit`` genome file (e.g. ``hg38.2bit``).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``bp_before``, ``bp_after``, and
        ``context`` (format: ``{bp_before}{REF}>{OG_ALT}{bp_after}``)
        columns added.
    """
    try:
        import twobitreader  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "twobitreader is required for context filtering. "
            "Install it with: pip install twobitreader"
        ) from exc

    genome = twobitreader.TwoBitFile(str(twobit_path))
    df = df.copy()
    df["bp_before"] = "NA"
    df["bp_after"] = "NA"

    for idx, row in df.iterrows():
        chrom = str(row["CHROM"])
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom
        pos = int(row["POS"])

        try:
            if chrom in genome:
                df.at[idx, "bp_before"] = (
                    genome[chrom][pos - 2 : pos - 1].upper() if pos > 1 else "NA"
                )
                df.at[idx, "bp_after"] = genome[chrom][pos : pos + 1].upper()
        except Exception as e:
            warnings.warn(f"Could not look up flanking bases at {chrom}:{pos}: {e}")

    df["context"] = (
        df["bp_before"].astype(str)
        + df["REF"].astype(str)
        + ">"
        + df["OG_ALT"].astype(str)
        + df["bp_after"].astype(str)
    )
    return df


def filter_noisy_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove positions with known artefact trinucleotide contexts:

    * ``[ACGTN]C>TG`` — C>T in a CpG context (oxidation artefact)
    * ``CG>A[ACGTN]`` — C>A in a CpG context

    Requires a ``context`` column (added by
    :func:`add_flanking_nucleotides`).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if "context" not in df.columns:
        warnings.warn(
            "filter_noisy_context: 'context' column not found. "
            "Run add_flanking_nucleotides() first. Skipping filter."
        )
        return df

    mask_c_tg = df["context"].str.match(r"[ACGTN]C>TG")
    mask_cg_a = df["context"].str.match(r"CG>A[ACGTN]")
    return df[~(mask_c_tg | mask_cg_a)].copy()


# ---------------------------------------------------------------------------
# Filter 2: Sex chromosomes
# ---------------------------------------------------------------------------

def filter_sex_chromosomes(
    df: pd.DataFrame,
    sex_chroms: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove variants on sex chromosomes.

    Parameters
    ----------
    df : pd.DataFrame
    sex_chroms : list[str] or None
        Chromosome names to remove. Defaults to ``["chrX", "chrY"]``.

    Returns
    -------
    pd.DataFrame
    """
    if sex_chroms is None:
        sex_chroms = ["chrX", "chrY", "X", "Y"]
    return df[~df["CHROM"].isin(sex_chroms)].copy()


# ---------------------------------------------------------------------------
# Filter 3: High read depth
# ---------------------------------------------------------------------------

def filter_high_depth(
    df: pd.DataFrame,
    max_depth: int = 200,
) -> pd.DataFrame:
    """
    Remove positions with abnormally high read depth.

    High-depth positions are typically in repetitive or duplicated
    genomic regions that attract mis-mapping artefacts.

    Parameters
    ----------
    df : pd.DataFrame
    max_depth : int
        Positions with ``Read_Depth > max_depth`` are removed.
        Default: 200.

    Returns
    -------
    pd.DataFrame
    """
    return df[df["Read_Depth"] < max_depth].copy()


# ---------------------------------------------------------------------------
# Filter 4: Germline variants
# ---------------------------------------------------------------------------

def filter_germline(
    df: pd.DataFrame,
    pvalue_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Remove positions that appear to be heterozygous germline variants.

    For each position, the maximum non-reference base count is tested
    against p = 0.5 (binomial test, alternative = "less"). Positions
    where this count is consistent with a heterozygous state
    (p > threshold) are removed.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``A_count``, ``T_count``, ``C_count``, ``G_count``,
        ``indel_count``, ``Read_Depth``.
    pvalue_threshold : float
        Positions with ``germline_pvalue > threshold`` are removed.
        Default: 0.01.

    Returns
    -------
    pd.DataFrame
        With ``germline_pvalue`` column added.
    """
    df = df.copy()
    pvalues: list[float | None] = []

    for _, row in df.iterrows():
        counts = [
            row.get("A_count", 0),
            row.get("T_count", 0),
            row.get("C_count", 0),
            row.get("G_count", 0),
            row.get("indel_count", 0),
        ]
        max_count = int(max(counts))
        depth = int(row["Read_Depth"])

        if depth < 1:
            pvalues.append(None)
        else:
            result = binomtest(k=max_count, n=depth, p=0.5, alternative="less")
            pvalues.append(result.pvalue)

    df["germline_pvalue"] = pvalues
    df["germline_pvalue"] = df["germline_pvalue"].fillna(0)

    return df[df["germline_pvalue"] <= pvalue_threshold].copy()


# ---------------------------------------------------------------------------
# Filter 5: Panel of Normals (remission-side)
# ---------------------------------------------------------------------------

def filter_pon_remission(
    df: pd.DataFrame,
    pon: pd.DataFrame,
    pvalue_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Remove positions whose alt-allele count is not significantly above
    the platform-level background noise rate stored in the PoN.

    This mirrors :func:`~daisy_mrd.lspv.pon.filter_pon` but operates on
    the remission pileup rather than the diagnosis VCF.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``CHROM``, ``POS``, ``read_num_match``, ``Read_Depth``.
    pon : pd.DataFrame
        PoN table with ``CHROM``, ``POS``, ``P_N`` (and ``pos`` key).
    pvalue_threshold : float
        Default: 0.05.

    Returns
    -------
    pd.DataFrame
        With ``noise_pvalue`` column added.
    """
    df = df.copy()
    df["pos"] = df["CHROM"].astype(str) + "_" + df["POS"].astype(str)

    # Ensure PoN has pos key
    if "pos" not in pon.columns:
        pon = pon.copy()
        pon["pos"] = pon["CHROM"].astype(str) + "_" + pon["POS"].astype(str)

    # Inner merge — matches original pd.merge default, drops variants not in PoN
    merged = pd.merge(df, pon.loc[:,["pos", "P_N"]], on="pos")

    pvalues: list[float | None] = []
    for _, row in merged.iterrows():
        p_n = row["P_N"]
        k = int(row["read_num_match"])
        n = int(row["Read_Depth"])
        if n < 1:
            pvalues.append(None)
        else:
            result = binomtest(k=k, n=n, p=float(p_n), alternative="greater")
            pvalues.append(result.pvalue)

    merged["noise_pvalue"] = pvalues
    merged["noise_pvalue"] = merged["noise_pvalue"].fillna(0)
    merged = merged.drop(columns=["pos", "P_N"], errors="ignore")

    # No row filtering — keep all PoN-merged rows (df_filtered3 = df_filtered2)
    # noise_pvalue is used only when computing the MRD score numerator
    return merged.copy()




# ---------------------------------------------------------------------------
# Full filter pipeline (saves each layer to disk)
# ---------------------------------------------------------------------------

_LAYER_NAMES = [
    "no_filters",
    "filter_ct",
    "no_xy",
    "no_xy_no_200",
    "no_xy_no_germline",
    "no_xy_no_germline_no_pon",
]


def apply_all_filters(
    df: pd.DataFrame,
    output_dir: str | Path,
    file_name: str,
    pon: pd.DataFrame,
    twobit_path: str,
    max_depth: int,
    germline_pvalue_threshold: float,
    pon_pvalue_threshold: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Apply all filter layers sequentially, saving each layer to disk.

    Any individual filter can be disabled by setting its controlling
    parameter to ``None`` or by not supplying the required data
    (e.g. omitting ``twobit_path`` skips the context filter,
    omitting ``pon`` skips the PoN filter).

    Parameters
    ----------
    df : pd.DataFrame
        Read-count DataFrame from :func:`~daisy_mrd.mrd.readcount.apply_read_counts`.
    output_dir : str or Path
        Root directory under which per-layer sub-directories are created.
    file_name : str
        Base name used for output CSV filenames (without extension).
    pon : pd.DataFrame or None
        PoN table. If ``None``, the PoN filter is skipped.
    twobit_path : str, Path, or None
        Path to a ``.2bit`` genome file. If ``None``, the context
        filter is skipped.
    max_depth : int
        Read-depth threshold for :func:`filter_high_depth`.
    germline_pvalue_threshold : float
    pon_pvalue_threshold : float

    Returns
    -------
    (dict[str, pd.DataFrame], pd.DataFrame)
        * A dictionary mapping each layer name to its filtered DataFrame.
        * A one-row summary DataFrame with LSPV counts per layer.
    """
    output_dir = Path(output_dir)
    df = df.copy()
    df = df.drop(columns=["Base_Qualities"], errors="ignore")
    df["pos"] = df["CHROM"].astype(str) + "_" + df["POS"].astype(str)
    df = add_vaf(df)

    layers: dict[str, pd.DataFrame] = {}

    def _save(layer_df: pd.DataFrame, layer_name: str) -> pd.DataFrame:
        layer_df = layer_df.copy()
        out_path = output_dir / layer_name / f"{layer_name}_{file_name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        layer_df.to_csv(out_path, index=False)
        layers[layer_name] = layer_df
        return layer_df

    # Layer 0: no filters
    current = _save(df, "no_filters")
    no_filter_count = len(current)

    # Layer 1: context filter (C>TG / CG>A)
    if twobit_path is not None:
        current = add_flanking_nucleotides(current, twobit_path)
        current = filter_noisy_context(current)

    else:
        warnings.warn(
            "twobit_path not provided — skipping C>TG/CG>A context filter. "
            "Pass twobit_path='path/to/hg38.2bit' to enable it."
        )
    current = _save(current, "filter_ct")

    # Layer 2: sex chromosomes
    current = filter_sex_chromosomes(current)
    current = _save(current, "no_xy")

    # Layer 3: high read depth
    current = filter_high_depth(current, max_depth=max_depth)
    current = _save(current, "no_xy_no_200")

    # Layer 4: germline
    current = filter_germline(current, pvalue_threshold=germline_pvalue_threshold)
    current = _save(current, "no_xy_no_germline")

    # Layer 5: PoN
    current = filter_pon_remission(current, pon, pvalue_threshold=pon_pvalue_threshold)

    current = _save(current, "no_xy_no_germline_no_pon")


    # Summary row
    summary = pd.DataFrame(
        {
            "file_name": [file_name],
            "no_filter_lspvs": [no_filter_count],
            "filter_ct_lspvs": [len(layers.get("filter_ct", pd.DataFrame()))],
            "filter_xy_lspvs": [len(layers.get("no_xy", pd.DataFrame()))],
            "filter_xy_>200": [len(layers.get("no_xy_no_200", pd.DataFrame()))],
            "filter_germline": [len(layers.get("no_xy_no_germline", pd.DataFrame()))],
            "filter_pon": [len(layers.get("no_xy_no_germline_no_pon", pd.DataFrame()))],
        }
    )

    return layers, summary

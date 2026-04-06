"""
daisy_mrd.lspv.filter
=====================
VCF-level hard filters applied before clonality analysis.

Functions
---------
filter_vcf_pass         : Keep only PASS variants (file → file)
remove_germline_variants: Drop variants present in gnomAD (AF ≥ 1e-3)
remove_rs               : Drop variants with dbSNP rs IDs
remove_indels           : Keep SNVs only (drop indels via VEP SO term)
apply_hard_filters      : Convenience wrapper that applies all three
                          DataFrame-level filters in sequence
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from daisy_mrd.lspv.query_germline_vcf import remove_germline_variants


# ---------------------------------------------------------------------------
# File-level filter (PASS)
# ---------------------------------------------------------------------------

def filter_vcf_pass(input_vcf: str | Path, output_vcf: str | Path) -> Path:
    """
    Write a new VCF containing only variants whose FILTER column is ``PASS``.

    All header lines (starting with ``#``) are preserved unchanged.

    Parameters
    ----------
    input_vcf : str or Path
        Path to the input VCF (plain or ``.gz``).
    output_vcf : str or Path
        Path where the filtered VCF will be written.

    Returns
    -------
    Path
        Path to the written output file.
    """
    input_vcf = Path(input_vcf)
    output_vcf = Path(output_vcf)

    written = 0
    with open(input_vcf) as infile, open(output_vcf, "w") as outfile:
        for line in infile:
            if line.startswith("#"):
                outfile.write(line)
                continue
            fields = line.strip().split("\t")
            if len(fields) > 6 and fields[6] == "PASS":
                outfile.write(line)
                written += 1

    return output_vcf


# ---------------------------------------------------------------------------
# DataFrame-level filters
# ---------------------------------------------------------------------------

# def variant_in_germline_vcf(germline_vcf: Path, chrom: str, pos: int, ref: str, alt: str) -> bool:
#     """
#     Check if a variant (chrom, pos, ref, alt) exists in an indexed VCF using bcftools.
#     """
#     region = f"{chrom}:{pos}-{pos}"
#     result = subprocess.run([
#             "bcftools", "view", "-r", region, str(germline_vcf)
#         ],
#         capture_output=True, text=True, check=True
#     )
#     for line in result.stdout.splitlines():
#         if line.startswith("#"):
#             continue
#
#         fields = line.split("\t")
#         rec_chrom = fields[0]
#         rec_pos = int(fields[1])
#         rec_ref = fields[3]
#         rec_alts = fields[4]
#
#         if (rec_chrom == chrom and rec_pos == pos and rec_ref == ref and alt in rec_alts):
#             return True
#     return False
#
#
# def remove_germline_variants(vcf_df: pd.DataFrame, germline_vcf_file: Path) -> pd.DataFrame:
#     # uses germline vcf file to exclude all variants exactly matching a variant in the vcf file
#     # maybe should wrap everything in a DB class
#     # germline_positions = set_of_all_germline_positions(germline_vcf_file)
#     vcf_df["KNOWN_GERMLINE_VARIANT"]=False
#     for idx, row in vcf_df.iterrows():
#         t=time.time()
#         if int(row.POS)==101442512:
#             croc=1
#         if variant_in_germline_vcf(germline_vcf_file, row.CHROM, row.POS, row.REF, row.ALT):
#             vcf_df.at[idx, 'KNOWN_GERMLINE_VARIANT'] = True
#         print(time.time() - t)
#     return vcf_df[~vcf_df["KNOWN_GERMLINE_VARIANT"]].drop(columns=["KNOWN_GERMLINE_VARIANT"]).reset_index(drop=True)




def remove_rs(vcf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove variants that carry a dbSNP ``rs`` identifier in the ID column.

    These are known germline variants regardless of their gnomAD frequency.

    Parameters
    ----------
    vcf_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    mask = ~vcf_df["ID"].astype(str).str.startswith("rs")
    return vcf_df[mask].copy()


def remove_indels(vcf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove indel variants detected via the ``VARIANT_TYPE`` tag in the INFO
    field (set by VEP or the variant caller).

    If ``VARIANT_TYPE`` is absent from INFO, the variant is kept (it is
    assumed to be an SNV).

    Parameters
    ----------
    vcf_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    def _is_indel(info_str: str) -> bool:
        for part in str(info_str).split(";"):
            if part.startswith("VARIANT_TYPE="):
                return "indel" in part.split("=", 1)[1].lower()
        return False

    mask = ~vcf_df["INFO"].apply(_is_indel)
    return vcf_df[mask].copy()


def apply_hard_filters(vcf_df: pd.DataFrame, germline_vcf_file: Path) -> pd.DataFrame:
    """
    Apply all three DataFrame-level hard filters in sequence:

    1. :func:`remove_germline_variants`
    2. :func:`remove_rs`
    3. :func:`remove_indels`

    Parameters
    ----------
    vcf_df : pd.DataFrame
        Annotated VCF DataFrame (must have ``GNOMAD`` and ``AF`` columns
        if gnomAD annotation was run; otherwise germline removal falls
        back to gnomAD-absent logic only).
    germline_vcf_file : Path
        Path to common germline mutations. All such mutations will be filtered out

    Returns
    -------
    pd.DataFrame
        Hard-filtered DataFrame.
    """
    df = vcf_df
    df = remove_germline_variants(df, germline_vcf_file) # not really the optimal way to do it, but not important
    df = remove_rs(df)
    df = remove_indels(df)
    return df

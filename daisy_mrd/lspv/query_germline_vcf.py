from pathlib import Path
import subprocess
import tempfile
import pandas as pd


def load_matching_germline_variants(
    germline_vcf: Path,
    query_df: pd.DataFrame,
) -> set[tuple[str, int, str, str]]:
    """
    Fetch all germline variants from germline_vcf that match any CHROM/POS in query_df,
    then return them as a set of (chrom, pos, ref, alt) tuples.
    """

    unique_regions = query_df.loc[:, ["CHROM", "POS"]].drop_duplicates()

    with tempfile.NamedTemporaryFile(mode="w", delete=True) as region_file:
        for row in unique_regions.itertuples(index=False):
            region_file.write(f"{row.CHROM}\t{int(row.POS)}\n")
        region_file.flush()

        result = subprocess.run(
            [
                "bcftools",
                "query",
                "-R",
                region_file.name,
                "-f",
                "%CHROM\t%POS\t%REF\t%ALT\n",
                str(germline_vcf),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    germline_variants: set[tuple[str, int, str, str]] = set()

    for line in result.stdout.splitlines():
        chrom, pos, ref, alts = line.split("\t")
        pos = int(pos)
        for alt in alts.split(","):
            germline_variants.add((chrom, pos, ref, alt))

    return germline_variants


def remove_germline_variants(
    vcf_df: pd.DataFrame,
    germline_vcf_file: Path,
) -> pd.DataFrame:
    """
    Remove all rows from vcf_df whose (CHROM, POS, REF, ALT) exactly match
    a variant in the indexed germline VCF.
    """

    df = vcf_df.copy()

    germline_variants = load_matching_germline_variants(germline_vcf_file, df)

    variant_keys = list(
        zip(
            df["CHROM"].astype(str),
            df["POS"].astype(int),
            df["REF"].astype(str),
            df["ALT"].astype(str),
        )
    )

    mask_known = pd.Series(
        (key in germline_variants for key in variant_keys),
        index=df.index,
    )

    return df.loc[~mask_known].reset_index(drop=True)
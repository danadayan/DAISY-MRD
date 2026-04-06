"""
daisy_mrd.lspv.annotate
"""

from __future__ import annotations


import re
from pathlib import Path
from typing import Dict


def verify_vep_annotations(vcf_path: Path) -> bool:
    # returns has_gnomad, has_vep
    with open(vcf_path, "r") as f:
        has_csq = False
        for line in f:
            if line.startswith("#"):
                # has_gnomad_af = has_gnomad_af or re.search(r"^##INFO=<ID=CSQ", line) is not None
                has_csq = has_csq or re.search(r"^##INFO=<ID=CSQ", line) is not None
            else:
                break
    return has_csq

def verify_every_line_passes(vcf_path: Path) -> bool:
    # verifies every variant in the vcf is a passing variant
    with open(vcf_path, "r") as f:
        for line in f:
            if not (line.startswith("#") or line.split()[6] == "PASS"):
                return False
    return True


def verify_vcf(vcf_path: Path) -> Dict[str, bool]:
    """
    Verify that a VCF file has VEP annotations, GNOMAD annotations, and is filtered.
    Parameters
    ----------
    vcf_path
        Path to the VCF file.
    Returns has_gnomad_af,
    -------
    """
    has_vep = verify_vep_annotations(vcf_path)
    return {"VEP": has_vep, "FILTERED": verify_every_line_passes(vcf_path)}


def check_vcf_ready_for_daisy(vcf_path: Path):
    vcf_features = verify_vcf(vcf_path)
    problematic_keys = [key for key in vcf_features if not vcf_features[key]]
    if len(problematic_keys) == 0:
        print("VCF is ready for Daisy")
    else:
        print(f"VCF is NOT ready for Daisy. It lacks {', '.join(problematic_keys)}")
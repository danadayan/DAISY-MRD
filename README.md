# DAISY-MRD

**Distributed Analysis of Integrated Sites for Yielding MRD**

A Python package for WGS-based measurable residual disease (MRD) monitoring in pediatric AML using leukemia-specific passenger variants (LSPVs).

---

## Background

DAISY-MRD identifies **Leukemia specific passenger variants (LSPVs)** - somatic mutations with no biological effect that are shared by the entire leukemic clone, and calculate an MRD score based in the patient specific LSPVs. 

The method is described in:

> Shiloh R\*, Dayan D\*, et al. *Passenger Variants as Pan-Clonal Markers of Pediatric AML: Implications for Molecular MRD.* Hemasphere, 2026.

---

## How it works

```
Diagnosis WGS (VCF)                 Remission BAM/CRAM
       │                                      │
  ┌────▼────────────────────┐    ┌────────────▼──────────────┐
  │    Step 1: LSPV ID      │    │    Step 2: MRD Scoring    │
  │                         │    │                           │
  │  • PASS filter          │    │  • samtools mpileup       │
  │  • gnomAD annotation    │    │    over LSPV positions    │
  │  • PoN filter           │    │  • Count reads            │
  │  • GMM clonality fit    │────▶  • Noise filtering        │
  │  • Remove coding vars   │    │  • DAISY-MRD score        │
  └─────────────────────────┘    └───────────────────────────┘
           │                                  │
     LSPVs CSV                     score = Σ(alt) / Σ(depth)
     GMM plot                      noise distribution plot
     Pie chart
```

**DAISY-MRD score** = total ALT-supporting reads across all LSPV positions ÷ total read depth across all LSPV positions. With ~300 LSPVs per patient at 100× WGS depth, this aggregates ~30,000 reads, giving a sensitivity of ~3 × 10⁻⁴.

---

## Installation

### GitHub

```bash
git clone https://github.com/maruvka-lab/daisy-mrd.git
cd daisy-mrd
pip install -e .
```

### PyPI (*coming soon*)

```bash
pip install daisy-mrd
```

### Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| numpy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| scipy | ≥ 1.10 |
| scikit-learn | ≥ 1.3 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |
| requests | ≥ 2.28 |
| samtools | ≥ 1.17 (external binary, required for Step 2) |

**Optional dependencies:**

| Package | Purpose | Install |
|---|---|---|
| `twobitreader` | C>TG / CG>A trinucleotide context filter in Step 2 | `pip install twobitreader` |
| Docker + `ensemblorg/ensembl-vep` | VEP annotation in Step 1 (skip if VCF already annotated) | `docker pull ensemblorg/ensembl-vep` |

---

## Quick Start

### Step 1 — Identify LSPVs from a diagnosis VCF

```python
from daisy_mrd import run_lspv_pipeline

lspv_result = run_lspv_pipeline(
    vcf_path                = DIAGNOSIS_VCF,
    output_dir              = f"results/{PATIENT_ID}/lspv/",
    patient_id              = PATIENT_ID,

    run_vep_annotation      = False, # False for annotated files, True if annotation required
    vcf_is_filtered         = True,  # True for filtered (FILTER == PASS)
    vcf_is_gnomad_annotated = True,  # True if gnomad annotation exist
    
)

print(result.summary)
#    patient  total_variants  clonal  subclonal  n_lspvs
# 0      001            1204     892        312      734

result.lspvs.head()       # DataFrame of LSPVs
result.fig_gmm.show()     # Gaussian Mixture Model plot
result.fig_pie.show()     # Clonal vs sub-clonal pie chart
```

### Step 2 — Compute DAISY-MRD score from a remission BAM/CRAM

```python
from daisy_mrd import run_mrd_single

mrd_result = run_mrd_single(
    lspv_csv       = lspv_csv,          # output from Step 1
    remission_bam  = REMISSION_CRAM,
    reference      = HUMAN_REFERENCE_GENOME,
    output_dir     = f"results/{PATIENT_ID}/mrd/",
    patient_id     = PATIENT_ID,

    samtools_path  = "/samtools",

    # Use the built-in Panel of Normals for noise filtering
    pon_path       = None,
    twobit_path  = "/data/hg38.2bit")


print(f"DAISY-MRD score : {mrd_result.score.score:.2e}")
print(f"Alt reads       : {mrd_result.score.total_alt_reads}")
print(f"Total depth     : {mrd_result.score.total_read_depth}")
print(f"LSPV positions  : {mrd_result.score.n_lspv_positions}")
```
**Output files:**

```
results/001/mrd/
├── 001_remission.pileup
├── 001_pileup.csv
├── 001_read_counts.csv
├── 001_filter_summary.csv
├── 001_daisy_mrd_score.csv
└── filter_layers/
    ├── no_filters/
    ├── filter_ct/
    ├── no_xy/
    ├── no_xy_no_200/
    ├── no_xy_no_germline/
    ├── no_xy_no_germline_no_pon/     ← used for final score
```


### Running a cohort

```python
from daisy_mrd import run_mrd_cohort

patients = [
    {
        "patient_id": "001",
        "lspv_csv": "results/001/001_lspvs.csv",
        "remission_bam": "data/001_remission.cram",
    },
    {
        "patient_id": "002",
        "lspv_csv": "results/002/002_lspvs.csv",
        "remission_bam": "data/002_remission.cram",
    },
]

scores_df = run_mrd_cohort(
    patients=patients,
    reference="GRCh38.fa",
    output_dir="results/cohort/",
)

scores_df.to_csv("daisy_mrd_scores.csv", index=False)
```

---


**Returns `MrdResult`:**

| Attribute | Type | Description |
|---|---|---|
| `score` | `MrdScore` | `.score`, `.total_alt_reads`, `.total_read_depth`, `.n_lspv_positions` |
| `filter_layers` | `dict[str, DataFrame]` | All 7 intermediate DataFrames, keyed by layer name |
| `filter_summary` | `pd.DataFrame` | LSPV count at each filter layer |
| `fig_noise` | `Figure` or `None` | Noise distribution plot (populated by cohort run) |
| `output_dir` | `Path` | Directory of saved outputs |


---

### `run_mrd_cohort()`

```python
scores_df = run_mrd_cohort(
    patients        = [...],             # list of dicts (see above)
    reference       = "GRCh38.fa",
    output_dir      = "results/cohort/",
    # All run_mrd_single() options are accepted as cohort-level defaults
    # and can be overridden per-patient inside each patient dict
    plot_noise      = True,              # Save noise distribution PDFs
)
```

Returns a `pd.DataFrame` with columns `patient_id`, `daisy_mrd_score`, `total_alt_reads`, `total_read_depth`, `n_lspv_positions`.

---

## Panel of Normals

A built-in PoN (built on the Ultima Genomics sequencing platform) is included. To use your own:

```python
run_lspv_pipeline(..., pon_path="/path/to/my_pon.csv")
run_mrd_single(...,    pon_path="/path/to/my_pon.csv")
```

**Required PoN CSV columns:**

| Column | Description |
|---|---|
| `CHROM` | Chromosome (e.g. `chr1`) |
| `POS` | 1-based position |
| `P_N` | Background noise rate (float in [0, 1]) |

---

## Using individual modules

Every function is importable on its own:

```python
# Step 1
from daisy_mrd.lspv.filter   import filter_vcf_pass, apply_hard_filters
from daisy_mrd.lspv.annotate import annotate_vcf, run_vep
from daisy_mrd.lspv.reads    import extract_info, get_reads, get_vaf
from daisy_mrd.lspv.pon      import load_pon, filter_pon
from daisy_mrd.lspv.gmm      import fit_gmm, get_clonal_peak_mean, label_clonality, plot_gmm
from daisy_mrd.lspv.identify import extract_lspvs, plot_clonality_pie
from daisy_mrd.utils         import read_vcf

# Step 2
from daisy_mrd.mrd.pileup    import run_mpileup, pileup_to_df, merge_lspv_alts
from daisy_mrd.mrd.readcount import apply_read_counts
from daisy_mrd.mrd.filters   import (
    add_flanking_nucleotides, filter_noisy_context,
    filter_sex_chromosomes, filter_high_depth,
    filter_germline, filter_pon_remission, filter_high_vaf,
    apply_all_filters,
)
from daisy_mrd.mrd.score     import compute_mrd_score, plot_noise_distributions
```

---

## Project structure

```
daisy-mrd/
├── daisy_mrd/
│   ├── __init__.py              # Package entry-point; exposes main functions
│   ├── utils.py                 # VCF reader, path helpers, PoN resolver
│   ├── data/
│   │   └── pon_default.csv      # Built-in Panel of Normals
│   ├── lspv/                    # Step 1: LSPV identification
│   │   ├── annotate.py          # VEP (optional) + gnomAD annotation
│   │   ├── filter.py            # PASS, germline, rs, indel hard filters
│   │   ├── gmm.py               # GMM fitting + clonality classification
│   │   ├── identify.py          # LSPV extraction + pie chart
│   │   ├── pipeline.py          # run_lspv_pipeline() orchestrator
│   │   ├── pon.py               # Panel of Normals (diagnosis-side)
│   │   └── reads.py             # DP / AD / VAF extraction from FORMAT
│   └── mrd/                     # Step 2: DAISY-MRD scoring
│       ├── filters.py           # 6 independent noise filter functions
│       ├── pipeline.py          # run_mrd_single() / run_mrd_cohort()
│       ├── pileup.py            # samtools mpileup + pileup parser
│       ├── readcount.py         # ALT read counting from pileup strings
│       └── score.py             # Score calculation + noise plots
├── LICENSE
├── CONTRIBUTING.md
└── pyproject.toml
```
---

## Citation

If you use DAISY-MRD in your research, please cite:

```
```

---

## License

© 2026 Dana Dayan, Yosef Maruvka — Maruvka Lab, Technion

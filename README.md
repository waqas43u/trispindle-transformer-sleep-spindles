# TriSpindle: Controlled Transformer Evaluation for Sleep Spindle Segmentation

[![Research code](https://img.shields.io/badge/status-research%20code-informational)](#research-scope)

TriSpindle is a research package for controlled comparison of Global Transformer, Sparse-Local Transformer, and Conformer encoder configurations for single-channel sleep spindle segmentation. It includes the MASS-SS2 development pipeline, a frozen DREAMS external-evaluation pipeline, compact verified results, publication figures, configurations, and audit records.

## Repository description

**Suggested GitHub repository name:** `trispindle-transformer-sleep-spindles`  
**Suggested description:** *Controlled Transformer-family comparison for sleep spindle segmentation with subject-wise MASS-SS2 evaluation, explainability, and frozen DREAMS external transfer.*

## What is included

```text
src/                 MASS-SS2 training/evaluation and frozen DREAMS inference scripts
results/             Compact numerical summaries, configurations, and audit artifacts
figures/             Manuscript-ready main and supplementary figures (PNG and PDF)
tables/              LaTeX tables used in the verified Results reconstruction
docs/                Data-access, reproducibility, and results guides
requirements.txt     Python dependencies from the archived experimental environment
```

The repository deliberately excludes raw EEG, sleep-stage files, expert annotations, model checkpoints, probability timelines, and server-specific paths. Compact summary tables retain public MASS record identifiers where needed to audit fixed subject-wise splits; they contain no EEG samples or continuous per-participant timelines. These choices respect data-access conditions, limit repository size, and retain essential provenance.

## Research scope

The primary internal endpoint is union-reference event F1 at IoU $\geq 0.20$ on 15 retained MASS-SS2 subjects, evaluated with subject-wise five-fold cross-validation. The Conformer achieved the highest descriptive internal mean (79.58% $\pm$ 2.82%), followed by Sparse-Local (77.93% $\pm$ 5.20%) and Global (64.56% $\pm$ 12.70%) configurations. These fold-level differences did not remain statistically significant after Holm correction.

For frozen MASS-to-DREAMS transfer, all learned weights, calibrators, and decoder parameters were restored without DREAMS adaptation. The Global Transformer had the highest union-reference excerpt-macro F1 at IoU $\geq 0.20$ (47.92%), while the Conformer had the highest tolerant F1 (62.85%). The ranking reversal is descriptive and reflects a cross-protocol comparison; it does not identify a causal source of the transfer difference.

## Installation

Python 3.11 was used for the archived experiments. Create an isolated environment, install a PyTorch build appropriate for your operating system/CUDA driver, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

See the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for an appropriate CPU or CUDA wheel. The exact archived package versions are recorded in `results/internal/*/package_versions.json` and `results/external/run_manifest.json`.

## Data access and execution

Obtain MASS-SS2 and DREAMS from their original providers and comply with their respective data-use conditions. Do not commit either dataset, annotations, or derived participant-level data to a public fork.

Inspect the command-line interfaces before running an experiment:

```bash
python src/transformer_mass_pipeline.py --help
python src/frozen_dreams_evaluation.py --help
```

The MASS pipeline implements training and subject-wise evaluation. The DREAMS script is deliberately an inference application: it imports shared definitions from the MASS pipeline, restores fold-specific artifacts, and prohibits training, calibration, threshold tuning, decoder tuning, or model selection on DREAMS.

Detailed data layout and reproducibility guidance are provided in [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Results and figures

- `results/internal/`: cross-fold performance, reference sensitivity, paired statistics, and representation-reliance summaries.
- `results/external/`: frozen DREAMS macro/micro comparisons, coverage policy, and boundary summaries.
- `results/audit/`: checkpoint/provenance and numerical-consistency documentation.
- `figures/main/`: main manuscript figures.
- `figures/supplementary/`: secondary figures, including DREAMS boundary reconstruction.
- `tables/`: ready-to-insert `booktabs` LaTeX tables.

Read [docs/RESULTS_GUIDE.md](docs/RESULTS_GUIDE.md) before interpreting or reusing metrics. In particular, do not treat the five MASS fold ensembles evaluated on the same DREAMS excerpts as independent external replications.

## Citation

Please cite the accompanying TriSpindle manuscript and this software release. A citation template is provided in `CITATION.cff`; update the version, authors, DOI, and release date before creating a public GitHub release or Zenodo archive.

## License and contributions

No public software license has been selected in this staging package. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) before publishing. Contributions, issues, and pull requests should follow [CONTRIBUTING.md](CONTRIBUTING.md).

## Research-use notice

This repository is intended for research reproducibility. It is not a medical device and has not been clinically validated for diagnosis, treatment, or autonomous clinical decision-making.

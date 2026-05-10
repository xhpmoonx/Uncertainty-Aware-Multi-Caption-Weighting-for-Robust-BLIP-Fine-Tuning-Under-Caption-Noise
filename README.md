# Uncertainty-Aware Multi-Caption Weighting for BLIP

This repository contains code, configuration files, and selected result summaries for a course project on uncertainty-aware multi-caption weighting for BLIP image captioning.

## Repository layout

- `code/blip_uncertainty_First_Implementation/` — external soft-weighting experiments on top of BLIP.
- `code/BLIP_dynamic_Second_Implementation/` — modified BLIP training loop with native weighted caption training.
- `code/BLIP_official_baseline/` — BLIP baseline code/configuration used for comparison.
- `docs/paper.pdf` — project paper.
- `docs/slides.pdf` — project slides.
- `LICENSES/BLIP_BSD_3_CLAUSE_LICENSE.txt` — third-party BLIP license notice.

## Not included

Large generated datasets, COCO image files, checkpoints, logs, virtual environments, and local HPC output folders are intentionally excluded. Recreate or download datasets/checkpoints using the scripts and configuration files.

## Third-party attribution

Parts of this repository are based on Salesforce BLIP. The BLIP-derived source code is distributed under the BSD 3-Clause License. See `LICENSES/BLIP_BSD_3_CLAUSE_LICENSE.txt` and the source-file headers.

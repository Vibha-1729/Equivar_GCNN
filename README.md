# Equivar GCNN: Predicting Atomic Born Effective Charges across the Periodic Table

This repository contains the production code, datasets, and benchmark results for **EquivarBECGNN**, an E(3)-equivariant graph neural network (GNN) built using **PyTorch Geometric (PyG)** and **e3nn** to predict 3x3 atomic Born Effective Charge (BEC) tensors directly from 3D crystal structures.

Based on the research paper:
> **Representing Born effective charges with equivariant graph convolutional neural networks**
> *Alex Kutana, Koji Shimizu, Satoshi Watanabe, and Ryoji Asahi* (*Scientific Reports*, 2025). [DOI: 10.1038/s41598-025-01250-5](https://doi.org/10.1038/s41598-025-01250-5)

---

## Repository Directory Structure

- data/raw/: Raw extxyz datasets (perovskites, ZrO2, Li3PO4, MP)
- data/processed/: Master deduplicated dataset and PyTorch checkpoints
- scripts/: Production Python scripts
- outputs/plots/: 500-epoch loss curves and parity plots
- outputs/csv_results/: Benchmark tables, ASR audits and anomalous BEC lists
- outputs/models/: 500-epoch and cross-validation model weight checkpoints (.pt)

---

## 500-Epoch Empirical Benchmark Results

All models were trained for 500 full epochs using an E(3)-equivariant Graph Neural Network with l <= 2 spherical harmonics (0e+1o+2e irreps), 32 Gaussian Cosine Envelope Radial Basis functions (r_c = 3.0 A), AdamW optimizer, and L1-norm MAE loss.

| Experiment Description | Train Split | Validation Split | Test Split | Final Test MAE (e) |
| :--- | :---: | :---: | :---: | :---: |
| **500-Epoch Master Combined Model** | 31,085 (80%) | 3,886 (10%) | 3,886 (10%) | **0.3854 e** |
| **500-Epoch Mendeley-Only Model** | 19,461 (80%) | 2,433 (10%) | 2,433 (10%) | **0.3821 e** |
| **500-Epoch MP-Only Model** | 11,624 (80%) | 1,453 (10%) | 1,453 (10%) | **0.5060 e** |
| **Cross-Transfer: Mendeley -> MP** | 24,327 (100%) | N/A | 14,530 (100%) | **6.8174 e** |
| **Cross-Transfer: MP -> Mendeley** | 14,530 (100%) | N/A | 24,327 (100%) | **0.3842 e** |

---

## Dataset Curation & Quality Audits
1. Fingerprint Deduplication: MD5 hash matching formula stoichiometry, atom count, and 3D atomic coordinates. Preserved distinct polymorphs (cubic vs monoclinic ZrO2, thermal MD snapshots); removed 34 exact coordinate duplicates.
2. Acoustic Sum Rule (ASR) Filter: Computed net unit-cell charge drift ||sum_i Z_i*||_F. Filtered 4,991 unphysical high-temperature MD snapshots with drift > 1.0 e.
3. Band Gap Requirement (Eg > 0 eV): Filtered out metallic structures where Born effective charge tensors are ill-defined.
4. Master Dataset: 38,857 unique crystal structures (1.55M+ atomic BEC tensors across 85 elements).

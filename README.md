# Equivar BEC GCNN: Predicting Atomic Born Effective Charges across the Periodic Table

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-PyTorch--Geometric-orange.svg)](https://pyg.org/)
[![e3nn Equivariant GNN](https://img.shields.io/badge/e3nn-Equivariant%20GNN-purple.svg)](https://e3nn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the production codebase, curated datasets, 500-epoch deep learning benchmarks, and evaluation scripts for **EquivarBECGNN**, an E(3)-equivariant graph neural network built with **PyTorch Geometric (PyG)** and **e3nn** to predict $3 \times 3$ atomic Born Effective Charge ($Z^*_{ij}$) tensors directly from 3D crystal structures.

Based on the research paper:
> **Representing Born effective charges with equivariant graph convolutional neural networks**  
> *Alex Kutana, Koji Shimizu, Satoshi Watanabe, and Ryoji Asahi* (*Scientific Reports*, 2025). [DOI: 10.1038/s41598-025-01250-5](https://doi.org/10.1038/s41598-025-01250-5)

---

## 📊 500-Epoch Empirical Benchmark Results

All models were trained for **500 full epochs** using an E(3)-equivariant Graph Neural Network with spherical harmonics up to $\ell = 2$ (`0e+1o+2e` irreps), 32 Gaussian Cosine Envelope Radial Basis functions ($r_c = 3.0\text{ \AA}$), AdamW optimizer, Cosine Annealing learning rate schedule, and Acoustic Sum Rule (ASR) zero-charge drift symmetry enforcement.

| Experiment Description | Train Dataset | Validation Dataset | Test Dataset | Final Test MAE ($e$) | Loss Plot |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **500-Epoch Master 3-Way Model (Mendeley + MP + JARVIS)** | 52,257 (80%) | 6,532 (10%) | 6,533 (10%) | **`0.4798 e`** | [`loss_curve_mendeley_mp_jarvis_500ep.png`](outputs/plots/loss_curve_mendeley_mp_jarvis_500ep.png) |
| **500-Epoch Master Combined Model (Mendeley + MP)** | 31,085 (80%) | 3,886 (10%) | 3,886 (10%) | **`0.3854 e`** | [`loss_curve_master_combined_500ep.png`](outputs/plots/loss_curve_master_combined_500ep.png) |
| **500-Epoch Mendeley-Only Model** | 19,461 (80%) | 2,433 (10%) | 2,433 (10%) | **`0.3821 e`** | [`loss_curve_mendeley_500ep.png`](outputs/plots/loss_curve_mendeley_500ep.png) |
| **500-Epoch MP-Only Model** | 11,624 (80%) | 1,453 (10%) | 1,453 (10%) | **`0.5060 e`** | [`loss_curve_mp_500ep.png`](outputs/plots/loss_curve_mp_500ep.png) |
| **500-Epoch JARVIS-Alone Model** | 21,172 (80%) | 2,646 (10%) | 2,647 (10%) | **`0.7279 e`** | [`loss_curve_jarvis_500ep.png`](outputs/plots/loss_curve_jarvis_500ep.png) |
| **Cross Transfer (MP $\rightarrow$ Mendeley Zero-Shot)** | 14,530 (100% MP) | N/A | 24,327 (100% Mendeley) | **`0.3842 e`** | [`loss_curve_transfer_mp_to_mendeley.png`](outputs/plots/loss_curve_transfer_mp_to_mendeley.png) |
| **Cross Transfer (Mendeley $\rightarrow$ MP Zero-Shot)** | 19,461 (100% Mendeley) | 2,433 (Mendeley Val) | 14,530 (100% MP) | **`2.8667 e`** | [`loss_curve_mendeley_to_mp_500ep_verified.png`](outputs/plots/loss_curve_mendeley_to_mp_500ep_verified.png) |

---

## 🔬 Physics & Cross-Transfer Analysis

- **Mendeley Dataset Precision**: Models trained on the Mendeley dataset achieve high accuracy (**`0.3821 e`** MAE) due to dense, low-noise sampling across 9 oxide elements ($\text{Pb, Sr, Ba, Ti, Zr, Hf, Nb, Ta, O}$).
- **Materials Project & JARVIS Diversity**: Materials Project covers **85 chemical elements** across 14,530 structures. JARVIS covers **89 chemical elements** across 26,465 insulator structures.
- **Master 3-Way Fusion**: Fusing Mendeley + Materials Project + JARVIS yields a robust, generalizable model covering **89 chemical elements** across 65,322 crystal structures with a test MAE of **`0.4798 e`**.

---

## 📁 Repository Directory Structure

```
.
├── data/
│   ├── raw/                  # Raw extxyz datasets (perovskites, ZrO2, Li3PO4, MP)
│   └── processed/            # Master deduplicated ExtXYZ datasets (38,857 & 65,322 structures)
├── scripts/                  # Production Python scripts and EquivarBECGNN model definition
│   ├── equivar_model.py      # Core PyG / e3nn E(3)-Equivariant GNN implementation
│   ├── 01_clean_and_deduplicate_master_dataset.py
│   ├── 04_run_500ep_full_experiments.py
│   ├── 30_mendeley_to_mp_500ep_transfer.py
│   ├── 40_fetch_and_prepare_jarvis_dataset.py
│   ├── 44_train_500ep_mendeley_mp_jarvis.py
│   ├── 45_train_500ep_jarvis_alone.py
│   └── 46_download_and_filter_gnome_insulators.py
└── outputs/
    ├── plots/                # 500-epoch loss curves and parity plots (.png)
    ├── csv_results/          # Benchmark tables, ASR audits, and anomalous BEC statistics (.csv)
    └── models/               # PyTorch model checkpoints (.pt)
```

---

## 🧹 Master Dataset Curation & Quality Audits

1. **Fingerprint Deduplication**: MD5 hashing combining chemical formula stoichiometry, total atom count, and 3D atomic coordinates. Preserved distinct polymorphs (e.g., cubic vs monoclinic $\text{ZrO}_2$ and thermal MD snapshots) while removing exact coordinate duplicates.
2. **Acoustic Sum Rule (ASR) Audit**: Net unit-cell charge drift $\|\sum_i Z_i^*\|_F \le 1.0 e$ enforced across all structures.
3. **Insulator Requirement ($E_g > 0\text{ eV}$)**: Filtered out metallic structures where Born Effective Charge tensors are ill-defined.
4. **Master 3-Way Dataset**: **65,322 unique crystal structures** containing **2.1M+ atomic BEC tensors** across **89 chemical elements**.

---

## ⚡ Execution Guide

### Environment Setup
```bash
conda create -n equivar python=3.10 -y
conda activate equivar
pip install torch torch-geometric e3nn numpy pandas matplotlib tqdm jarvis-tools
```

### Reproducing 500-Epoch Benchmarks
```bash
# Run Master 3-Way 500-epoch model (Mendeley + MP + JARVIS)
python scripts/44_train_500ep_mendeley_mp_jarvis.py

# Run JARVIS-alone 500-epoch model
python scripts/45_train_500ep_jarvis_alone.py
```

---

## 📜 License & Citation

Distributed under the **MIT License**.

If you use this codebase or dataset in your research, please cite:
```bibtex
@article{kutana2025representing,
  title={Representing Born effective charges with equivariant graph convolutional neural networks},
  author={Kutana, Alex and Shimizu, Koji and Watanabe, Satoshi and Asahi, Ryoji},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={1250},
  year={2025},
  publisher={Nature Publishing Group}
}
```

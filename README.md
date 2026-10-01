# Equivar BEC GCNN: Predicting Atomic Born Effective Charges across the Periodic Table

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-PyTorch--Geometric-orange.svg)](https://pyg.org/)
[![e3nn Equivariant GNN](https://img.shields.io/badge/e3nn-Equivariant%20GNN-purple.svg)](https://e3nn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the production codebase, curated datasets, 500-epoch deep learning benchmarks, and screening pipelines for **EquivarBECGNN**, an E(3)-equivariant graph neural network built with **PyTorch Geometric (PyG)** and **e3nn** to predict full $3 \times 3$ atomic Born Effective Charge ($Z^*_{ij}$) tensors directly from 3D crystal structures.

---

## 📊 Retrained 500-Epoch Production Benchmarks

All models were trained for **500 full epochs** using an E(3)-equivariant Graph Neural Network with spherical harmonics up to $\ell = 2$ (`0e+1o+2e` irreps), 32 Gaussian Cosine Envelope Radial Basis functions ($r_c = 5.0\text{ \AA}$), AdamW optimizer, Cosine Annealing learning rate schedule, and Acoustic Sum Rule (ASR) zero-charge drift symmetry enforcement:

| Experiment Benchmark | Crystals | Atoms | Diagonal MAE ($e$) | Off-Diagonal MAE ($e$) | Trace MAE ($e$) | **Overall Test MAE ($e$)** | Loss Plot |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Unified Master 3-Way Model (MP + Mendeley + JARVIS)** | **48,186** | **1,815,989** | **`0.3079`** | **`0.2275`** | **`0.1529`** | **`0.2543`** | [`loss_curve_combined.png`](combined_master/plots/loss_curve_combined_500ep_baseline.png) |
| **Mendeley Oxides Baseline** | 29,318 | 1,512,011 | **`0.2360`** | **`0.2240`** | **`0.0831`** | **`0.2280`** | [`loss_curve_mendeley.png`](mendeley_oxides/plots/loss_curve_mendeley_500ep_baseline.png) |
| **Materials Project Baseline** | 13,436 | 240,156 | **`0.6158`** | **`0.1885`** | **`0.4617`** | **`0.3309`** | [`loss_curve_mp.png`](materials_project/plots/loss_curve_mp_500ep_baseline.png) |
| **JARVIS-Alone Baseline** | 4,303 | 63,516 | **`0.6481`** | **`0.1609`** | **`0.5161`** | **`0.3233`** | [`loss_curve_jarvis.png`](jarvis/plots/loss_curve_jarvis_500ep_baseline.png) |

---

## 🔬 Dataset Deduplication & Crystallographic Diversity Metrics

1. **Cross-Dataset Deduplication**: Evaluated with `pymatgen.analysis.structure_matcher.StructureMatcher` (fractional tolerance $= 0.2$, lattice angle tolerance $= 5^\circ$). Confirmed **0 cross-dataset duplicate structures** between Materials Project, Mendeley, and JARVIS.
2. **Symmetry Coverage**: Covers **202 out of 230 space groups** ($87.8\%$), **all 7 crystal systems**, and **86 chemical elements**.
3. **Composite Diversity Index (CDI)**:
   $$\text{CDI} = \frac{1}{4}\left(H_{\text{elem}} + H_{\text{spg}} + H_{\text{sys}} + \frac{N_{\text{spg}}}{230}\right) = \mathbf{0.4340}$$

---

## ⚡ High-Throughput Discovery on Google DeepMind GNoME

The trained Unified Master Model was deployed on **Google DeepMind's GNoME dataset** (554,054 crystal structures) streamed directly from compressed archives in GPU memory:
- Filtered for confirmed semiconductors ($E_g > 0\,\text{eV}$ and $E_g \ne \infty$), yielding **385,895 target materials**.
- Evaluates the **full $3 \times 3$ tensor** (all 6 off-diagonal elements, Frobenius shear norms, and dynamical asymmetry metrics).
- Discovered verified novel giant anomalous BEC semiconductors ($|\bar{Z}^* - q_{\text{nominal}}| \ge 2.0\,e$) across gapped polar lattices.

---

## 📁 Repository Directory Structure

```
.
├── combined_master/        # Unified 3-way master benchmark, diversity metrics & verified anomaly catalogs
├── materials_project/      # Materials Project baseline summaries, atomic positions & model checkpoint
├── mendeley_oxides/        # Mendeley Oxides baseline summaries & model checkpoint
├── jarvis/                 # JARVIS baseline summaries, retrained model checkpoint & loss plots
├── gnome_discovery/        # GNoME semiconductor 3x3 predictions & verified candidate discoveries
└── scripts/                # Production training, screening, and inference scripts
```

## 🛠️ Production Scripts

```
scripts/
├── 50_train_500ep_mp_baseline.py               # Materials Project baseline training
├── 51_train_500ep_mendeley_baseline.py         # Mendeley Oxides baseline training
├── 52_train_500ep_jarvis_baseline.py           # Retrained JARVIS baseline training
├── 53_train_500ep_combined_baseline.py         # Unified 3-way combined master baseline training
├── audit_deduplication.py                      # Pymatgen StructureMatcher cross-dataset deduplication
├── extract_symmetry_and_diversity.py           # Space group distribution & CDI calculator
├── parallel_compute_nominal_oxi_and_anomalous_bec.py  # Parallel BVAnalyzer oxidation states & BEC screener
└── run_gnome_all_components_discovery.py       # High-throughput full 3x3 GNoME tensor discovery
```

---

## 🚀 Execution Guide

```bash
# Train Unified Master Baseline (48,186 crystals)
python scripts/53_train_500ep_combined_baseline.py

# Run Full 3x3 BEC Tensor Discovery on GNoME Semiconductors
python scripts/run_gnome_all_components_discovery.py --batch-size 15000 --out-suffix verified
```

---

## 📜 License
Distributed under the **MIT License**.

# Equivar BEC GCNN: Predicting Atomic Born Effective Charges across the Periodic Table

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-PyTorch--Geometric-orange.svg)](https://pyg.org/)
[![e3nn Equivariant GNN](https://img.shields.io/badge/e3nn-Equivariant%20GNN-purple.svg)](https://e3nn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the production codebase, curated datasets, 500-epoch deep learning benchmarks, and high-throughput screening pipelines for **EquivarBECGNN**, an $E(3)$-equivariant graph neural network built with **PyTorch Geometric (PyG)** and **e3nn** to predict full $3 \times 3$ atomic Born Effective Charge ($Z^*_{ij}$) tensors directly from 3D crystal structures.

---

## 📊 Retrained 500-Epoch Production Benchmarks

All models were trained for **500 full epochs** using an $E(3)$-equivariant Graph Neural Network with spherical harmonics up to $\ell = 2$ (`0e+1o+2e` irreps), 32 Gaussian Cosine Envelope Radial Basis functions ($r_c = 5.0\text{ \AA}$), AdamW optimizer, Cosine Annealing learning rate schedule, and Acoustic Sum Rule (ASR) zero-charge drift symmetry enforcement:

| Experiment Benchmark | Crystals | Atoms | Diagonal MAE ($e$) | Off-Diagonal MAE ($e$) | Trace MAE ($e$) | **Overall Test MAE ($e$)** | Loss Plot |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Unified Master 3-Way Model (MP + Mendeley + JARVIS)** | **48,186** | **1,815,989** | **`0.3079`** | **`0.2275`** | **`0.1529`** | **`0.2543`** | [`loss_curve_combined.png`](combined_master/plots/loss_curve_combined_500ep_baseline.png) |
| **Mendeley Oxides Baseline** | 29,318 | 1,512,011 | **`0.2360`** | **`0.2240`** | **`0.0831`** | **`0.2280`** | [`loss_curve_mendeley.png`](mendeley_oxides/plots/loss_curve_mendeley_500ep_baseline.png) |
| **Materials Project Baseline** | 13,436 | 240,156 | **`0.6158`** | **`0.1885`** | **`0.4617`** | **`0.3309`** | [`loss_curve_mp.png`](materials_project/plots/loss_curve_mp_500ep_baseline.png) |
| **JARVIS-Alone Baseline** | 4,303 | 63,516 | **`0.6481`** | **`0.1609`** | **`0.5161`** | **`0.3233`** | [`loss_curve_jarvis.png`](jarvis/plots/loss_curve_jarvis_500ep_baseline.png) |

---

## 🌐 High-Throughput Screening: Dataset Constraints, Stability & Bandgap Audit

High-throughput inference of full $3 \times 3$ BEC tensors was deployed across four major datasets. Below is the exact audit of thermodynamic stability ($E_{\text{hull}} \le 0$), electronic bandgaps ($E_g > 0$), and Acoustic Sum Rule (ASR) compliance:

| Dataset | Total Raw Entries | Stability Metric | Strict Hull ($E \le 0$) Count | Metastable ($E \le 0.05\text{ eV}$) | Bandgap ($E_g > 0$) Availability | Evaluated Stable Semiconductors | ASR Compliance ($\sum Z^* = 0$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GNoME** (Google DeepMind) | 554,054 | `Decomposition Energy Per Atom` | **544,596** (98.3%) | 544,596 (strict $\le 0$ used) | **YES** (400,460 with $E_g > 0$) | **385,490** (actinides $Z \ge 95$ removed) | **100.0%** (385,490 / 385,490) |
| **Alexandria** (PBEsol 3D) | 4,800,000 | `e_above_hull` | **48,022** (1.0%) | **230,204** (4.8%) | **YES** (~1.5M with $E_g > 0$; ~68% metals) | **230,204** (metastable) / **48,022** (strict ground state) | **100.0%** (230,204 / 230,204) |
| **OMat24** (Meta FAIR) | 100,544 | *Not stored in LMDB* | Mapped via Alexandria | Mapped via Alexandria | **Mapped via Alexandria parents** (7,492 semiconductors) | **99,976** (relaxed frames) / **7,492** (verified $E_g > 0$) | **100.0%** (99,976 / 99,976) |
| **AMMQ3** (Mixed + Parents) | 227 | OQMD convex hull | 227 (100.0%) | 227 (100.0%) | **YES** (All 227 have $E_g > 0$) | **227** | **100.0%** (227 / 227) |

### Key Scientific & Methodological Insights:
1. **Origin of Negative Convex Hull in GNoME ($E_{\text{decomp}} < 0$)**: Google DeepMind defined decomposition energy relative to the *pre-existing* Materials Project hull. Novel discoveries lying below the old hull have negative decomposition energy ($E < 0$).
2. **Absence of Negative Hull in Alexandria ($E_{\text{above hull}} \ge 0$)**: Alexandria computes the convex hull self-consistently over its own dataset. On a self-consistent convex hull, ground-state phases lie on the hull ($E = 0.000\text{ eV/atom}$); negative values are mathematically impossible.
3. **OMat24 Bandgap Derivation**: Meta FAIR designed OMat24 for potential fitting (forces/energies) and did not record bandgap $E_g$ in the ASE LMDB. However, **100.0% of parent structures are labeled with Alexandria IDs (`agm...`)**. Cross-referencing against Alexandria shows that **7.5% (7,492 frames)** are verified semiconductors ($E_g > 0$), while **92.5% (93,051 frames)** are metallic ($E_g = 0$).

---

## 🔍 Nominal Oxidation State Audit: Eliminating "Zero_Fallback" & Fractional Artifacts

Anomalous Born Effective Charge is defined as $\Delta Z^* = Z^*_{\text{predicted}} - q_{\text{nominal}}$. Our automated chemical assignment audit revealed:

| Dataset | Total Materials | Verified `BVAnalyzer` | `Composition_Guesses` | `Zero_Fallback` ($q_{\text{nom}} = 0$) | Genuine Anomalies (No Zero_Fallback) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GNoME** | 385,490 | 84,584 (21.9%) | 37,651 (9.8%) | 263,255 (68.3%) | **84,620** |
| **Alexandria** | 230,204 | 189,215 (82.2%) | 0 (0.0%) | 40,989 (17.8%) | **45,452** |
| **OMat24** | 99,976 | 23,707 (23.7%) | 0 (0.0%) | 76,269 (76.3%) | **11,047** |

* **Zero_Fallback Exclusion**: In complex intermetallics and borides (e.g. $\text{TaCuRe}$, $\text{Ho}_4\text{Al}_{18}\text{FePdPt}_4$), bond valence and composition algorithms cannot define formal valences, defaulting to $q_{\text{nominal}} = 0.0$. This artificially inflated anomalies ($|Z^* - 0| \ge 2.0e$). **Zero_Fallback materials are strictly excluded from clean parity plots and verified candidate catalogs.**
* **Fractional Charges (5.46% in GNoME)**: Caused by formula-averaging over mixed-valence elements (e.g., $\text{Fe}^{2.67+}$ in $\text{Fe}_3\text{O}_4$) or covalent clusters (e.g., $\text{B}^{-1.5}$ in $\text{B}_6$ boride cages). Clean parity plots filter these non-integer cases to guarantee discrete integer ticks on the $x$-axis.

---

## 📁 Repository Directory Structure

```
.
├── dataset_constraints_and_stability_audit_summary.csv   # Master 3-way stability & constraints audit
├── gnome_constraints_and_filtering_summary.csv           # GNoME filtering stage breakdown
├── alexandria_constraints_and_filtering_summary.csv      # Alexandria filtering stage breakdown
├── omat24_constraints_and_filtering_summary.csv          # OMat24 filtering stage breakdown
├── combined_master/        # Unified 3-way master benchmark & verified anomaly catalogs
├── materials_project/      # Materials Project baseline summaries & model checkpoint
├── mendeley_oxides/        # Mendeley Oxides baseline summaries & model checkpoint
├── jarvis/                 # JARVIS baseline summaries & retrained model checkpoint
├── gnome_discovery/        # GNoME semiconductor 3x3 predictions, parity plots & candidate discoveries
├── alexandria/             # Alexandria semiconductor 3x3 predictions & parity plots
├── omat24/                 # OMat24 relaxed trajectory 3x3 predictions & parity plots
└── scripts/                # Clean production training, screening, and plotting scripts
```

---

## 📜 License
Distributed under the **MIT License**.

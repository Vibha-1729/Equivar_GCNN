# Equivar Evaluation Pipeline: Predicting Atomic Born Effective Charges

This repository contains the evaluation pipeline for **Equivar**, an $O(3)$-equivariant graph convolutional neural network (GCNN) designed to predict tensors of atomic Born effective charges ($Z^*$) directly from 3D crystal structures. 

Based on the research paper:  
> **Representing Born effective charges with equivariant graph convolutional neural networks**  
> *Alex Kutana, Koji Shimizu, Satoshi Watanabe, and Ryoji Asahi* (*Scientific Reports*, 2025). [DOI: 10.1038/s41598-025-01250-5](https://doi.org/10.1038/s41598-025-01250-5)

---

## Key Features

* **Exact Physical Equivariance:** Utilizes spherical harmonics ($Y_{lm}$) and $O(3)$ irreducible representations ($0e \oplus 1e \oplus 2e$) to guarantee 100% rotational covariance under 3D spatial rotations.
* **Acoustic Sum Rule (ASR):** Enforces exact charge neutrality ($\sum_i Z^*_i = \mathbf{0}$) as an automated post-processing mean shift.
* **CPU-Optimized & PyTorch 2.x Compatible:** Includes custom TorchScript operator fallbacks, allowing fast inference on standard CPUs without requiring C++ compilation or GPU dependencies.

---

## Inputs and Outputs

* **Input:** 3D crystal structures in Extended XYZ (`extxyz`) format (`data/frames.xyz`).
* **Pre-trained Weights:** PyTorch JIT model weights (`BM1.pt` or `BM2.pt`).
* **Output:** $3 \times 3$ Born effective charge tensors per atom in CSV format (`data/evaluated.csv`):
  
  `Columns: [ids, Z11, Z12, Z13, Z21, Z22, Z23, Z31, Z32, Z33]`

---

## Quick Start / How to Run

### 1. Clone & Install
```bash
git clone https://github.com/Vibha-1729/Equivar_GCNN.git
cd Equivar_GCNN/equivar_eval
pip install -e .
```

### 2. Prepare Data & Configuration
Place your target crystal structure (`frames.xyz`) and model weights (`BM1.pt`) inside the `data/` directory, and update `config.yaml`:

```yaml
data_dir: "data"
saved_model_path: "data/BM1.pt"
ouput_path: "data/evaluated.csv"
```

### 3. Run Predictions
```bash
python -m equivar_eval.scripts.evaluate
```

---

## Reference & Citation
If you use this code or model in your research, please cite the original study:
```bibtex
@article{kutana2025representing,
  title={Representing Born effective charges with equivariant graph convolutional neural networks},
  author={Kutana, Alex and Shimizu, Koji and Watanabe, Satoshi and Asahi, Ryoji},
  journal={Scientific Reports},
  volume={15},
  year={2025},
  publisher={Nature Publishing Group}
}
```

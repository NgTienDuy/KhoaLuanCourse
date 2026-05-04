# Raman Physics-Informed AI MVP

> A physics-informed deep learning system for Raman spectroscopy that predicts
> compound composition, identifies chemical bonds, and flags out-of-distribution
> (OOD) samples — without sacrificing interpretability.

**Status:** 🚧 Active development (14-day MVP sprint, Day 1 of 14)

---

## What it does

Given a single 1D Raman spectrum (raw or preprocessed), the system returns:

1. **Normalized characteristic spectrum** (after AsLS + cosmic-ray + SG + SNV)
2. **Peak table** with position (cm⁻¹), intensity, FWHM, and assigned chemical bond/mode
3. **Quantitative composition** — % of each of 6 known compounds (Ala, Asn, Asp, Glu, His, GlcN)
4. **OOD score** + novelty flag for compounds outside the training set
5. **Reconstructed spectrum** computed from predicted composition (physics validation)

The differentiator vs. black-box baselines: a deterministic, inspectable
**bond-mapping engine** (`engine/`) plus a physics-based reconstruction loss
that enforces the Beer–Lambert linearity assumption.

---

## Quickstart

### 1. Clone & install

```bash
git clone <your-repo-url> Raman-Physics-AI
cd Raman-Physics-AI

# Python 3.10 recommended
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd.exe):
# .venv\Scripts\activate.bat
# macOS / Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note (CUDA users):** install PyTorch separately with the matching CUDA wheel
> from https://pytorch.org/get-started/locally/ **before** running
> `pip install -r requirements.txt`.

### 2. Place the dataset

Drop the original CSV into `data/raw/data.csv` (4378 × 1031).

### 3. Build processed splits

```bash
python scripts/prepare_data.py
```

### 4. Train

```bash
python -m src.training.train --config configs/train_config.yaml
```

### 5. Single-spectrum inference

```python
from src.inference.predict import predict
import numpy as np

spectrum = np.load("data/raw/example.npy")  # shape (1024,)
report = predict(spectrum)
print(report["composition"])       # dict of compound -> fraction
print(report["ood_flag"])          # True/False
print(report["peaks"])             # list of {wavenumber, intensity, fwhm, bond}
```

### 6. (Optional) Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

## Project layout

See [`docs/REPORT.md`](docs/REPORT.md) and the project structure document for
full folder rationale. Quick orientation:

| Folder | What lives here |
|---|---|
| `engine/` | **Deterministic** symbolic modules (bond DB, peak extraction, novelty). Not learned. |
| `src/` | **Learnable** code: data loaders, model, training loop, inference pipeline. |
| `configs/` | YAML hyperparameters and paths. One source of truth. |
| `data/raw/` | Original CSV (gitignored). |
| `data/processed/` | Cached preprocessed tensors (gitignored, regenerable). |
| `tests/` | pytest suite. |
| `results/` | Experiment outputs: training logs, benchmark tables, figures, reports. |
| `docs/` | Methodology report, theory derivations, thesis chapter. |

---

## Compounds supported (MVP)

| # | Name | Notes |
|---|---|---|
| 1 | Alanine (Ala) | Smallest amino acid, few discriminative peaks |
| 2 | Asparagine (Asn) | |
| 3 | Aspartic Acid (Asp) | |
| 4 | Glutamic Acid (Glu) | |
| 5 | Histidine (His) | Easiest — strong imidazole ring fingerprint |
| 6 | Glucosamine (GlcN) | Only non-amino-acid; pyranose ring at ~1080/1100 cm⁻¹ |

---

## Roadmap (14-day sprint)

- **Day 1–2 (Groundwork):** repo, dataloader, preprocessing, bond DB, SOTA review *(in progress)*
- **Day 3–6 (Modeling):** ResNet1D backbone, quantification head, reconstruction module, MC Dropout
- **Day 7 (Mid-checkpoint):** GO/NO-GO gate on validation MAE
- **Day 8–10 (Inference):** OOD scoring, peak extraction, symbolic mapping, report generator
- **Day 11–12 (Eval):** benchmark vs. PCA+SVM and ResNet-only baselines, sanity figures
- **Day 13 (Polish):** optional Streamlit dashboard, external benchmark
- **Day 14:** test suite green, REPORT.md complete, tag `v0.1.0-mvp`

---

## Citation

If you use this work, please cite:

```bibtex
@software{raman_physics_ai_2026,
  title  = {Raman Physics-Informed AI MVP},
  year   = {2026},
  note   = {Undergraduate thesis project},
  url    = {https://github.com/NgTienDuy/KhoaLuanCourse}
}
```

Domain references:
- De Gelder *et al.* (2007), *J. Raman Spectroscopy* — amino-acid peak assignments
- Karniadakis *et al.* (2021), *Nat. Rev. Phys.* — physics-informed ML methodology
- Socrates (2004), *Infrared and Raman Characteristic Group Frequencies*, 3rd ed.

---

## License

[MIT](LICENSE)

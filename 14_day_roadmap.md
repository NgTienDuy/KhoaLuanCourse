# Raman Physics-Informed AI MVP — 14-Day Roadmap

> **Working assumption:** ~8.5 productive hours per day. Total budget: ~120 hours.
> **Architecture:** Plan X (minimal), upgradeable to Plan Y if Day 7 checkpoint passes comfortably.
> **Critical path tasks** are marked with [CP]. Slipping any of these slips the whole MVP.

---

## Week 1: Foundations & Core Engine

### Day 1 — Bootstrap & Audit (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | T01: SOTA audit (review old 10 models) | 4 | `docs/sota_review.md` — gap analysis 1-pager |
| Afternoon | T02: New repo scaffold [CP] | 3 | New GitHub repo with structure, `requirements.txt`, README skeleton |
| Late | Read AlphaFold/Dreamer-style sections of provided survey for inspiration | 1 | Mental model |

**End-of-day checkpoint:** `git push` of clean skeleton. Repo has folders ready (see Cấu trúc thư mục document).

**If you fall behind:** Skip the survey reading; resume Day 2.

---

### Day 2 — Data Pipeline (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | T03: Port DataLoader from old repo [CP] | 5 | `src/data/dataloader.py` working, prints (4378, 1024) tensor |
| Afternoon | T04: Classical preprocessing module | 3 | `src/data/preprocess.py` — baseline + cosmic + smoothing + SNV |

**End-of-day checkpoint:** Run a smoke test: load CSV → preprocess → output shape verified. Save 5 example preprocessed spectra as PNG in `results/sanity/`.

**Stretch:** Start T05 (bond mapping DB review) if time permits.

---

### Day 3 — Reference Spectra & Splits (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | T06: Extract reference pure spectra (mean per compound) | 3 | `engine/reference_spectra.npy` (6, 1024) |
| Morning-Mid | T07: Implement split strategy A (composition OOD) [CP] | 4 | `src/data/splits.py` with `get_split_A()` returning train/val/test indices |
| Afternoon | T05: Verify bond mapping DB (use provided seed JSON) | 1 | `engine/bond_mapping.json` validated |

**End-of-day checkpoint:** 
- Reference spectra plotted (6 lines, distinct shapes) → `results/sanity/pure_spectra.png`
- Split sanity: train ≈ 3400, val ≈ 480, test ≈ 480 (for 60/20/20); or 42/6/6 vials.

**Stretch:** T09 metrics module skeleton.

---

### Day 4 — Backbone & Quantification Head (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | T10: 1D-ResNet backbone (port + adapt) [CP] | 6 | `src/models/backbone.py` — outputs 256-dim feature |
| Afternoon | T11: Quantification head with simplex output [CP] | 2 | `src/models/heads.py` — softmax, sum-to-1 |

**End-of-day checkpoint:**
- Forward pass on dummy input `(2, 1, 1024)` returns `(2, 256)` then `(2, 6)` summing to 1.
- No NaN, no shape errors.

**Watch for:** Dimensional mismatches (1024 vs your actual feature length). Hard-code, don't infer.

---

### Day 5 — Reconstruction & Physics Loss (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | T12: Reconstruction module [CP] | 5 | `src/models/reconstruction.py` — `s_recon = Σ(α_i · scale_i · pure_i)` |
| Afternoon | T13: Physics loss function | 3 | `src/training/losses.py` — `physics_loss = MSE(s_in, s_recon) + λ·cosine_dist` |

**End-of-day checkpoint:**
- Reconstruction module forward pass: input `α=(0.5, 0.3, 0.2, 0, 0, 0)` → output spectrum visually plausible (linear combo of first 3 references).
- Physics loss returns scalar > 0 on random input.

**Critical decision today:** Set initial `λ` (cosine weight) to 0.3. Will tune later. Set `β` (physics vs quant ratio) to 0.5 in combined loss.

---

### Day 6 — Training Pipeline (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| All day | T14: Combined loss + T15: Train loop v1 [CP] | 8 | `src/training/train.py` runs end-to-end on 1 epoch without crash |

**Components today:**
- Optimizer: Adam, lr=1e-3
- Scheduler: cosine annealing over 50 epochs
- Logging: wandb or simple CSV (`results/training_log.csv`)
- Checkpoint: save best val MAE to `checkpoints/best.pt`
- Augmentation T16: integrate into dataloader (random shift ±10 cm⁻¹, intensity ±10%, Gaussian noise σ=0.005)

**End-of-day checkpoint:** Run 5 epochs. Loss should decrease. Val MAE printed every epoch.

---

### Day 7 — MID-CHECKPOINT 🚨

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | Train for 30-50 epochs (let it run) | 0.5 active + 4 passive | Trained model, training curves |
| Afternoon | T17: Evaluate on val set [CP] | 4 | `results/midcheckpoint_report.md` |

**🚨 GO/NO-GO GATE 🚨**

| Metric | Pass | Borderline | Fail (REGROUP) |
|---|---|---|---|
| Val ID Accuracy | ≥ 85% | 70-85% | < 70% |
| Val Quantification MAE | ≤ 0.04 | 0.04-0.06 | > 0.06 |
| Loss curve | Monotonically decreasing | Plateau early | Spiking / NaN |

**Decision tree:**
- **All PASS:** Proceed to Day 8 (OOD module). Consider Plan Y upgrades for Day 13.
- **Borderline:** Proceed but DROP optional tasks (T08, T28, T29). Focus on must-haves only.
- **Fail:** STOP. Spend Day 8 morning debugging. Most likely culprits: (1) bug in physics loss sign, (2) bug in reconstruction (check pure spectra normalization), (3) learning rate too high.

**Critical:** Don't proceed to OOD until quantification is working. OOD on top of broken quant = double-broken.

---

## Week 2: OOD, Explanation, Benchmarking

### Day 8 — MC Dropout & OOD Score (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | T18: MC Dropout wrapper [CP] | 5 | `src/models/uncertainty.py` — `predict_with_uncertainty(model, x, n=50)` |
| Afternoon | T19: OOD score function | 3 | `src/inference/ood.py` — combined recon + variance score |

**End-of-day checkpoint:**
- Run on 10 ID samples + 10 OOD samples (using split B if implemented, else mock OOD by injecting Gaussian noise).
- Print OOD scores. Visually verify ID samples have lower scores.

**If split B (T08) wasn't done:** Implement minimal version today — hold out Histidine, retrain (or use existing model and just evaluate). Histidine has 4 discriminative peaks, easiest OOD case.

---

### Day 9 — Peak Extraction (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | T20: Peak extraction with Voigt fitting | 5 | `engine/peak_extractor.py` — input spectrum → list of (pos, intensity, FWHM) |
| Afternoon | T21: Symbolic mapper (peak → bond DB) | 3 | `engine/symbolic_mapper.py` — input peak list → enriched annotations |

**Tools:**
- `scipy.signal.find_peaks` for initial detection (height + prominence thresholds)
- `lmfit.models.VoigtModel` for refined fitting per peak
- Tolerance window from `bond_mapping.json` (per-entry `tolerance_cm-1`)

**End-of-day checkpoint:**
- Run pipeline on 1 pure Alanine spectrum: should detect ~5-8 peaks, all matched to DB.
- Run on 1 pure Histidine: should detect P004, P008, P014, P015 (the discriminative 4).
- Save annotated plots to `results/sanity/peak_demo_<compound>.png`.

---

### Day 10 — Novelty Localization (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | T22: Novelty peak localizer | 4 | `engine/novelty_locator.py` — flags unmatched peaks |
| Afternoon | Integration test: full predict() pipeline | 4 | `src/inference/predict.py` ties it all together |

**`predict(spectrum)` returns:**
```python
{
  "preprocessed_spectrum": np.array,
  "composition": {"Alanine": 0.42, "Glycine": 0.38, ...},
  "reconstructed_spectrum": np.array,
  "reconstruction_cosine_sim": 0.97,
  "ood_score": 0.12,
  "ood_flag": False,  # True if score > threshold
  "peaks": [{"position": 1003, "intensity": 0.8, "fwhm": 12, "matched_to": "P004", "bond": "imidazole ring breathing", "compounds": ["Histidine"]}, ...],
  "novel_peaks": [],  # Non-empty only if OOD
  "metadata": {"model_version": "0.1", "timestamp": "..."}
}
```

**End-of-day checkpoint:** Call `predict()` on 3 test samples (1 ID, 1 mild OOD, 1 fully novel). All return valid dicts. Save outputs to `results/predict_demos/`.

---

### Day 11 — Benchmarks (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | T23: PCA+SVM baseline | 4 | `src/models/baselines/pca_svm.py` trained, evaluated |
| Afternoon | T24: 1D-ResNet without physics baseline + T25: Comparison table | 4 | `results/benchmark_table.md`, `results/benchmark_table.csv` |

**Comparison matrix:**

| Model | ID Acc | Quant MAE | OOD AUROC | CVR | Inference Time |
|---|---|---|---|---|---|
| PCA+SVM | ? | ? | N/A | High | Fast |
| 1D-ResNet (no physics) | ? | ? | ? | High | Medium |
| **Ours (physics-informed)** | **?** | **?** | **?** | **Low** | Medium |

**End-of-day checkpoint:** Numbers populated. Bold cells where "Ours" wins. Save plots: confusion matrix, MAE bar chart, OOD ROC curves.

---

### Day 12 — Visualization & Report Generator (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | T26: Report generator (Markdown + JSON) | 4 | `src/inference/report.py` — `generate_report(predict_output) → md_string` |
| Afternoon | T27: Visualization module | 4 | `src/inference/visualize.py` — overlay plots, peak annotations |

**Report template (per spectrum):**
```
# Raman Spectrum Analysis Report
**Date:** 2026-XX-XX
**Sample ID:** xxx

## Composition
- Alanine: 42.3% (uncertainty: ±2.1%)
- Glycine: 38.1% (uncertainty: ±1.8%)
- ...

## Peak Analysis
Detected 12 peaks. 11 matched to known compounds, 1 unmatched.

| Position (cm⁻¹) | Intensity | FWHM | Bond/Mode | Compound |
|---|---|---|---|---|
| 1003.2 | 0.82 | 11.5 | Imidazole ring breathing | Histidine |
...

## Physics Validation
- Reconstruction cosine similarity: 0.971
- Constraint Violation: ❌ Within tolerance ✓

## OOD Assessment
- OOD Score: 0.12 (threshold: 0.35)
- Status: IN-DISTRIBUTION ✓
- Novel peaks detected: 0

## Visualization
[input vs reconstructed plot]
[peak annotation plot]
```

**End-of-day checkpoint:** Generate reports for 3 demo samples. PDFs saved to `results/reports/demo_*.pdf` (use markdown-to-pdf or just markdown).

---

### Day 13 — STRETCH or POLISH (≈8h)

**If Day 7 was PASS:** Pick ONE of:

| Option | Task | Hours | Value |
|---|---|---|---|
| **A (recommended)** | T28: Streamlit dashboard | 6-8 | Highest impact for thesis defense |
| B | T29: VAE augmentation | 6-8 | Improves model quality, may bump accuracy 1-2% |
| C | T30: External dataset test (Bacteria-ID, RRUFF) | 4-6 | Strong evidence for novelty detection |
| D | T31: Architecture Y upgrade (add VAE branch) | 8 | Risky; may break working pipeline |

**If Day 7 was BORDERLINE:** Skip stretch. Use Day 13 to polish:
- Re-run experiments with better hyperparameters
- Generate more visualizations
- Write more thorough documentation

**End-of-day checkpoint:** Whatever you picked, must be working end-to-end. Don't leave half-finished features.

---

### Day 14 — Documentation & Defense Prep (≈8h)

| Slot | Task | Hours | Deliverable |
|---|---|---|---|
| Morning | T32: Test suite | 4 | `tests/` folder, `pytest` runs green |
| Afternoon | T33: English methodology report | 4 | `docs/REPORT.md` (2-3 pages) — ready for thesis chapter or paper |

**Final report structure:**
1. **Abstract** (200 words)
2. **Introduction & Motivation** (problem: black-box AI in spectral analysis)
3. **Method** (architecture diagram, loss formulation, training protocol)
4. **Experiments** (datasets, splits, baselines)
5. **Results** (the benchmark table from Day 11)
6. **Discussion** (when does it work, when does it fail, future directions)
7. **References**

**Concurrent T34:** Update README.md with:
- One-paragraph project summary
- Quickstart (3-5 commands to reproduce results)
- Citation block
- Acknowledgments

**End-of-day checkpoint:** 
- `git tag v0.1.0-mvp` and push.
- Send self a "completion email" with links to: repo, results folder, report PDF.

---

## Risk Management & Contingencies

### Top 5 Risks (in order of likelihood)

| Risk | Probability | Mitigation |
|---|---|---|
| Day 7 fails (model not converging) | 30% | Have simplified version of architecture ready: pure regression head, no physics loss. Drop to that. |
| OOD module gives random results (Day 8-9) | 25% | Fall back to pure reconstruction error as OOD score. Skip MC Dropout if it doesn't work in 4h. |
| Peak fitting unstable (Voigt convergence issues) | 20% | Use Lorentzian-only or just `find_peaks` parameters without fitting. Note limitation in report. |
| Streamlit dashboard takes too long | 30% | Start it on Day 13; if not 70% done by end of day, scrap and use Jupyter notebook demo instead. |
| External dataset format incompatibility | 50% | Skip T30 unless you have data already downloaded by Day 11. |

### Hard Stops (where to call it good enough)

- If Day 9 ends with peak extraction kinda-working but glitchy on edge cases → ship it, note in report.
- If reconstruction cosine sim is 0.85-0.92 (not 0.97) → ship it, lower the bar in report description ("≥ 0.85 considered acceptable").
- If you only get ID metrics done and OOD is rough → ship ID story, note OOD as "future work."

---

## Daily Standup Template (for self-discipline)

Each morning, write 3 lines:

```
Hôm nay tôi sẽ: [task ID + name, e.g., T15 Train loop v1]
Tôi đã làm hôm qua: [recap]
Blocker: [bug? unclear spec? tired? — be honest]
```

Each evening, before closing laptop:

```
Done: [✓/✗ for each planned task]
Lessons: [1-2 lines about what surprised you]
Tomorrow's first 30min: [exact next action so morning is frictionless]
```

This is the difference between a 14-day MVP and a 21-day MVP.

---

## Final Recommendation

**Stick to the plan. Resist scope creep.** Every paradigm you skip (EGNN, Neural Operators, Geometric AI) is a paragraph in your thesis future-work section, not a failure. The MVP that ships > the MVP that's perfect.

When Day 14 ends and you push the final commit, you should have:

1. ✅ Working physics-informed model with quantification + OOD
2. ✅ Bond mapping & symbolic explanation
3. ✅ Benchmarks vs 2 baselines
4. ✅ Reports for demo samples
5. ✅ Test suite
6. ✅ English methodology document
7. ✅ (Maybe) Streamlit dashboard
8. ✅ Clear v2 roadmap (the things you didn't get to)

That's a thesis. That's a publication. That's a foundation.

Good luck.

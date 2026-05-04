# data/raw/

Place the original `data.csv` (4378 × 1031) here.

This directory is **gitignored** for the actual data files. Only this README
and a `.gitkeep` are tracked.

## Expected file
- `data.csv` — header row + 4378 sample rows
  - columns 1–1024: Raman intensities at wavenumbers 801.62 → 931.28 cm⁻¹
  - column 1025: `vial #` (sample identifier; `a01..a48` for mixtures, full compound names for pure)
  - columns 1026–1031: mixing ratios for Alanine, Asparagine, Aspartic Acid, Glutamic Acid, Histidine, Glucosamine

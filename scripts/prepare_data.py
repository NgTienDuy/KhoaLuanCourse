"""One-time data preparation script.

Steps:
    1. Load `data/raw/data.csv` via `src.data.dataloader.load_raw_csv`
    2. Build the requested split scheme(s) per `configs/data_config.yaml`
    3. Persist each split to `data/splits/split_{name}.json`
    4. (NEW in T04) Apply the classical preprocessing pipeline to all spectra
    5. (NEW in T04) Cache preprocessed tensors to `data/processed/spectra_full.pt`
       (single file with the full preprocessed array; downstream code slices by
       split indices). This is much smaller and cleaner than per-split caches.

Usage
-----
    # Defaults (split scheme from data_config.yaml, preprocessing on):
    python scripts/prepare_data.py

    # Override split scheme:
    python scripts/prepare_data.py --scheme A
    python scripts/prepare_data.py --scheme all

    # Skip preprocessing cache (only build splits):
    python scripts/prepare_data.py --no-preprocess

Output files
------------
    data/splits/split_A.json
    data/splits/split_A_prime.json
    data/splits/split_B.json
    data/processed/spectra_full.pt           — preprocessed (N, P) torch.float32
    data/processed/labels.pt                 — (N, C) torch.float32
    data/processed/wavenumbers.npy           — (P,) float64 (Raman shift cm⁻¹)
    data/processed/preprocess_meta.json      — pipeline hyperparameters used
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Make `src` importable when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.data.dataloader import load_raw_csv  # noqa: E402
from src.data.splits import select_split, save_split_to_json  # noqa: E402

log = logging.getLogger("prepare_data")


def _save_torch_tensor(arr: np.ndarray, path: Path) -> None:
    """Save a numpy array as a torch tensor (lazy import to avoid hard dep)."""
    import torch
    t = torch.from_numpy(arr.astype(np.float32, copy=False))
    torch.save(t, path)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Prepare splits and preprocessed cache.")
    parser.add_argument("--default-config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs/data_config.yaml"))
    parser.add_argument("--scheme", default=None,
                        help="Override split scheme. One of: A, A_prime, B, all.")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip the preprocessing cache step (only build splits).")
    args = parser.parse_args()

    with open(args.default_config) as f:
        defaults = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    if args.scheme is not None:
        data_cfg["split"]["scheme"] = args.scheme

    # ── Step 1: load raw CSV ──
    log.info("Step 1: load CSV")
    table = load_raw_csv(
        csv_path=PROJECT_ROOT / defaults["paths"]["data_raw_csv"],
        compound_full_names=defaults["compounds"]["full_names"],
        laser_wl_nm=defaults["wavenumber"]["laser_wavelength_nm"],
        expected_num_points=defaults["wavenumber"]["expected_num_points"],
    )

    # ── Step 2 & 3: build + save splits ──
    log.info("Step 2: build split(s)")
    splits = select_split(table, data_cfg)
    log.info(f"  Built {len(splits)} split(s): {list(splits.keys())}")

    log.info("Step 3: save splits to data/splits/")
    splits_dir = PROJECT_ROOT / defaults["paths"]["data_splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, sp in splits.items():
        out_path = splits_dir / f"split_{name}.json"
        save_split_to_json(sp, out_path)

    # ── Step 4: classical preprocessing (full table) ──
    if args.no_preprocess:
        log.info("Step 4 SKIPPED — flag --no-preprocess.")
        return 0

    log.info("Step 4: apply classical preprocessing to all 4378 spectra "
             "(cosmic → AsLS → SG → SNV)")
    from src.data.preprocess import preprocess_batch

    pp_cfg = data_cfg.get("preprocessing", {})
    pp_kwargs = dict(
        cosmic_threshold=float(pp_cfg.get("cosmic_threshold", 5.0)),
        asls_lam=float(pp_cfg.get("asls_lam", 1.0e5)),
        asls_p=float(pp_cfg.get("asls_p", 0.01)),
        asls_max_iter=int(pp_cfg.get("asls_max_iter", 30)),
        savgol_window=int(pp_cfg.get("savgol_window", 11)),
        savgol_polyorder=int(pp_cfg.get("savgol_polyorder", 3)),
        snv_eps=float(pp_cfg.get("snv_eps", 1.0e-8)),
    )
    log.info(f"  pipeline kwargs: {pp_kwargs}")

    t0 = time.time()
    spectra_pp = preprocess_batch(table.spectra, is_preprocessed=False, **pp_kwargs)
    elapsed = time.time() - t0
    log.info(f"  done in {elapsed:.1f}s  (~{elapsed/table.num_samples*1000:.1f} ms/spectrum)")

    # ── Step 5: cache to data/processed/ ──
    log.info("Step 5: cache preprocessed tensors to data/processed/")
    proc_dir = PROJECT_ROOT / defaults["paths"]["data_processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    spec_path = proc_dir / "spectra_full.pt"
    label_path = proc_dir / "labels.pt"
    wn_path = proc_dir / "wavenumbers.npy"
    meta_path = proc_dir / "preprocess_meta.json"
    vial_path = proc_dir / "vial_ids.npy"

    try:
        _save_torch_tensor(spectra_pp, spec_path)
        _save_torch_tensor(table.labels, label_path)
    except ImportError:
        log.warning("  torch not installed; saving as .npy instead.")
        spec_path = spec_path.with_suffix(".npy")
        label_path = label_path.with_suffix(".npy")
        np.save(spec_path, spectra_pp.astype(np.float32, copy=False))
        np.save(label_path, table.labels.astype(np.float32, copy=False))

    np.save(wn_path, table.wavenumbers.astype(np.float64, copy=False))
    np.save(vial_path, table.vial_ids)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "pipeline_kwargs": pp_kwargs,
            "num_spectra": int(table.num_samples),
            "num_points": int(table.num_points),
            "num_compounds": int(table.num_compounds),
            "preprocessing_seconds": round(elapsed, 2),
            "compound_names": table.compound_names,
        }, f, indent=2)

    for p in [spec_path, label_path, wn_path, vial_path, meta_path]:
        if p.exists():
            log.info(f"  ✓ saved {p.relative_to(PROJECT_ROOT)} "
                     f"({p.stat().st_size / 1024:.0f} KB)")

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

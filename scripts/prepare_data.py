"""One-time data preparation script.

What this script does TODAY (Day 1, after T03):
    1. Load `data/raw/data.csv` via `src.data.dataloader.load_raw_csv`
    2. Build the requested split scheme(s) per `configs/data_config.yaml`
    3. Persist each split to `data/splits/split_{name}.json`

What this script will additionally do AFTER T04:
    4. Apply the classical preprocessing pipeline to all spectra
    5. Cache the preprocessed tensors to `data/processed/{train,val,test}.pt`

Usage
-----
    # Default configs (configs/default.yaml + configs/data_config.yaml):
    python scripts/prepare_data.py

    # Override the split scheme on the command line:
    python scripts/prepare_data.py --scheme A
    python scripts/prepare_data.py --scheme all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `src` importable when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from src.data.dataloader import load_raw_csv  # noqa: E402
from src.data.splits import select_split, save_split_to_json  # noqa: E402

log = logging.getLogger("prepare_data")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Prepare splits and (later) preprocessed tensors.")
    parser.add_argument("--default-config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--data-config",    default=str(PROJECT_ROOT / "configs/data_config.yaml"))
    parser.add_argument("--scheme", default=None,
                        help="Override split scheme. One of: A, A_prime, B, all.")
    args = parser.parse_args()

    with open(args.default_config) as f:
        defaults = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    if args.scheme is not None:
        data_cfg["split"]["scheme"] = args.scheme

    log.info("Step 1: load CSV")
    table = load_raw_csv(
        csv_path=PROJECT_ROOT / defaults["paths"]["data_raw_csv"],
        compound_full_names=defaults["compounds"]["full_names"],
        laser_wl_nm=defaults["wavenumber"]["laser_wavelength_nm"],
        expected_num_points=defaults["wavenumber"]["expected_num_points"],
    )

    log.info("Step 2: build split(s)")
    splits = select_split(table, data_cfg)
    log.info(f"  Built {len(splits)} split(s): {list(splits.keys())}")

    log.info("Step 3: save splits to data/splits/")
    splits_dir = PROJECT_ROOT / defaults["paths"]["data_splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, sp in splits.items():
        out_path = splits_dir / f"split_{name}.json"
        save_split_to_json(sp, out_path)

    # Step 4 & 5 are reserved for after T04 (preprocessing pipeline).
    log.info(
        "Step 4 (preprocess + cache to data/processed/) — DEFERRED to after T04 "
        "(see src/data/preprocess.py)."
    )
    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

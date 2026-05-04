"""Three reproducible split schemes for the Raman amino-acid dataset.

Schemes
-------
**A — composition-level OOD** (default per project spec):
    Group rows by `vial #`. Randomly assign whole vials to train / val / test
    in proportions (num_train_vials, num_val_vials, num_test_vials), e.g.
    42 / 6 / 6. The test set sees compositions never present in training.

**A' — sample-level random**:
    Plain shuffled split at the row level (60 / 20 / 20 by default).
    Less rigorous OOD but more data per split. Good for sanity-check upper-bound.

**B — component-level OOD**:
    Hold out ALL rows whose `vial #` produces (or contains) one specific
    compound — e.g. Histidine. Test set therefore contains a compound the
    model never saw during training. The most aggressive OOD.

Each scheme produces a `SplitIndices` instance with three numpy arrays of
row indices into the full `RawSpectraTable`. They are also serialisable to
JSON via `save_split_to_json()` so splits are reproducible across runs.

Author: Day-1 sprint (T03)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.data.dataloader import RawSpectraTable, SplitIndices, is_pure_vial

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────────────────────────

def _filter_pure(
    table: RawSpectraTable,
    include_pure: bool,
) -> np.ndarray:
    """Return the row indices eligible to participate in the split.

    If `include_pure` is False, pure-compound rows are excluded.
    """
    eligible = np.arange(table.num_samples)
    if include_pure:
        return eligible
    pure_mask = table.pure_mask()
    excluded = int(pure_mask.sum())
    log.info(f"  Excluding {excluded} pure-sample rows from split eligibility.")
    return eligible[~pure_mask]


def _vial_to_rows_map(
    table: RawSpectraTable,
    eligible_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map each unique vial-id → array of its row indices, restricted to eligible rows."""
    out: dict[str, np.ndarray] = {}
    eligible_set = set(eligible_indices.tolist())
    for vial in np.unique(table.vial_ids):
        rows = np.where(table.vial_ids == vial)[0]
        rows = rows[np.isin(rows, list(eligible_set))]
        if rows.size > 0:
            out[str(vial)] = rows
    return out


def _vial_contains_compound(
    table: RawSpectraTable,
    vial: str,
    compound_name: str,
    threshold: float = 0.0,
) -> bool:
    """True if any row with this vial-id has the given compound's fraction > threshold."""
    if compound_name not in table.compound_names:
        raise ValueError(
            f"Unknown compound '{compound_name}'. "
            f"Available: {table.compound_names}"
        )
    col_idx = table.compound_names.index(compound_name)
    rows = np.where(table.vial_ids == vial)[0]
    if rows.size == 0:
        return False
    return bool((table.labels[rows, col_idx] > threshold).any())


# ───────────────────────────────────────────────────────────────────────────
#  Scheme A — composition-level OOD (whole-vial groups)
# ───────────────────────────────────────────────────────────────────────────

def split_A_composition_ood(
    table: RawSpectraTable,
    *,
    num_train_vials: int = 42,
    num_val_vials: int = 6,
    num_test_vials: int = 6,
    include_pure: bool = True,
    seed: int = 42,
) -> SplitIndices:
    """Split by whole vials. Test/val vials hold compositions not seen at training time."""
    eligible = _filter_pure(table, include_pure)
    vial_to_rows = _vial_to_rows_map(table, eligible)
    vials = sorted(vial_to_rows.keys())
    n = len(vials)
    requested = num_train_vials + num_val_vials + num_test_vials
    if requested > n:
        raise ValueError(
            f"Scheme A: requested {requested} vials "
            f"({num_train_vials}+{num_val_vials}+{num_test_vials}) "
            f"but only {n} eligible vials exist."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(vials)
    train_vials = perm[:num_train_vials]
    val_vials   = perm[num_train_vials : num_train_vials + num_val_vials]
    test_vials  = perm[num_train_vials + num_val_vials :
                       num_train_vials + num_val_vials + num_test_vials]

    train_idx = np.concatenate([vial_to_rows[v] for v in train_vials])
    val_idx   = np.concatenate([vial_to_rows[v] for v in val_vials])
    test_idx  = np.concatenate([vial_to_rows[v] for v in test_vials])

    sp = SplitIndices(
        train=np.sort(train_idx),
        val=np.sort(val_idx),
        test=np.sort(test_idx),
        scheme="A_composition_ood",
    )
    log.info(
        f"  Scheme A: {len(train_vials)} train vials / {len(val_vials)} val / "
        f"{len(test_vials)} test → {sp.summary()}"
    )
    return sp


# ───────────────────────────────────────────────────────────────────────────
#  Scheme A' — sample-level random
# ───────────────────────────────────────────────────────────────────────────

def split_A_prime_random(
    table: RawSpectraTable,
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    include_pure: bool = True,
    seed: int = 42,
) -> SplitIndices:
    """Plain random row-level split. Spectra from the same vial may appear
    across train/val/test."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Scheme A': fractions must sum to 1.0, got "
            f"{train_frac + val_frac + test_frac}"
        )
    eligible = _filter_pure(table, include_pure)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(eligible)
    n = perm.size
    cut1 = int(round(train_frac * n))
    cut2 = int(round((train_frac + val_frac) * n))
    sp = SplitIndices(
        train=np.sort(perm[:cut1]),
        val=np.sort(perm[cut1:cut2]),
        test=np.sort(perm[cut2:]),
        scheme="A_prime_random",
    )
    log.info(f"  Scheme A': {sp.summary()}")
    return sp


# ───────────────────────────────────────────────────────────────────────────
#  Scheme B — component-level OOD (hold out one compound entirely)
# ───────────────────────────────────────────────────────────────────────────

def split_B_component_ood(
    table: RawSpectraTable,
    *,
    holdout_compound: str = "Histidine",
    val_frac_within_train_pool: float = 0.15,
    include_pure: bool = True,
    seed: int = 42,
) -> SplitIndices:
    """Reserve every vial that contains `holdout_compound` (>0) for the TEST set.

    Of the remaining (train-pool) vials, a fraction `val_frac_within_train_pool`
    is set aside for validation; the rest is the training set. This guarantees
    the model never sees the held-out compound at training or validation time.
    """
    eligible = _filter_pure(table, include_pure)
    vial_to_rows = _vial_to_rows_map(table, eligible)
    vials = sorted(vial_to_rows.keys())

    test_vials, train_pool_vials = [], []
    for v in vials:
        if _vial_contains_compound(table, v, holdout_compound, threshold=0.0):
            test_vials.append(v)
        else:
            train_pool_vials.append(v)

    if not test_vials:
        raise ValueError(
            f"Scheme B: no eligible vial contains '{holdout_compound}'. "
            f"Cannot build holdout test set."
        )
    if not train_pool_vials:
        raise ValueError(
            f"Scheme B: every eligible vial contains '{holdout_compound}'. "
            f"Train pool would be empty."
        )

    rng = np.random.default_rng(seed)
    train_pool_vials = list(rng.permutation(train_pool_vials))
    n_val = max(1, int(round(val_frac_within_train_pool * len(train_pool_vials))))
    val_vials   = train_pool_vials[:n_val]
    train_vials = train_pool_vials[n_val:]

    train_idx = np.concatenate([vial_to_rows[v] for v in train_vials])
    val_idx   = np.concatenate([vial_to_rows[v] for v in val_vials])
    test_idx  = np.concatenate([vial_to_rows[v] for v in test_vials])

    sp = SplitIndices(
        train=np.sort(train_idx),
        val=np.sort(val_idx),
        test=np.sort(test_idx),
        scheme=f"B_component_ood_holdout={holdout_compound}",
    )
    log.info(
        f"  Scheme B (holdout={holdout_compound}): {len(train_vials)} train vials / "
        f"{len(val_vials)} val / {len(test_vials)} test → {sp.summary()}"
    )
    return sp


# ───────────────────────────────────────────────────────────────────────────
#  JSON persistence — reproducibility across runs / scripts
# ───────────────────────────────────────────────────────────────────────────

def save_split_to_json(split: SplitIndices, path: str | Path) -> None:
    """Persist a split to a JSON file (lists of int indices)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scheme": split.scheme,
        "train_indices": split.train.tolist(),
        "val_indices":   split.val.tolist(),
        "test_indices":  split.test.tolist(),
        "n_train":       int(len(split.train)),
        "n_val":         int(len(split.val)),
        "n_test":        int(len(split.test)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info(f"  Saved split to {path}")


def load_split_from_json(path: str | Path) -> SplitIndices:
    """Load a previously-saved split."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return SplitIndices(
        train=np.array(payload["train_indices"], dtype=np.int64),
        val=np.array(payload["val_indices"], dtype=np.int64),
        test=np.array(payload["test_indices"], dtype=np.int64),
        scheme=payload.get("scheme", ""),
    )


# ───────────────────────────────────────────────────────────────────────────
#  Convenience: build all three splits in one call
# ───────────────────────────────────────────────────────────────────────────

def build_all_splits(
    table: RawSpectraTable,
    config: dict,
) -> dict[str, SplitIndices]:
    """Build A, A', and B splits using settings from `config['split']`.

    Parameters
    ----------
    config
        Parsed `data_config.yaml` dictionary.

    Returns
    -------
    Dictionary {'A': ..., 'A_prime': ..., 'B': ...}.
    """
    cfg = config["split"]
    include_pure = bool(cfg.get("include_pure_in_split", True))

    A = split_A_composition_ood(
        table,
        num_train_vials=cfg["A"]["num_train_vials"],
        num_val_vials=cfg["A"]["num_val_vials"],
        num_test_vials=cfg["A"]["num_test_vials"],
        include_pure=include_pure,
        seed=cfg["A"]["seed"],
    )
    A_prime = split_A_prime_random(
        table,
        train_frac=cfg["A_prime"]["train_frac"],
        val_frac=cfg["A_prime"]["val_frac"],
        test_frac=cfg["A_prime"]["test_frac"],
        include_pure=include_pure,
        seed=cfg["A_prime"]["seed"],
    )
    B = split_B_component_ood(
        table,
        holdout_compound=cfg["B"]["holdout_compound"],
        val_frac_within_train_pool=cfg["B"].get("val_frac_within_train_pool", 0.15),
        include_pure=include_pure,
        seed=cfg["B"]["seed"],
    )
    return {"A": A, "A_prime": A_prime, "B": B}


def select_split(
    table: RawSpectraTable,
    config: dict,
) -> dict[str, SplitIndices]:
    """Return one or all splits depending on `config['split']['scheme']`.

    * scheme="A"        → {"A": ...}
    * scheme="A_prime"  → {"A_prime": ...}
    * scheme="B"        → {"B": ...}
    * scheme="all"      → all three (uses build_all_splits)
    """
    scheme = config["split"]["scheme"]
    if scheme == "all":
        return build_all_splits(table, config)
    elif scheme == "A":
        return {"A": split_A_composition_ood(
            table,
            num_train_vials=config["split"]["A"]["num_train_vials"],
            num_val_vials=config["split"]["A"]["num_val_vials"],
            num_test_vials=config["split"]["A"]["num_test_vials"],
            include_pure=config["split"].get("include_pure_in_split", True),
            seed=config["split"]["A"]["seed"],
        )}
    elif scheme == "A_prime":
        return {"A_prime": split_A_prime_random(
            table,
            train_frac=config["split"]["A_prime"]["train_frac"],
            val_frac=config["split"]["A_prime"]["val_frac"],
            test_frac=config["split"]["A_prime"]["test_frac"],
            include_pure=config["split"].get("include_pure_in_split", True),
            seed=config["split"]["A_prime"]["seed"],
        )}
    elif scheme == "B":
        return {"B": split_B_component_ood(
            table,
            holdout_compound=config["split"]["B"]["holdout_compound"],
            val_frac_within_train_pool=config["split"]["B"].get(
                "val_frac_within_train_pool", 0.15),
            include_pure=config["split"].get("include_pure_in_split", True),
            seed=config["split"]["B"]["seed"],
        )}
    else:
        raise ValueError(f"Unknown split scheme: {scheme!r}")

"""Symbolic peak → bond mapping.

The `BondMapper` class loads the seed bond database (`engine/bond_mapping.json`)
and exposes deterministic lookup methods used by:

* the inference report generator (annotate detected peaks with bond names)
* the novelty locator (decide which detected peaks are unmatched)
* the chemistry-feature post-processor for thesis figures

Nothing here is learned — every output is a direct function of the JSON DB
plus the input wavenumber. That deterministic, inspectable nature is the
project's interpretability anchor: the user can edit the JSON and the model's
explanations update with no retraining required.

Author: Day-2 sprint (T05).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
#  DB entry dataclass — typed, serialisable, easy to reason about
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BondEntry:
    """One row of the bond-mapping database."""
    id: str
    wavenumber_cm_inv: float
    tolerance_cm_inv: float
    bond: str
    mode: str
    compounds: tuple[str, ...]
    discriminative_for: tuple[str, ...]
    notes: str = ""

    def matches(self, wavenumber: float, tolerance: float | None = None) -> bool:
        """True iff `wavenumber` is within this entry's tolerance window."""
        tol = self.tolerance_cm_inv if tolerance is None else tolerance
        return abs(wavenumber - self.wavenumber_cm_inv) <= tol

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "wavenumber_cm_inv": self.wavenumber_cm_inv,
            "tolerance_cm_inv": self.tolerance_cm_inv,
            "bond": self.bond,
            "mode": self.mode,
            "compounds": list(self.compounds),
            "discriminative_for": list(self.discriminative_for),
            "notes": self.notes,
        }


# ───────────────────────────────────────────────────────────────────────────
#  BondMapper
# ───────────────────────────────────────────────────────────────────────────

class BondMapper:
    """Read-only lookup over the bond-mapping JSON database.

    Construction is light (parses ~30 entries); the resulting object is
    inexpensive to keep around. The DB is treated as immutable after load —
    edit the JSON and re-instantiate to refresh.

    Examples
    --------
    >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
    >>> hits = mapper.match_peak(1003)
    >>> hits[0].id
    'P004'
    >>> hits[0].bond
    'Imidazole ring breathing'
    """

    def __init__(
        self,
        entries: Sequence[BondEntry],
        *,
        compound_canonical_names: Sequence[str] | None = None,
        tolerance_default_cm_inv: float = 8.0,
        schema_version: str = "1.0",
    ) -> None:
        self.entries: list[BondEntry] = list(entries)
        self.tolerance_default_cm_inv = float(tolerance_default_cm_inv)
        self.schema_version = schema_version
        self.compound_canonical_names: list[str] = (
            list(compound_canonical_names) if compound_canonical_names else []
        )
        # Build helper indexes
        self._by_id: dict[str, BondEntry] = {e.id: e for e in self.entries}
        self._by_compound: dict[str, list[BondEntry]] = {}
        for e in self.entries:
            for c in e.compounds:
                self._by_compound.setdefault(c, []).append(e)

    # ── Constructors ─────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str | Path) -> "BondMapper":
        """Load a BondMapper from a JSON file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Bond DB not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "BondMapper":
        """Build a BondMapper from a parsed dictionary (used by from_json)."""
        if "entries" not in data:
            raise ValueError("Bond DB JSON missing 'entries' key.")
        entries = [
            BondEntry(
                id=e["id"],
                wavenumber_cm_inv=float(e["wavenumber_cm_inv"]),
                tolerance_cm_inv=float(e.get(
                    "tolerance_cm_inv",
                    data.get("tolerance_default_cm_inv", 8.0),
                )),
                bond=e["bond"],
                mode=e.get("mode", ""),
                compounds=tuple(e.get("compounds", ())),
                discriminative_for=tuple(e.get("discriminative_for", ())),
                notes=e.get("notes", ""),
            )
            for e in data["entries"]
        ]
        return cls(
            entries,
            compound_canonical_names=data.get("compound_canonical_names"),
            tolerance_default_cm_inv=float(data.get("tolerance_default_cm_inv", 8.0)),
            schema_version=data.get("schema_version", "1.0"),
        )

    # ── Core API ─────────────────────────────────────────────────────

    def match_peak(
        self,
        wavenumber: float,
        *,
        intensity: float | None = None,
        tolerance: float | None = None,
    ) -> list[BondEntry]:
        """Return all entries whose tolerance window contains `wavenumber`.

        Parameters
        ----------
        wavenumber : float
            Peak position in cm⁻¹.
        intensity : float, optional
            Peak intensity. Currently unused by the matching logic but kept
            for forward compatibility (e.g. the engine can later weight or
            filter by intensity). Logged for debugging.
        tolerance : float, optional
            If given, overrides each entry's per-row tolerance. Useful for
            broad/loose searches during exploration.

        Returns
        -------
        list[BondEntry]
            All matching entries, sorted by ascending |wavenumber − entry.wn|.
            Empty list if no match (caller can treat that as 'novelty').
        """
        if intensity is not None:
            log.debug(f"match_peak({wavenumber}) intensity={intensity}")
        hits = [
            e for e in self.entries
            if e.matches(wavenumber, tolerance=tolerance)
        ]
        hits.sort(key=lambda e: abs(wavenumber - e.wavenumber_cm_inv))
        return hits

    def match_peaks(
        self,
        wavenumbers: Iterable[float],
        *,
        tolerance: float | None = None,
    ) -> list[list[BondEntry]]:
        """Vectorised version of `match_peak`."""
        return [self.match_peak(wn, tolerance=tolerance) for wn in wavenumbers]

    def get_compound_fingerprint(self, compound_name: str) -> dict:
        """Return all DB entries associated with the given compound.

        The result splits entries into:
            * 'discriminative' — entries listing this compound in
              `discriminative_for` (likely uniquely characteristic).
            * 'supporting'     — entries listing the compound in `compounds`
              but NOT in `discriminative_for` (less specific signals).

        Returns
        -------
        dict
            {
              "compound": str,
              "discriminative": list[dict],
              "supporting":     list[dict],
              "n_total":        int,
            }
        """
        bucket = self._by_compound.get(compound_name, [])
        if not bucket and self.compound_canonical_names \
                and compound_name not in self.compound_canonical_names:
            log.warning(
                f"Compound '{compound_name}' not found in DB; canonical names: "
                f"{self.compound_canonical_names}"
            )
        discriminative = [e for e in bucket if compound_name in e.discriminative_for]
        supporting     = [e for e in bucket if compound_name not in e.discriminative_for]
        return {
            "compound": compound_name,
            "discriminative": [e.to_dict() for e in discriminative],
            "supporting":     [e.to_dict() for e in supporting],
            "n_total": len(bucket),
        }

    def lookup_by_id(self, peak_id: str) -> BondEntry:
        """Return the entry with the given ID; raises KeyError if absent."""
        try:
            return self._by_id[peak_id]
        except KeyError as e:
            raise KeyError(f"Peak ID {peak_id!r} not in bond DB.") from e

    def all_compounds(self) -> list[str]:
        """List every compound name referenced by any entry."""
        return sorted(self._by_compound.keys())

    # ── Validation ────────────────────────────────────────────────────

    def validate_db(self, raise_on_error: bool = False) -> dict:
        """Run integrity checks. Returns a dict of {check: ok/fail/details}.

        Checks
        ------
        * IDs are unique.
        * IDs follow `Pxxx` pattern (e.g. P001, P042).
        * Wavenumbers are positive and finite.
        * Tolerances are positive.
        * Every `discriminative_for` compound is also listed in `compounds`.
        * Every compound name (in `compounds` and `discriminative_for`) is in
          `compound_canonical_names` if that list is provided.
        * Tolerance values look reasonable (1 ≤ tol ≤ 30 cm⁻¹).

        Parameters
        ----------
        raise_on_error : bool, default False
            If True, raise `ValueError` on the first failure. Otherwise return
            a status dict with all problems collected.
        """
        import math
        import re

        problems: list[str] = []
        id_pattern = re.compile(r"^P\d{3,}$")

        # 1. Unique IDs
        ids = [e.id for e in self.entries]
        dup = {x for x in ids if ids.count(x) > 1}
        if dup:
            problems.append(f"Duplicate IDs: {sorted(dup)}")

        # 2. ID format
        for e in self.entries:
            if not id_pattern.match(e.id):
                problems.append(f"Bad ID format (expect Pxxx): {e.id!r}")

        # 3. Numeric sanity
        for e in self.entries:
            if not (math.isfinite(e.wavenumber_cm_inv) and e.wavenumber_cm_inv > 0):
                problems.append(f"{e.id}: invalid wavenumber {e.wavenumber_cm_inv}")
            if not (e.tolerance_cm_inv > 0):
                problems.append(f"{e.id}: non-positive tolerance {e.tolerance_cm_inv}")
            if not (1 <= e.tolerance_cm_inv <= 30):
                problems.append(
                    f"{e.id}: tolerance {e.tolerance_cm_inv} outside [1, 30] cm⁻¹"
                )

        # 4. discriminative_for ⊆ compounds
        for e in self.entries:
            extra = set(e.discriminative_for) - set(e.compounds)
            if extra:
                problems.append(
                    f"{e.id}: discriminative_for has compounds not in compounds list: {extra}"
                )

        # 5. Compound names known
        if self.compound_canonical_names:
            allowed = set(self.compound_canonical_names)
            for e in self.entries:
                bad = (set(e.compounds) | set(e.discriminative_for)) - allowed
                if bad:
                    problems.append(
                        f"{e.id}: unknown compound(s) {bad}; allowed: {sorted(allowed)}"
                    )

        ok = not problems
        report = {
            "ok": ok,
            "n_entries": len(self.entries),
            "n_problems": len(problems),
            "problems": problems,
        }
        if not ok and raise_on_error:
            raise ValueError("Bond DB validation failed:\n  - " + "\n  - ".join(problems))
        return report

    # ── Misc ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (
            f"BondMapper(n_entries={len(self.entries)}, "
            f"schema={self.schema_version!r}, "
            f"compounds={self.all_compounds()})"
        )

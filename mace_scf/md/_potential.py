"""Scalar voltage observations shared by the MD controller and logger."""
from __future__ import annotations

import numpy as np


def potential_difference(results, vacuum_potential_key, fermi_level_key) -> float:
    """Return phi_vac - mu/e in volts from two finite scalar observations.

    phi_vac uses the electron-referenced voltage convention; mu is in eV, so
    mu/e has the same numerical value in volts. Existing DFT/model label values
    therefore need no rescaling. A target may require an initial-frame offset.
    """
    values = []
    for key in (vacuum_potential_key, fermi_level_key):
        if key not in results:
            raise RuntimeError(f"Calculator did not return {key!r}")
        try:
            value = np.asarray(results[key], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Calculator result {key!r} must be one finite scalar") from exc
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError(f"Calculator result {key!r} must be one finite scalar")
        values.append(float(value.reshape(-1)[0]))
    difference = values[0] - values[1]
    if not np.isfinite(difference):
        raise ValueError("Vacuum and Fermi levels must give a finite potential difference")
    return difference

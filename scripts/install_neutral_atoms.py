#!/usr/bin/env python3
"""Install the DFT neutral atoms in a trained model: python install_neutral_atoms.py MODEL FRAMES OUT [INDEX].

FRAMES are frames of scripts/collect_vasp_data.py, with proto_fourier_potential and
vasp_pscenc (or neutral_atom_alpha); INDEX selects them (default ':'). Counter
charges of the saved model then need nothing else, in training or in MD.
"""
import sys

import torch
from ase.io import read

from mace_scf.electrostatics.potential import fit_neutral_atoms, install_neutral_atoms


def main(model_path, frames_path, out_path, index=":"):
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    images = read(frames_path, index=index)
    images = [images] if hasattr(images, "numbers") else images
    rows, report = fit_neutral_atoms([(a.cell.array, a.positions, a.numbers, a.info) for a in images])
    print(report)
    numbers = [int(z) for z in model.atomic_numbers]
    missing = sorted(set(numbers) - set(rows))
    if missing:
        print(f"No neutral atoms for elements {missing}: counter charges refuse frames that contain them")
    install_neutral_atoms(model.field_dependent_charges_map, numbers, rows)
    if getattr(model, "backbone_layout", "mul_ir") == "ir_mul":
        # Save the portable e3nn layout; the calculator and MD drivers accelerate it again.
        from mace_scf.utils.foundation import decelerate_backbone
        decelerate_backbone(model)
    torch.save(model, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])

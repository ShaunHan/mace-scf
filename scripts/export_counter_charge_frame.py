"""Export one MD geometry and its prescribed source for DFT relabeling."""
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

from mace_scf.data.new_atomic_data import COUNTER_CHARGE_KEYS, counter_charge_data


TRAJECTORY = 'nvtphi.traj'
FRAME = -1
OUTPUT_DIRECTORY = 'counter_charge_single_point'


def export_frame(atoms, directory):
    """Save source inputs, not predicted electronic or mechanical labels.

    The published VASP local-potential plugin can reproduce this charge profile.
    Its potential, ionic energy/force corrections and slab boundaries must be
    included when obtaining the reference labels. This function does not run DFT.
    """
    source = counter_charge_data(atoms.info, atoms.pbc)
    if float(source['counter_charge_width']) == 0.:
        raise ValueError('The frame has no specified counter-charge profile')
    keys = (*COUNTER_CHARGE_KEYS, 'total_charge', 'external_field',
            *[f'{prefix}_{axis}frac' for prefix in ('vacuum', 'dipole_correction') for axis in 'xyz'])
    controls = {key: np.asarray(atoms.info[key]).tolist() for key in keys if key in atoms.info}
    if 'total_charge' not in controls or not np.isfinite(controls['total_charge']):
        raise ValueError('The frame must store its finite total_charge')
    controls['counter_charge_width'] = float(source['counter_charge_width'])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    geometry = atoms.copy()
    geometry.calc = None
    geometry.info = controls
    for key in tuple(geometry.arrays):
        if key not in ('numbers', 'positions'):
            del geometry.arrays[key]
    write(directory/'POSCAR', geometry, format='vasp', direct=True, sort=False, vasp5=True)
    write(directory/'frame.xyz', geometry, format='extxyz', write_results=False)
    (directory/'counter_charge.json').write_text(json.dumps(controls, indent=2)+'\n')
    print(f'Wrote {directory}. Set NELECT = neutral valence-electron count - {controls["total_charge"]:.12g}.')


if __name__ == '__main__':
    export_frame(read(TRAJECTORY, index=FRAME), OUTPUT_DIRECTORY)

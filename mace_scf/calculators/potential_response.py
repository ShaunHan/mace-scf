import mace.data
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from mace.tools import AtomicNumberTable, torch_geometric, torch_tools

import mace_scf.data

SHE_ABSOLUTE_V = 4.44  # Trasatti's absolute SHE potential at 298.15 K, IUPAC
D_BLOCK = frozenset([*range(21, 31), *range(39, 49), *range(57, 81), *range(89, 113)])


def slab_axis(atoms):
    """The non-periodic axis of a slab, the unit normal of its periodic plane and the cell height along it."""
    axes = np.flatnonzero(~np.asarray(atoms.pbc, dtype=bool))
    if len(axes) != 1:
        raise ValueError(f'A constant potential slab needs one non-periodic axis, not pbc = {atoms.pbc.tolist()}')
    axis = int(axes[0])
    normal = np.cross(*np.delete(atoms.cell.array, axis, axis=0))
    normal = normal*np.sign(atoms.cell[axis]@normal)/np.linalg.norm(normal)
    return axis, normal, float(atoms.cell[axis]@normal)


def slab_controls(atoms, sheet_height=6., sheet_width=1., cut=None):
    """The dipole correction plane and a counter charge sheet of a slab, from its geometry.

    Both lie along the one non-periodic axis of the slab. The plane lies in the middle of the largest empty gap
    along it unless cut is given. The electrode surface is the outermost d-block atom, or O bonded to one, on
    the side that holds the solvent; the sheet lies sheet_height beyond it.
    """
    axis, _, length = slab_axis(atoms)
    z = atoms.get_scaled_positions(wrap=True)[:, axis]
    if cut is None:
        order = np.sort(z)
        gaps = np.diff(np.append(order, order[0]+1.))
        i = int(np.argmax(gaps))
        cut = (order[i]+.5*gaps[i]) % 1.
    height = ((z-cut) % 1.)*length
    numbers = atoms.get_atomic_numbers()
    electrode = np.isin(numbers, list(D_BLOCK))
    if not electrode.any():
        raise ValueError('No d-block electrode atoms to place the counter charge sheet')
    surface = list(np.flatnonzero(electrode))
    for o in np.flatnonzero(numbers == 8):
        if (atoms.get_distances(int(o), np.flatnonzero(electrode), mic=True) < 2.3).any():
            surface.append(int(o))
    top, bottom = height[surface].max(), height[surface].min()
    sheet = top+sheet_height if (height > top).sum() >= (height < bottom).sum() else bottom-sheet_height
    letter = 'xyz'[axis]
    return {f'dipole_correction_{letter}frac': float(cut),
            f'counter_charge_{letter}frac': float((cut+sheet/length) % 1.), 'counter_charge_width': float(sheet_width)}


class MACEPotentialResponse(Calculator):
    """ASE calculator of a PotentialResponse model at the total charge, counter charge and field in atoms.info.

    With constant_potential the counter charge stands for solvated ions and acts through electrostatics only,
    and a slab without one gets a sheet counter_charge_height above its electrode surface, along its one
    non-periodic axis, on the first call, kept for the trajectory (slab_controls).
    """

    implemented_properties = ['energy', 'free_energy', 'forces', 'dipole', 'charges', 'fermi_level',
                              'vacuum_potential', 'work_function', 'cell_voltage', 'she_offset',
                              'electrode_potential_she']

    def __init__(self, model_path, device='cuda', default_dtype='float64', constant_potential=False,
                 counter_charge_height=6., counter_charge_width=1., enable_cueq=False, **kwargs):
        Calculator.__init__(self, **kwargs)
        torch_tools.set_default_dtype(default_dtype)
        self.device = torch_tools.init_device(device)
        self.model = torch.load(f=model_path, map_location=self.device, weights_only=False).to(self.device)
        if enable_cueq:
            from mace_scf.utils.foundation import accelerate_backbone
            accelerate_backbone(self.model, self.device)
        self.model.counter_charge_overlap = not constant_potential
        self.model.eval().requires_grad_(False)
        self.constant_potential = constant_potential
        self.counter_charge_height = float(counter_charge_height)
        self.counter_charge_width = float(counter_charge_width)
        self.z_table = AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        self.keyspec = mace.data.KeySpecification(info_keys={
            'total_charge': 'total_charge', 'counter_charge': 'counter_charge',
            'counter_charge_width': 'counter_charge_width', 'counter_charge_fraction': 'counter_charge_zfrac',
            'dipole_correction_fraction': 'dipole_correction_zfrac', 'external_field': 'external_field'},
            arrays_keys={})

    def check_state(self, atoms, tol=1e-15):
        changes = Calculator.check_state(self, atoms, tol)
        if self.atoms is not None and any(np.any(np.asarray(atoms.info.get(key)) != np.asarray(self.atoms.info.get(key)))
                                          for key in self.keyspec.info_keys.values()):
            changes.append('info')
        return changes

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if self.constant_potential and atoms is not None:
            letter = 'xyz'[slab_axis(atoms)[0]]
            self.keyspec.info_keys.update(counter_charge_fraction=f'counter_charge_{letter}frac',
                                          dipole_correction_fraction=f'dipole_correction_{letter}frac')
            if f'counter_charge_{letter}frac' not in atoms.info:
                atoms.info.update(slab_controls(atoms, self.counter_charge_height, self.counter_charge_width))
        Calculator.calculate(self, atoms, properties, system_changes)
        config = mace.data.config_from_atoms(self.atoms, key_specification=self.keyspec)
        data = mace_scf.data.ExtAtomicData.from_config(config, z_table=self.z_table, cutoff=float(self.model.r_max))
        batch = torch_geometric.batch.Batch.from_data_list([data]).to(self.device)
        output = self.model(batch.to_dict(), compute_force=True)
        output = {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in output.items()}
        fermi_level, vacuum_potential = float(output['fermi_level'][0]), float(output['vacuum_potential'][0])
        work_function, cell_voltage = vacuum_potential-fermi_level, float(output['cell_voltage'][0])
        self.results = {
            'energy': float(output['energy'][0]),
            'free_energy': float(output['energy'][0]),
            'forces': output['forces'].cpu().numpy(),
            'dipole': output['dipole'][0].cpu().numpy(),
            'charges': output['density_coefficients'][:, 0].cpu().numpy(),
            'fermi_level': fermi_level,
            'vacuum_potential': vacuum_potential,
            'work_function': work_function,
            'cell_voltage': cell_voltage,
            'she_offset': work_function-cell_voltage,
            'electrode_potential_she': work_function-SHE_ABSOLUTE_V,
        }
        for key in ('fourier_potential', 'k_vectors_grid_shape', 'k_vectors_mask'):
            self.results[key] = output[key].cpu().numpy()

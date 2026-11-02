import mace.data
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from mace.tools import AtomicNumberTable, torch_geometric, torch_tools

import mace_scf.data

SHE_ABSOLUTE_V = 4.44  # Trasatti's absolute SHE potential at 298.15 K, IUPAC


class MACEPotentialResponse(Calculator):
    """ASE calculator of a PotentialResponse model at the total charge, counter charge and field in atoms.info.

    A slab is periodic along two cell vectors. Along the third, atoms.info gives as fractions the dipole
    correction plane (dipole_correction_fraction) and the counter charge sheet (counter_charge_fraction, of
    width counter_charge_width); info_keys maps these names to other keys of atoms.info, as the info_keys of a
    training config do. With constant_potential the counter charge stands for solvated ions and acts through
    electrostatics only. With band_edges it also returns the valence band maximum, the conduction band minimum
    and the band gap.
    """

    implemented_properties = ['energy', 'free_energy', 'forces', 'dipole', 'charges', 'fermi_level',
                              'vacuum_potential', 'work_function', 'cell_voltage', 'she_offset',
                              'electrode_potential_she', 'valence_band_maximum', 'conduction_band_minimum',
                              'band_gap']

    def __init__(self, model_path, device='cuda', default_dtype='float64', constant_potential=False,
                 enable_cueq=False, band_edges=False, info_keys=None, **kwargs):
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
        self.band_edges = bool(band_edges)
        self.z_table = AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        names = ('total_charge', 'counter_charge', 'counter_charge_width', 'counter_charge_fraction',
                 'dipole_correction_fraction', 'external_field')
        self.keyspec = mace.data.KeySpecification(info_keys={**{name: name for name in names}, **(info_keys or {})},
                                                  arrays_keys={})

    def check_state(self, atoms, tol=1e-15):
        changes = Calculator.check_state(self, atoms, tol)
        if self.atoms is not None and any(np.any(np.asarray(atoms.info.get(key)) != np.asarray(self.atoms.info.get(key)))
                                          for key in self.keyspec.info_keys.values()):
            changes.append('info')
        return changes

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        keys = self.keyspec.info_keys
        if not self.atoms.pbc.all():
            needed = ['dipole_correction_fraction']
            if self.constant_potential or self.atoms.info.get(keys['counter_charge'], 0.):
                needed += ['counter_charge_fraction', 'counter_charge_width']
            missing = [keys[name] for name in needed if keys[name] not in self.atoms.info]
            if missing:
                raise ValueError(f'A slab needs {missing} in atoms.info, as fractions along its non-periodic axis')
        config = mace.data.config_from_atoms(self.atoms, key_specification=self.keyspec)
        data = mace_scf.data.ExtAtomicData.from_config(config, z_table=self.z_table, cutoff=float(self.model.r_max))
        batch = torch_geometric.batch.Batch.from_data_list([data]).to(self.device)
        output = self.model(batch.to_dict(), compute_force=True, band_edges=self.band_edges)
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
        if self.band_edges:
            for key in ('valence_band_maximum', 'conduction_band_minimum', 'band_gap'):
                self.results[key] = float(output[key][0])

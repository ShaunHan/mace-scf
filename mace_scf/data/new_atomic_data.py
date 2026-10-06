###########################################################################################
#
#     replace AtomicData to allow for extra fields
#
###########################################################################################

from typing import Optional, Sequence

import torch.utils.data

from mace.tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    torch_geometric,
    voigt_to_matrix,
)

from mace.data.utils import Configuration
import inspect
from mace.data import AtomicData
from scipy.constants import pi

from .neighborhood import get_neighborhood
from mace.data import KeySpecification
import numpy as np


_ATOMIC_DATA_PARAMETERS = inspect.signature(AtomicData.__init__).parameters

def update_keyspec_from_kwargs(keyspec, keydict) -> KeySpecification:
    # convert command line style property_key arguments into a keyspec
    infos = [
        "energy_key",
        "stress_key",
        "virials_key",
        "dipole_key",
        "head_key",
        "fermi_level_key",
        "fermi_level_weight_key",
        "external_field_key",
        "polarizability_key",
        "fourier_density_key",
        "fourier_potential_key",
        "fourier_proto_potential_key",
        "workfunction_key",
        "vacuum_zfrac_key",
        "dipole_correction_zfrac_key",
        "potcar_match_key",
        "workfunction_weight_key",
        "potential_weight_key",
    ]
    arrays = [
        "forces_key",
        "charges_key",
        "enegs_key",
        "hardness_key",
        "atomic_multipoles_key",
    ]
    info_keys = {}
    arrays_keys = {}
    for key in infos:
        if key in keydict:
            info_keys[key[:-4]] = keydict[key]
    for key in arrays:
        if key in keydict:
            arrays_keys[key[:-4]] = keydict[key]
    keyspec.update(info_keys=info_keys, arrays_keys=arrays_keys)
    return keyspec


class ExtAtomicData(AtomicData):
    density_coefficients: torch.Tensor
    density_coefficients_weight: torch.Tensor
    rcell: torch.Tensor
    volume: torch.Tensor
    pbc: torch.Tensor
    total_charge: torch.Tensor
    external_field: torch.Tensor
    fermi_level: torch.Tensor
    fermi_level_weight: torch.Tensor
    enegs: torch.Tensor
    hardness: torch.Tensor
    fourier_density: torch.Tensor
    fourier_density_shape: torch.Tensor
    fourier_density_weight: torch.Tensor
    fourier_potential: torch.Tensor
    fourier_potential_shape: torch.Tensor
    fourier_potential_weight: torch.Tensor
    potential_weight: torch.Tensor
    fourier_proto_potential: torch.Tensor
    fourier_proto_potential_shape: torch.Tensor
    fourier_proto_potential_weight: torch.Tensor
    workfunction: torch.Tensor
    workfunction_weight: torch.Tensor
    vacuum_zfrac: torch.Tensor
    dipole_correction_zfrac: torch.Tensor
    potcar_match: torch.Tensor

    def __init__(
        self,
        **kwargs,
    ):
        # ``mace.tools.torch_geometric.Batch.get_example`` reconstructs a
        # single graph by calling the original data class with no arguments
        # and then assigning the sliced attributes.  Upstream ``AtomicData``
        # intentionally has a strict, fully required constructor because MACE
        # never de-batches its training batches.  MACE-VOLT's bounded-memory
        # force path does de-batch a logical optimizer batch, so the subclass
        # must support the standard PyG empty-construction protocol.
        #
        # Bypass only the strict AtomicData validation for this empty
        # reconstruction placeholder.  Normal user/data-loader construction
        # still follows the complete AtomicData constructor and all shape
        # checks below.  Calling Data.__init__ directly is robust across MACE
        # versions whose AtomicData required-argument list differs.
        if not kwargs:
            torch_geometric.data.Data.__init__(self)
            return

        # new bits
        density_coefficients = kwargs.pop("density_coefficients", None)
        density_coefficients_weight = kwargs.pop(
            "density_coefficients_weight", None
        )  # [,]
        electrostatic_potentials = kwargs.pop("electrostatic_potentials", None)
        electrostatic_potentials_weight = kwargs.pop(
            "electrostatic_potentials_weight", None
        )  # [,]
        rcell = kwargs.pop("rcell", None)  # [3,3]
        volume = kwargs.pop("volume", None)  # [1]
        pbc = kwargs.pop("pbc", None)  # [3,1]
        total_charge = kwargs.pop("total_charge", None)  # [1]
        external_field = kwargs.pop("external_field", None)  # [3]
        fermi_level = kwargs.pop("fermi_level", None)  # [1]
        fermi_level_weight = kwargs.pop("fermi_level_weight", None)  # [,]
        # polarizability = kwargs.pop("polarizability", None) # [3,3]
        # polarizability_weight = kwargs.pop("polarizability_weight", None) #[1]
        cluster_batch = kwargs.pop("cluster_batch", None)
        cluster_loss_weight = kwargs.pop("cluster_loss_weight", None)
        enegs = kwargs.pop("enegs", None)
        hardness = kwargs.pop("hardness", None)
        fourier_density = kwargs.pop("fourier_density", None)
        fourier_density_shape = kwargs.pop("fourier_density_shape", None)
        fourier_density_weight = kwargs.pop("fourier_density_weight", None)
        fourier_potential = kwargs.pop("fourier_potential", None)
        fourier_potential_shape = kwargs.pop("fourier_potential_shape", None)
        fourier_potential_weight = kwargs.pop("fourier_potential_weight", None)
        potential_weight = kwargs.pop("potential_weight", None)
        if potential_weight is not None:
            potential_weight = torch.as_tensor(potential_weight)
            if potential_weight.numel() != 3:
                raise ValueError(
                    "potential_weight must contain exactly three components; "
                    f"got shape {tuple(potential_weight.shape)}"
                )
            # Upstream MACE treats every one-dimensional extra property as a
            # per-atom column and promotes (3,) to (3, 1).  This quantity is a
            # graph-level Cartesian selector, so canonicalize either form.
            potential_weight = potential_weight.reshape(3)
        fourier_proto_potential = kwargs.pop("fourier_proto_potential", None)
        fourier_proto_potential_shape = kwargs.pop("fourier_proto_potential_shape", None)
        fourier_proto_potential_weight = kwargs.pop("fourier_proto_potential_weight", None)
        workfunction = kwargs.pop("workfunction", None)
        workfunction_weight = kwargs.pop("workfunction_weight", None)
        vacuum_zfrac = kwargs.pop("vacuum_zfrac", None)
        dipole_correction_zfrac = kwargs.pop("dipole_correction_zfrac", None)
        potcar_match = kwargs.pop("potcar_match", None)
        # MACE-develop added required, optional-valued magnetic targets.
        for key in ("magforces_weight", "magmom", "magforces"):
            if key in _ATOMIC_DATA_PARAMETERS:
                kwargs.setdefault(key, None)
        super().__init__(**kwargs)
        assert (
            density_coefficients_weight is None
            or len(density_coefficients_weight.shape) == 0
        )
        assert (
            electrostatic_potentials_weight is None
            or len(electrostatic_potentials_weight.shape) == 0
        )
        assert (
            electrostatic_potentials is None or electrostatic_potentials.shape[-1] == 1
        )
        assert total_charge is None or len(total_charge.shape) == 0
        assert external_field is None or external_field.shape == torch.Size([1, 3])
        assert fermi_level is None or len(fermi_level.shape) == 0
        assert fermi_level_weight is None or len(fermi_level_weight.shape) == 0
        # assert polarizability is None or polarizability.shape == torch.Size([1,3,3])
        # assert polarizability_weight is None or len(polarizability_weight.shape) == 0
        assert cluster_loss_weight is None or len(cluster_loss_weight.shape) == 0
        assert workfunction is None or workfunction.ndim == 0
        assert workfunction_weight is None or workfunction_weight.ndim == 0
        assert vacuum_zfrac is None or vacuum_zfrac.ndim == 0
        assert dipole_correction_zfrac is None or dipole_correction_zfrac.ndim == 0
        assert potcar_match is None or potcar_match.ndim == 0
        if potential_weight is not None:
            if not torch.all(torch.isfinite(potential_weight)) or torch.any(
                potential_weight < 0
            ):
                raise ValueError(
                    "potential_weight must be finite and nonnegative; got "
                    f"{potential_weight.tolist()}"
                )

        # Aggregate data
        data = {
            "density_coefficients": density_coefficients,
            "density_coefficients_weight": density_coefficients_weight,
            "electrostatic_potentials": electrostatic_potentials,
            "electrostatic_potentials_weight": electrostatic_potentials_weight,
            "volume": volume,
            "rcell": rcell,
            "pbc": pbc,
            "total_charge": total_charge,
            "external_field": external_field,
            "fermi_level": fermi_level,
            "fermi_level_weight": fermi_level_weight,
            # "polarizability": polarizability,
            # "polarizability_weight": polarizability_weight,
            "cluster_batch": cluster_batch,
            "cluster_loss_weight": cluster_loss_weight,
            "enegs": enegs,
            "hardness": hardness,
            "fourier_density": fourier_density,
            "fourier_density_shape": fourier_density_shape,
            "fourier_density_weight": fourier_density_weight,
            "fourier_potential": fourier_potential,
            "fourier_potential_shape": fourier_potential_shape,
            "fourier_potential_weight": fourier_potential_weight,
            "potential_weight": potential_weight,
            "fourier_proto_potential": fourier_proto_potential,
            "fourier_proto_potential_shape": fourier_proto_potential_shape,
            "fourier_proto_potential_weight": fourier_proto_potential_weight,
            "workfunction": workfunction,
            "workfunction_weight": workfunction_weight,
            "vacuum_zfrac": vacuum_zfrac,
            "dipole_correction_zfrac": dipole_correction_zfrac,
            "potcar_match": potcar_match,
        }
        for key, value in data.items():
            setattr(self, key, value)

    @classmethod
    def from_config(
        cls,
        config: Configuration,
        z_table: AtomicNumberTable,
        cutoff: float,
        heads: Optional[list] = None,
        atomic_multipoles_max_l: int = 0,
    ) -> "ExtAtomicData":
        # Call the base class explicitly.  ``super().from_config`` preserves
        # the subclass binding of the classmethod, so recent MACE versions try
        # to construct ExtAtomicData while passing through all extra
        # properties.  Their generic one-dimensional-property rule promotes
        # the graph-level potential selector from (3,) to (3, 1), before this
        # method has a chance to canonicalize it.
        atomic_data = AtomicData.from_config(
            config, z_table, cutoff, heads=heads
        )
        num_atoms = len(config.atomic_numbers)

        # redo cell
        edge_index, shifts, unit_shifts, cell = get_neighborhood(
            positions=config.positions, cutoff=cutoff, pbc=config.pbc, cell=config.cell
        )
        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )
        atomic_data.cell = cell

        density_coefficients = (
            torch.tensor(
                np.atleast_2d(config.properties.get("atomic_multipoles").T).T[
                    ..., : (atomic_multipoles_max_l + 1) ** 2
                ],
                dtype=torch.get_default_dtype(),
            )
            if config.properties.get("atomic_multipoles") is not None
            else torch.zeros(
                (num_atoms, (atomic_multipoles_max_l + 1) ** 2),
                dtype=torch.get_default_dtype(),
            )
        )
        density_coefficients_weight = (
            torch.tensor(
                config.property_weights.get("atomic_multipoles"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("atomic_multipoles") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        electrostatic_potentials = (
            torch.tensor(config.properties.get("electrostatic_potentials")).unsqueeze(
                -1
            )
            if config.properties.get("electrostatic_potentials") is not None
            else torch.zeros((num_atoms, 1))
        )
        electrostatic_potentials_weight = (
            torch.tensor(
                config.property_weights.get("electrostatic_potentials"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("electrostatic_potentials") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )
        volume = (
            torch.abs(torch.linalg.det(atomic_data.cell))
            if atomic_data.cell is not None
            else None
        )
        rcell = (
            2 * pi * torch.linalg.inv(atomic_data.cell.mT)
            if volume is not None and float(volume) > 1.0e-12
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )
        pbc = (
            torch.tensor(np.asarray(config.pbc, dtype=bool).tolist(), dtype=torch.bool)
            if config.pbc is not None
            else torch.tensor([False, False, False], dtype=torch.bool)
        )
        total_charge = (
            torch.tensor(
                config.properties.get("total_charge"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("total_charge") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        external_field = (
            torch.tensor(
                config.properties.get("external_field"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("external_field") is not None
            else torch.tensor([3 * [0.0]], dtype=torch.get_default_dtype())
        )
        external_field = torch.atleast_2d(external_field)
        fermi_level = (
            torch.tensor(
                config.properties.get("fermi_level"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("fermi_level") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        fermi_level_weight_value = config.property_weights.get("fermi_level")
        if fermi_level_weight_value is None:
            fermi_level_weight_value = config.properties.get("fermi_level_weight")
        if fermi_level_weight_value is None:
            fermi_level_weight_value = config.properties.get(
                "config_fermi_level_weight"
            )
        if fermi_level_weight_value is None:
            fermi_level_weight_value = config.properties.get(
                "config_workfunction_weight",
                1.0 if config.properties.get("fermi_level") is not None else 0.0,
            )
            if config.properties.get("fermi_level") is None:
                fermi_level_weight_value = 0.0
        fermi_level_weight = torch.tensor(
            fermi_level_weight_value, dtype=torch.get_default_dtype()
        )
        """ polarizability = (
            voigt_to_matrix(
                torch.tensor(config.properties.get("polarizability"), dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if config.properties.get("polarizability") is not None
            else torch.zeros((1,3,3), dtype=torch.get_default_dtype())
        ) """
        polarizability_weight = (
            torch.tensor(
                config.property_weights.get("polarizability"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("polarizability") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        cluster_batch = (
            torch.tensor(config.properties.get("molID"), dtype=torch.long)
            if config.properties.get("molID") is not None
            else torch.zeros((num_atoms,), dtype=torch.long)
        )
        cluster_loss_weight = (
            torch.tensor(
                config.property_weights.get("molID"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("molID") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        enegs = (
            torch.tensor(
                config.properties.get("enegs"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("enegs") is not None
            else torch.zeros((num_atoms))
        )
        hardness = (
            torch.tensor(
                config.properties.get("hardness"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("hardness") is not None
            else torch.zeros((num_atoms))
        )

        def read_fft(name):
            value = config.properties.get(name)
            if value is None:
                return (
                    torch.zeros((0,), dtype=torch.cdouble),
                    torch.zeros((3,), dtype=torch.long),
                    False,
                )
            value = torch.as_tensor(value, dtype=torch.get_default_dtype())
            if value.ndim != 4 or value.shape[0] != 2:
                raise ValueError(
                    f"{name} must have shape [2, nx, ny, nz], got {tuple(value.shape)}"
                )
            complex_value = torch.complex(value[0], value[1])
            return (
                complex_value.reshape(-1),
                torch.tensor(complex_value.shape, dtype=torch.long),
                True,
            )

        fourier_density, fourier_density_shape, has_fourier_density = read_fft(
            "fourier_density"
        )
        fourier_potential, fourier_potential_shape, has_fourier_potential = read_fft(
            "fourier_potential"
        )
        (
            fourier_proto_potential,
            fourier_proto_potential_shape,
            has_fourier_proto_potential,
        ) = read_fft("fourier_proto_potential")
        fourier_density_weight = torch.tensor(
            config.property_weights.get(
                "fourier_density", 1.0 if has_fourier_density else 0.0
            ),
            dtype=torch.get_default_dtype(),
        )
        fourier_potential_weight = torch.tensor(
            config.property_weights.get(
                "fourier_potential", 1.0 if has_fourier_potential else 0.0
            ),
            dtype=torch.get_default_dtype(),
        )
        potential_weight_value = config.properties.get("potential_weight")
        if potential_weight_value is None:
            potential_weight_value = config.properties.get(
                "config_potential_weight"
            )
        if potential_weight_value is None:
            # Observation selection is independent of electrostatic boundary conditions.
            potential_weight_value = np.ones(3, dtype=float)
        potential_weight = torch.as_tensor(
            potential_weight_value, dtype=torch.get_default_dtype()
        ).reshape(-1)
        if potential_weight.shape != torch.Size([3]):
            raise ValueError(
                "config_potential_weight must be a three-component vector, got "
                f"shape {tuple(potential_weight.shape)}"
            )
        if not torch.all(torch.isfinite(potential_weight)) or torch.any(
            potential_weight < 0.0
        ):
            raise ValueError(
                "config_potential_weight must be finite and nonnegative, got "
                f"{potential_weight.tolist()}"
            )
        if has_fourier_potential and not torch.any(potential_weight > 0.0):
            raise ValueError(
                "A Fourier-potential target requires at least one positive "
                "config_potential_weight component"
            )
        fourier_proto_potential_weight = torch.tensor(
            config.property_weights.get(
                "fourier_proto_potential",
                1.0 if has_fourier_proto_potential else 0.0,
            ),
            dtype=torch.get_default_dtype(),
        )
        workfunction_value = config.properties.get("workfunction")
        workfunction = torch.tensor(
            0.0 if workfunction_value is None else workfunction_value,
            dtype=torch.get_default_dtype(),
        )
        workfunction_weight_value = config.property_weights.get("workfunction")
        if workfunction_weight_value is None:
            workfunction_weight_value = config.properties.get("workfunction_weight")
        if workfunction_weight_value is None:
            workfunction_weight_value = config.properties.get(
                "config_workfunction_weight",
                1.0 if workfunction_value is not None else 0.0,
            )
        workfunction_weight = torch.tensor(
            workfunction_weight_value, dtype=torch.get_default_dtype()
        )
        vacuum_zfrac_value = config.properties.get("vacuum_zfrac")
        vacuum_zfrac = torch.tensor(
            0.5 if vacuum_zfrac_value is None else vacuum_zfrac_value,
            dtype=torch.get_default_dtype(),
        )
        dipole_correction_zfrac_value = config.properties.get(
            "dipole_correction_zfrac"
        )
        dipole_correction_zfrac = torch.tensor(
            0.5
            if dipole_correction_zfrac_value is None
            else dipole_correction_zfrac_value,
            dtype=torch.get_default_dtype(),
        )
        potcar_match_value = config.properties.get("potcar_match")
        potcar_match = torch.tensor(
            -1.0 if potcar_match_value is None else potcar_match_value,
            dtype=torch.get_default_dtype(),
        )
        for name, property_weight in (
            ("fermi_level", fermi_level_weight),
            ("fourier_density", fourier_density_weight),
            ("fourier_potential", fourier_potential_weight),
            ("fourier_proto_potential", fourier_proto_potential_weight),
            ("workfunction", workfunction_weight),
        ):
            if not torch.isfinite(property_weight) or property_weight.item() < 0.0:
                raise ValueError(
                    f"{name} weight must be finite and nonnegative, got "
                    f"{property_weight.item()}"
                )
        if (
            fermi_level_weight.item() > 0.0
            and config.properties.get("fermi_level") is None
        ):
            raise ValueError(
                "Positive fermi_level weight requires a fermi_level target"
            )
        if torch.all(pbc) and workfunction_weight.item() != 0.0:
            raise ValueError("Fully periodic configurations must have zero workfunction weight")
        if workfunction_weight.item() > 0.0:
            if workfunction_value is None:
                raise ValueError(
                    "Positive workfunction weight requires a workfunction target"
                )
            if not torch.isfinite(vacuum_zfrac):
                raise ValueError("Positive workfunction weight requires a finite vacuum_zfrac")
            if not 0.0 <= float(vacuum_zfrac) < 1.0:
                raise ValueError(
                    f"vacuum_zfrac must be in [0, 1), got {float(vacuum_zfrac)}"
                )
            if not torch.isfinite(dipole_correction_zfrac) or not (
                0.0 <= float(dipole_correction_zfrac) < 1.0
            ):
                raise ValueError(
                    "Positive workfunction weight requires "
                    "dipole_correction_zfrac in [0, 1)"
                )
            open_axes = torch.nonzero(~pbc.to(torch.bool), as_tuple=False).reshape(-1)
            if open_axes.numel() != 1:
                raise ValueError(
                    "Positive workfunction weight requires exactly one nonperiodic axis; "
                    f"got pbc={pbc.tolist()}"
                )
            if workfunction_value is None or not torch.isfinite(workfunction):
                raise ValueError("Positive workfunction weight requires a finite workfunction target")
        return cls(
            edge_index=atomic_data.edge_index,
            positions=atomic_data.positions,
            shifts=atomic_data.shifts,
            unit_shifts=atomic_data.unit_shifts,
            cell=cell,  # use cell from new neighbourhood fn
            node_attrs=atomic_data.node_attrs,
            weight=atomic_data.weight,
            head=atomic_data.head,
            energy_weight=atomic_data.energy_weight,
            forces_weight=atomic_data.forces_weight,
            stress_weight=atomic_data.stress_weight,
            virials_weight=atomic_data.virials_weight,
            dipole_weight=atomic_data.dipole_weight,
            charges_weight=atomic_data.charges_weight,
            forces=atomic_data.forces,
            energy=atomic_data.energy,
            stress=atomic_data.stress,
            virials=atomic_data.virials,
            dipole=atomic_data.dipole,
            charges=atomic_data.charges,
            polarizability=atomic_data.polarizability,
            polarizability_weight=polarizability_weight,
            elec_temp=atomic_data.elec_temp,  # new things below
            density_coefficients=density_coefficients,
            density_coefficients_weight=density_coefficients_weight,
            electrostatic_potentials=electrostatic_potentials,
            electrostatic_potentials_weight=electrostatic_potentials_weight,
            volume=volume,
            rcell=rcell,
            pbc=pbc,
            total_charge=total_charge,
            external_field=external_field,
            fermi_level=fermi_level,
            fermi_level_weight=fermi_level_weight,
            cluster_batch=cluster_batch,
            cluster_loss_weight=cluster_loss_weight,
            enegs=enegs,
            hardness=hardness,
            fourier_density=fourier_density,
            fourier_density_shape=fourier_density_shape,
            fourier_density_weight=fourier_density_weight,
            fourier_potential=fourier_potential,
            fourier_potential_shape=fourier_potential_shape,
            fourier_potential_weight=fourier_potential_weight,
            potential_weight=potential_weight,
            fourier_proto_potential=fourier_proto_potential,
            fourier_proto_potential_shape=fourier_proto_potential_shape,
            fourier_proto_potential_weight=fourier_proto_potential_weight,
            workfunction=workfunction,
            workfunction_weight=workfunction_weight,
            vacuum_zfrac=vacuum_zfrac,
            dipole_correction_zfrac=dipole_correction_zfrac,
            potcar_match=potcar_match,
        )

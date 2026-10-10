###########################################################################################
#
#     replace AtomicData to allow for extra fields
#
###########################################################################################

from typing import Optional, Sequence
from dataclasses import replace

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
COUNTER_CHARGE_KEYS = ("counter_charge", "counter_charge_center", "counter_charge_width")


def counter_charge_data(properties):
    """Read a prescribed Gaussian source, independently of electronic labels.

    The center is Cartesian in Angstrom. Slab periodicity defines a Gaussian
    sheet normal to the open direction; other cells use a localized Gaussian.
    Zero width is an internal marker for an absent source, never a user width.
    """
    charge = float(properties.get("counter_charge") or 0.)
    center = properties.get("counter_charge_center")
    if not np.isfinite(charge):
        raise ValueError("counter_charge must be finite, in elementary charge units")
    width = 0.
    if center is not None:
        width = properties.get("counter_charge_width")
        width = 1. if width is None else float(width)
        if not np.isfinite(width) or width <= 0.:
            raise ValueError("counter_charge_width must be a positive finite Gaussian sigma in Angstrom")
    elif charge != 0. or properties.get("counter_charge_width") is not None:
        raise ValueError("A counter charge requires counter_charge_center in Cartesian Angstrom")
    center = np.zeros(3) if center is None else np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("counter_charge_center must contain three finite Cartesian coordinates")
    return {key: torch.as_tensor(value, dtype=torch.get_default_dtype()) for key, value in {
        "counter_charge": charge, "counter_charge_center": center[None],
        "counter_charge_width": width}.items()}

def plane_fraction(properties, prefix, pbc):
    """Read one fractional cell-axis plane, consistent with slab periodicity."""
    specified = [(i, properties[f"{prefix}_{axis}frac"]) for i, axis in enumerate("xyz")
                 if properties.get(f"{prefix}_{axis}frac") is not None]
    if len(specified) > 1:
        raise ValueError(f"Specify only one of {prefix}_xfrac, {prefix}_yfrac, {prefix}_zfrac")
    if not specified:
        return .5
    axis, value = specified[0]
    value = float(value)
    if not np.isfinite(value) or not 0. <= value < 1.:
        raise ValueError(f"{prefix}_{'xyz'[axis]}frac must be finite and in [0, 1)")
    open_axes = np.flatnonzero(~np.asarray(pbc, dtype=bool))
    if len(open_axes) == 1 and axis != open_axes[0]:
        raise ValueError(f"{prefix}_{'xyz'[axis]}frac conflicts with the nonperiodic cell axis")
    return value


def slab_positions(config):
    """Select the slab image bounded by its specified dipole discontinuity.

    VASP wraps coordinates even along the direction that becomes nonperiodic.
    Rejoin that slab before constructing neighbors or electronic moments.
    Integer lattice translations leave every Fourier observation unchanged.
    Ordinary open structures without a specified correction plane are untouched.
    """
    pbc = np.asarray(config.pbc, dtype=bool)
    if pbc.sum() != 2 or not any(config.properties.get(
            f"dipole_correction_{axis}frac") is not None for axis in "xyz"):
        return config.positions
    plane = plane_fraction(config.properties, "dipole_correction", pbc)
    axis = np.flatnonzero(~pbc)[0]
    center = (plane + .5) % 1.
    fractional = config.positions @ np.linalg.inv(config.cell)
    images = np.floor(fractional[:, axis] - center + .5)
    return config.positions - images[:, None] * config.cell[axis]


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
        "vacuum_potential_key",
        *[f"{prefix}_{axis}frac_key" for prefix in ("vacuum", "dipole_correction") for axis in "xyz"],
        "potcar_match_key",
        "vacuum_potential_weight_key",
        *[f"{key}_key" for key in COUNTER_CHARGE_KEYS],
    ]
    arrays = [
        "forces_key",
        "charges_key",
        "enegs_key",
        "hardness_key",
        "atomic_multipoles_key",
    ]
    info_keys = {key: keyspec.info_keys.get(key, key) for key in COUNTER_CHARGE_KEYS}
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
    fourier_proto_potential: torch.Tensor
    fourier_proto_potential_shape: torch.Tensor
    fourier_proto_potential_weight: torch.Tensor
    vacuum_potential: torch.Tensor
    vacuum_potential_weight: torch.Tensor
    vacuum_fraction: torch.Tensor
    dipole_correction_fraction: torch.Tensor
    potcar_match: torch.Tensor
    counter_charge: torch.Tensor
    counter_charge_center: torch.Tensor
    counter_charge_width: torch.Tensor

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
        fourier_proto_potential = kwargs.pop("fourier_proto_potential", None)
        fourier_proto_potential_shape = kwargs.pop("fourier_proto_potential_shape", None)
        fourier_proto_potential_weight = kwargs.pop("fourier_proto_potential_weight", None)
        vacuum_potential = kwargs.pop("vacuum_potential", None)
        vacuum_potential_weight = kwargs.pop("vacuum_potential_weight", None)
        vacuum_fraction = kwargs.pop("vacuum_fraction", None)
        dipole_correction_fraction = kwargs.pop("dipole_correction_fraction", None)
        potcar_match = kwargs.pop("potcar_match", None)
        counter = {key: kwargs.pop(key, None) for key in COUNTER_CHARGE_KEYS}
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
        assert vacuum_potential is None or vacuum_potential.ndim == 0
        assert vacuum_potential_weight is None or vacuum_potential_weight.ndim == 0
        assert vacuum_fraction is None or vacuum_fraction.ndim == 0
        assert dipole_correction_fraction is None or dipole_correction_fraction.ndim == 0
        assert potcar_match is None or potcar_match.ndim == 0

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
            "fourier_proto_potential": fourier_proto_potential,
            "fourier_proto_potential_shape": fourier_proto_potential_shape,
            "fourier_proto_potential_weight": fourier_proto_potential_weight,
            "vacuum_potential": vacuum_potential,
            "vacuum_potential_weight": vacuum_potential_weight,
            "vacuum_fraction": vacuum_fraction,
            "dipole_correction_fraction": dipole_correction_fraction,
            "potcar_match": potcar_match,
            **counter,
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
        config = replace(config, positions=slab_positions(config),
                         cell=None if config.cell is None else config.cell.copy())
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
        atomic_data.edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        atomic_data.shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
        atomic_data.unit_shifts = torch.as_tensor(unit_shifts, dtype=torch.get_default_dtype())

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
            fermi_level_weight_value = float(config.properties.get("fermi_level") is not None)
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
        fourier_proto_potential_weight = torch.tensor(
            config.property_weights.get(
                "fourier_proto_potential",
                1.0 if has_fourier_proto_potential else 0.0,
            ),
            dtype=torch.get_default_dtype(),
        )
        vacuum_potential_value = config.properties.get("vacuum_potential")
        vacuum_potential = torch.tensor(
            0.0 if vacuum_potential_value is None else vacuum_potential_value,
            dtype=torch.get_default_dtype(),
        )
        vacuum_potential_weight_value = config.property_weights.get("vacuum_potential")
        if vacuum_potential_weight_value is None:
            vacuum_potential_weight_value = config.properties.get("vacuum_potential_weight")
        if vacuum_potential_weight_value is None:
            vacuum_potential_weight_value = config.properties.get(
                "config_vacuum_potential_weight",
                1.0 if vacuum_potential_value is not None else 0.0,
            )
        vacuum_potential_weight = torch.tensor(
            vacuum_potential_weight_value, dtype=torch.get_default_dtype()
        )
        vacuum_fraction = torch.tensor(
            plane_fraction(config.properties, "vacuum", pbc), dtype=torch.get_default_dtype())
        dipole_correction_fraction = torch.tensor(
            plane_fraction(config.properties, "dipole_correction", pbc), dtype=torch.get_default_dtype())
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
            ("vacuum_potential", vacuum_potential_weight),
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
        if torch.all(pbc) and vacuum_potential_weight.item() != 0.0:
            raise ValueError("Fully periodic configurations must have zero vacuum_potential weight")
        if vacuum_potential_weight.item() > 0.0:
            if vacuum_potential_value is None:
                raise ValueError(
                    "Positive vacuum_potential weight requires a vacuum_potential target"
                )
            open_axes = torch.nonzero(~pbc.to(torch.bool), as_tuple=False).reshape(-1)
            if open_axes.numel() != 1:
                raise ValueError(
                    "Positive vacuum_potential weight requires exactly one nonperiodic axis; "
                    f"got pbc={pbc.tolist()}"
                )
            if not torch.isfinite(vacuum_potential):
                raise ValueError("Positive vacuum_potential weight requires a finite vacuum_potential target")
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
            fourier_proto_potential=fourier_proto_potential,
            fourier_proto_potential_shape=fourier_proto_potential_shape,
            fourier_proto_potential_weight=fourier_proto_potential_weight,
            vacuum_potential=vacuum_potential,
            vacuum_potential_weight=vacuum_potential_weight,
            vacuum_fraction=vacuum_fraction,
            dipole_correction_fraction=dipole_correction_fraction,
            potcar_match=potcar_match,
            **counter_charge_data(config.properties),
        )

import torch

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch
from mace.tools.scatter import scatter_sum, scatter_mean
from copy import deepcopy

from mace.modules.loss import (
    weighted_mean_squared_error_energy,
    mean_squared_error_forces,
    weighted_mean_squared_stress,
)
from .utils import compute_effective_index


def weighted_mean_squared_error_charge(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # charge: [n_graphs, ]
    configs_weight = ref.weight # [n_graphs, ]
    num_graphs = ref.ptr.numel() - 1
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]) # [n_graphs, ]
    total_charge = scatter_sum(
        src=pred["density_coefficients"][:,0], index=ref["batch"], dim=-1, dim_size=num_graphs
    ) # [n_graphs, ]
    return torch.mean(configs_weight * torch.square( (ref["total_charge"] - total_charge) / num_atoms)) 


def weighted_mean_squared_error_charge_extensive(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # charge: [n_graphs, ]
    configs_weight = ref.weight # [n_graphs, ]
    num_graphs = ref.ptr.numel() - 1
    total_charge = scatter_sum(
        src=pred["density_coefficients"][:,0], index=ref["batch"], dim=-1, dim_size=num_graphs
    ) # [n_graphs, ]
    return torch.mean(configs_weight * torch.square(ref["total_charge"] - total_charge))  


def weighted_mean_squared_error_fermi(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # fermi level: [n_graphs, ]
    configs_weight = ref.weight  # [n_graphs, ]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]) # [n_graphs, ]
    assert ref["fermi_level"].shape == pred['fermi_level'].shape
    return torch.mean(configs_weight * torch.square( (ref["fermi_level"] - pred['fermi_level']) / num_atoms))


def weighted_mean_squared_error_fermi_extensive(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # fermi level: [n_graphs, ]
    configs_weight = ref.weight  # [n_graphs, ]
    assert ref["fermi_level"].shape == pred['fermi_level'].shape
    return torch.mean(configs_weight * torch.square( (ref["fermi_level"] - pred['fermi_level']) ))


def weighted_mean_squared_error_dma(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # dma: [n_atoms, many]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_density_coefficients_weight = torch.repeat_interleave(
        ref.density_coefficients_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    assert ref["density_coefficients"].shape == pred["density_coefficients"].shape
    sq_error = torch.square(
        ref["density_coefficients"] - pred["density_coefficients"]
    )  # [n_nodes, (max_l+1)**2]

    return torch.mean(configs_weight * configs_density_coefficients_weight * sq_error)


def weighted_mean_squared_error_dipole(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # dipole: [n_graphs, 3]
    configs_weight = ref.weight.view(-1, 1)  # [n_graphs, 1]
    configs_dipole_weight = ref.dipole_weight.view(-1, 3)  # [n_graphs, 3]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1)  # [n_graphs, 1]
    return torch.mean(
        configs_weight
        * configs_dipole_weight
        * torch.square((ref["dipole"] - pred["dipole"]) / num_atoms)
    )


def weighted_mean_squared_error_dipole_extensive(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # dipole: [n_graphs, 3]
    configs_weight = ref.weight.view(-1, 1)  # [n_graphs, 1]
    configs_dipole_weight = ref.dipole_weight.view(-1, 3)  # [n_graphs, 3]
    return torch.mean(
        configs_weight
        * configs_dipole_weight
        * torch.square(ref["dipole"] - pred["dipole"])
    )


def weighted_mean_squared_error_esp(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # esp: [n_atoms, 1]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_electrostatic_potentials_weight = torch.repeat_interleave(
        ref.electrostatic_potentials_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    sq_error = torch.square(
        pred["esps_dft"].view(-1, 1) - pred["esps"].view(-1, 1)
    )  # [n_nodes, 1]

    return torch.mean(configs_weight * configs_electrostatic_potentials_weight * sq_error)


def weighted_mean_squared_error_field_feats(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # esp: [n_atoms, 1]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_electrostatic_potentials_weight = torch.repeat_interleave(
        ref.electrostatic_potentials_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    sq_error = torch.mean(torch.square(
        pred["feats_no_mu_model"] - pred["feats_no_mu_dft"]
    ), axis=-1).view(-1,1)  # [n_nodes, 1]

    return torch.mean(configs_weight * configs_electrostatic_potentials_weight * sq_error)


def weighted_mean_squared_error_polarizability(
    ref: Batch, pred: TensorDict
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1) 
    configs_polarizability_weight = ref.polarizability_weight.view(-1, 1, 1)  
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  
    sq_error = torch.square(
        (ref["polarizability"].view(-1, 3, 3) - pred["polarizability"]) / num_atoms
    )
    return torch.mean(configs_weight * configs_polarizability_weight * sq_error)


def weighted_mean_squared_cluster_virial_extensive(
    ref: Batch, pred: TensorDict
) -> torch.Tensor:
    num_graphs = ref.ptr.numel() - 1
    mol_ids, unique_combinations = compute_effective_index([ref["batch"], ref["cluster_batch"]])
    
    pred_molecule_forces = scatter_sum(pred["forces"], mol_ids, dim=0)
    ref_molecule_forces = scatter_sum(ref["forces"], mol_ids, dim=0)
    molecule_centers = scatter_mean(ref["positions"], mol_ids, dim=0)

    cluster_delta_work = torch.sum(
        molecule_centers * (pred_molecule_forces - ref_molecule_forces), dim=1, keepdim=True
    ) # [n_mols, 1]

    config_delta_work = scatter_sum(
        cluster_delta_work, unique_combinations[:,0], dim=0, dim_size=num_graphs
    ) # [n_graphs, 1]

    configs_weight = ref.weight.view(-1, 1)  # [n_graphs, 1]
    configs_cluster_virial_weight = ref.cluster_loss_weight.view(-1, 1)  # [n_graphs, 1]

    return torch.mean(configs_weight * configs_cluster_virial_weight * torch.square(config_delta_work))


def weighted_mean_squared_cluster_virial(
    ref: Batch, pred: TensorDict
) -> torch.Tensor:
    num_graphs = ref.ptr.numel() - 1
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1) # [n_graphs, 1]
    mol_ids, unique_combinations = compute_effective_index([ref["batch"], ref["cluster_batch"]])
    
    pred_molecule_forces = scatter_sum(pred["forces"], mol_ids, dim=0)
    ref_molecule_forces = scatter_sum(ref["forces"], mol_ids, dim=0)
    molecule_centers = scatter_mean(ref["positions"], mol_ids, dim=0)

    cluster_delta_work = torch.sum(
        molecule_centers * (pred_molecule_forces - ref_molecule_forces), dim=1, keepdim=True
    ) # [n_mols, 1]

    config_delta_work = scatter_sum(
        cluster_delta_work, unique_combinations[:,0], dim=0, dim_size=num_graphs
    ) # [n_graphs, 1]

    configs_weight = ref.weight.view(-1, 1)  # [n_graphs, 1]
    configs_cluster_virial_weight = ref.cluster_loss_weight.view(-1, 1)  # [n_graphs, 1]

    return torch.mean(configs_weight * configs_cluster_virial_weight * torch.square(config_delta_work) / num_atoms)


def weighted_mean_squared_molecular_forces(
    ref: Batch, pred: TensorDict
) -> torch.Tensor:
    num_graphs = ref.ptr.numel() - 1
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1) # [n_graphs, 1]
    mol_ids, unique_combinations = compute_effective_index([ref["batch"], ref["cluster_batch"]])
    
    pred_molecule_forces = scatter_sum(pred["forces"], mol_ids, dim=0) # [n_mols, 3]
    ref_molecule_forces = scatter_sum(ref["forces"], mol_ids, dim=0) # [n_mols, 3]
    err = scatter_sum(
        (pred_molecule_forces - ref_molecule_forces)**2, unique_combinations[:,0], dim=0, dim_size=num_graphs
    ) # [n_graphs, 1]

    configs_weight = ref.weight.view(-1, 1)  # [n_graphs, 1]
    configs_forces_weight = ref.forces_weight.view(-1, 1)  # [n_graphs, 1]

    return torch.mean(configs_weight * configs_forces_weight * err / num_atoms)


def fermi_level_gradient_function(
    ref: Batch, pred: TensorDict
) -> torch.Tensor:
    return torch.mean(torch.relu(pred["dq_dmu"]))


def final_terms_fixedpoint_scf_stability(
    ref: Batch, pred: TensorDict
) -> torch.Tensor:
    num_graphs = ref.ptr.numel() - 1
    difference = pred["charges_history"][...,-1] - pred["charges_history"][...,-2]
    return torch.mean(
        torch.abs(difference)
    )


class FixedPointStability(torch.nn.Module):
    def __init__(self, smooth=True, beta=10.0, offset=1.0):
        super().__init__()
        if smooth:
            self.activation = torch.nn.Softplus(beta=beta, threshold=20.0)
        else:
            self.activation = torch.nn.relu
        self.offset = offset    
    
    def __call__(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        if "perturbing_vector" not in pred:
            # The probe runs in training updates only; validation reports the other terms.
            return ref.positions.new_zeros(())
        num_graphs = ref.ptr.numel() - 1
        # scf_random_vector_stability_loss: [n_graphs, 1]
        perturbation_norms = scatter_sum(
            pred["perturbing_vector"] ** 2, ref["batch"], dim=0, dim_size=num_graphs
        )
        output_norms = scatter_sum(
            pred["perturbed_density"] ** 2, ref["batch"], dim=0, dim_size=num_graphs
        )
        return torch.mean(self.activation(output_norms / perturbation_norms - self.offset))



def spectral_errors(ref, pred, key):
    """Per-graph Parseval MSE of a Fourier field and its label at the same resolution.

    Potentials are compared modulo a constant; the density keeps k=0, its total charge.
    """
    label = 'fourier_potential' if key.endswith('potential') else 'fourier_density'
    weights = ref.weight*getattr(ref, label+'_weight')
    target = pred.get(key+'_dft')
    if target is None:
        if bool((weights > 0).any()):
            raise ValueError('Missing Fourier observation for '+key)
        return pred['energy']*0.
    active = pred['k_vectors_mask'] & pred[label+'_dft_mask'] & (weights > 0)[:, None]
    if label == 'fourier_potential':
        active = active & (pred['k_vectors'].square().sum(-1) > 0)
    difference = pred[key]-target
    if bool((active[..., None] & ~torch.isfinite(difference)).any()):
        raise FloatingPointError('Nonfinite observed Fourier label or prediction: '+key)
    difference = torch.where(active[..., None], difference, 0.)
    return difference.square().sum((-1, -2))/pred['k_vectors_grid_shape'].prod().to(difference).square()


def vacuum_observation_weight(ref):
    """Measured vacuum labels on two-periodic slabs."""
    slab = ref.pbc.reshape(-1, 3).bool().sum(-1) == 2
    return ref.weight*ref.vacuum_potential_weight*slab


class WeightedFourierDensity(torch.nn.Module):
    """Raw density, or density at the Gaussian resolution of the model (smeared) or of the far field."""
    def __init__(self, reference='smeared'):
        super().__init__()
        keys = {'raw': 'fourier_density', 'smeared': 'smooth_fourier_density', 'farfield': 'fourier_farfield_density'}
        if reference not in keys:
            raise ValueError('fourier_density reference must be raw, smeared or farfield')
        self.reference, self.key = reference, keys[reference]

    def forward(self, ref, pred):
        weights = ref.weight*ref.fourier_density_weight
        return (spectral_errors(ref, pred, self.key)/1.e-6*weights).sum()/weights.sum().clamp_min(1.e-30)


class WeightedFourierPotential(torch.nn.Module):
    """Raw and/or Gaussian-smoothed deformation potential, modulo a constant."""
    def __init__(self, reference='smeared'):
        super().__init__()
        keys = {'raw': ('fourier_potential',), 'smeared': ('smooth_fourier_potential',),
                'raw+smeared': ('fourier_potential', 'smooth_fourier_potential')}
        if reference not in keys:
            raise ValueError('fourier_potential reference must be raw, smeared or raw+smeared')
        self.reference, self.keys = reference, keys[reference]

    def forward(self, ref, pred):
        weights = ref.weight*ref.fourier_potential_weight
        square = sum(spectral_errors(ref, pred, key) for key in self.keys)
        return (square*weights).sum()/weights.sum().clamp_min(1.e-30)


class WeightedScalarLoss(torch.nn.Module):
    """Absolute scalar MSE, normalized by the observed label weight."""
    key = ''

    def weights(self, ref):
        if self.key == 'vacuum_potential':
            return vacuum_observation_weight(ref).reshape(-1)
        if self.key == 'work_function':
            return (vacuum_observation_weight(ref)*ref.fermi_level_weight).reshape(-1)
        return (ref.weight*getattr(ref, self.key+'_weight')).reshape(-1)

    def label(self, ref):
        if self.key == 'work_function':
            return ref.vacuum_potential-ref.fermi_level
        return getattr(ref, self.key)

    def errors(self, ref, pred):
        value = pred[self.key].reshape(-1)
        weight = self.weights(ref).to(value)
        error = value-self.label(ref).reshape(-1).to(value)
        if bool(((weight > 0) & ~torch.isfinite(error)).any()):
            raise FloatingPointError('Nonfinite observed '+self.key+' error')
        return torch.where(weight > 0, error, 0.), weight

    def forward(self, ref, pred):
        error, weight = self.errors(ref, pred)
        return (weight*error.square()).sum()/weight.sum().clamp_min(1.e-30)


class WeightedFermiLevel(WeightedScalarLoss):
    key = 'fermi_level'


class WeightedVacuumPotential(WeightedScalarLoss):
    key = 'vacuum_potential'


class WeightedWorkFunction(WeightedScalarLoss):
    """Vacuum level minus Fermi level, independent of the potential origin."""
    key = 'work_function'


_LOSS_FUNCTIONS = {
    "fourier_density": WeightedFourierDensity,
    "fourier_potential": WeightedFourierPotential,
    "vacuum_potential": WeightedVacuumPotential,
    "work_function": WeightedWorkFunction,
    "energy_per_atom": weighted_mean_squared_error_energy,
    "forces": mean_squared_error_forces,
    "stress": weighted_mean_squared_stress,
    "atomic_multipoles": weighted_mean_squared_error_dma,
    "total_charge": weighted_mean_squared_error_charge_extensive,
    "dipole": weighted_mean_squared_error_dipole_extensive,
    "total_charge_per_atom": weighted_mean_squared_error_charge,
    "dipole_per_atom": weighted_mean_squared_error_dipole,
    "polarizability": weighted_mean_squared_error_polarizability,
    "fermi_level_per_atom": weighted_mean_squared_error_fermi,
    "fermi_level": WeightedFermiLevel,
    "esps": weighted_mean_squared_error_esp,
    "cluster_virial_per_atom": weighted_mean_squared_cluster_virial,
    "cluster_virial": weighted_mean_squared_cluster_virial_extensive,
    "molecular_forces": weighted_mean_squared_molecular_forces,
    "fixedpoint_scf_stability": FixedPointStability,
    "fermi_level_gradient": fermi_level_gradient_function,
    "final_terms_fixedpoint_scf_stability": final_terms_fixedpoint_scf_stability,
    "field_features": weighted_mean_squared_error_field_feats,
}


class WeightedLoss(torch.nn.Module):
    def __init__(
        self,
        _weights_and_options,
    ):
        super().__init__()
        self.loss_weights = {}
        self.loss_fns = {}
        weights_and_options = deepcopy(_weights_and_options)
        for name, options in weights_and_options.items():
            if not name in _LOSS_FUNCTIONS:
                raise ValueError(f"requested `{name}` in loss, which is not recognised")
            if isinstance(options, (int, float)):
                options = {"weight": options}
            self.loss_weights[name] = options.pop("weight")
            function = _LOSS_FUNCTIONS[name]
            if options or (isinstance(function, type) and issubclass(function, torch.nn.Module)):
                self.loss_fns[name] = function(**options)
            else:
                self.loss_fns[name] = function

    def denominators(self, ref):
        """Observation count of each term in a batch, to pool a loss over batches."""
        result = {}
        for name, function in self.loss_fns.items():
            if isinstance(function, WeightedScalarLoss):
                result[name] = function.weights(ref).sum()
            elif name in ("fourier_density", "fourier_potential"):
                result[name] = (ref.weight*getattr(ref, name+"_weight")).sum()
            else:
                atoms = name in ("forces", "atomic_multipoles", "esps", "field_features")
                result[name] = ref.weight.new_tensor(float(len(ref.positions) if atoms else ref.num_graphs))
        return result

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        loss = 0.
        # set weights to discount non-converged scf
        if "loss_weight_modifier" in pred:
            data_weight = torch.clone(ref.weight)
            ref.weight = ref.weight * pred["loss_weight_modifier"]
        for name, func in self.loss_fns.items():
            if self.loss_weights[name]:
                loss += self.loss_weights[name] * func(ref, pred)
        if "loss_weight_modifier" in pred:
            ref.weight = data_weight
        return loss

    def __repr__(self):
        string = f"{self.__class__.__name__}("
        for name in self.loss_fns:
            string += f"{name}_weight={self.loss_weights[name]}, "
            if hasattr(self.loss_fns[name], "reference"):
                string += f"{name}_reference={self.loss_fns[name].reference}, "
        return string + ")"

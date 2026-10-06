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
import logging


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
        num_graphs = ref.ptr.numel() - 1
        # scf_random_vector_stability_loss: [n_graphs, 1]
        perturbation_norms = scatter_sum(
            pred["perturbing_vector"] ** 2, ref["batch"], dim=0, dim_size=num_graphs
        )
        output_norms = scatter_sum(
            pred["perturbed_density"] ** 2, ref["batch"], dim=0, dim_size=num_graphs
        )
        return torch.mean(self.activation(output_norms / perturbation_norms - self.offset))



def _label_mean(error, weights):
    """Mean over observed labels; missing labels never dilute the objective."""
    weights = torch.broadcast_to(weights, error.shape)
    valid = (weights > 0) & torch.isfinite(error) & torch.isfinite(weights)
    if bool(((weights > 0) & ~torch.isfinite(error)).any()):
        raise FloatingPointError("A positively weighted label or prediction is nonfinite")
    safe = torch.where(valid, error, 0.)
    weight = torch.where(valid, weights, 0.)
    return (safe.square()*weight).sum()/weight.sum().clamp_min(1.e-30)


def spectral_errors(ref, pred, key, full_spectrum=True):
    """Per-graph Parseval MSE in physical real-space units."""
    weight_key = ('fourier_density' if key == 'fourier_farfield_density' else
                  'fourier_potential' if key == 'fourier_total_potential' else key)
    weights = (ref.weight*getattr(ref, weight_key+'_weight') if ref is not None else
               pred[key].new_ones(len(pred[key])))
    if key == 'fourier_total_potential' and ref is not None:
        weights = weights*ref.fourier_proto_potential_weight
    target = pred.get(key+"_dft")
    if target is None:
        if bool((weights > 0).any()):
            raise ValueError("Missing Fourier observation for " + key)
        return pred["energy"]*0.
    difference = pred[key]-target
    active = pred["k_vectors_mask"]
    if key+"_dft_mask" in pred:
        active = active & pred[key+"_dft_mask"]
    if key in ("fourier_potential", "fourier_total_potential"):
        active = active & (pred["k_vectors"].square().sum(-1)>0)
    active = active & (weights > 0)[:, None]
    if bool((active[...,None] & ~torch.isfinite(difference)).any()):
        raise FloatingPointError('Nonfinite observed Fourier label or prediction: '+key)
    mode_weight = (pred.get("potential_mode_weight", active.to(difference))
                   if key == "fourier_potential" and not full_spectrum else active.to(difference))
    difference = torch.where(active[..., None], difference, 0.)
    size = pred["k_vectors_grid_shape"].prod().to(difference)
    return (difference.square().sum(-1)*mode_weight).sum(-1)/size.square()


def weighted_fourier_density(ref, pred):
    # Density is expressed in millielectrons/Angstrom^3 to keep weights legible.
    weight = ref.weight*ref.fourier_density_weight
    square = spectral_errors(ref, pred, "fourier_density")/1.e-6
    return (square*weight).sum()/weight.sum().clamp_min(1.e-30)


def weighted_fourier_potential(ref, pred):
    weight = ref.weight*ref.fourier_potential_weight
    square = spectral_errors(ref, pred, "fourier_potential")
    return (square*weight).sum()/weight.sum().clamp_min(1.e-30)


def vacuum_reference_weights(ref, reference):
    """v366 same-gauge vacuum target EF_DFT + WF_DFT, from labels only."""
    zero = reference.reshape(-1).new_zeros(reference.numel())
    ef, wf = getattr(ref, 'fermi_level', None), getattr(ref, 'workfunction', None)
    if ef is None or wf is None:
        return zero, zero
    ef, wf = ef.reshape(-1).to(reference), wf.reshape(-1).to(reference)
    ew = getattr(ref, 'fermi_level_weight', torch.ones_like(ef)).reshape(-1)
    ww = getattr(ref, 'workfunction_weight', torch.ones_like(wf)).reshape(-1)
    weight = ref.weight.reshape(-1)*ww
    active = (ew>0)&(weight>0)
    active &= ref.pbc.reshape(-1,3).bool().sum(-1) == 2
    if bool((active & ~(torch.isfinite(ef)&torch.isfinite(wf)&torch.isfinite(ew)&torch.isfinite(weight))).any()):
        raise FloatingPointError('Nonfinite positively weighted EF/WF vacuum reference')
    return torch.where(active, ef+wf, 0.), torch.where(active, weight, 0.)


class WeightedFourierDensity(torch.nn.Module):
    """Parseval error of the original or explicitly smoothed density observer."""
    def __init__(self, metric='l2', reference='density'):
        super().__init__()
        if metric != 'l2' or reference not in ('density', 'multipoles', 'farfield'):
            raise ValueError('fourier_density supports metric=l2 and reference=density, multipoles or farfield')
        self.reference = reference

    def extra_repr(self):
        return 'reference='+self.reference

    def forward(self, ref, pred):
        key = 'fourier_farfield_density' if self.reference == 'farfield' else 'fourier_density'
        square = spectral_errors(ref, pred, key)/1.e-6
        weights = ref.weight*ref.fourier_density_weight
        return (square*weights).sum()/weights.sum().clamp_min(1.e-30)


class WeightedFourierPotential(torch.nn.Module):
    """Real-space potential MSE plus optional same-gauge vacuum MSE.

    The Fourier representation evaluates Parseval's identity. The vacuum term
    uses EF_DFT + WF_DFT; it never differentiates predicted EF or a WF residual.
    Spatial and vacuum observations have separate availability/normalization.
    """
    def __init__(self, vacuum_weight=0., full_spectrum=True):
        super().__init__()
        import math
        if (isinstance(vacuum_weight, bool) or not isinstance(vacuum_weight, (float,int))
                or not math.isfinite(vacuum_weight) or vacuum_weight < 0):
            raise ValueError('fourier_potential.vacuum_weight must be finite and nonnegative')
        if not isinstance(full_spectrum, bool):
            raise TypeError('fourier_potential.full_spectrum must be bool')
        self.vacuum_weight, self.full_spectrum = float(vacuum_weight), full_spectrum

    def extra_repr(self):
        return f'vacuum_weight={self.vacuum_weight:g}, full_spectrum={self.full_spectrum}'

    def statistics(self, ref, pred):
        weights = ref.weight*ref.fourier_potential_weight
        square = spectral_errors(ref, pred, 'fourier_potential', self.full_spectrum)
        parts = {'spatial': ((square*weights).sum(), weights.sum())}
        if self.vacuum_weight:
            value = pred['vacuum_potential'].reshape(-1)
            target, weights = vacuum_reference_weights(ref, value)
            if bool(((weights>0)&~torch.isfinite(value)).any()):
                raise FloatingPointError('Nonfinite prediction on an observed vacuum label')
            error = torch.where(weights>0, value-target, 0.)
            parts['vacuum'] = (self.vacuum_weight*(weights*error.square()).sum(), weights.sum())
        return parts

    def forward(self, ref, pred):
        return sum(numerator/denominator.clamp_min(1.e-30)
                   for numerator,denominator in self.statistics(ref,pred).values())


def weighted_fermi_level(ref, pred):
    return _label_mean(pred["fermi_level"]-ref.fermi_level, ref.weight*ref.fermi_level_weight)


def weighted_workfunction(ref, pred):
    return _label_mean(pred["workfunction"]-ref.workfunction, ref.weight*ref.workfunction_weight)


def vacuum_observation_weight(ref):
    """Only two-periodic slabs have a vacuum plane observation."""
    slab = ref.pbc.reshape(-1, 3).sum(-1) == 2
    return ref.weight*ref.fourier_potential_weight*ref.fourier_proto_potential_weight*slab


def weighted_vacuum_potential(ref, pred):
    weight = vacuum_observation_weight(ref)
    if "vacuum_potential_dft" not in pred:
        if bool((weight>0).any()):
            raise ValueError("Vacuum supervision requires both deformation and proto spectra")
        return pred["energy"].sum()*0.
    return _label_mean(pred["vacuum_potential"]-pred["vacuum_potential_dft"], weight)


_LOSS_FUNCTIONS = {
    "fourier_density": WeightedFourierDensity,
    "fourier_potential": WeightedFourierPotential,
    "workfunction": weighted_workfunction,
    "vacuum_potential": weighted_vacuum_potential,
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
    "fermi_level": weighted_fermi_level,
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
                options = {'weight': options}
            self.loss_weights[name] = options.pop("weight")
            function = _LOSS_FUNCTIONS[name]
            if isinstance(function, type) and issubclass(function, torch.nn.Module):
                self.loss_fns[name] = function(**options)
            elif options:
                self.loss_fns[name] = function(**options)
            else:
                self.loss_fns[name] = function
        
    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        loss = 0.
        logstring = "loss breakdown: "
        # set weights to discount non-converged scf
        if "loss_weight_modifier" in pred:
            data_weight = torch.clone(ref.weight)
            ref.weight = ref.weight * pred["loss_weight_modifier"]
        for name, func in self.loss_fns.items():
            if self.loss_weights[name] == 0:
                continue
            loss_component = self.loss_weights[name] * func(ref, pred)
            loss += loss_component
            logstring += f'{name}: {loss_component}, ' 
        logging.debug(logstring)
        if "loss_weight_modifier" in pred:
            ref.weight = data_weight
        return loss

    def __repr__(self):
        string = f"{self.__class__.__name__}("
        for name in self.loss_fns:
            string += f"{name}_weight={self.loss_weights[name]}, "
            function = self.loss_fns[name]
            if isinstance(function, (WeightedFourierDensity, WeightedFourierPotential)):
                string += f'{name} options: {function.extra_repr()}, '
        return string + ")"

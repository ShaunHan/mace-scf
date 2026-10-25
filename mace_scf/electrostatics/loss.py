import math
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
from .desmearing import field_loss_options


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



def spectral_errors(ref, pred, key, apply_smearing=False):
    """Per-graph Parseval MSE; potential fields are compared modulo a constant.

    Raw and zero-mean potential labels give the same nonzero-mode objective.
    Density retains k=0 because it measures total charge, not a voltage gauge.
    """
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
    if apply_smearing:
        # The analytic model already uses finite-width sources. This option
        # reproduces training on Gaussian-smoothed DFT labels, now internally.
        # Never filter the prediction a second time or invert a Gaussian.
        density = key in ('fourier_density', 'fourier_farfield_density')
        width = pred['density_smearing_width' if density else 'potential_smearing_width']
        k2 = pred['k_vectors'].square().sum(-1)
        target = target*torch.exp(-.5*width**2*k2)[...,None]
    difference = pred[key]-target
    active = pred["k_vectors_mask"]
    if key+"_dft_mask" in pred:
        active = active & pred[key+"_dft_mask"]
    if key in ("fourier_potential", "fourier_total_potential"):
        active = active & (pred["k_vectors"].square().sum(-1)>0)
    active = active & (weights > 0)[:, None]
    if bool((active[...,None] & ~torch.isfinite(difference)).any()):
        raise FloatingPointError('Nonfinite observed Fourier label or prediction: '+key)
    difference = torch.where(active[..., None], difference, 0.)
    size = pred["k_vectors_grid_shape"].prod().to(difference)
    return (difference.square().sum(-1)*active).sum(-1)/size.square()


def vacuum_observation_weight(ref):
    """Independent measured vacuum labels on two-periodic slabs."""
    slab = ref.pbc.reshape(-1, 3).bool().sum(-1) == 2
    return ref.weight*ref.vacuum_potential_weight*slab


def vacuum_reference_weights(ref, reference):
    """Return the independently stored same-gauge vacuum observation."""
    target = ref.vacuum_potential.reshape(-1).to(reference)
    weight = vacuum_observation_weight(ref).reshape(-1).to(reference)
    observed = weight > 0
    if bool((observed & ~(torch.isfinite(target) & torch.isfinite(weight))).any()):
        raise FloatingPointError('Nonfinite observed vacuum potential')
    return torch.where(observed, target, 0.), torch.where(observed, weight, 0.)


class WeightedFourierDensity(torch.nn.Module):
    """Exactly one density reference; the frozen output map adds no losses."""
    def __init__(self, reference=None, desmearing=None, reference_width=None, **legacy):
        super().__init__()
        unknown = set(legacy)-{'apply_smearing'}
        if unknown:
            raise TypeError(f'Unknown density loss options: {sorted(unknown)}')
        options = dict(legacy)
        if reference is not None:
            options['reference'] = reference
        if desmearing is not None:
            options['desmearing'] = desmearing
        self.reference, self.desmearing, self.apply_smearing = field_loss_options(options)
        self.reference_width = None if reference_width is None else float(reference_width)
        if self.reference_width is not None and (self.reference != 'farfield'
                or not math.isfinite(self.reference_width) or self.reference_width <= 0):
            raise ValueError('reference_width must be finite and positive, with reference=farfield')

    def extra_repr(self):
        width='' if self.reference_width is None else f', reference_width={self.reference_width} Angstrom'
        return f'reference={self.reference}, desmearing={self.desmearing}'+width

    def reference_fields(self, pred):
        """Change the density observation resolution, retaining the live SCF."""
        if self.reference_width is None:
            return pred
        source_width=pred['density_smearing_width']
        if self.reference_width < float(source_width):
            raise ValueError('Density reference_width must be at least the Gaussian density source width')
        filter=torch.exp(-.5*(self.reference_width**2-source_width**2)*pred['k_vectors'].square().sum(-1))[...,None]
        result=dict(pred)
        for source,target in (('fourier_density','fourier_farfield_density'),
                ('smooth_fourier_density','smooth_fourier_farfield_density'),
                ('fourier_density_dft','fourier_farfield_density_dft')):
            if source in pred:result[target]=pred[source]*filter
        if 'fourier_density_dft_mask' in pred:
            result['fourier_farfield_density_dft_mask']=pred['fourier_density_dft_mask']
        return result

    def forward(self, ref, pred):
        pred=self.reference_fields(pred)
        key = 'fourier_farfield_density' if self.reference == 'farfield' else 'fourier_density'
        if self.apply_smearing and 'smooth_'+key in pred:
            pred = dict(pred, **{key: pred['smooth_'+key]})
        square = spectral_errors(ref, pred, key, self.apply_smearing)/1.e-6
        weights = ref.weight*ref.fourier_density_weight
        return (square*weights).sum()/weights.sum().clamp_min(1.e-30)


class WeightedFourierPotential(torch.nn.Module):
    """Selected Parseval MSE, or their explicit sum for raw+smeared."""
    def __init__(self, desmearing=None, reference=None, **legacy):
        super().__init__()
        unknown = set(legacy)-{'apply_smearing'}
        if unknown:
            raise TypeError(f'Unknown potential loss options: {sorted(unknown)}')
        options = dict(legacy)
        if reference is not None:
            options['reference'] = reference
        if desmearing is not None:
            options['desmearing'] = desmearing
        self.reference, self.desmearing, self.apply_smearing = field_loss_options(options, density=False)

    def extra_repr(self):
        return f'reference={self.reference}, desmearing={self.desmearing}'

    def statistics(self, ref, pred):
        weights = ref.weight*ref.fourier_potential_weight
        square = pred['energy']*0.
        if self.reference in ('raw', 'raw+smeared'):
            square = spectral_errors(ref, pred, 'fourier_potential', False)
        if self.reference in ('smeared', 'raw+smeared'):
            smooth = dict(pred, fourier_potential=pred.get('smooth_fourier_potential', pred['fourier_potential']))
            square = square+spectral_errors(ref, smooth, 'fourier_potential', True)
        return {'spatial': ((square*weights).sum(), weights.sum())}

    def forward(self, ref, pred):
        numerator, denominator = self.statistics(ref, pred)['spatial']
        return numerator/denominator.clamp_min(1.e-30)


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
        if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
            raise ValueError('Scalar observation weights must be finite and nonnegative')
        return torch.where(weight > 0, error, 0.), weight

    def moments(self, ref, pred):
        """Sufficient statistics for the absolute loss and bias diagnostics."""
        error, weight = self.errors(ref, pred)
        return torch.stack((weight.sum(), (weight*error).sum(),
                            (weight*error.square()).sum()))

    def from_moments(self, moments):
        weight, _, second = moments.unbind()
        return second/weight.clamp_min(1.e-30)

    def forward(self, ref, pred, normalizers=None):
        error, weight = self.errors(ref, pred)
        denominator = weight.sum() if normalizers is None else normalizers[(self.key, '')].to(error)
        return (weight*error.square()).sum()/denominator.clamp_min(1.e-30)


class WeightedFermiLevel(WeightedScalarLoss):
    key = 'fermi_level'


class WeightedVacuumPotential(WeightedScalarLoss):
    key = 'vacuum_potential'


class WeightedWorkFunction(WeightedScalarLoss):
    """Gauge-invariant vacuum minus Fermi level; EF/vacuum fix the gauge."""
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
        self.definition = {name: deepcopy(value) if isinstance(value, dict) else {'weight': value}
                           for name, value in weights_and_options.items()}
        for name, options in weights_and_options.items():
            if name == 'cell_voltage':
                raise ValueError('cell_voltage is no longer a training target: NVT-Phi uses the cell voltage '
                                 'as the energy derivative dE/dQ along Q_cc = -Q, which needs no label. '
                                 'Remove cell_voltage from the loss.')
            if not name in _LOSS_FUNCTIONS:
                raise ValueError(f"requested `{name}` in loss, which is not recognised")
            if isinstance(options, (int, float)):
                options = {'weight': options}
            obsolete = {'relative', 'mean_weight'} & options.keys()
            if obsolete:
                raise ValueError(f'Remove {sorted(obsolete)} from loss {name}; '
                                 'scalar labels now always use absolute MSE.')
            self.loss_weights[name] = options.pop("weight")
            if name in ('energy_per_atom', 'forces', 'stress'):
                reference = options.pop('reference', 'dft')
                if reference != 'dft':
                    raise ValueError('Energy, force and stress targets must use the supplied DFT convention; '
                                     'no vacuum-energy label conversion or second energy target is used.')
            function = _LOSS_FUNCTIONS[name]
            if isinstance(function, type) and issubclass(function, torch.nn.Module):
                self.loss_fns[name] = function(**options)
            elif options:
                self.loss_fns[name] = function(**options)
            else:
                self.loss_fns[name] = function
            if isinstance(self.loss_fns[name], WeightedScalarLoss):
                # Internal checkpoint semantics, not a configurable loss option.
                self.definition[name]['objective'] = 'absolute_mse'
            if name in ('fourier_density', 'fourier_potential'):
                # Store the resolved target convention so a resumed objective
                # remains unambiguous when defaults evolve.
                self.definition[name].pop('apply_smearing', None)
                self.definition[name]['reference'] = self.loss_fns[name].reference
                self.definition[name]['desmearing'] = self.loss_fns[name].desmearing

    @classmethod
    def from_checkpoint(cls, definition):
        """Read an equivalent saved objective, or request a new loss baseline.

        Old explicit relative:false settings are exactly the absolute loss.
        A removed objective cannot reproduce its historical best loss; resume
        must re-evaluate that checkpoint on the current objective instead.
        """
        definition = deepcopy(definition)
        if 'cell_voltage' in definition:
            return None
        for name, options in definition.items():
            if name in ('fermi_level', 'vacuum_potential', 'work_function') and (
                    not isinstance(options, dict) or not (
                        options.get('objective') == 'absolute_mse'
                        or options.get('relative') is False)):
                return None
            if isinstance(options, dict):
                if options.get('relative', False) or 'mean_weight' in options:
                    return None
                options.pop('relative', None)
                if name in ('fermi_level', 'vacuum_potential', 'work_function'):
                    options.pop('objective', None)
        return cls(definition)
        
    def normalizers(self, ref: Batch):
        """Full-batch denominators for exact graph-wise gradient accumulation.

        Native means include unobserved zero-weight entries; electronic label
        means instead divide by observed weight. Keep that distinction when
        fitting a large optimizer batch in smaller device batches.
        """
        result={}
        atom_means={'forces','atomic_multipoles','esps','field_features'}
        label_means={'fourier_density'}
        unsupported={'fixedpoint_scf_stability','fermi_level_gradient','final_terms_fixedpoint_scf_stability'}
        for name,function in self.loss_fns.items():
            if not self.loss_weights[name]:continue
            if name in unsupported:
                raise ValueError(f'Graph accumulation is not defined for trajectory loss {name}')
            if isinstance(function, WeightedScalarLoss):
                result[(name, '')] = function.weights(ref).sum()
            elif isinstance(function,WeightedFourierPotential):
                result[(name,'spatial')]=(ref.weight*ref.fourier_potential_weight).sum()
            elif name in label_means:
                result[(name,'')]=(ref.weight*getattr(ref,name+'_weight')).sum()
            else:
                result[(name,'')]=ref.weight.new_tensor(len(ref.positions) if name in atom_means else ref.num_graphs)
        return result

    @property
    def supports_graph_accumulation(self):
        trajectory={'fixedpoint_scf_stability','fermi_level_gradient','final_terms_fixedpoint_scf_stability'}
        return not any(self.loss_weights.get(name,0) for name in trajectory)

    def forward(self, ref: Batch, pred: TensorDict, normalizers=None) -> torch.Tensor:
        loss = 0.
        local_normalizers=self.normalizers(ref) if normalizers is not None else None
        # set weights to discount non-converged scf
        if "loss_weight_modifier" in pred:
            if normalizers is not None:
                raise ValueError('Prediction-dependent convergence weights require an unsplit batch')
            data_weight = torch.clone(ref.weight)
            ref.weight = ref.weight * pred["loss_weight_modifier"]
        for name, func in self.loss_fns.items():
            if self.loss_weights[name] == 0:
                continue
            if isinstance(func, WeightedScalarLoss):
                value = func(ref, pred, normalizers=normalizers)
            elif normalizers is not None and isinstance(func,WeightedFourierPotential):
                value=sum(numerator/normalizers[(name,part)].to(numerator).clamp_min(1.e-30)
                          for part,(numerator,_) in func.statistics(ref,pred).items())
            elif normalizers is not None:
                local=local_normalizers[(name,'')]
                value=func(ref,pred)*local/normalizers[(name,'')].to(local).clamp_min(1.e-30)
            else:
                value=func(ref,pred)
            loss_component = self.loss_weights[name] * value
            loss += loss_component
        if "loss_weight_modifier" in pred:
            ref.weight = data_weight
        return loss+self.reference_loss(ref,pred,normalizers)

    def reference_loss(self, ref, pred, normalizers=None):
        """Auxiliary electronic supervision; absent from deployment evaluation."""
        loss = 0.
        reference = pred.get('reference_response')
        if reference is not None:
            from types import SimpleNamespace
            # The auxiliary uses the existing electronic weights. Missing
            # teachers contribute zero under the SAME full-batch denominator,
            # including when an optimizer batch is split for device memory.
            for name, mask in reference['reference_masks'].items():
                if not self.loss_weights.get(name,0):
                    continue
                masked = SimpleNamespace(**ref.to_dict())
                masked.weight = ref.weight*mask.to(ref.weight)
                if name == 'vacuum_potential':
                    count, base = vacuum_observation_weight(masked).sum(), vacuum_observation_weight(ref).sum()
                else:
                    label_weight = getattr(ref,name+'_weight')
                    count, base = (masked.weight*label_weight).sum(), (ref.weight*label_weight).sum()
                if normalizers is not None:
                    base = normalizers[(name,'spatial' if name=='fourier_potential' else '')].to(base)
                function = self.loss_fns[name]
                if isinstance(function, WeightedScalarLoss):
                    denominators = self.normalizers(ref) if normalizers is None else normalizers
                    value = function(masked, reference, normalizers=denominators)
                else:
                    value = function(masked,reference)*count/base.clamp_min(1.e-30)
                loss += self.loss_weights[name]*value
        return loss

    def __repr__(self):
        string = f"{self.__class__.__name__}("
        for name in self.loss_fns:
            string += f"{name}_weight={self.loss_weights[name]}, "
            function = self.loss_fns[name]
            if isinstance(function, (WeightedFourierDensity, WeightedFourierPotential)):
                string += f'{name} options: {function.extra_repr()}, '
        return string + ")"

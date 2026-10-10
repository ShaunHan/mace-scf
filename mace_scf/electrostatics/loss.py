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
    """Parseval error of the original or explicitly smoothed density observer."""
    def __init__(self, reference='density'):
        super().__init__()
        if reference not in ('density', 'farfield'):
            raise ValueError('fourier_density reference must be density or farfield')
        self.reference = reference

    def extra_repr(self):
        return 'reference='+self.reference

    def forward(self, ref, pred):
        key = 'fourier_farfield_density' if self.reference == 'farfield' else 'fourier_density'
        square = spectral_errors(ref, pred, key)/1.e-6
        weights = ref.weight*ref.fourier_density_weight
        return (square*weights).sum()/weights.sum().clamp_min(1.e-30)


class WeightedFourierPotential(torch.nn.Module):
    """Real-space MSE of the retained potential after uniform-offset alignment."""
    def statistics(self, ref, pred):
        weights = ref.weight*ref.fourier_potential_weight
        square = spectral_errors(ref, pred, 'fourier_potential')
        return {'spatial': ((square*weights).sum(), weights.sum())}

    def forward(self, ref, pred):
        numerator, denominator = self.statistics(ref, pred)['spatial']
        return numerator/denominator.clamp_min(1.e-30)


class RelativeScalarLoss(torch.nn.Module):
    """Scalar differences with one free additive gauge per optimizer batch.

    For equal weights the relative objective is the mean squared error of
    all pair differences, divided by two. Its expectation is the population
    error variance, independently of batch size. One label supplies no pair.
    This changes the gradients, not just the reported absolute error.
    """
    key = ''

    def __init__(self, relative=True):
        super().__init__()
        if not isinstance(relative, bool):
            raise TypeError('relative must be a boolean')
        self.relative = relative

    def extra_repr(self):
        return f'relative={self.relative}'

    def weights(self, ref):
        return (vacuum_observation_weight(ref) if self.key == 'vacuum_potential'
                else ref.weight*ref.fermi_level_weight).reshape(-1)

    def errors(self, ref, pred):
        value = pred[self.key].reshape(-1)
        weight = self.weights(ref).to(value)
        error = value-getattr(ref, self.key).reshape(-1).to(value)
        if bool(((weight > 0) & ~torch.isfinite(error)).any()):
            raise FloatingPointError('Nonfinite observed '+self.key+' error')
        if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
            raise ValueError('Scalar observation weights must be finite and nonnegative')
        return torch.where(weight > 0, error, 0.), weight

    def moments(self, ref, pred):
        error, weight = self.errors(ref, pred)
        return torch.stack((weight.sum(), (weight*error).sum(),
                            (weight*error.square()).sum(), weight.square().sum()))

    def from_moments(self, moments):
        weight, first, second, square_weight = moments.unbind()
        safe = weight.clamp_min(1.e-30)
        if self.relative:
            denominator = weight-square_weight/safe
            return torch.where(denominator > 0,
                (second-first.square()/safe).clamp_min(0.)/denominator.clamp_min(1.e-30), second*0.)
        return second/safe

    def forward(self, ref, pred, normalizers=None, reference=False):
        error, weight = self.errors(ref, pred)
        total = weight.sum()
        if self.relative:
            center_key = (self.key, 'reference_mean' if reference else 'mean')
            if normalizers is not None and center_key in normalizers:
                center = normalizers[center_key].to(error)
            else:
                # A detached optimal offset has the identical derivative:
                # the weighted sum of centered residuals is zero.
                center = (error*weight).sum().detach()/total.clamp_min(1.e-30)
                if normalizers is not None and not reference:
                    full = normalizers[(self.key, '')].to(total)
                    if not torch.isclose(total, full):
                        raise ValueError('Relative losses require the whole optimizer-batch mean before microbatch backward')
            denominator = (normalizers[(self.key, 'pairs')].to(error) if normalizers is not None
                           else total-weight.square().sum()/total.clamp_min(1.e-30))
            value = (weight*(error-center).square()).sum()
            return torch.where(denominator > 0, value/denominator.clamp_min(1.e-30), value*0.)
        denominator = total if normalizers is None else normalizers[(self.key, '')].to(error)
        return (weight*error.square()).sum()/denominator.clamp_min(1.e-30)


class WeightedFermiLevel(RelativeScalarLoss):
    key = 'fermi_level'


class WeightedVacuumPotential(RelativeScalarLoss):
    key = 'vacuum_potential'


_LOSS_FUNCTIONS = {
    "fourier_density": WeightedFourierDensity,
    "fourier_potential": WeightedFourierPotential,
    "vacuum_potential": WeightedVacuumPotential,
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
        
    def normalizers(self, ref: Batch):
        """Full-batch denominators for exact graph-wise gradient accumulation.

        Native means include unobserved zero-weight entries; electronic label
        means instead divide by observed weight. Keep that distinction when
        fitting a large optimizer batch in smaller device batches.
        """
        result={}
        atom_means={'forces','atomic_multipoles','esps','field_features'}
        label_means={'fourier_density','fermi_level'}
        unsupported={'fixedpoint_scf_stability','fermi_level_gradient','final_terms_fixedpoint_scf_stability'}
        for name,function in self.loss_fns.items():
            if not self.loss_weights[name]:continue
            if name in unsupported:
                raise ValueError(f'Graph accumulation is not defined for trajectory loss {name}')
            if isinstance(function, RelativeScalarLoss):
                weight = function.weights(ref)
                total = weight.sum()
                result[(name, '')] = total
                if function.relative:
                    result[(name, 'pairs')] = total-weight.square().sum()/total.clamp_min(1.e-30)
            elif isinstance(function,WeightedFourierPotential):
                result[(name,'spatial')]=(ref.weight*ref.fourier_potential_weight).sum()
            elif name in label_means:
                result[(name,'')]=(ref.weight*getattr(ref,name+'_weight')).sum()
            elif name=='vacuum_potential':
                result[(name,'')]=vacuum_observation_weight(ref).sum()
            else:
                result[(name,'')]=ref.weight.new_tensor(len(ref.positions) if name in atom_means else ref.num_graphs)
        return result

    @property
    def relative_scalar_losses(self):
        return {name: fn for name, fn in self.loss_fns.items()
                if self.loss_weights[name] and isinstance(fn, RelativeScalarLoss) and fn.relative}

    def relative_moments(self, ref, pred):
        """Small sufficient statistics for an exact memory-bounded replay."""
        from types import SimpleNamespace
        result = {}
        for name, function in self.relative_scalar_losses.items():
            result[(name, 'mean')] = function.moments(ref, pred)[:2].detach()
            branch = pred.get('reference_response')
            if branch is not None and name in branch['reference_masks']:
                masked = SimpleNamespace(**ref.to_dict())
                masked.weight = ref.weight*branch['reference_masks'][name].to(ref.weight)
                result[(name, 'reference_mean')] = function.moments(masked, branch)[:2].detach()
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
            if isinstance(func, RelativeScalarLoss):
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
                if name != 'fermi_level' and getattr(self.loss_fns.get('fermi_level'), 'relative', False):
                    # An absolute DFT chemical potential cannot constrain a
                    # grand-canonical auxiliary in a freely chosen EF gauge.
                    # The measured-field -> relative EF auxiliary remains valid.
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
                if isinstance(function, RelativeScalarLoss):
                    denominators = self.normalizers(ref) if normalizers is None else normalizers
                    value = function(masked, reference, normalizers=denominators, reference=True)
                else:
                    value = function(masked,reference)*count/base.clamp_min(1.e-30)
                loss += self.loss_weights[name]*value
        return loss

    def __repr__(self):
        string = f"{self.__class__.__name__}("
        for name in self.loss_fns:
            string += f"{name}_weight={self.loss_weights[name]}, "
            function = self.loss_fns[name]
            if isinstance(function, (WeightedFourierDensity, RelativeScalarLoss)):
                string += f'{name} options: {function.extra_repr()}, '
        return string + ")"

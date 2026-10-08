import numpy as np
import pytest
import torch
from e3nn import o3
from mace import modules
from mace.data import config_from_atoms, KeySpecification
from mace.tools import AtomicNumberTable, torch_geometric
from ase import Atoms

from mace_scf.data import ExtAtomicData
from mace_scf.electrostatics.fixed_point_core import FixedPointCore
from mace_scf.electrostatics.field_blocks import StrictQuadraticFieldEnergyReadout
from mace_scf.electrostatics.potential import (
    VariationalResponse, SpectralGeometry, conjugate_gradient, evaluate_variational, unroll_electronic,
)


@pytest.mark.parametrize('zero', [False, True])
def test_finite_iteration_first_second_derivatives_and_checkpoint(zero):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(13)
    matrix = torch.randn(1, 6, 6)*.2
    coulomb = matrix@matrix.mT
    hardness = torch.full((1, 3, 2), 2., requires_grad=True)
    rhs = (torch.zeros_like(hardness) if zero else torch.randn_like(hardness)).requires_grad_()
    constraint = torch.zeros_like(rhs)
    constraint[..., 0] = 1.
    diag = coulomb.diagonal(dim1=-2, dim2=-1).reshape_as(rhs)

    def solve(value, local, checkpoint=False):
        operator = lambda state: local*state+(coulomb@state.flatten(1)[..., None]).reshape_as(state)
        return unroll_electronic(operator, value, local, diag, constraint, 3, checkpoint)

    assert torch.autograd.gradcheck(solve, (rhs, hardness))
    assert torch.autograd.gradgradcheck(solve, (rhs, hardness))
    full = solve(rhs, hardness)
    short = solve(rhs, hardness, True)
    torch.testing.assert_close(full, short)
    for result in (full, short):
        gradient = torch.autograd.grad(result.square().sum()+result[0, 0, 1], (rhs, hardness), create_graph=True)
        assert torch.isfinite(torch.autograd.grad(sum(g.square().sum() for g in gradient), rhs)[0]).all()
        assert gradient[0].abs().max() > 0


@pytest.mark.parametrize('steps', [1, 3, 12])
def test_finite_energy_force_ef_conjugacy(steps):
    model = small_model()
    data = small_data()
    result = evaluate_variational(model, data, steps=steps, mode='unroll_scf', training=True)
    epsilon = 1.e-5
    charge_energies, position_energies = [], []
    for sign in (-1, 1):
        charge_energies.append(evaluate_variational(model, small_data(.1+sign*epsilon), steps=steps,
                                                   mode='unroll_scf', compute_force=False)['energy'])
        shifted = small_data()
        shifted['positions'][0, 2] += sign*epsilon
        position_energies.append(evaluate_variational(model, shifted, steps=steps,
                                                     mode='unroll_scf', compute_force=False)['energy'])
    torch.testing.assert_close(-(charge_energies[1]-charge_energies[0])/(2*epsilon), result['fermi_level'], atol=2.e-7, rtol=2.e-5)
    torch.testing.assert_close(-(position_energies[1]-position_energies[0])/(2*epsilon), result['forces'][0, 2:3], atol=2.e-7, rtol=2.e-5)
    loss = result['forces'].square().sum()+result['workfunction'].square().sum()
    loss.backward(inputs=[p for p in model.parameters() if p.requires_grad])
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_finite_checkpoint_mode_matches_forward_and_parameter_gradients():
    model = small_model()
    parameters = [p for p in model.parameters() if p.requires_grad]
    results, gradients = [], []
    for mode in ('unroll_scf', 'shortcut_scf'):
        output = evaluate_variational(model, small_data(), steps=3, mode=mode, training=True)
        objective = output['energy'].square().sum()+output['forces'].square().sum()+output['workfunction'].square().sum()
        gradients.append(torch.autograd.grad(objective, parameters, allow_unused=True))
        results.append(output)
    for key in ('energy', 'forces', 'fermi_level', 'workfunction', 'density_coefficients'):
        torch.testing.assert_close(results[0][key], results[1][key])
    for left, right in zip(*gradients):
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, atol=1.e-10, rtol=1.e-9)


def test_finite_trajectory_rotation_invariance():
    model = small_model()
    original = small_data()
    rotated = small_data()
    rotation = o3.rand_matrix()
    for key in ('positions', 'cell', 'shifts', 'external_field'):
        rotated[key] = rotated[key]@rotation.T
    before = evaluate_variational(model, original, steps=3, mode='unroll_scf')
    after = evaluate_variational(model, rotated, steps=3, mode='unroll_scf')
    for key in ('energy', 'fermi_level', 'workfunction'):
        torch.testing.assert_close(before[key], after[key], atol=1.e-9, rtol=1.e-9)
    torch.testing.assert_close(before['forces']@rotation.T, after['forces'], atol=1.e-9, rtol=1.e-9)


@pytest.mark.parametrize('relative', [True, False])
@pytest.mark.parametrize('kind', ['variational','coupled','coupled_proto'])
def test_atomic_vacuum_reference_fit(relative,kind):
    from mace_scf.electrostatics.potential import initialize_vacuum_reference
    if kind == 'variational':
        model=small_model()
    else:
        from .test_coupled_response import coupled_model
        model=coupled_model()
        if kind == 'coupled_proto':
            model.field_dependent_charges_map.proto_fitted.fill_(True)
            model.field_dependent_charges_map.proto_coefficients.fill_(13.)
    reference=torch.tensor([100.,-35.])
    graphs=[]
    for i,(hydrogens,height) in enumerate(((1,12.),(2,12.),(1,15.),(3,18.))):
        atoms=Atoms('O'+'H'*hydrogens,positions=[[3.,3.,4.]]+[[3.4+j*.3,3.,4.5] for j in range(hydrogens)],
                    cell=[7,7,height],pbc=[1,1,0])
        graph=small_data(atoms=atoms,batched=False)
        geometry=SpectralGeometry(model,torch_geometric.Batch.from_data_list([graph]).to_dict(),graph.positions)
        shape=torch.tensor(geometry.shape)
        # Odd shifted Fourier grids, with every retained axial mode observed.
        graph.fourier_potential=torch.zeros(int(shape.prod()),2)
        # A raw DFT constant must not contaminate the deformation plane fit.
        graph.fourier_potential[0,0]=(i+1)*17.*shape.prod()
        graph.fourier_potential_shape=shape
        graph.fourier_potential_weight=torch.tensor(1.)
        graph.fourier_proto_potential_weight=torch.tensor(0.)
        graph.vacuum_potential_weight=torch.tensor(1.)
        graph.vacuum_potential=(graph.node_attrs.sum(0)@reference)/torch.linalg.det(graph.cell).abs()
        if relative: graph.vacuum_potential += 7.
        graphs.append(graph)
    loader=torch_geometric.dataloader.DataLoader(graphs,batch_size=2)
    initialize_vacuum_reference(model,loader,'cpu',relative=relative)
    torch.testing.assert_close(model.field_dependent_charges_map.vacuum_reference_integrals,reference,atol=1.e-8,rtol=1.e-8)


def test_variational_reference_preserves_charge_conjugacy():
    model=small_model()
    model.field_dependent_charges_map.vacuum_reference_integrals.copy_(torch.tensor([100.,-35.]))
    result=evaluate_variational(model,small_data(),steps=12,mode='unroll_scf')
    eps=1.e-5
    a=evaluate_variational(model,small_data(.1-eps),steps=12,mode='unroll_scf')['energy']
    b=evaluate_variational(model,small_data(.1+eps),steps=12,mode='unroll_scf')['energy']
    torch.testing.assert_close(-(b-a)/(2*eps),result['fermi_level'],atol=2.e-7,rtol=2.e-5)


def small_model(widths=(1.5, 3.)):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(7)
    model = FixedPointCore(r_max=4., num_bessel=4, num_polynomial_cutoff=5,
        max_ell=1, interaction_cls=modules.RealAgnosticResidualInteractionBlock,
        interaction_cls_first=modules.RealAgnosticResidualInteractionBlock,
        num_interactions=1, num_elements=2, hidden_irreps=o3.Irreps('4x0e+4x1o'),
        MLP_irreps=o3.Irreps('4x0e'), atomic_energies=np.zeros(2), avg_num_neighbors=3.,
        atomic_numbers=[1, 8], correlation=2, gate=torch.nn.functional.silu,
        atom_density_scaling=np.ones(2), radial_MLP=[4, 4],
        atomic_multipoles_max_l=1, atomic_multipoles_smearing_width=1.5,
        field_feature_max_l=1, field_feature_widths=[1.5, 3.],
        kspace_cutoff_factor=1., include_electrostatic_self_interaction=True,
        fixed_point_update_config={'type': VariationalResponse, 'potential_widths': widths},
        field_readout_config={'type': StrictQuadraticFieldEnergyReadout})
    model.lr_source_maps.requires_grad_(False)
    model.local_electron_energy.requires_grad_(False)
    with torch.no_grad():
        response = model.field_dependent_charges_map
        response.drive.weight.normal_(std=.1)
        response.species_drive[1, 0] = .5
    return model


def small_data(charge=.1, batched=True, atoms=None):
    if atoms is None:
        atoms = Atoms('OHH', positions=[[3., 3., 4.], [3.7, 3., 4.5], [2.7, 3.8, 4.]], cell=[7., 7., 12.], pbc=[1,1,0])
    atoms.info.update(total_charge=charge, external_field=np.array([0., 0., .02]), fermi_level=-.4)
    keys = KeySpecification(info_keys={'total_charge':'total_charge', 'external_field':'external_field', 'fermi_level':'fermi_level'}, arrays_keys={})
    data = ExtAtomicData.from_config(config_from_atoms(atoms, key_specification=keys), z_table=AtomicNumberTable([1,8]), cutoff=4., atomic_multipoles_max_l=1)
    return torch_geometric.Batch.from_data_list([data]).to_dict() if batched else data


def test_projected_solver_and_double_backward():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    a = torch.randn(1, 9, 9)
    a = (a@a.mT+torch.eye(9)).requires_grad_()
    b = torch.randn(1, 3, 3, requires_grad=True)
    c = torch.zeros_like(b)
    c[...,0] = 1.
    op = lambda x: (a@x.flatten(1)[...,None]).reshape_as(x)
    x, mu, residual = conjugate_gradient(op, b, a.diagonal(dim1=-2,dim2=-1).reshape_as(b),50,c,torch.tensor([.2]))
    torch.testing.assert_close((x*c).sum(), torch.tensor(.2), atol=1.e-12, rtol=0)
    assert residual.max() < 1.e-10
    g = torch.autograd.grad(x.square().sum(), b, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(g.sum(), a)[0]).all()


def test_official_gto_parity_and_moment_free_completion():
    from graph_longrange.gto_utils import GTOBasis
    model = small_model()
    data = small_data()
    geom = SpectralGeometry(model, data, data['positions'])
    k = geom.wave.flatten(0,1)
    k2 = geom.k2.flatten()
    zero = (k2==0).to(k2)
    native = GTOBasis(1,[1.5],float(model.kspace_cutoff),'multipoles')(k,k2,zero)
    native = torch.view_as_complex(native.contiguous()).reshape(*geom.k2.shape,4)
    torch.testing.assert_close(native*geom.mask[...,None], geom.basis[...,:4])
    assert geom.basis[:,0,4:].abs().max() == 0


def test_mixed_bulk_slab_batch_has_no_boundary_cross_talk():
    model = small_model()
    slab = small_data(batched=False)
    bulk = slab.clone()
    bulk.pbc = torch.ones_like(bulk.pbc, dtype=torch.bool)
    bulk.external_field.zero_()
    single = evaluate_variational(model, torch_geometric.Batch.from_data_list([bulk]).to_dict())
    mixed = evaluate_variational(model, torch_geometric.Batch.from_data_list([slab, bulk]).to_dict())
    for key in ('energy', 'fermi_level', 'vacuum_potential'):
        torch.testing.assert_close(single[key][0], mixed[key][1], rtol=1.e-9, atol=1.e-9)
    torch.testing.assert_close(single['forces'], mixed['forces'][3:], rtol=1.e-9, atol=1.e-9)
    assert single['vacuum_potential'].item() == 0.


def test_finite_force_parameter_gradient_and_charge_conjugacy():
    model = small_model()
    data = small_data()
    result = evaluate_variational(model, data, steps=50, training=True)
    assert torch.isfinite(result['forces']).all()
    assert result['scf_residual'].max() < 1.e-8
    loss = result['forces'].square().sum()+result['workfunction'].square().sum()
    loss.backward(inputs=[p for p in model.parameters() if p.requires_grad])
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    eps = 1.e-5
    energies=[]
    for q in (.1-eps,.1+eps):
        energies.append(evaluate_variational(model,small_data(q),steps=50,compute_force=False)['energy'])
    torch.testing.assert_close(-(energies[1]-energies[0])/(2*eps),result['fermi_level'],atol=2.e-6,rtol=2.e-5)


def test_no_reference_label_leakage():
    model=small_model()
    data=small_data()
    before=evaluate_variational(model,data,compute_force=False)
    data['fermi_level']=torch.tensor([999.])
    data['density_coefficients']=torch.randn_like(data['density_coefficients'])*100
    after=evaluate_variational(model,data,compute_force=False)
    for key in ('energy','dipole','fermi_level','workfunction'):
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)


def test_energy_potential_are_adjoint_on_the_actual_slab_grid():
    model=small_model()
    data=small_data()
    data['dipole_correction_fraction']=torch.tensor([.73])
    geom=SpectralGeometry(model,data,data['positions'])
    state=torch.randn_like(geom.constraint)*.1
    _,deformation,_=geom.potentials(state)
    torch.testing.assert_close(geom.hartree(state),
        -geom.adjoint(deformation-geom.applied),atol=2.e-12,rtol=2.e-12)


@pytest.mark.parametrize('steps', [50])
def test_force_and_stress_finite_differences(steps):
    model = small_model()
    reference = evaluate_variational(model, small_data(), steps=steps, compute_stress=True)
    epsilon = 1.e-5
    energies, strained = [], []
    for sign in (-1, 1):
        data = small_data()
        data['positions'][1, 2] += sign*epsilon
        energies.append(evaluate_variational(model, data, steps=steps, compute_force=False)['energy'])
        data = small_data()
        deformation = torch.eye(3)
        deformation[0, 0] += sign*epsilon
        for key in ('positions', 'cell', 'shifts'):
            data[key] = data[key]@deformation
        strained.append(evaluate_variational(model, data, steps=steps, compute_force=False)['energy'])
    torch.testing.assert_close(-(energies[1]-energies[0])/(2*epsilon),
        reference['forces'][1, 2:3], atol=2.e-7, rtol=2.e-5)
    volume = torch.linalg.det(small_data()['cell'].reshape(-1,3,3)).abs()
    torch.testing.assert_close((strained[1]-strained[0])/(2*epsilon*volume),
        reference['stress'][:, 0, 0], atol=2.e-7, rtol=2.e-5)


def test_rotation_and_batch_invariance():
    model = small_model()
    data = small_data()
    reference = evaluate_variational(model, data)
    rotation = o3.rand_matrix()
    rotated = small_data()
    for key in ('positions', 'cell', 'shifts', 'external_field'):
        rotated[key] = rotated[key]@rotation.T
    result = evaluate_variational(model, rotated)
    for key in ('energy', 'workfunction', 'fermi_level'):
        torch.testing.assert_close(result[key], reference[key], atol=2.e-10, rtol=2.e-10)
    torch.testing.assert_close(result['forces'], reference['forces']@rotation.T, atol=2.e-10, rtol=2.e-10)
    torch.testing.assert_close(result['dipole'], reference['dipole']@rotation.T, atol=2.e-10, rtol=2.e-10)
    graphs = []
    second = Atoms('OH', positions=[[3., 3., 4.], [3.7, 3., 4.5]], cell=[8., 8., 14.], pbc=[1,1,0])
    for charge, atoms in ((.1, None), (-.2, second)):
        graphs.append(small_data(charge, batched=False, atoms=atoms))
    batched = torch_geometric.Batch.from_data_list(graphs).to_dict()
    result = evaluate_variational(model, batched)
    for i, (charge, atoms) in enumerate(((.1, None), (-.2, second))):
        single = evaluate_variational(model, small_data(charge, atoms=atoms))
        for key in ('energy', 'workfunction', 'fermi_level', 'dipole'):
            torch.testing.assert_close(result[key][i:i+1], single[key], atol=3.e-10, rtol=3.e-10)


def test_padded_fft_modes_are_not_zero_valued_observations():
    from mace_scf.electrostatics.potential import target_mode_mask
    from mace_scf.electrostatics.loss import spectral_errors
    model=small_model()
    result=evaluate_variational(model,small_data(),compute_force=False)
    result['fourier_potential_dft']=result['fourier_potential'].detach().clone()
    modes=SpectralGeometry(model,small_data(),small_data()['positions']).modes
    mask=target_mode_mask(torch.tensor([[3,3,3]]),modes)
    result['fourier_potential_dft_mask']=mask
    result['fourier_potential_dft'][~mask] += 1.e5
    assert spectral_errors(None,result,'fourier_potential').max()==0


def test_zero_initial_density_has_a_nonzero_learning_gradient():
    model=small_model(widths=())
    with torch.no_grad():
        model.field_dependent_charges_map.drive.weight.zero_()
        model.field_dependent_charges_map.species_drive.zero_()
    data=small_data(charge=0.)
    data['external_field'].zero_()
    result=evaluate_variational(model,data,training=True,compute_force=False)
    assert result['density_coefficients'].abs().max()==0
    target=torch.zeros_like(result['density_coefficients'])
    target[:,0]=torch.tensor([-.2,.1,.1])
    loss=(result['density_coefficients']-target).square().sum()
    loss.backward()
    gradient=model.field_dependent_charges_map.drive.weight.grad
    assert torch.isfinite(gradient).all() and gradient.norm()>1.e-6


@pytest.mark.parametrize('zero', [True, False])
def test_implicit_solver_first_and_second_derivatives(zero):
    from mace_scf.electrostatics.potential import _ElectronicSolve
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(13)
    rhs=torch.zeros(1,2,2) if zero else torch.randn(1,2,2)
    rhs.requires_grad_()
    hardness=(1+torch.rand_like(rhs)).requires_grad_()
    phase=torch.randn(1,3,2,dtype=torch.complex128,requires_grad=True)
    basis=torch.randn(1,3,2,dtype=torch.complex128,requires_grad=True)
    coulomb=torch.rand(1,3,requires_grad=True)
    moment=torch.randn_like(rhs,requires_grad=True)
    factor=torch.ones(1,requires_grad=True)
    constraint=torch.zeros_like(rhs)
    constraint[...,0]=1.
    diagonal=torch.ones_like(rhs)
    solve=lambda *args: _ElectronicSolve.apply(*args,constraint,diagonal,50)
    inputs=(rhs,hardness,phase,basis,coulomb,moment,factor)
    assert torch.autograd.gradcheck(solve,inputs,fast_mode=True)
    assert torch.autograd.gradgradcheck(solve,inputs,fast_mode=True)


def test_unconverged_implicit_state_is_not_silently_used():
    with pytest.raises(RuntimeError,match='did not converge'):
        evaluate_variational(small_model(),small_data(),steps=1)


@pytest.mark.parametrize('profile', ['gaussian', 'slab'])
def test_counter_charge_adjoint_and_force(profile):
    model = small_model()
    data = small_data()
    data['counter_charge'] = torch.tensor([-.1])
    if profile == 'slab':
        data['counter_slab_bounds'] = torch.tensor([[9., 11.]])
    else:
        data['counter_charge_center'] = torch.tensor([[3., 3., 10.]])
        data['counter_charge_width'] = torch.tensor([1.])
    geom = SpectralGeometry(model, data, data['positions'])
    state = torch.randn_like(geom.constraint)*.1
    _, deformation, _ = geom.potentials(state)
    torch.testing.assert_close(geom.hartree(state)-geom.adjoint(geom.counter),
        -geom.adjoint(deformation-geom.applied), atol=2.e-12, rtol=2.e-12)
    result = evaluate_variational(model, data, steps=50)
    energies=[]
    for shift in (-1.e-5, 1.e-5):
        moved = {key: value.detach().clone() for key, value in data.items()}
        moved['positions'][1, 2] += shift
        energies.append(evaluate_variational(model, moved, compute_force=False)['energy'])
    torch.testing.assert_close(-(energies[1]-energies[0])/2.e-5,
        result['forces'][1, 2:3], atol=2.e-7, rtol=2.e-5)


def test_grand_canonical_conjugacy():
    model = small_model()
    fixed = evaluate_variational(model, small_data())
    data = small_data()
    data['fermi_level'] = fixed['fermi_level'].detach()
    grand = evaluate_variational(model, data, constant_charge=False)
    for key in ('total_charge', 'energy', 'dipole', 'forces'):
        torch.testing.assert_close(grand[key], fixed[key], atol=3.e-9, rtol=3.e-8)

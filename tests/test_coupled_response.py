"""Coupled response transfer, physical observers and spectral loss contracts."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3
from mace.tools import torch_geometric

from mace_scf.electrostatics.coupled import CoupledResponse, CoupledGeometry, evaluate_coupled
from mace_scf.electrostatics.coupled import (
    factor_linear_system, solve_factored_system, implicit_root, RootOptions, SCFConvergenceError,
)
from mace_scf.electrostatics.loss import WeightedLoss, WeightedFourierPotential, spectral_errors
from tests.test_spectral_response import small_model, small_data


def test_fused_angular_transport_has_the_same_first_and_second_derivatives():
    from mace_scf.electrostatics.coupled import angular_transport
    torch.manual_seed(370)
    positions=torch.randn(5,3,dtype=torch.float64,requires_grad=True)
    edges=torch.tensor([[0,1,2,3,4,0,0],[1,2,0,4,3,1,0]])
    shifts=torch.randn(edges.shape[1],3,dtype=torch.float64)*.1
    batch=torch.tensor([0,0,0,1,1]);slot=torch.tensor([0,1,2,0,1])
    kwargs=dict(positions=positions,edge_index=edges,shifts=shifts,batch=batch,slot=slot,
                graphs=2,max_nodes=3,cutoff=6.,width=3.)
    sender,receiver=edges
    vector=positions[sender]-positions[receiver]-shifts
    square=vector.square().sum(-1)
    weight=torch.exp(-square/18.)*(1.-square/36.).clamp_min(0.).pow(3)
    index=(batch[receiver]*3+slot[receiver])*3+slot[sender]
    direction=positions.new_zeros(18,3).index_add(0,index,weight[:,None]*vector/3.).reshape(2,3,3,3)
    raw=positions.new_zeros(18).index_add(0,index,weight).reshape(2,3,3)
    denom=1.+raw.sum(-1,keepdim=True)
    scalar=raw/denom
    expected=torch.cat((scalar[...,None],direction/denom[...,None]),-1)
    actual=angular_transport(**kwargs)
    torch.testing.assert_close(actual,expected,rtol=1.e-12,atol=1.e-12)
    def derivatives(value):
        first=torch.autograd.grad(value.square().sum(),positions,create_graph=True,retain_graph=True)[0]
        second=torch.autograd.grad(first.square().sum(),positions,retain_graph=True)[0]
        return first,second
    for a,b in zip(derivatives(actual),derivatives(expected)):
        torch.testing.assert_close(a,b,rtol=1.e-10,atol=1.e-12)


def test_spectral_metric_is_the_real_space_inverse_fft_rmse():
    torch.manual_seed(370)
    shape=(5,5,7)
    field=torch.randn(2,*shape,dtype=torch.float64)
    field-=field.mean((1,2,3),keepdim=True)
    coefficients=torch.fft.fftshift(torch.fft.fftn(field,dim=(1,2,3)),dim=(1,2,3))
    modes=torch.stack(torch.meshgrid(*(torch.fft.fftshift(torch.fft.fftfreq(n))*n for n in shape),indexing='ij'),-1).reshape(-1,3)
    pred={'energy':torch.zeros(2),'fourier_potential':torch.view_as_real(coefficients.reshape(2,-1)),
          'fourier_potential_dft':torch.zeros(2,modes.shape[0],2),
          'k_vectors':modes[None].expand(2,-1,-1),'k_vectors_mask':torch.ones(2,len(modes),dtype=torch.bool),
          'k_vectors_grid_shape':torch.tensor(shape)}
    torch.testing.assert_close(spectral_errors(None,pred,'fourier_potential'),field.square().mean((1,2,3)),rtol=1.e-12,atol=1.e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cueq_backend_preserves_fields_forces_gradients_and_export(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('A CUDA-enabled PyTorch device is unavailable in this environment')
    pytest.importorskip('cuequivariance_torch')
    if device == 'cuda':
        pytest.importorskip('cuequivariance_ops_torch')
    from mace_scf.utils.foundation import accelerate_backbone
    from mace_scf.utils.run_train_utils import get_param_options
    model=coupled_model().to(device)
    data={k:v.to(device) if torch.is_tensor(v) else v for k,v in small_data().items()}
    before=evaluate_coupled(model,deepcopy(data),steps=12,training=True)
    loss=before['forces'].square().sum()+before['fermi_level'].square().sum()
    grad=torch.autograd.grad(loss,model.field_dependent_charges_map.common_level.weight)[0]
    random_state=torch.random.get_rng_state()
    accelerate_backbone(model,device)
    assert torch.equal(random_state,torch.random.get_rng_state())
    assert model.backbone_layout == 'ir_mul'
    after=evaluate_coupled(model,deepcopy(data),steps=12,training=True)
    for key in ('energy','forces','fermi_level','fourier_potential','density_coefficients'):
        torch.testing.assert_close(after[key],before[key],atol=1.e-8,rtol=1.e-8)
    backbone_parameters=tuple(model.interactions.parameters())+tuple(model.products.parameters())
    force_gradients=torch.autograd.grad(after['forces'].square().sum(),backbone_parameters,
                                       retain_graph=True,allow_unused=True)
    assert any(g is not None and bool(g.abs().max()>0) for g in force_gradients)
    assert all(g is None or bool(torch.isfinite(g).all()) for g in force_gradients)
    grad2=torch.autograd.grad(after['forces'].square().sum()+after['fermi_level'].square().sum(),
                             model.field_dependent_charges_map.common_level.weight)[0]
    torch.testing.assert_close(grad,grad2,atol=1.e-8,rtol=1.e-8)
    args=SimpleNamespace(model='FixedPoint',weight_decay=.001,local_charges_weight_decay=0.,
                         field_block_weight_decay=.01,lr=.001,amsgrad=False,beta=.9,beta_two=.999)
    groups=get_param_options(model,args)['params']
    owned=[id(p) for group in groups for p in group['params']]
    assert len(owned)==len(set(owned))
    assert set(owned)=={id(p) for p in model.parameters() if p.requires_grad}
    path=tmp_path/'cueq.model';torch.save(model,path)
    loaded=torch.load(path,map_location=device,weights_only=False)
    result=evaluate_coupled(loaded,deepcopy(data),steps=12)
    torch.testing.assert_close(result['forces'],after['forces'],atol=1.e-9,rtol=1.e-9)


def test_common_level_cannot_feed_charge_roundoff():
    model=coupled_model();data=small_data()
    before=evaluate_coupled(model,deepcopy(data),steps=12)
    with torch.no_grad():model.field_dependent_charges_map.common_level.weight.add_(1.e8)
    after=evaluate_coupled(model,deepcopy(data),steps=12)
    torch.testing.assert_close(before['density_coefficients'],after['density_coefficients'],atol=0.,rtol=0.)
    torch.testing.assert_close(before['forces'],after['forces'],atol=0.,rtol=0.)
    assert not torch.equal(before['fermi_level'],after['fermi_level'])


@pytest.mark.parametrize('mode', ['unroll_scf', 'shortcut_scf', 'implicit'])
def test_relative_replay_matches_full_force_training_gradient(mode):
    from mace_scf.utils.train import _relative_batch_centers
    from mace_scf.utils.model_training_wrappers import FixedPointWrapper
    from mace_scf.electrostatics.fixed_point_state import FixedPointTrainingOptions, FixedPointSCFOptions
    model = coupled_model()
    graphs = [small_data(charge=q,batched=False) for q in (.1,-.2)]
    for i,g in enumerate(graphs):
        g.fermi_level = torch.tensor(.3*i)
        g.fermi_level_weight = torch.tensor(1.)
        g.vacuum_potential = torch.tensor(.4*i)
        g.vacuum_potential_weight = torch.tensor(1.)
    wrapper = FixedPointWrapper(None, {'forces':True}, FixedPointTrainingOptions(
        mode=mode,scf=FixedPointSCFOptions(num_scf_steps=50 if mode == 'implicit' else 3,
                                        scf_tolerance=1.e-11,mixing_parameter=.5)))
    loss = WeightedLoss({'forces':500,'fermi_level':100,'vacuum_potential':100})
    full = torch_geometric.Batch.from_data_list(graphs)
    parameters = [p for p in model.parameters() if p.requires_grad]
    value = loss(full, wrapper(model, full.to_dict(), training=True))
    gradients = torch.autograd.grad(value, parameters, allow_unused=True)
    normalizers = loss.normalizers(full)
    rng = torch.random.get_rng_state()
    flags = [p.requires_grad for p in model.parameters()]
    pieces = [torch_geometric.Batch.from_data_list([g]) for g in graphs]
    normalizers.update(_relative_batch_centers(model,wrapper,loss,pieces,'cpu'))
    assert wrapper.output_args['forces']
    assert [p.requires_grad for p in model.parameters()] == flags
    torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0.,atol=0.)
    total = 0.
    model.zero_grad(set_to_none=True)
    for graph in graphs:
        piece = torch_geometric.Batch.from_data_list([graph])
        local = loss(piece,wrapper(model,piece.to_dict(),training=True),normalizers=normalizers)
        total += local.detach()
        torch.autograd.backward(local,inputs=parameters)
    torch.testing.assert_close(total,value,atol=1.e-10,rtol=1.e-10)
    for p,g in zip(parameters,gradients):
        if g is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad,g,atol=1.e-9,rtol=1.e-8)


def test_optional_proto_reference_is_one_shared_gauge_and_serializes(tmp_path):
    model = coupled_model()
    before = evaluate_coupled(model, small_data(), steps=3)
    r = model.field_dependent_charges_map
    with torch.no_grad(): r.vacuum_reference_integrals.copy_(torch.tensor([50.,-20.]))
    after = evaluate_coupled(model, small_data(), steps=3)
    shift = (2*50.-20.)/(7*7*12)
    for key in ('fermi_level','vacuum_potential'):
        torch.testing.assert_close(after[key]-before[key],torch.tensor([shift]))
    for key in ('workfunction','forces','density_coefficients','fourier_potential'):
        torch.testing.assert_close(after[key],before[key],atol=1.e-12,rtol=1.e-12)
    path=tmp_path/'reference.model';torch.save(model,path)
    saved=torch.load(path,weights_only=False)
    restored=evaluate_coupled(saved,small_data(),steps=3)
    torch.testing.assert_close(restored['vacuum_potential'],after['vacuum_potential'])
    # Optional spatial proto data must not change the scalar observation.
    r.proto_fitted.fill_(True)
    actual=evaluate_coupled(model,small_data(),steps=3)
    torch.testing.assert_close(actual['vacuum_potential'],after['vacuum_potential'])


def test_absent_counter_charge_has_no_zero_times_proto_force_graph():
    model = coupled_model()
    model.field_dependent_charges_map.proto_fitted.fill_(True)
    model.field_dependent_charges_map.proto_coefficients.fill_(2.)
    data = small_data()
    data['positions'].requires_grad_(True)
    geom = CoupledGeometry(model, data, data['positions'])
    assert geom.observer_proto.requires_grad
    assert not geom.counter.requires_grad
    assert not geom.counter_energy.requires_grad
    assert torch.count_nonzero(geom.counter) == 0
    assert torch.count_nonzero(geom.counter_energy) == 0


@pytest.mark.parametrize('mode', ['unroll_scf', 'shortcut_scf', 'implicit'])
def test_neutral_proto_reference_preserves_response_and_wf(mode):
    model = coupled_model()
    response = model.field_dependent_charges_map
    options = dict(mode=mode, steps=50 if mode == 'implicit' else 4, training=True)
    before = evaluate_coupled(model, small_data(), **options)
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradient = torch.autograd.grad(before['workfunction'].sum()+before['forces'].square().sum(),
                                   parameters, allow_unused=True)
    response.proto_fitted.fill_(True)
    response.proto_coefficients.copy_(torch.linspace(-80.,0.,64).expand(2,-1))
    response.vacuum_reference_integrals.fill_(80.)
    after = evaluate_coupled(model, small_data(), **options)
    new_gradient = torch.autograd.grad(after['workfunction'].sum()+after['forces'].square().sum(),
                                       parameters, allow_unused=True)
    for key in ('energy','forces','density_coefficients','fourier_potential','workfunction','scf_residual'):
        torch.testing.assert_close(after[key],before[key],atol=1.e-11,rtol=1.e-11)
    for a,b in zip(gradient,new_gradient):
        if a is None: assert b is None
        else: torch.testing.assert_close(a,b,atol=1.e-10,rtol=1.e-10)
    for key in ('fermi_level','vacuum_potential'):
        torch.testing.assert_close(after[key]-before[key],torch.tensor([240./(7*7*12)]))
    assert not torch.equal(after['fourier_total_potential'],before['fourier_total_potential'])
    data = small_data()
    g = CoupledGeometry(model,data,data['positions'])
    plane = g.plane(torch.view_as_complex((after['fourier_total_potential']/g.ngrid).contiguous()))
    torch.testing.assert_close(plane+after['potential_reference'],after['vacuum_potential'])
    # Changing the observation plane must never change a bulk chemical level.
    data['vacuum_fraction'] = torch.tensor([.37])
    moved = evaluate_coupled(model,data,**options)
    torch.testing.assert_close(moved['fermi_level'],after['fermi_level'])


def test_saved_response_reference_prevents_silent_checkpoint_reinterpretation():
    model = coupled_model()
    response = model.field_dependent_charges_map
    assert response.get_extra_state()['response_reference'] == 'neutral'
    response.set_extra_state({'deployment_mode':'unroll_scf'})
    assert response.response_reference == 'total'
    state = model.state_dict()
    copy = coupled_model()
    copy.load_state_dict(state)
    assert copy.field_dependent_charges_map.response_reference == 'total'


@pytest.mark.parametrize('mode', ['unroll_scf', 'shortcut_scf', 'implicit'])
@pytest.mark.parametrize('proto', [False, True])
def test_scalar_references_preserve_physics_relative_gradients_and_export(mode, proto, tmp_path):
    model = coupled_model()
    response = model.field_dependent_charges_map
    response.proto_fitted.fill_(proto)
    graphs = [small_data(charge=q,batched=False) for q in (.1,-.2)]
    for i, graph in enumerate(graphs):
        graph.fermi_level = torch.tensor(float(i))
        graph.fermi_level_weight = torch.tensor(1.)
        graph.vacuum_potential = torch.tensor(.5*i)
        graph.vacuum_potential_weight = torch.tensor(1.)
    batch = torch_geometric.Batch.from_data_list(graphs)
    options = dict(steps=50 if mode == 'implicit' else 3, mode=mode,training=True,tolerance=1.e-10)
    before = evaluate_coupled(model,deepcopy(batch.to_dict()),**options)
    loss = WeightedLoss({'fermi_level':100,'vacuum_potential':100})
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradient = torch.autograd.grad(loss(batch,before),parameters,allow_unused=True)
    response.scalar_reference.copy_(torch.tensor([3.5,-.2]))
    after = evaluate_coupled(model,deepcopy(batch.to_dict()),**options)
    new_gradient = torch.autograd.grad(loss(batch,after),parameters,allow_unused=True)
    for a,b in zip(gradient,new_gradient):
        if a is None: assert b is None
        else: torch.testing.assert_close(a,b,atol=1.e-9,rtol=1.e-9)
    for key in ('energy','forces','dipole','density_coefficients','fourier_potential','fourier_total_potential','scf_residual'):
        torch.testing.assert_close(before[key],after[key],atol=1.e-12,rtol=1.e-12)
    for key,shift in (('fermi_level',3.5),('vacuum_potential',-.2),('workfunction',-3.7)):
        torch.testing.assert_close(after[key]-before[key],torch.full((2,),shift),atol=1.e-12,rtol=1.e-12)
    geometry=CoupledGeometry(model,batch.to_dict(),batch.positions)
    plane=geometry.plane(torch.view_as_complex((after['fourier_total_potential']/geometry.ngrid).contiguous()))
    torch.testing.assert_close(plane+after['potential_reference'],after['vacuum_potential'],atol=1.e-12,rtol=1.e-12)
    path=tmp_path/'referenced.model';torch.save(model,path)
    restored=torch.load(path,weights_only=False)
    prediction=evaluate_coupled(restored,deepcopy(batch.to_dict()),**options)
    torch.testing.assert_close(prediction['workfunction'],after['workfunction'])
    state=model.state_dict()
    del state['field_dependent_charges_map.scalar_reference']
    restored.load_state_dict(state,strict=True)
    assert not bool(restored.field_dependent_charges_map.scalar_reference.any())


def test_field_only_probes_respect_no_grad_after_force_evaluation():
    model=coupled_model();data=small_data()
    reference=evaluate_coupled(model,data,steps=12)
    assert data['positions'].requires_grad
    with torch.no_grad():
        probe=evaluate_coupled(model,data,steps=12,compute_force=False)
        forced=evaluate_coupled(model,data,steps=12,compute_force=True)
    for key,value in probe.items():
        assert not torch.is_tensor(value) or not value.requires_grad,key
    for key in ('energy','fermi_level','fourier_potential'):
        torch.testing.assert_close(probe[key],reference[key],rtol=1.e-12,atol=1.e-12)
    torch.testing.assert_close(forced['forces'],reference['forces'],rtol=1.e-12,atol=1.e-12)


def test_recurrent_total_field_equals_component_observer():
    from mace_scf.electrostatics.coupled import prepare_coupled, total_spectrum, evaluate_coefficients
    model=coupled_model();data=small_data();local=model.local_part(data,False)
    geometry,initial,*_=prepare_coupled(model,data,local)
    state=(initial+torch.randn_like(initial)*.01).detach().requires_grad_()
    total=total_spectrum(state,geometry.tensors,geometry.kernels,1.5)
    parts=evaluate_coefficients(state,geometry.tensors,geometry.kernels,1.5)[0]
    torch.testing.assert_close(total,parts,atol=1.e-12,rtol=1.e-12)
    a=torch.autograd.grad(total.square().sum(),state,retain_graph=True)[0]
    b=torch.autograd.grad(parts.square().sum(),state)[0]
    torch.testing.assert_close(a,b,atol=1.e-11,rtol=1.e-11)


def coupled_model(widths=(1.5, 3.), local_energy=False):
    model = small_model()
    model.field_dependent_charges_map = CoupledResponse(
        node_feats_irreps='4x0e+4x1o', charges_irreps='0e+1o', num_elements=2,
        potential_widths=widths, include_local_energy=local_energy)
    model.lr_source_maps.requires_grad_(True)
    with torch.no_grad():
        for block in model.lr_source_maps:
            output=getattr(block,'linear_2',getattr(block,'linear',None))
            for parameter in output.parameters():parameter.zero_()
        r = model.field_dependent_charges_map
        r.species_level[1] = .5
        r.scalar_out.weight.normal_(std=.01)
        r.vector_out.weight.normal_(std=.01)
        r.local_source_scalar.weight.normal_(std=.01)
        if local_energy:
            r.energy_readout[-1].weight.normal_(std=.01)
    return model


def test_electronic_reference_is_not_weight_decayed():
    from mace_scf.utils.run_train_utils import get_param_options
    m=coupled_model()
    args=SimpleNamespace(model='FixedPoint',weight_decay=.001,local_charges_weight_decay=0.,
                         field_block_weight_decay=.1,lr=.003,amsgrad=False,beta=.9,beta_two=.999)
    groups=get_param_options(m,args)['params']
    decay={id(p):g['weight_decay'] for g in groups for p in g['params']}
    r=m.field_dependent_charges_map
    assert decay[id(r.common_level.weight)]==.1
    for p in (r.species_level,r.species_source,r.scalar_out.bias,r.chemical_scalar.bias):
        assert decay[id(p)]==0.


@pytest.mark.parametrize('widths', [(), (1.5, 3.)])
@pytest.mark.parametrize('functional', [False, True])
def test_one_screened_step_solves_the_full_affine_response(widths, functional):
    """A frozen nonlinear response must not leave the completion one step behind."""
    from mace_scf.electrostatics.coupled import prepare_coupled, screen_moments
    model = coupled_model(widths)
    r = model.field_dependent_charges_map
    with torch.no_grad():
        for name in ('state_scalar', 'state_vector', 'neighbor_scalar', 'neighbor_vector',
                     'neighbor_divergence', 'neighbor_gradient'):
            getattr(r, name).weight.zero_()
    data = small_data()
    local = model.local_part(data, compute_force=False)
    g, initial, args, update, _ = prepare_coupled(model, data, local, functional=functional)
    state = initial+torch.randn_like(initial)*.1
    raw = update.raw(state, *args)
    solved = update(state, *args)
    torch.testing.assert_close(update.raw(solved, *args), solved, atol=2.e-12, rtol=2.e-12)
    torch.testing.assert_close(solved[..., 0].sum(-1), data['total_charge'], atol=1.e-12, rtol=0.)
    old = screen_moments(state, raw, args[7], g.present, args[9:13],
                         materialized=not functional)
    if widths:
        assert float((update.raw(old, *args)-old).detach().abs().max()) > 1.e-5
    else:
        torch.testing.assert_close(solved, old, atol=0., rtol=0.)


def test_screening_preserves_nonlinear_roots_and_their_force_derivatives(monkeypatch):
    from mace_scf.electrostatics import coupled
    model = coupled_model(local_energy=True)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    new = evaluate_coupled(model, small_data(), steps=70, training=True, mode='implicit', tolerance=1.e-11)
    new_grad = torch.autograd.grad(new['forces'].square().sum()+new['fermi_level'].square().sum(), params, allow_unused=True)
    monkeypatch.setattr(coupled, 'screen_coupled',
                        lambda z,p,r,k,m,f,**kw: coupled.screen_moments(z,p,r,m,f,**kw))
    old = evaluate_coupled(model, small_data(), steps=70, training=True, mode='implicit', tolerance=1.e-11)
    old_grad = torch.autograd.grad(old['forces'].square().sum()+old['fermi_level'].square().sum(), params, allow_unused=True)
    for key in ('energy', 'forces', 'fermi_level', 'vacuum_potential', 'density_coefficients'):
        torch.testing.assert_close(new[key], old[key], atol=1.e-8, rtol=1.e-8)
    for a, b in zip(new_grad, old_grad):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=1.e-8, rtol=1.e-7)


@pytest.mark.parametrize('conditioning',[False,True])
@pytest.mark.parametrize('relative',[False,True])
def test_loss_accumulation_has_identical_value_and_gradient_with_missing_labels(conditioning, relative):
    graphs=[]
    for i in range(3):
        atoms=Atoms('O'+'H'*(i+1), positions=[[3.,3.,4.]]+[[3.5+j*.3,3.,4.5] for j in range(i+1)], cell=[7.,7.,12.],pbc=[1,1,0])
        g=small_data(charge=.1*i,batched=False,atoms=atoms)
        g.weight=torch.tensor(float(i+1));g.forces_weight=torch.tensor(float(i!=1))
        g.fermi_level=torch.tensor(.2*i);g.fermi_level_weight=torch.tensor(float(i!=2))
        g.vacuum_potential=torch.tensor(.3*i);g.vacuum_potential_weight=torch.tensor(float(i!=1))
        g.fourier_potential_weight=torch.tensor(float(i!=1));graphs.append(g)
    loss=WeightedLoss({'energy_per_atom':100,'forces':500,'dipole':10,
                       'fermi_level':{'weight':100,'relative':relative},
                       'fourier_potential':100,'vacuum_potential':{'weight':100,'relative':relative}})
    full=torch_geometric.Batch.from_data_list(graphs)
    theta=torch.tensor(.2,requires_grad=True)
    def prediction(batch):
        n=batch.num_graphs;z=theta*torch.ones(n)
        result = {'energy':z,'forces':theta*torch.ones_like(batch.forces),'dipole':z[:,None].expand(-1,3),
                'fermi_level':z,'vacuum_potential':2*z,'fourier_potential':z[:,None,None].expand(-1,3,2),
                'fourier_potential_dft':torch.zeros(n,3,2),'k_vectors_mask':torch.ones(n,3,dtype=torch.bool),
                'k_vectors':torch.tensor([[[0.,0.,0.],[0.,0.,1.],[0.,0.,-1.]]]).expand(n,-1,-1),
                'k_vectors_grid_shape':torch.tensor([1,1,3])}
        if conditioning:
            reference = dict(result)
            reference['vacuum_potential'] = 3*z
            reference['fermi_level'] = z*.5
            reference['reference_masks'] = {key:batch.total_charge>.05 for key in
                                           ('fermi_level','vacuum_potential','fourier_potential')}
            result['reference_response'] = reference
        return result
    reference=loss(full,prediction(full));grad=torch.autograd.grad(reference,theta)[0]
    normalizers=loss.normalizers(full)
    for key, moments in loss.relative_moments(full,prediction(full)).items():
        normalizers[key]=moments[1]/moments[0].clamp_min(1.e-30)
    for parts in ((graphs[:1],graphs[1:]),([graphs[0]],[graphs[1]],[graphs[2]])):
        value=0.;gradient=0.
        for part in parts:
            batch=torch_geometric.Batch.from_data_list(part)
            item=loss(batch,prediction(batch),normalizers=normalizers)
            value+=item.detach();gradient+=torch.autograd.grad(item,theta)[0]
        torch.testing.assert_close(value,reference)
        torch.testing.assert_close(gradient,grad)


@pytest.mark.parametrize('widths', [(), (1.5,3.)])
@pytest.mark.parametrize('mode', ['unroll_scf','shortcut_scf','implicit'])
def test_modes_force_loss_and_full_field_gradients(widths, mode):
    model = coupled_model(widths)
    result = evaluate_coupled(model, small_data(), steps=50 if mode=='implicit' else 12,
                              training=True, mode=mode)
    torch.testing.assert_close(result['total_charge'], torch.tensor([.1]), atol=1.e-12, rtol=0.)
    assert result['scf_residual'].max()<1.e-6
    loss = result['forces'].square().sum()+result['fermi_level'].square().sum()
    if widths:
        loss = loss+result['workfunction'].square().sum()+result['fourier_potential'].square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.field_dependent_charges_map.state_scalar.weight.grad.abs().max()>0.
    assert model.lr_source_maps[0].linear_2.weight.grad.abs().max()>0.
    if widths:
        assert model.field_dependent_charges_map.local_source_scalar.weight.grad.abs().max()>0.


def test_checkpointed_gradient_and_converged_root_parity():
    model = coupled_model()
    parameters=[p for p in model.parameters() if p.requires_grad]
    values=[]; gradients=[]
    for mode in ('unroll_scf','shortcut_scf','implicit'):
        out=evaluate_coupled(model,small_data(),steps=50,training=True,mode=mode)
        objective=out['forces'].square().sum()+out['workfunction'].square().sum()
        gradients.append(torch.autograd.grad(objective,parameters,allow_unused=True))
        values.append(out)
    for i in (1,2):
        for key in ('energy','forces','workfunction','fourier_potential'):
            torch.testing.assert_close(values[0][key],values[i][key],atol=1.e-6,rtol=1.e-6)
        for a,b in zip(gradients[0],gradients[i]):
            if a is None:
                assert b is None
            else:
                torch.testing.assert_close(a,b,atol=2.e-6,rtol=2.e-5)


@pytest.mark.parametrize('mixing', [.25, .5])
def test_damped_fifty_step_shortcut_keeps_force_and_parameter_derivatives(mixing):
    model = coupled_model()
    parameters = [p for p in model.parameters() if p.requires_grad]
    results = []
    for mode in ('unroll_scf', 'shortcut_scf'):
        out = evaluate_coupled(model, small_data(), steps=50, mixing=mixing,
                               training=True, mode=mode)
        objective = out['forces'].square().sum()+out['workfunction'].square().sum()
        gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
        results.append((out, gradients))
    for key in ('energy', 'forces', 'fermi_level', 'vacuum_potential', 'fourier_potential'):
        torch.testing.assert_close(results[0][0][key], results[1][0][key], atol=1.e-12, rtol=1.e-12)
    for first, second in zip(results[0][1], results[1][1]):
        if first is None:
            assert second is None
        else:
            torch.testing.assert_close(first, second, atol=1.e-10, rtol=1.e-9)


def test_nonfinite_scalar_observation_is_rejected():
    from mace_scf.electrostatics.coupled import SCFNumericalError
    model = coupled_model()
    model.field_dependent_charges_map.scalar_reference[0] = float('nan')
    with pytest.raises(SCFNumericalError, match='fermi_level'):
        evaluate_coupled(model, small_data(), steps=3)


def test_nonfinite_force_is_rejected_even_with_finite_state(monkeypatch):
    from mace_scf.electrostatics.coupled import SCFNumericalError
    model = coupled_model()
    original = torch.autograd.grad
    def invalid_derivative(*args, **kwargs):
        return tuple(torch.full_like(value, float('inf')) if value is not None else None
                     for value in original(*args, **kwargs))
    monkeypatch.setattr(torch.autograd, 'grad', invalid_derivative)
    with pytest.raises(SCFNumericalError, match='forces'):
        evaluate_coupled(model, small_data(), steps=3)


@pytest.mark.parametrize('local_energy',[False,True])
def test_force_matches_finite_energy_derivative(local_energy):
    model=coupled_model(local_energy=local_energy)
    output=evaluate_coupled(model,small_data(),steps=4,training=True)
    epsilon=1.e-5; energies=[]
    for sign in (-1,1):
        data=small_data();data['positions'][0,2]+=sign*epsilon
        energies.append(evaluate_coupled(model,data,steps=4,compute_force=False)['energy'])
    torch.testing.assert_close(-(energies[1]-energies[0])/(2*epsilon),output['forces'][0,2:3],atol=3.e-7,rtol=2.e-5)
    if local_energy:
        assert output['electron_energy'].abs().max()>0.


def test_rotation_and_heterogeneous_batch_invariance():
    model=coupled_model()
    first=small_data(batched=False)
    second=small_data(charge=-.2,batched=False,atoms=Atoms('OH',positions=[[2,2,3],[2.8,2,3.2]],cell=[8,8,14],pbc=[1,1,0]))
    batch=torch_geometric.Batch.from_data_list([first,second]).to_dict()
    together=evaluate_coupled(model,batch,steps=12)
    for i,item in enumerate((first,second)):
        separate=evaluate_coupled(model,torch_geometric.Batch.from_data_list([item]).to_dict(),steps=12)
        for key in ('energy','fermi_level','workfunction'):
            torch.testing.assert_close(together[key][i:i+1],separate[key],atol=1.e-9,rtol=1.e-9)
        torch.testing.assert_close(together['forces'][batch['ptr'][i]:batch['ptr'][i+1]],separate['forces'],atol=1.e-9,rtol=1.e-9)
    # Atom-normalized force losses and their full parameter derivatives must
    # also be invariant to the device partition, including different FFT grids.
    for graph in (first,second):
        graph.energy_weight=torch.tensor(1.);graph.forces_weight=torch.tensor(1.)
    loss=WeightedLoss({'energy_per_atom':100,'forces':500,'fermi_level':{'weight':100,'relative':False}})
    full=torch_geometric.Batch.from_data_list([first,second])
    parameters=tuple(p for p in model.parameters() if p.requires_grad)
    out=evaluate_coupled(model,full.to_dict(),steps=3,training=True)
    reference=loss(full,out)
    gradients=torch.autograd.grad(reference,parameters,allow_unused=True)
    accumulated=[torch.zeros_like(p) for p in parameters]
    for graph in (first,second):
        part=torch_geometric.Batch.from_data_list([graph])
        out=evaluate_coupled(model,part.to_dict(),steps=3,training=True)
        objective=loss(part,out,normalizers=loss.normalizers(full))
        for a,g in zip(accumulated,torch.autograd.grad(objective,parameters,allow_unused=True)):
            if g is not None:a.add_(g)
    for a,g in zip(accumulated,gradients):
        torch.testing.assert_close(a,torch.zeros_like(a) if g is None else g,atol=1.e-9,rtol=1.e-9)
    rotation=o3.rand_matrix();original=small_data();rotated=small_data()
    for key in ('positions','cell','shifts','external_field'):
        rotated[key]=rotated[key]@rotation.T
    a=evaluate_coupled(model,original,steps=12);b=evaluate_coupled(model,rotated,steps=12)
    for key in ('energy','fermi_level','workfunction'):
        torch.testing.assert_close(a[key],b[key],atol=1.e-9,rtol=1.e-9)
    torch.testing.assert_close(a['forces']@rotation.T,b['forces'],atol=1.e-9,rtol=1.e-9)


def test_labels_do_not_enter_the_scf_map():
    model=coupled_model();data=small_data();other=deepcopy(data)
    other['fermi_level'].fill_(12345.);other['vacuum_potential'].fill_(-10000.)
    a=evaluate_coupled(model,data,steps=12);b=evaluate_coupled(model,other,steps=12)
    for key in ('energy','forces','workfunction','fermi_level','density_coefficients','fourier_potential'):
        torch.testing.assert_close(a[key],b[key],rtol=0.,atol=0.)


def test_common_chemical_level_is_charge_null_direction():
    model=coupled_model();data=small_data()
    a=evaluate_coupled(model,data,steps=50)
    with torch.no_grad():
        model.field_dependent_charges_map.species_level.add_(.5)
    # Species shifts also change the field descriptor's driving reference, so
    # test the exact KKT null direction at fixed shared hidden features.
    r=model.field_dependent_charges_map
    raw=torch.tensor([[.3,.1,-.2]]);hidden=torch.randn(1,3,64)
    attrs=data['node_attrs'][None];soft=torch.tensor([[.01,.02,.03]]);mask=torch.ones_like(raw)
    left=r.chemical_levels(raw,hidden,attrs,soft,mask)
    with torch.no_grad():r.species_level.add_(.5)
    right=r.chemical_levels(raw,hidden,attrs,soft,mask)
    from mace_scf.electrostatics.coupled import charge_closure
    q,mu,*_=charge_closure(torch.zeros_like(raw),left,soft,torch.tensor([.1]),mask)
    q1,mu1,*_=charge_closure(torch.zeros_like(raw),right,soft,torch.tensor([.1]),mask)
    torch.testing.assert_close(q,q1,atol=1.e-15,rtol=0.)
    torch.testing.assert_close(mu1-mu,torch.tensor([.5]))


def test_cached_linear_solve_second_derivative():
    torch.set_default_dtype(torch.float64)
    a=torch.tensor([[[3.,.2],[.1,2.]]],requires_grad=True)
    b=torch.tensor([[[.4],[.7]]],requires_grad=True)
    def solve(a,b):
        lu,pivots=factor_linear_system(a)
        return solve_factored_system(a,lu,pivots,b)
    assert torch.autograd.gradcheck(solve,(a,b))
    assert torch.autograd.gradgradcheck(solve,(a,b))


def test_implicit_never_accepts_an_unconverged_fixed_step():
    initial=torch.zeros(1,1,1)
    with pytest.raises(SCFConvergenceError):
        implicit_root(lambda x:x+1.,initial,options=RootOptions(max_steps=2))


def observations():
    torch.set_default_dtype(torch.float64)
    n=3**3
    ref=SimpleNamespace(weight=torch.tensor([1.,2.,1.]),
        fourier_potential_weight=torch.tensor([1.,1.,0.]),
        fourier_density_weight=torch.ones(3),fourier_proto_potential_weight=torch.ones(3),
        fermi_level=torch.tensor([1.,3.,7.]),fermi_level_weight=torch.tensor([1.,0.,1.]),
        vacuum_potential=torch.tensor([3.,7.,12.]),vacuum_potential_weight=torch.tensor([1.,0.,0.]),
        pbc=torch.tensor([[1,1,0],[1,1,0],[1,1,1]],dtype=torch.bool))
    torch.manual_seed(10)
    real=torch.randn(3,3,3,3)
    spectrum=torch.view_as_real(torch.fft.fftn(real,dim=(-3,-2,-1))).reshape(3,n,2)
    modes=torch.cartesian_prod(*(torch.fft.fftfreq(3)*3 for _ in range(3)))
    pred={'energy':torch.zeros(3), 'fourier_potential':spectrum.clone().requires_grad_(),
          'fourier_potential_dft':torch.zeros_like(spectrum),
          'k_vectors_mask':torch.ones(3,n,dtype=torch.bool),'k_vectors':modes[None].expand(3,-1,-1),
          'k_vectors_grid_shape':torch.tensor([3,3,3]),
          'vacuum_potential':torch.tensor([4.,1000.,1000.],requires_grad=True),
          'fermi_level':torch.tensor([100.,200.,300.],requires_grad=True)}
    return ref,pred,real


def test_parseval_and_vacuum_loss_gradients():
    ref,pred,real=observations()
    mse=(real-real.mean((-3,-2,-1),keepdim=True)).square().mean((-3,-2,-1))
    torch.testing.assert_close(spectral_errors(ref,pred,'fourier_potential')[:2],mse[:2])
    loss=WeightedLoss({'fourier_potential':1,'vacuum_potential':{'weight':10,'relative':False}})(ref,pred)
    torch.testing.assert_close(loss,(mse[0]+2*mse[1])/3+10.)
    loss.backward()
    torch.testing.assert_close(pred['vacuum_potential'].grad,torch.tensor([20.,0.,0.]))
    assert pred['fermi_level'].grad is None


def test_masked_labels_and_invalid_observed_labels():
    ref,pred,real=observations()
    objective=WeightedLoss({'fourier_potential':1,'vacuum_potential':1})
    baseline=objective(ref,pred)
    with torch.no_grad():pred['fourier_potential'][2].fill_(float('nan'))
    ref.vacuum_potential[1]=float('nan')
    torch.testing.assert_close(objective(ref,pred),baseline)
    ref.vacuum_potential[0]=float('nan')
    with pytest.raises(FloatingPointError):objective(ref,pred)


def test_validation_weighted_global_wf_and_batch_partition():
    from mace_scf.utils.train import evaluate
    graphs=[]
    for i in range(1,4):
        g=small_data(charge=float(i),batched=False)
        g.weight=torch.tensor(float(i));g.vacuum_potential=torch.tensor(0.);g.vacuum_potential_weight=torch.tensor(1.)
        g.fermi_level=torch.tensor(0.);g.fermi_level_weight=torch.tensor(1.)
        g.fourier_potential_weight=torch.tensor(float(i<3))
        graphs.append(g)
    def wrapper(model,data,**kwargs):
        charge=data['total_charge'].reshape(-1)
        predicted=torch.zeros(len(charge),3,2);predicted[:,1,0]=charge*3
        return {'energy':charge*0.,'workfunction':charge*2,'vacuum_potential':charge*3,
                'fermi_level':charge,'fourier_potential':predicted,'fourier_potential_dft':predicted*0,
                'k_vectors_mask':torch.ones(len(charge),3,dtype=torch.bool),
                'k_vectors':torch.tensor([[[0.,0.,0.],[0.,0.,1.],[0.,0.,-1.]]]).expand(len(charge),-1,-1),
                'k_vectors_grid_shape':torch.tensor([1,1,3])}
    values=[]
    for batch_size in (1,2,3):
        loader=torch_geometric.dataloader.DataLoader(graphs,batch_size=batch_size,shuffle=False)
        loss,metric=evaluate(torch.nn.Linear(1,1),wrapper,WeightedLoss({'fourier_potential':100,'vacuum_potential':100}),None,loader,'cpu')
        values.append((loss,metric))
        assert metric['rmse_wf_abs']==pytest.approx(24.**.5)
        assert metric['rmse_wf_rel']==pytest.approx((24.-(28./6)**2)**.5)
        assert metric['wf_bias']==pytest.approx(28./6)
        assert loss==pytest.approx(100*(3.+90./11.))
        assert metric['rmse_esp_vac_rel']==pytest.approx(5.**.5)
        assert metric['rmse_fermi_level_rel']==pytest.approx((6.-(14./6)**2)**.5)
    for value in values[1:]:assert value[0]==pytest.approx(values[0][0])


def test_saved_deployment_and_optional_diagnostics(tmp_path):
    import sys
    from unittest.mock import patch
    from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
    from mace_scf.utils import create_scf_convergence_summary
    model=coupled_model();model.field_dependent_charges_map.deployment_mixing.fill_(.4)
    result=evaluate_coupled(model,small_data(),steps=50)
    path=tmp_path/'coupled.model';torch.save(model,path)
    atoms=Atoms('OHH',positions=small_data()['positions'].numpy(),cell=[7,7,12],pbc=[1,1,0])
    atoms.info.update(total_charge=.1,external_field=np.array([0.,0.,.02]))
    atoms.calc=MACEFixedPointSCF(str(path),device='cpu')
    np.testing.assert_allclose(atoms.get_potential_energy(),result['energy'].detach()[0],rtol=1.e-10,atol=1.e-10)
    np.testing.assert_allclose(atoms.get_forces(),result['forces'].detach(),rtol=1.e-10,atol=1.e-10)
    assert atoms.calc.results['num_scf_steps']==50
    loader=torch_geometric.dataloader.DataLoader([small_data(batched=False)],batch_size=1)
    with patch.dict(sys.modules,{'mace_scf.utils.diagnostics':None}):
        report=create_scf_convergence_summary(model,{'valid':loader},{'forces':True},'cpu',{})
    assert 'WF_50_minus_100_eV' in report


@pytest.mark.parametrize('foundation',[False,True])
def test_readout_conditioning_preserves_function_and_fits_force_units(foundation):
    from mace_scf.utils.foundation import condition_readouts, set_foundation_stage
    model=coupled_model()
    model.register_buffer('readout_feature_units',torch.ones(1,16))
    if foundation:
        model.register_buffer('foundation_element_map',torch.eye(2))
        model.register_buffer('foundation_atomic_numbers',model.atomic_numbers.clone())
    graph=small_data(batched=False)
    def local_force():
        data=torch_geometric.Batch.from_data_list([graph]).to_dict()
        energy=model.local_part(data,compute_force=True).energies.sum()
        return energy,-torch.autograd.grad(energy,data['positions'])[0]
    before,force=local_force()
    graph.forces=force.detach()*2
    graph.forces_weight=torch.tensor(1.)
    loader=torch_geometric.dataloader.DataLoader([graph],batch_size=1)
    condition_readouts(model,loader,'cpu')
    after,result=local_force()
    torch.testing.assert_close(result,force*(2 if foundation else 1),atol=1.e-12,rtol=1.e-10)
    torch.testing.assert_close(after,before*(2 if foundation else 1),atol=1.e-12,rtol=1.e-10)
    assert model.readout_feature_units.min()>=1.
    if foundation:
        set_foundation_stage(model,True)
        assert all(not p.requires_grad for p in model.products.parameters())
        assert all(p.requires_grad for p in model.readouts.parameters())
        set_foundation_stage(model,False)
        assert all(p.requires_grad for p in model.products.parameters())


def test_rejected_prior_keeps_conditioned_hidden_energy_features():
    from mace import modules
    from mace_scf.utils.foundation import condition_readouts
    model=coupled_model()
    model.readouts[0]=modules.NonLinearReadoutBlock(o3.Irreps('4x0e+4x1o'),o3.Irreps('4x0e'),torch.nn.functional.silu)
    model.register_buffer('readout_feature_units',torch.ones(1,16))
    model.register_buffer('foundation_element_map',torch.eye(2))
    model.register_buffer('foundation_atomic_numbers',model.atomic_numbers.clone())
    graph=small_data(batched=False);data=torch_geometric.Batch.from_data_list([graph]).to_dict()
    energy=model.local_part(data,compute_force=True).energies.sum()
    graph.forces=torch.autograd.grad(energy,data['positions'])[0].detach()  # anticorrelated prior
    graph.forces_weight=torch.tensor(1.)
    first=model.readouts[0].linear_1.weight.detach().clone()
    loader=torch_geometric.dataloader.DataLoader([graph],batch_size=1)
    condition_readouts(model,loader,'cpu')
    assert not bool(model.readouts[0].linear_2.weight.any())
    torch.testing.assert_close(model.readouts[0].linear_1.weight,first,rtol=0.,atol=0.)


def test_finite_trajectory_stress_matches_strained_energy():
    model=coupled_model();data=small_data()
    result=evaluate_coupled(model,data,steps=4,compute_force=False,compute_stress=True)
    epsilon=1.e-5;energies=[]
    for sign in (-1,1):
        shifted=small_data();strain=torch.eye(3);strain[0,0]+=sign*epsilon
        for key in ('positions','cell','shifts'):shifted[key]=shifted[key]@strain
        energies.append(evaluate_coupled(model,shifted,steps=4,compute_force=False)['energy'])
    volume=torch.linalg.det(data['cell'].reshape(3,3)).abs()
    torch.testing.assert_close((energies[1]-energies[0])/(2*epsilon*volume),result['stress'][:,0,0],atol=1.e-8,rtol=2.e-5)

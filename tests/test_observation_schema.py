"""Independent vacuum labels, axis conventions and loss/data migration."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io.extxyz import key_val_dict_to_str, key_val_str_to_dict
from mace.data import KeySpecification, config_from_atoms
from mace.tools import AtomicNumberTable

from mace_scf.data import ExtAtomicData
from mace_scf.data.new_atomic_data import plane_fraction
from mace_scf.electrostatics.coupled import evaluate_coupled
from mace_scf.electrostatics.loss import WeightedLoss, WeightedVacuumPotential
from tests.test_coupled_response import coupled_model
from tests.test_spectral_response import small_data

ROOT = Path(__file__).resolve().parents[1]


def migration_module():
    spec = spec_from_file_location('dataset_migration', ROOT/'tmp_update_dataset.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_preserves_spectra_atoms_weights_and_unrelated_metadata(tmp_path):
    script = migration_module()
    metadata = {'Properties':'species:S:1:pos:R:3:workfuncion:R:1:forces:R:3',
                'Lattice':np.eye(3)*8, 'pbc':[True, False, True],
                'vasp_workfunction':5., 'vasp_efermi':-3.,
                'config_workfunction_weight':.5, 'vacuum_zfrac':.7,
                'dipole_correction_zfrac':.9, 'custom_metadata':'preserve',
                'vasp_phi_fftn_shifted_1.5':np.arange(54.).reshape(2,3,3,3),
                'macevolt_potential_reference':'same_gauge'}
    source, output = tmp_path/'old.xyz', tmp_path/'new.xyz'
    original = '1\n'+key_val_dict_to_str(metadata)+'\nH 1 2 3 100 4 5 6\n'
    source.write_text(original)
    assert script.update_dataset(source, output) == 1
    assert source.read_text() == original
    lines = output.read_text().splitlines()
    info = key_val_str_to_dict(lines[1])
    assert 'workfunc' not in output.read_text()
    assert info['vacuum_potential'] == 2.
    assert info['config_vacuum_potential_weight'] == .5
    assert info['config_fermi_level_weight'] == .5
    assert info['vacuum_yfrac'] == .7
    assert info['dipole_correction_yfrac'] == .9
    assert info['custom_metadata'] == 'preserve'
    assert info['potential_reference'] == 'same_gauge'
    np.testing.assert_array_equal(info['vasp_phi_fftn_shifted_1.5'], metadata['vasp_phi_fftn_shifted_1.5'])
    assert lines[2] == 'H 1 2 3 4 5 6'
    assert script.clean_info(info).keys() == info.keys()
    with pytest.raises(FileExistsError):
        script.update_dataset(source, output)
    with pytest.raises(ValueError, match='disagrees'):
        script.clean_info(dict(metadata, vacuum_potential=9.))


@pytest.mark.parametrize('axis', range(3))
def test_fraction_names_and_model_plane_are_cell_axis_covariant(axis):
    torch.set_default_dtype(torch.float64)
    atoms = Atoms('OH', positions=[[3,3,4],[3.5,3,4.5]], cell=[7,7,12], pbc=[1,1,0])
    permutation = ([2,0,1], [1,2,0], [0,1,2])[axis]
    atoms.set_cell(atoms.cell.array[permutation][:,permutation])
    atoms.positions = atoms.positions[:,permutation]
    atoms.pbc = np.asarray([1,1,0], dtype=bool)[permutation]
    # Cyclic permutations preserve handedness and rotate the slab normal.
    atoms.info.update(vacuum_potential=2., total_charge=.1)
    atoms.info[f'vacuum_{"xyz"[axis]}frac'] = .7
    atoms.info[f'dipole_correction_{"xyz"[axis]}frac'] = .9
    keys = {key:key for key in atoms.info}
    graph = ExtAtomicData.from_config(config_from_atoms(atoms, key_specification=KeySpecification(
        info_keys=keys, arrays_keys={})), z_table=AtomicNumberTable([1,8]), cutoff=4., atomic_multipoles_max_l=1)
    assert float(graph.vacuum_fraction) == .7
    assert float(graph.dipole_correction_fraction) == .9
    assert float(graph.vacuum_potential_weight) == 1.
    assert float(graph.fermi_level_weight) == 0.
    from mace.tools import torch_geometric
    model = coupled_model()
    rotated = evaluate_coupled(model,torch_geometric.Batch.from_data_list([graph]).to_dict(),steps=3)
    original = deepcopy(graph)
    inverse = np.argsort(permutation)
    original.positions = graph.positions[:,inverse]
    original.cell = graph.cell[inverse][:,inverse]
    original.shifts = graph.shifts[:,inverse]
    original.pbc = graph.pbc[:,inverse] if graph.pbc.ndim == 2 else graph.pbc[inverse]
    baseline = evaluate_coupled(model,torch_geometric.Batch.from_data_list([original]).to_dict(),steps=3)
    for key in ('vacuum_potential','fermi_level','energy'):
        torch.testing.assert_close(rotated[key],baseline[key],rtol=1.e-9,atol=1.e-9)
    torch.testing.assert_close(rotated['forces'][:,inverse],baseline['forces'],rtol=1.e-9,atol=1.e-9)
    with pytest.raises(ValueError, match='only one'):
        plane_fraction({'vacuum_xfrac':.2,'vacuum_yfrac':.3}, 'vacuum', atoms.pbc)


def test_independent_vacuum_loss_does_not_require_fermi_or_spectral_labels():
    from types import SimpleNamespace
    ref = SimpleNamespace(vacuum_potential=torch.tensor([2.,float('nan')]),
                          vacuum_potential_weight=torch.tensor([1.,0.]),
                          weight=torch.ones(2), pbc=torch.tensor([[1,1,0],[1,1,1]]))
    value = torch.tensor([3.,7.], requires_grad=True)
    loss = WeightedVacuumPotential(relative=False)(ref, {'vacuum_potential':value})
    torch.testing.assert_close(loss, torch.tensor(1.))
    torch.testing.assert_close(torch.autograd.grad(loss,value)[0],torch.tensor([2.,0.]))
    with pytest.raises(ValueError, match='not recognised'):
        WeightedLoss({'workfunction':1})
    with pytest.raises(TypeError):
        WeightedLoss({'fourier_potential':{'weight':1,'vacuum_weight':1}})


def test_reference_labels_cannot_change_deployment_predictions():
    model = coupled_model()
    data = small_data()
    first = evaluate_coupled(model,deepcopy(data),steps=12)
    data['fermi_level'].fill_(999.)
    data['vacuum_potential'].fill_(-999.)
    second = evaluate_coupled(model,deepcopy(data),steps=12)
    for key in ('energy','forces','fermi_level','vacuum_potential','workfunction','fourier_potential'):
        torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)


def test_collector_has_no_removed_label_and_uses_same_gauge(tmp_path):
    path = ROOT/'scripts/collect_vasp_data.py'
    assert 'workfunction' not in path.read_text().lower()
    pytest.importorskip('pymatgen.io.vasp')
    spec=spec_from_file_location('collector',path);module=module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'INCAR').write_text('DIPOL = 0.0 0.0 0.0\n')
    for axis in range(3):
        pbc=np.ones(3,dtype=bool);pbc[axis]=False
        atoms=Atoms('H',positions=[[1,1,1]],cell=[8,8,8],pbc=pbc)
        atoms.info[f'vacuum_{"xyz"[axis]}frac']=.25
        atoms.info[f'dipole_correction_{"xyz"[axis]}frac']=.5
        field=np.broadcast_to(np.arange(8.).reshape([8 if i==axis else 1 for i in range(3)]),(8,8,8))
        value,fraction,slope=module.compute_vacuum_potential(atoms,tmp_path,field)
        assert value == -1.5
        assert fraction == .25
        assert slope == 1.
        assert module.open_axis_from_info(atoms.info)==axis


def test_planar_collector_math_without_vasp_io(tmp_path):
    """Exercise all axes and the gauge arithmetic independently of pymatgen."""
    import ast
    from types import SimpleNamespace
    tree = ast.parse((ROOT/'scripts/collect_vasp_data.py').read_text())
    names = {'compute_vacuum_potential', 'open_axis_from_info'}
    namespace = {'np':np, 'Incar':SimpleNamespace(from_file=lambda _: {'DIPOL':[0.,0.,0.]})}
    functions = ast.Module(body=[node for node in tree.body
                                if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
    exec(compile(functions,'collector_math','exec'),namespace)
    for axis in range(3):
        pbc = np.ones(3,dtype=bool);pbc[axis] = False
        atoms = Atoms('H',positions=[[1,1,1]],cell=[8,8,8],pbc=pbc)
        atoms.info.update({f'vacuum_{"xyz"[axis]}frac':.25,
                           f'dipole_correction_{"xyz"[axis]}frac':.5})
        grid = np.broadcast_to(np.arange(8.).reshape([8 if i==axis else 1 for i in range(3)]),(8,8,8))
        assert namespace['compute_vacuum_potential'](atoms,tmp_path,grid) == (-1.5,.25,1.)
        assert namespace['open_axis_from_info'](atoms.info) == axis


@pytest.mark.parametrize('ef,vacuum', [(False,False),(True,False),(False,True),(True,True)])
def test_wf_metric_requires_both_independent_labels(ef,vacuum):
    from mace.tools import torch_geometric
    from mace_scf.utils.train import evaluate
    from mace_scf.utils.model_training_wrappers import FixedPointWrapper
    from mace_scf.electrostatics.fixed_point_state import FixedPointTrainingOptions,FixedPointSCFOptions
    model = coupled_model()
    graph = small_data(batched=False)
    graph.fermi_level_weight.fill_(float(ef))
    graph.vacuum_potential_weight.fill_(float(vacuum))
    loader = torch_geometric.dataloader.DataLoader([graph],batch_size=1)
    wrapper = FixedPointWrapper(None,{'forces':True},FixedPointTrainingOptions(
        mode='unroll_scf',scf=FixedPointSCFOptions(num_scf_steps=2)))
    _,metrics = evaluate(model,wrapper,WeightedLoss({'forces':1.}),None,loader,'cpu')
    assert ('rmse_wf_abs' in metrics) == (ef and vacuum)
    assert ('rmse_esp_vac' in metrics) == vacuum


def test_vacuum_only_main_log(caplog):
    import logging
    from types import SimpleNamespace
    from mace_scf.utils.train import valid_err_log
    with caplog.at_level(logging.INFO):
        valid_err_log(1.,{'rmse_esp_vac':.1,'esp_vacuum_enabled':True},
                      SimpleNamespace(log=lambda _: None),'ElectrostaticRMSE',0)
    assert 'RMSE_ESPvac(abs/rel)=100.0000/n/a mV' in caplog.text
    assert 'RMSE_WF' not in caplog.text


def test_cold_reference_initialization_uses_independent_vacuum_label():
    from mace.tools import torch_geometric
    from mace_scf.electrostatics.coupled import CoupledResponse, initialize_reference
    from mace_scf.electrostatics.fixed_point_state import FixedPointTrainingOptions, FixedPointSCFOptions
    model = coupled_model()
    model.field_dependent_charges_map = CoupledResponse(
        node_feats_irreps='4x0e+4x1o',charges_irreps='0e+1o',num_elements=2,
        potential_widths=[1.5,3.])
    graph = small_data(batched=False)
    graph.fermi_level_weight.zero_()
    graph.fourier_potential_weight.zero_()
    graph.vacuum_potential_weight.fill_(1.)
    batch = torch_geometric.Batch.from_data_list([graph])
    before = evaluate_coupled(model,batch.to_dict(),steps=12,compute_force=False)
    graph.vacuum_potential = before['vacuum_potential'][0].detach()+.1
    objective = WeightedLoss({'vacuum_potential':{'weight':100.,'relative':False}})
    batch = torch_geometric.Batch.from_data_list([graph])
    initial = float(objective(batch,before).detach())
    flags = [p.requires_grad for p in model.parameters()]
    options = FixedPointTrainingOptions(mode='unroll_scf',scf=FixedPointSCFOptions(num_scf_steps=12))
    initialize_reference(model,[batch],'cpu',options,{'vacuum_potential':{'weight':100.,'relative':False}})
    after = evaluate_coupled(model,batch.to_dict(),steps=12,compute_force=False)
    assert float(objective(batch,after).detach()) < initial*.01
    assert flags == [p.requires_grad for p in model.parameters()]


def test_relative_cold_fit_is_invariant_to_label_offsets():
    from mace.tools import torch_geometric
    from mace_scf.electrostatics.coupled import CoupledResponse, initialize_reference
    from mace_scf.electrostatics.fixed_point_state import FixedPointTrainingOptions, FixedPointSCFOptions
    model=coupled_model()
    model.field_dependent_charges_map=CoupledResponse(
        node_feats_irreps='4x0e+4x1o',charges_irreps='0e+1o',num_elements=2,potential_widths=[1.5,3.])
    original=deepcopy(model)
    with torch.no_grad():
        model.field_dependent_charges_map.species_level.copy_(torch.tensor([.2,-.4]))
        model.field_dependent_charges_map.species_source.copy_(torch.tensor([[.3,-.1],[.1,.2]]))
    graphs=[]
    for i,n in enumerate((1,2,3,2)):
        atoms=Atoms('O'+'H'*n,positions=[[3,3,4]]+[[3.4+.3*j,3,4.5] for j in range(n)],
                    cell=[7,7,12+2*i],pbc=[1,1,0])
        g=small_data(charge=.1*i,atoms=atoms,batched=False)
        pred=evaluate_coupled(model,torch_geometric.Batch.from_data_list([g]).to_dict(),steps=12,compute_force=False)
        for key in ('fermi_level','vacuum_potential'):
            setattr(g,key,pred[key][0].detach())
            setattr(g,key+'_weight',torch.tensor(1.))
        graphs.append(g)
    options=FixedPointTrainingOptions(mode='unroll_scf',scf=FixedPointSCFOptions(num_scf_steps=12))
    predictions=[]
    for offset in (0.,10.):
        candidate=deepcopy(original)
        shifted=deepcopy(graphs)
        for g in shifted:
            g.fermi_level+=offset;g.vacuum_potential-=2*offset
        # Deliberately fit through singleton loader batches: centering is
        # over the entire calibration set, never independently per piece.
        loader=torch_geometric.dataloader.DataLoader(shifted,batch_size=1)
        initialize_reference(candidate,loader,'cpu',options,{'fermi_level':100,'vacuum_potential':100})
        output=evaluate_coupled(candidate,torch_geometric.Batch.from_data_list(graphs).to_dict(),steps=12,compute_force=False)
        predictions.append(torch.stack((output['fermi_level'],output['vacuum_potential']),-1).detach())
    torch.testing.assert_close(predictions[0],predictions[1],atol=2.e-7,rtol=2.e-7)


def reference_data(model):
    data = small_data()
    with torch.no_grad():
        prediction = evaluate_coupled(model,data,steps=50,compute_force=False)
    for key,values in [('fourier_density',prediction['fourier_density']),
                       ('fourier_potential',prediction['fourier_potential']),
                       ('fourier_proto_potential',prediction['fourier_total_potential']-prediction['fourier_potential'])]:
        data[key] = torch.view_as_complex(values.contiguous()).flatten()
        data[key+'_shape'] = prediction['k_vectors_grid_shape'][None]
        data[key+'_weight'] = torch.ones(1)
    data['fermi_level'] = prediction['fermi_level']+.02
    data['vacuum_potential'] = prediction['vacuum_potential']-.01
    data['vacuum_potential_weight'] = torch.ones(1)
    return data


@pytest.mark.parametrize('mode',['unroll_scf','shortcut_scf','implicit'])
def test_conditional_loss_has_finite_gradients_without_changing_free_predictions(mode):
    from types import SimpleNamespace
    model = coupled_model()
    data = reference_data(model)
    before = evaluate_coupled(model,deepcopy(data),steps=50,training=True,mode=mode)
    after = evaluate_coupled(model,deepcopy(data),steps=50,training=True,mode=mode,reference_conditioning=True)
    for key in ('energy','forces','fermi_level','vacuum_potential','fourier_potential','total_charge'):
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    assert 'reference_response' in after
    assert 'workfunction' not in after['reference_response']
    torch.testing.assert_close(after['total_charge'],data['total_charge'])
    assert after['reference_response']['reference_charge_error'].abs().max()>0
    ref = SimpleNamespace(**data,to_dict=lambda:data)
    objective = WeightedLoss({'fermi_level':{'weight':100,'relative':False},
                              'vacuum_potential':{'weight':100,'relative':False},'fourier_potential':100,
                              'fourier_density':{'weight':10,'reference':'farfield'}})
    value = objective(ref,after)+after['forces'].square().sum()
    value.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.field_dependent_charges_map.common_level.weight.grad.abs().max()>0


@pytest.mark.parametrize('reference_mode', ['fermi_level','electronic'])
def test_validation_wrapper_never_conditions_on_reference_labels(reference_mode):
    from mace_scf.electrostatics.fixed_point_state import FixedPointTrainingOptions,FixedPointSCFOptions
    from mace_scf.utils.model_training_wrappers import FixedPointWrapper
    model = coupled_model()
    data = reference_data(model)
    options = FixedPointTrainingOptions(mode='unroll_scf',reference_conditioning=reference_mode,
                                       scf=FixedPointSCFOptions(num_scf_steps=12))
    wrapper = FixedPointWrapper(None,{'forces':True},options)
    baseline = wrapper(model,deepcopy(data),training=False)
    data['fermi_level'].add_(9.)
    data['vacuum_potential'].sub_(11.)
    data['fourier_potential'].mul_(2.)
    prediction = wrapper(model,deepcopy(data),training=False)
    assert 'reference_response' not in prediction
    for key in ('energy','forces','fermi_level','vacuum_potential','workfunction'):
        torch.testing.assert_close(baseline[key],prediction[key],rtol=0,atol=0)
    assert prediction['scf_steps'].item()==50


@pytest.mark.parametrize('mode',['unroll_scf','shortcut_scf','implicit'])
def test_ef_only_conditioning_matches_paired_ef_gradients_without_extra_density_solve(mode):
    from types import SimpleNamespace
    model = coupled_model()
    data = reference_data(model)
    ref = SimpleNamespace(**data,to_dict=lambda:data)
    objective = WeightedLoss({'fermi_level':100.})
    parameters = list(model.field_dependent_charges_map.parameters())
    values, gradients = [], []
    for conditioning in ('electronic','fermi_level'):
        prediction = evaluate_coupled(model,deepcopy(data),steps=50,training=True,
                                      compute_force=False,mode=mode,reference_conditioning=conditioning)
        branch = prediction['reference_response']
        if conditioning == 'fermi_level':
            assert set(branch['reference_masks']) == {'fermi_level'}
            assert 'reference_charge_error' not in branch
            assert 'fourier_density' not in branch
        value = objective(ref,prediction)
        values.append(value.detach())
        gradients.append(torch.autograd.grad(value,parameters,allow_unused=True))
    torch.testing.assert_close(*values,rtol=0,atol=0)
    for a,b in zip(*gradients):
        if a is None or b is None:
            assert a is None and b is None
        else:
            torch.testing.assert_close(a,b,rtol=1.e-11,atol=1.e-11)


@pytest.mark.parametrize('reference_mode',['none','fermi_level','electronic'])
def test_reference_conditioning_configuration(reference_mode):
    from mace_scf.electrostatics.fixed_point_options import validate_fixed_point_training_options
    parsed = validate_fixed_point_training_options({'mode':'unroll_scf','scf':{'num_scf_steps':3},
                                                    'reference_conditioning':reference_mode})
    assert parsed.reference_conditioning == reference_mode
    with pytest.raises(ValueError,match='reference_conditioning'):
        validate_fixed_point_training_options({'mode':'unroll_scf','scf':{'num_scf_steps':3},'reference_conditioning':'unknown'})


def test_no_fermi_observations_means_no_conditional_branch():
    model = coupled_model()
    data = small_data()
    data['fermi_level_weight'].zero_()
    data['fermi_level'].fill_(float('nan'))
    result = evaluate_coupled(model,data,steps=3,training=True,reference_conditioning=True)
    assert 'reference_response' not in result
    assert torch.isfinite(result['forces']).all()


def test_reference_operator_first_and_second_parameter_derivatives():
    from types import SimpleNamespace
    model = coupled_model()
    data = reference_data(model)
    ref = SimpleNamespace(**data,to_dict=lambda:data)
    objective = WeightedLoss({'fermi_level':1,'fourier_potential':1,'vacuum_potential':1,
                              'fourier_density':{'weight':1,'reference':'farfield'}})
    parameter = model.field_dependent_charges_map.scalar_out.weight
    saved = parameter.detach().clone()
    torch.manual_seed(371)
    direction = torch.randn_like(parameter); direction /= direction.norm()
    def value():
        return objective(ref,evaluate_coupled(model,deepcopy(data),steps=12,training=True,
                                             compute_force=False,reference_conditioning=True))
    first = torch.autograd.grad(value(),parameter,create_graph=True)[0]
    slope = float((first*direction).sum().detach())
    second = torch.autograd.grad((first*direction).sum(),parameter)[0]
    curvature = float((second*direction).sum())
    values, slopes = [], []
    epsilon = 1.e-5
    try:
        for sign in (-1.,1.):
            with torch.no_grad():
                parameter.copy_(saved+sign*epsilon*direction)
            item = value()
            values.append(float(item.detach()))
            slopes.append(float((torch.autograd.grad(item,parameter)[0]*direction).sum()))
    finally:
        with torch.no_grad():
            parameter.copy_(saved)
    assert slope == pytest.approx((values[1]-values[0])/(2*epsilon),rel=1.e-5,abs=1.e-8)
    assert curvature == pytest.approx((slopes[1]-slopes[0])/(2*epsilon),rel=1.e-5,abs=1.e-8)


def test_optional_gradient_audit_preserves_parameters_and_grad_buffers():
    from mace.tools import torch_geometric
    from mace_scf.electrostatics.fixed_point_state import FixedPointTrainingOptions,FixedPointSCFOptions
    from mace_scf.utils.model_training_wrappers import FixedPointWrapper
    from mace_scf.utils.diagnostics import objective_gradients
    model = coupled_model()
    wrapper = FixedPointWrapper(None,{'forces':True},FixedPointTrainingOptions(
        mode='unroll_scf',scf=FixedPointSCFOptions(num_scf_steps=3)))
    loader = torch_geometric.dataloader.DataLoader([small_data(batched=False)],batch_size=1)
    parameters = [p.detach().clone() for p in model.parameters()]
    random_state = torch.random.get_rng_state()
    report = objective_gradients(model,wrapper,WeightedLoss({'forces':1,'dipole':1}),loader,'cpu')
    assert torch.equal(random_state,torch.random.get_rng_state())
    assert report['terms']['forces']['gradient_norm']>0
    for before,after in zip(parameters,model.parameters()):
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        assert after.grad is None

"""Training/deployment contracts independent of fitted validation accuracy."""
import random

import numpy as np
import pytest
import torch
from torch_ema import ExponentialMovingAverage
from mace.tools.checkpoint import CheckpointState

from mace_scf.utils.train import CheckpointHandler, evaluate
from mace_scf.utils.model_training_wrappers import FixedPointWrapper
from mace_scf.electrostatics.fixed_point_state import FixedPointTrainingOptions, FixedPointSCFOptions
from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
from .test_spectral_response import small_model, small_data


def test_iridium_recipes_keep_the_step_curriculum_and_final_deployment_budget():
    from pathlib import Path
    import yaml
    from .test_coupled_response import coupled_model
    steps = int(coupled_model().field_dependent_charges_map.deployment_steps)
    configs = list(Path(__file__).parents[1].glob('config_IrO2*.yaml'))
    assert configs
    for path in configs:
        config = yaml.safe_load(path.read_text())
        assert config['restart_latest'] is False
        stages = list(config['train_schedule'].values())
        assert stages[-1]['fixed_point_training_options']['scf']['num_scf_steps'] == steps == 50
        for stage in stages:
            options = stage['fixed_point_training_options']
            assert options['mode'] in ('unroll_scf', 'shortcut_scf')
            assert options['scf']['num_scf_steps'] <= steps
            assert options['scf']['mixing_parameter'] == .5
            assert stage['lr'] >= .001


def test_short_training_budget_is_supported_without_observation_banner(caplog):
    import logging
    from .test_coupled_response import coupled_model
    model = coupled_model()
    wrapper = FixedPointWrapper(None, {'forces':False}, FixedPointTrainingOptions(
        mode='unroll_scf', scf=FixedPointSCFOptions(num_scf_steps=2)))
    with caplog.at_level(logging.INFO):
        out = wrapper(model, small_data(), training=True)
        wrapper(model, small_data(), training=True)
    assert 'Observation convention:' not in caplog.text
    assert 'Use the deployment step count' not in caplog.text
    assert '2 training steps; 50 validation/deployment steps' in caplog.text
    assert out['scf_steps'].item() == 2


def test_log_uses_requested_density_and_dipole_names(caplog):
    import logging
    from mace_scf.utils.train import valid_err_log
    class Logger:
        def log(self,metrics):pass
    metrics={'rmse_rho':.001,'rmse_mu_per_atom':.002,'rmse_esp':.1,'rmse_esp_vac':.2,
             'rmse_esp_vac_rel':.1,'rmse_fermi_level':.2,'rmse_fermi_level_rel':.05,
             'esp_vacuum_enabled':True,'rmse_wf_abs':.15,'rmse_wf_rel':.14}
    with caplog.at_level(logging.INFO):valid_err_log(1.,metrics,Logger(),'ElectrostaticRMSE',5)
    assert 'RMSE_dip_per_atom=2.0000 meA' in caplog.text
    assert 'RMSE_rho=1.0000 me/A^3' in caplog.text
    assert 'RMSE_WF(abs/rel)=150.0000/140.0000 meV' in caplog.text
    assert 'RMSE_EF(abs/rel)=200.0000/50.0000 meV' in caplog.text
    assert 'RMSE_ESPvac(abs/rel)=200.0000/100.0000 mV' in caplog.text
    assert 'RMSE_ESP=100.0000 mV' in caplog.text
    assert 'RMSE_MU' not in caplog.text and 'RMSE_RHO' not in caplog.text




@pytest.mark.parametrize('relative_ef,relative_vac', [(True,True),(True,False),(False,True)])
def test_scalar_reference_calibration_uses_training_ema_and_preserves_state(relative_ef,relative_vac):
    from copy import deepcopy
    from mace.tools import torch_geometric
    from mace_scf.utils.train import calibrate_scalar_references
    from mace_scf.electrostatics.loss import WeightedLoss
    from .test_coupled_response import coupled_model
    model=coupled_model()
    model.field_dependent_charges_map.deployment_steps.fill_(4)
    wrapper=FixedPointWrapper(None,{'forces':True},FixedPointTrainingOptions(
        mode='unroll_scf',scf=FixedPointSCFOptions(num_scf_steps=2)))
    graphs=[small_data(charge=q,batched=False) for q in (.1,-.2,.3)]
    for graph,weight,noise in zip(graphs,(1.,2.,1.),(.1,-.1,.1)):
        batch=torch_geometric.Batch.from_data_list([graph])
        before=wrapper(model,deepcopy(batch.to_dict()),training=False)
        graph.fermi_level=before['fermi_level'].detach()[0]+3.5+(noise if weight==1. else noise/2)
        graph.fermi_level_weight=torch.tensor(1.)
        graph.vacuum_potential=before['vacuum_potential'].detach()[0]-.2+noise
        graph.vacuum_potential_weight=torch.tensor(1.)
        graph.weight=torch.tensor(weight)
    # The last graph has a missing EF label; its NaN must not enter the fit.
    graphs[-1].fermi_level_weight.zero_();graphs[-1].fermi_level.fill_(float('nan'))
    loader=torch_geometric.dataloader.DataLoader(graphs,batch_size=2,shuffle=True)
    ema=ExponentialMovingAverage(model.parameters(),decay=.99)
    with torch.no_grad():model.field_dependent_charges_map.common_level.weight.add_(.2)
    model.node_embedding.requires_grad_(False);model.train()
    raw=[p.clone().detach() for p in model.parameters()]
    flags=[p.requires_grad for p in model.parameters()]
    rng=torch.random.get_rng_state()
    loss=WeightedLoss({'fermi_level':{'weight':100,'relative':relative_ef},
                       'vacuum_potential':{'weight':100,'relative':relative_vac}})
    report=calibrate_scalar_references(model,wrapper,loss,ema,loader,'cpu')
    expected=torch.tensor([3.5 if relative_ef else 0.,-.2 if relative_vac else 0.])
    torch.testing.assert_close(model.field_dependent_charges_map.scalar_reference,expected,atol=1.e-12,rtol=1.e-12)
    assert model.training and wrapper.output_args['forces']
    assert flags==[p.requires_grad for p in model.parameters()]
    torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0.,rtol=0.)
    for a,b in zip(raw,model.parameters()):torch.testing.assert_close(a,b,atol=0.,rtol=0.)
    for key,target in (('fermi_level',.005**.5),('vacuum_potential',.1)):
        if key in report:assert report[key]['panel_RMSE_after_eV']==pytest.approx(target,abs=1.e-12)
    repeated=calibrate_scalar_references(model,wrapper,loss,ema,loader,'cpu')
    torch.testing.assert_close(model.field_dependent_charges_map.scalar_reference,expected,atol=1.e-12,rtol=1.e-12)
    for key in ('fermi_level','vacuum_potential'):
        if key in repeated:assert abs(repeated[key]['shift_eV'])<1.e-12
    absolute=WeightedLoss({'fermi_level':{'weight':100,'relative':False},
                           'vacuum_potential':{'weight':100,'relative':False}})
    assert calibrate_scalar_references(model,wrapper,absolute,ema,loader,'cpu') is None
    class Broken:
        output_args={'forces':True}
        def __call__(self,*args,**kwargs):
            raise RuntimeError('calibration interrupted')
    with pytest.raises(RuntimeError,match='calibration interrupted'):
        calibrate_scalar_references(model,Broken(),loss,ema,loader,'cpu')
    torch.testing.assert_close(model.field_dependent_charges_map.scalar_reference,expected,atol=1.e-12,rtol=1.e-12)
    assert model.training and flags==[p.requires_grad for p in model.parameters()]
    torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0.,rtol=0.)
    for a,b in zip(raw,model.parameters()):torch.testing.assert_close(a,b,atol=0.,rtol=0.)


def test_ir_finetune_has_no_epoch_ten_freeze_transition():
    from pathlib import Path
    import yaml
    config=yaml.safe_load((Path(__file__).parents[1]/'config_IrO2_finetune.yaml').read_text())
    stages=list(config['train_schedule'].values())
    assert stages[0]['start']==0 and stages[0]['end']==49
    assert all(not stage.get('freeze_foundation_backbone',False) for stage in stages)
    assert all(stage['lr']>=.001 for stage in stages)
    assert stages[-1]['fixed_point_training_options']['scf']['num_scf_steps']==50




def test_allocation_retry_preserves_update_and_discards_partial_gradients(monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    from ase import Atoms
    from mace.tools.torch_geometric import Batch
    from mace_scf.electrostatics.loss import WeightedLoss
    from mace_scf.utils.train import take_step, _evaluation_batches
    torch.set_default_dtype(torch.float64)
    graphs=[]
    for i in range(4):
        atoms=Atoms('O'+'H'*(i+1),positions=[[3.,3.,4.]]+[[3.5+j*.3,3.,4.5] for j in range(i+1)],
                    cell=[7.,7.,12.],pbc=[1,1,0])
        graph=small_data(atoms=atoms,batched=False)
        graph.energy=torch.tensor(.1*i);graph.energy_weight=torch.tensor(1.)
        graph.forces_weight=torch.tensor(float(i!=1));graph.fermi_level_weight=torch.tensor(float(i!=2))
        graphs.append(graph)
    batch=Batch.from_data_list(graphs)
    model=torch.nn.Linear(3,1)
    model.field_dependent_charges_map=SimpleNamespace(coupled=True)
    reference=deepcopy(model)
    loss=WeightedLoss({'energy_per_atom':100,'forces':500,'fermi_level':100})
    def prediction(model,data,**kwargs):
        n=len(data['ptr'])-1
        return {'energy':model.bias.expand(n),'forces':model.weight.expand_as(data['positions']),
                'fermi_level':model.bias.expand(n)}
    reference_optimizer=torch.optim.AdamW(reference.parameters(),lr=.001)
    reference_ema=ExponentialMovingAverage(reference.parameters(),decay=.9)
    take_step(reference,prediction,loss,batch.clone(),reference_optimizer,reference_ema,.5,'cpu')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    ema=ExponentialMovingAverage(model.parameters(),decay=.9)
    # Exercise CUDA retry decisions with CPU tensors; the real CUDA path has
    # its own hardware-dependent test. Fail once AFTER a partial backward.
    original_to=Batch.to
    monkeypatch.setattr(Batch,'to',lambda self,device,*a,**kw:original_to(self,'cpu',*a,**kw))
    tensor_to=torch.Tensor.to
    def simulated_device_copy(value,*args,**kwargs):
        if args and isinstance(args[0],(str,torch.device)) and torch.device(args[0]).type=='cuda':
            args=('cpu',*args[1:])
        return tensor_to(value,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',simulated_device_copy)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda device:torch.random.get_rng_state())
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda state,device:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    calls=[]
    def wrapper(model,data,**kwargs):
        count=len(data['ptr'])-1;calls.append(count)
        torch.rand(())
        if count==4 or (count==2 and calls.count(2)==2):
            raise torch.cuda.OutOfMemoryError('simulated allocation failure')
        return prediction(model,data,**kwargs)
    torch.manual_seed(17)
    expected_rng=torch.random.get_rng_state()
    for _ in graphs:torch.rand(())
    expected_next=torch.rand(())
    torch.random.set_rng_state(expected_rng)
    _,metrics=take_step(model,wrapper,loss,batch.clone(),optimizer,ema,.5,'cuda')
    # Four scalar-only predictions precede the four gradient microbatches;
    # that prepass restores the RNG and never steps the optimizer/EMA.
    assert calls==[4,2,2,1,1,1,1,1,1,1,1]
    assert metrics['device_batch_size']==1
    assert ema.num_updates==reference_ema.num_updates==1
    torch.testing.assert_close(torch.rand(()),expected_next,atol=0.,rtol=0.)
    for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,atol=1.e-14,rtol=1.e-14)
    for a,b in zip(ema.shadow_params,reference_ema.shadow_params):torch.testing.assert_close(a,b,atol=1.e-14,rtol=1.e-14)
    # A short final loader batch must not permanently reduce device capacity.
    optimizer._device_batch_size=4
    take_step(model,prediction,loss,Batch.from_data_list(graphs[:1]),optimizer,None,.5,'cuda')
    assert optimizer._device_batch_size==4
    before=[p.detach().clone() for p in model.parameters()]
    def numerical_failure(*args,**kwargs):raise FloatingPointError('not an allocation failure')
    with pytest.raises(FloatingPointError,match='not an allocation'):
        take_step(model,numerical_failure,loss,batch.clone(),optimizer,ema,.5,'cuda')
    for a,b in zip(model.parameters(),before):torch.testing.assert_close(a,b,atol=0.,rtol=0.)
    assert ema.num_updates==1
    with pytest.raises(torch.cuda.OutOfMemoryError):
        take_step(model,lambda *a,**k:(_ for _ in ()).throw(torch.cuda.OutOfMemoryError('single graph')),
                  loss,Batch.from_data_list(graphs[:1]),optimizer,ema,.5,'cuda')
    # Validation similarly visits every graph exactly once after allocation retry.
    def validation(model,data,**kwargs):
        if len(data['ptr'])-1>2:raise torch.cuda.OutOfMemoryError('validation allocation')
        return prediction(model,data,**kwargs)
    output=list(_evaluation_batches(model,validation,[batch.clone()],'cuda',None))
    assert [piece.num_graphs for piece,_ in output]==[2,2]
    assert validation._evaluation_device_batch_size==2
    torch.testing.assert_close(torch.cat([piece.energy for piece,_ in output]),batch.energy)


def test_raw_resume_and_ema_deployment_are_distinct(tmp_path):
    torch.set_default_dtype(torch.float64)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    ema = ExponentialMovingAverage(model.parameters(), decay=.9)
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.ones(2)).square().sum().backward()
        optimizer.step()
        ema.update()
    raw = [p.detach().clone() for p in model.parameters()]
    shadow = [p.clone() for p in ema.shadow_params]
    assert any(not torch.equal(a,b) for a,b in zip(raw,shadow))
    state = CheckpointState(model, optimizer, scheduler)
    handler = CheckpointHandler(directory=str(tmp_path), tag='resume_test', keep=False, ema=ema)
    handler.best_loss = .4
    handler.best_epoch = 2
    handler.save(state, 2)
    handler.save_progress(state, 2)
    random_next = random.random(), np.random.rand(), torch.rand(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(10.)
    optimizer.param_groups[0]['lr'] = 1.
    assert handler.load_latest(state, device='cpu') == 2
    assert optimizer.param_groups[0]['lr'] == .003
    assert handler.best_loss == .4
    assert random.random() == random_next[0]
    assert np.random.rand() == random_next[1]
    torch.testing.assert_close(torch.rand(1), random_next[2], rtol=0, atol=0)
    for a,b in zip(model.parameters(),raw):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    deployed = CheckpointHandler(directory=str(tmp_path), tag='resume_test', keep=False, deployment=True)
    assert deployed.load_latest(state, device='cpu') == 2
    for a,b in zip(model.parameters(),shadow):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_validation_restores_frozen_parameters_after_failure():
    model = torch.nn.Linear(2, 1)
    model.bias.requires_grad_(False)
    model.train()
    def bad_wrapper(*args, **kwargs):
        raise RuntimeError('intentional validation failure')
    class Loss:
        loss_fns = {}
    class Batch:
        def to(self,device):
            return self
        def to_dict(self):
            return {}
    with pytest.raises(RuntimeError, match='intentional'):
        evaluate(model,bad_wrapper,Loss(),None,[Batch()],'cpu')
    assert model.training
    assert model.weight.requires_grad
    assert not model.bias.requires_grad


def test_validation_averages_once_and_restores_after_failure():
    from contextlib import contextmanager
    model = torch.nn.Linear(2, 1)
    raw = model.weight.detach().clone()
    class Average:
        entries = 0
        @contextmanager
        def average_parameters(self):
            self.entries += 1
            with torch.no_grad():
                model.weight.add_(1.)
            try:
                yield
            finally:
                with torch.no_grad():
                    model.weight.copy_(raw)
    average = Average()
    class Batch:
        def to(self, device):
            return self
        def to_dict(self):
            return {}
    class Loss:
        loss_fns = {}
    def wrapper(*args, **kwargs):
        assert kwargs['ema'] is None
        torch.testing.assert_close(model.weight, raw+1.)
        raise RuntimeError('intentional validation failure')
    with pytest.raises(RuntimeError, match='intentional'):
        evaluate(model, wrapper, Loss(), average, [Batch(), Batch()], 'cpu')
    assert average.entries == 1
    torch.testing.assert_close(model.weight, raw, rtol=0., atol=0.)


def test_checkpoint_refuses_reinterpreted_backend_moments(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    state = CheckpointState(model, optimizer, scheduler)
    handler = CheckpointHandler(directory=str(tmp_path), tag='backend', keep=False)
    handler.save_progress(state, 0)
    model.backbone_layout = 'ir_mul'
    with pytest.raises(ValueError, match='different tensor-product layouts'):
        handler.load_latest(state, device='cpu')


@pytest.mark.parametrize('mode', ['implicit', 'unroll_scf', 'shortcut_scf'])
def test_validation_and_calculator_use_deployment_step_count(tmp_path, mode):
    from ase import Atoms
    from mace_scf.electrostatics.potential import evaluate_variational
    model=small_model()
    options=FixedPointTrainingOptions(mode=mode,scf=FixedPointSCFOptions(num_scf_steps=100,constant_charge=True))
    wrapper=FixedPointWrapper(None,{'forces':True,'stress':False,'virials':False},options)
    training=wrapper(model,small_data(),training=True)
    validation=wrapper(model,small_data(),training=False)
    assert training['scf_steps'].item()==100
    assert validation['scf_steps'].item()==50
    path=tmp_path/'model.model'
    torch.save(model,path)
    atoms=Atoms('OHH',positions=small_data()['positions'].numpy(),cell=[7,7,12],pbc=[1,1,0])
    atoms.info.update(total_charge=.1,external_field=np.array([0.,0.,.02]),fermi_level=-.4)
    atoms.calc=MACEFixedPointSCF(model_path=str(path),device='cpu',default_dtype='float64',pbc_handling='mixed_periodic',scf_options={'constant_charge':True})
    energy=atoms.get_potential_energy()
    assert atoms.calc.results['num_scf_steps']==50
    np.testing.assert_allclose(energy,validation['energy'].detach().numpy()[0],rtol=2.e-10,atol=2.e-10)
    np.testing.assert_allclose(atoms.get_forces(),validation['forces'].detach().numpy(),rtol=2.e-10,atol=2.e-10)
    # ASE normally ignores atoms.info; charge/voltage changes must invalidate it.
    atoms.info['total_charge']=-.2
    changed=atoms.get_potential_energy()
    ref=evaluate_variational(model,small_data(-.2))
    np.testing.assert_allclose(changed,ref['energy'].detach().numpy()[0],rtol=2.e-10,atol=2.e-10)
    atoms.info.update(counter_charge=.2,counter_charge_center=np.array([3.,3.,10.]))
    assert np.isfinite(atoms.get_potential_energy())
    from ase import units
    from mace_scf.md import NVTPhiLangevin
    dynamics=NVTPhiLangevin(atoms,timestep=.01*units.fs,temperature_K=0.,friction=.01/units.fs,
        target_potential=float(atoms.calc.results['workfunction'])+.1,
        rng=np.random.default_rng(17))
    dynamics.run(2)
    assert np.isfinite(atoms.positions).all()
    assert np.isfinite(dynamics.current_potential)
    assert atoms.info['counter_charge']==-atoms.info['total_charge']
    assert atoms.calc.results['num_scf_steps']==50


def test_diagnostics_module_is_optional():
    import sys
    from unittest.mock import patch
    from mace.tools import torch_geometric
    from mace_scf.utils import create_scf_convergence_summary
    model = small_model()
    model.field_dependent_charges_map.deployment_mode = 'unroll_scf'
    model.train()
    flags = [p.requires_grad for p in model.parameters()]
    loader = torch_geometric.dataloader.DataLoader([small_data(batched=False)], batch_size=1)
    with patch.dict(sys.modules, {'mace_scf.utils.diagnostics': None}):
        result = create_scf_convergence_summary(model, {'valid': loader}, {'forces': True}, 'cpu', {})
    assert 'mode=unroll_scf' in result
    assert 'residual_50' in result and 'WF_50_minus_100_eV' in result
    assert model.training
    assert [p.requires_grad for p in model.parameters()] == flags


def test_md_package_keeps_public_and_submodule_imports():
    from mace_scf.md import NVTPhiLangevin, NVTPhiVelocityVerlet, NVTPhiMDLogger
    from mace_scf.md.nvtphi_langevin import NVTPhiLangevin as Langevin
    from mace_scf.md.nvtphi_verlet import NVTPhiVelocityVerlet as Verlet
    from mace_scf.md.logger import NVTPhiMDLogger as Logger
    assert (NVTPhiLangevin, NVTPhiVelocityVerlet, NVTPhiMDLogger) == (Langevin, Verlet, Logger)


def test_yaml_false_is_not_true_for_amsgrad(tmp_path):
    from mace_scf.utils.extend_arg_parse import extended_arg_parser
    parser=extended_arg_parser()
    path=tmp_path/'config.yaml'
    path.write_text("name: test\namsgrad: false\ntrain_schedule: '{}'\nheads: '{}'\nerror_table: PerAtomRMSE\n")
    args=parser.parse_args(['--config',str(path)])
    assert args.amsgrad is False


def test_bulk_does_not_dilute_vacuum_observation():
    from types import SimpleNamespace
    from mace_scf.electrostatics.loss import WeightedVacuumPotential
    ref = SimpleNamespace(weight=torch.ones(2), vacuum_potential=torch.tensor([1.,0.]),
                          vacuum_potential_weight=torch.ones(2), fourier_potential_weight=torch.ones(2),
                          fourier_proto_potential_weight=torch.ones(2),
                          pbc=torch.tensor([[True, True, False], [True, True, True]]))
    pred = {'vacuum_potential': torch.tensor([2., 0.]),
            'vacuum_potential_dft': torch.tensor([1., 0.])}
    torch.testing.assert_close(WeightedVacuumPotential(relative=False)(ref, pred), torch.tensor(1.))


def test_relative_scalar_losses_are_weighted_pair_differences():
    from types import SimpleNamespace
    from mace_scf.electrostatics.loss import WeightedLoss
    ref = SimpleNamespace(weight=torch.tensor([1.,2.,3.,1.]),
        fermi_level=torch.tensor([2.,4.,8.,float('nan')]), fermi_level_weight=torch.tensor([1.,1.,1.,0.]))
    prediction = torch.tensor([3.,3.,11.,float('nan')], requires_grad=True)
    loss = WeightedLoss({'fermi_level':10})
    value = loss(ref, {'fermi_level':prediction})
    w = ref.weight[:3]; e = prediction[:3]-ref.fermi_level[:3]
    pair = ((e[:,None]-e[None,:]).square()*w[:,None]*w[None,:]).sum()/(2*(w.sum().square()-w.square().sum()))
    torch.testing.assert_close(value, 10*pair)
    shifted = prediction+7.
    torch.testing.assert_close(loss(ref, {'fermi_level':shifted}), value)
    grad, = torch.autograd.grad(value, prediction)
    torch.testing.assert_close(grad.sum(), torch.zeros(()), atol=1.e-12, rtol=0.)
    assert grad[-1] == 0
    absolute = WeightedLoss({'fermi_level':{'weight':10,'relative':False}})
    torch.testing.assert_close(absolute(ref, {'fermi_level':prediction}), 10*(e.square()*w).sum()/w.sum())
    with pytest.raises(TypeError, match='boolean'):
        WeightedLoss({'fermi_level':{'weight':1,'relative':'False'}})


def test_relative_scalar_microbatches_preserve_loss_and_gradient():
    from types import SimpleNamespace
    from mace_scf.electrostatics.loss import WeightedLoss
    def reference(indices):
        return SimpleNamespace(weight=torch.tensor([1.,2.,1.])[indices],
            fermi_level=torch.tensor([1.,2.,-1.])[indices],
            fermi_level_weight=torch.tensor([1.,0.,1.])[indices])
    loss = WeightedLoss({'fermi_level':{'weight':100}})
    theta = torch.tensor([2.,1.,2.], requires_grad=True)
    full = reference(slice(None))
    value = loss(full, {'fermi_level':theta})
    gradient, = torch.autograd.grad(value, theta)
    norms = loss.normalizers(full)
    moments = loss.relative_moments(full, {'fermi_level':theta})
    norms.update({key: m[1]/m[0].clamp_min(1.e-30) for key,m in moments.items()})
    partial = sum(loss(reference(slice(i,i+1)), {'fermi_level':theta[i:i+1]}, normalizers=norms) for i in range(3))
    part_grad, = torch.autograd.grad(partial,theta)
    torch.testing.assert_close(partial,value)
    torch.testing.assert_close(part_grad,gradient)
    assert partial > 0
    singleton = loss(reference(slice(0,1)), {'fermi_level':theta[:1]})
    assert singleton == 0
    with pytest.raises(ValueError, match='whole optimizer-batch mean'):
        loss(reference(slice(0,1)), {'fermi_level':theta[:1]}, normalizers=loss.normalizers(full))






def test_saved_solver_policy_and_original_v367_state_dict():
    source = small_model()
    target = small_model()
    source.field_dependent_charges_map.deployment_mode = 'unroll_scf'
    state = source.state_dict()
    target.load_state_dict(state)
    assert target.field_dependent_charges_map.deployment_mode == 'unroll_scf'
    del state['field_dependent_charges_map._extra_state']
    target.load_state_dict(state, strict=True)
    assert target.field_dependent_charges_map.deployment_mode == 'implicit'


def test_finite_validation_can_report_an_unconverged_state(tmp_path):
    model = small_model()
    with torch.no_grad():
        model.field_dependent_charges_map.hardness_scale.mul_(1.e-4)
    options = FixedPointTrainingOptions(mode='unroll_scf', scf=FixedPointSCFOptions(num_scf_steps=1))
    wrapper = FixedPointWrapper(None, {'forces':True, 'stress':False, 'virials':False}, options)
    output = wrapper(model, small_data(), training=False)
    assert output['scf_steps'].item() == 50
    assert output['scf_residual'].item() > options.scf.scf_tolerance
    assert torch.isfinite(output['forces']).all()
    from ase import Atoms
    path = tmp_path/'finite.model'
    torch.save(model, path)
    atoms = Atoms('OHH', positions=small_data()['positions'].numpy(), cell=[7, 7, 12], pbc=[1, 1, 0])
    atoms.info.update(total_charge=.1, external_field=np.array([0., 0., .02]))
    atoms.calc = MACEFixedPointSCF(str(path), device='cpu')
    np.testing.assert_allclose(atoms.get_potential_energy(), output['energy'].detach().numpy()[0], rtol=1.e-9, atol=1.e-9)
    np.testing.assert_allclose(atoms.get_forces(), output['forces'].detach().numpy(), rtol=1.e-9, atol=1.e-9)
    assert not atoms.calc.results['scf_converged']
    atoms.calc = MACEFixedPointSCF(str(path), device='cpu', ignore_nonconverged=False)
    with pytest.raises(RuntimeError, match='Electronic residual'):
        atoms.get_potential_energy()


@pytest.mark.parametrize('keep',[False,True])
def test_archived_bad_epochs_do_not_replace_best_deployment(tmp_path,keep):
    model=torch.nn.Linear(1,1,bias=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    state=CheckpointState(model,optimizer,scheduler)
    handler=CheckpointHandler(directory=str(tmp_path),tag='archive',keep=keep)
    with torch.no_grad():model.weight.fill_(2.)
    handler.best_epoch=2;handler.best_loss=.1
    handler.save(state,2,keep_last=True)
    with torch.no_grad():model.weight.fill_(5.)
    handler.save(state,5,keep_last=True)
    handler.save_progress(state,5)
    assert len(list(tmp_path.glob('*.pt')))==2
    deployment=CheckpointHandler(directory=str(tmp_path),tag='archive',keep=keep,deployment=True)
    assert deployment.load_latest(state,device='cpu')==2
    torch.testing.assert_close(model.weight,torch.full_like(model.weight,2.))
    resume=CheckpointHandler(directory=str(tmp_path),tag='archive',keep=keep)
    assert resume.load_latest(state,device='cpu')==5
    assert resume.best_epoch==2
    torch.testing.assert_close(model.weight,torch.full_like(model.weight,5.))



def test_validation_exception_archives_failed_epoch(tmp_path,monkeypatch):
    import importlib
    module=importlib.import_module('mace_scf.utils.train')
    model=torch.nn.Linear(1,1,bias=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    handler=CheckpointHandler(directory=str(tmp_path),tag='failed',keep=True)
    monkeypatch.setattr(module,'take_step',lambda *a,**k:(0.,{'grad_norm_before_clip':1.,'gradient_retained':1.,'graphs':1,'grad_clip_applied':False}))
    monkeypatch.setattr(module,'calibrate_scalar_references',lambda *a,**k:None)
    def fail(*a,**k):raise RuntimeError('test SCF failure')
    monkeypatch.setattr(module,'evaluate',fail)
    with pytest.raises(RuntimeError,match='test SCF failure'):
        module.train(model,None,None,[None],[None],optimizer,scheduler,0,0,10,handler,None,5,'cpu','ElectrostaticRMSE',save_all_checkpoints=True)
    paths=list(tmp_path.glob('*.pt'))
    assert len(paths)==1 and 'epoch-0' in paths[0].name
    saved=torch.load(paths[0],weights_only=False)
    assert saved['best_epoch'] is None and saved['best_loss']==float('inf')
    assert not handler.progress_path.with_suffix('.best').exists()
    deployment=CheckpointHandler(directory=str(tmp_path),tag='failed',keep=True,deployment=True)
    assert deployment.load_latest(CheckpointState(model,optimizer,scheduler),device='cpu') is None


@pytest.mark.parametrize('old_archive_all',[False,True])
def test_resume_old_archives_preserves_best_when_retention_is_enabled(tmp_path,old_archive_all):
    model=torch.nn.Linear(1,1,bias=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    state=CheckpointState(model,optimizer,scheduler)
    handler=CheckpointHandler(directory=str(tmp_path),tag='old',keep=True)
    # The previous release saved running best_loss but no best_epoch marker.
    for epoch in ([0,2,5] if old_archive_all else [2]):
        with torch.no_grad():model.weight.fill_(float(epoch))
        value=handler.builder.create_checkpoint(state)
        value['best_loss']=.2 if epoch==0 else .1
        torch.save(value,tmp_path/f'old_epoch-{epoch}.pt')
    with torch.no_grad():model.weight.fill_(5.)
    value=handler.builder.create_checkpoint(state)
    value.update(best_loss=.1,epoch=5)
    handler.progress_path.parent.mkdir(parents=True)
    torch.save(value,handler.progress_path)
    assert handler.load_latest(state,device='cpu')==5
    assert handler.best_epoch==2
    with torch.no_grad():model.weight.fill_(10.)
    handler.save(state,10,keep_last=True)
    deployment=CheckpointHandler(directory=str(tmp_path),tag='old',keep=True,deployment=True)
    assert deployment.load_latest(state,device='cpu')==2
    torch.testing.assert_close(model.weight,torch.full_like(model.weight,2.))

def test_stability_outliers_are_debug_only_and_do_not_change_metrics(caplog):
    import logging
    from mace.tools import torch_geometric
    from mace_scf.electrostatics.loss import WeightedLoss
    graphs = [small_data(charge=float(i+1), batched=False) for i in range(3)]
    for graph in graphs:
        graph.forces.zero_()
        graph.forces_weight.fill_(1.)
    def wrapper(model, data, **kwargs):
        charge = data['total_charge']
        return {'energy':charge*0., 'forces':charge[data['batch'], None].expand(-1, 3),
                'scf_residual':.1/charge}
    def check(level, batch_size):
        caplog.clear()
        loader = torch_geometric.dataloader.DataLoader(graphs, batch_size=batch_size, shuffle=False)
        with caplog.at_level(level):
            result = evaluate(torch.nn.Linear(1,1), wrapper, WeightedLoss({'forces':1}), None, loader, 'cpu')
        messages = [record.getMessage() for record in caplog.records if 'stability outliers' in record.getMessage()]
        return result, messages
    before, quiet = check(logging.INFO, 3)
    after, debug = check(logging.DEBUG, 2)
    assert not quiet and len(debug)==1
    assert before[0]==pytest.approx(after[0])
    assert before[1]['rmse_f']==pytest.approx(after[1]['rmse_f'])
    assert "by force contribution=[{'index': 2" in debug[0]
    assert "by SCF residual=[{'index': 0" in debug[0]
    assert 'zero-based evaluation order' in debug[0]
def test_voltage_diagnostics_center_once_and_preserve_input(caplog):
    import logging
    from copy import deepcopy
    from mace_scf.utils.train import _voltage_diagnostics
    rows = [{'index':i, 'weight':w, 'wf_error':e, 'ef_error':0.,
             'vacuum_error':e, 'dipole_step_error':e-2.}
            for i,(w,e) in enumerate(((1.,1.),(2.,3.),(1.,1.)))]
    original = deepcopy(rows)
    with caplog.at_level(logging.DEBUG):
        _voltage_diagnostics(rows, 'validation')
    assert rows == original
    messages = [record.getMessage() for record in caplog.records]
    assert "[{'index': 1" in messages[0]
    assert "'relative_WF_SSE_fraction': 0.5" in messages[0]
    assert 'correlation with WF residual=1;' in messages[1]

def test_gradient_diagnostics_preserve_gradients_and_track_missing_owners():
    import torch
    from mace_scf.utils.train import _gradient_diagnostics
    a = torch.nn.Parameter(torch.tensor([1., 2.]))
    b = torch.nn.Parameter(torch.tensor([3.]))
    a.grad = torch.tensor([3., 4.])
    optimizer = torch.optim.AdamW([{'name':'field', 'params':[a]}, {'name':'local', 'params':[b]}], lr=.001)
    report = _gradient_diagnostics(optimizer)
    assert report[0]['norm'] == 5.
    assert report[0]['squared_norm_fraction'] == 1.
    assert report[0]['gradient_values'] == 2
    assert report[1]['trainable_values'] == 1
    assert report[1]['gradient_values'] == 0
    torch.testing.assert_close(a.grad, torch.tensor([3., 4.]))
    assert b.grad is None and not optimizer.state


def test_voltage_diagnostics_reuse_evaluation_for_open_x_slabs(caplog):
    import logging
    from mace.tools import torch_geometric
    from mace_scf.electrostatics.loss import WeightedLoss
    graphs = [small_data(charge=float(i+1), batched=False) for i in range(3)]
    for graph in graphs:
        graph.pbc = torch.tensor([[False, True, True]])
        graph.fermi_level_weight.fill_(1.)
        graph.vacuum_potential_weight.fill_(1.)
        graph.dipole_weight = torch.tensor([[1., 0., 0.]])
        graph.dipole.zero_()

    calls = []
    def wrapper(model, data, **kwargs):
        calls.append(len(data['ptr'])-1)
        error = data['total_charge']
        ef = data['fermi_level']+error
        vac = data['vacuum_potential']+2*error
        dipole = torch.stack((error, error*0., error*0.), dim=-1)
        return {'fermi_level':ef, 'vacuum_potential':vac,
                'workfunction':vac-ef, 'dipole':dipole}

    def check(level, batch_size):
        caplog.clear()
        calls.clear()
        loader = torch_geometric.dataloader.DataLoader(graphs, batch_size=batch_size, shuffle=False)
        with caplog.at_level(level):
            result = evaluate(torch.nn.Linear(1, 1), wrapper,
                              WeightedLoss({'fermi_level':1, 'vacuum_potential':1}),
                              None, loader, 'cpu')
        return result, list(calls), [record.getMessage() for record in caplog.records]

    quiet, quiet_calls, _ = check(logging.INFO, 3)
    debug, debug_calls, messages = check(logging.DEBUG, 2)
    assert quiet_calls == [3] and debug_calls == [2, 1]
    assert quiet[0] == pytest.approx(debug[0])
    assert quiet[1]['rmse_wf_rel'] == pytest.approx(debug[1]['rmse_wf_rel'])
    assert any('correlation with WF residual=1;' in message for message in messages)



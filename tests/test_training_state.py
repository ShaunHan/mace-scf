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


def test_validation_and_calculator_use_deployment_step_count(tmp_path):
    from ase import Atoms
    from mace_scf.electrostatics.potential import evaluate_variational
    model=small_model()
    options=FixedPointTrainingOptions(mode='implicit',scf=FixedPointSCFOptions(num_scf_steps=100,constant_charge=True))
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
    from mace_scf.utils import create_scf_convergence_summary
    with patch.dict(sys.modules, {'mace_scf.utils.diagnostics': None}):
        assert 'not installed' in create_scf_convergence_summary(None)


def test_yaml_false_is_not_true_for_amsgrad(tmp_path):
    from mace_scf.utils.extend_arg_parse import extended_arg_parser
    parser=extended_arg_parser()
    path=tmp_path/'config.yaml'
    path.write_text("name: test\namsgrad: false\ntrain_schedule: '{}'\nheads: '{}'\nerror_table: PerAtomRMSE\n")
    args=parser.parse_args(['--config',str(path)])
    assert args.amsgrad is False


def test_bulk_does_not_dilute_vacuum_observation():
    from types import SimpleNamespace
    from mace_scf.electrostatics.loss import weighted_vacuum_potential
    ref = SimpleNamespace(weight=torch.ones(2), fourier_potential_weight=torch.ones(2),
                          fourier_proto_potential_weight=torch.ones(2),
                          pbc=torch.tensor([[True, True, False], [True, True, True]]))
    pred = {'vacuum_potential': torch.tensor([2., 0.]),
            'vacuum_potential_dft': torch.tensor([1., 0.])}
    torch.testing.assert_close(weighted_vacuum_potential(ref, pred), torch.tensor(1.))

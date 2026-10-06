"""Optional inverse-design and training probes; safe to remove for deployment."""
import logging
import numpy as np
import torch
from mace.tools import torch_geometric


def _panel(loader, maximum):
    indices = np.linspace(0, len(loader.dataset)-1, min(maximum,len(loader.dataset))).round().astype(int)
    return torch_geometric.dataloader.DataLoader([loader.dataset[int(i)] for i in np.unique(indices)], batch_size=1, shuffle=False)


def _probe_features(model, data, output):
    """Model-only geometry/field summaries, never reference-valued inputs."""
    phi = torch.view_as_complex(output["fourier_total_potential"].contiguous())
    phi = phi/output["k_vectors_grid_shape"].prod()
    wave = output["k_vectors"][0]
    phase = torch.exp(1j*(wave@data["positions"].T))
    sample = (phi[0,:,None]*phase).real.sum(0)
    field = (1j*phi[0,:,None,None]*phase[...,None]*wave[:,None]).real.sum(0)
    composition = data["node_attrs"].mean(0)
    dipole = output["dipole"].square().sum().sqrt()/data["volume"].reshape(-1)[0]
    return torch.cat((sample.new_ones(1), composition,
        torch.stack((sample.mean(), sample.std(correction=0), field.square().mean().sqrt(), dipole))))


def _ridge_report(train_x, train_y, valid_x, valid_y):
    """Select regularization on a training-only inner split; report development."""
    if len(train_x)<12 or len(valid_x)<2:
        return {"skipped":"need >=12 observed training and >=2 validation graphs"}
    x,y,v,z=(a.double() for a in (train_x,train_y,valid_x,valid_y))
    inner=torch.arange(len(x))%4==0
    means=x[~inner].mean(0)
    scale=x[~inner].std(0).clamp_min(1.e-6)
    means[0]=0.
    scale[0]=1.
    a,b=(x-means)/scale,(v-means)/scale
    baseline=float(y[inner].square().mean())
    candidates=[(baseline,None)]
    for ridge in (1.e-3,.1,10.,1000.):
        gram=a[~inner].T@a[~inner]+ridge*torch.eye(a.shape[1],dtype=a.dtype)
        coefficient=torch.linalg.solve(gram,a[~inner].T@y[~inner])
        candidates.append((float((y[inner]-a[inner]@coefficient).square().mean()),ridge))
    score,ridge=min(candidates,key=lambda item:item[0])
    if ridge is None:
        return {"selected":"no correction", "inner_RMSE":baseline**.5,
                "development_RMSE":float(z.square().mean().sqrt())}
    coefficient=torch.linalg.solve(a.T@a+ridge*torch.eye(a.shape[1],dtype=a.dtype),a.T@y)
    return {"selected_ridge":ridge,"training_before":float(y.square().mean().sqrt()),
            "training_after":float((y-a@coefficient).square().mean().sqrt()),
            "development_before":float(z.square().mean().sqrt()),
            "development_after":float((z-b@coefficient).square().mean().sqrt()),
            "selection":"training inner split only; diagnostic readout NOT installed"}


def audit_training(model, wrapper, loss, ema, train_loader, valid_loader, device, epoch):
    """Bounded, optional deployment/gap/low-k/identifiability audit."""
    import json
    from contextlib import nullcontext
    from mace_scf.electrostatics.potential import evaluate_electronic
    from mace_scf.electrostatics.coupled_solver import SCFConvergenceError, SCFNumericalError
    if not getattr(getattr(model,"field_dependent_charges_map",None),"spectral",False):
        return
    rows={}
    feature_rows={}
    residual_rows={}
    flags=[p.requires_grad for p in model.parameters()]
    was_training=model.training
    rng=torch.random.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model.eval()
        model.requires_grad_(False)
        with ema.average_parameters() if ema is not None else nullcontext():
            for label,loader in (("train",train_loader),("validation",valid_loader)):
                errors=[]
                features=[]
                targets=[]
                low=[]
                drifts=[]
                charges=[]
                force_squares=[]
                species_errors={}
                charge_responses=[]
                reference_gaps=[]
                for index,batch in enumerate(_panel(loader,32 if label=="train" else 16)):
                    batch=batch.to(device)
                    data=batch.to_dict()
                    output=evaluate_electronic(model,data,steps=50,compute_force=True)
                    atom_error=(output["forces"].detach()-batch.forces).square()
                    observed=(batch.weight*batch.forces_weight)[batch.batch]>0
                    force_squares.append(atom_error[observed].reshape(-1).cpu())
                    elements=model.atomic_numbers[batch.node_attrs.argmax(-1)]
                    for element in elements[observed].unique():
                        selected=atom_error[observed & (elements==element)]
                        species_errors.setdefault(str(int(element)),[]).append(selected.reshape(-1).cpu())
                    charges.append(float(batch.total_charge.reshape(-1)[0]))
                    if index<2:
                        with torch.no_grad():
                            values=[]
                            training_steps = wrapper.scf_options.num_scf_steps
                            for steps in sorted({training_steps,50,100}):
                                try:
                                    result=evaluate_electronic(model,data,steps=steps,compute_force=False)
                                except RuntimeError as exc:
                                    if not isinstance(exc,(SCFConvergenceError,SCFNumericalError)) and 'Electronic linear solve did not converge' not in str(exc):
                                        raise
                                    values.append({'steps':steps, 'converged':False, 'reason':str(exc)})
                                    continue
                                values.append({"steps":steps,"WF_eV":float(result['workfunction'][0]),
                                    "EF_eV":float(result['fermi_level'][0]),"residual":float(result['scf_residual'][0])})
                            drifts.append(values)
                            shifted=[]
                            for increment in (-.01,.01):
                                charged=dict(data)
                                charged['total_charge']=data['total_charge']+increment
                                try:
                                    result=evaluate_electronic(model,charged,steps=50,compute_force=False)
                                except RuntimeError as exc:
                                    if not isinstance(exc,(SCFConvergenceError,SCFNumericalError)) and 'Electronic linear solve did not converge' not in str(exc):
                                        raise
                                    charge_responses.append({'increment':increment,'failure':str(exc)})
                                    shifted=[]
                                    break
                                shifted.append(torch.stack((result['fermi_level'][0],result['workfunction'][0])))
                            if len(shifted)==2:
                                charge_responses.append(((shifted[1]-shifted[0])/.02).cpu().tolist())
                    with torch.no_grad():
                        if (float(batch.workfunction_weight[0])>0 and float(batch.fermi_level_weight[0])>0
                                and int(batch.pbc.sum())==2):
                            vacuum_target = batch.fermi_level[0]+batch.workfunction[0]
                            error=torch.stack((output['fermi_level'][0]-batch.fermi_level[0],
                                output['vacuum_potential'][0]-vacuum_target,
                                output['workfunction'][0]-batch.workfunction[0]))
                            if 'vacuum_potential_dft' in output and float(batch.fourier_proto_potential_weight[0]*batch.fourier_potential_weight[0])>0:
                                reference_gaps.append(float(output['vacuum_potential_dft'][0]-vacuum_target))
                            errors.append(error.cpu())
                            features.append(_probe_features(model,data,output).cpu())
                            targets.append((-error[2]).cpu())
                        if 'fourier_potential_dft' in output and float(batch.fourier_potential_weight[0])>0:
                            diff=(output['fourier_potential']-output['fourier_potential_dft'])/output['k_vectors_grid_shape'].prod()
                            k2=output['k_vectors'].square().sum(-1)
                            use=output['k_vectors_mask']&(k2>0)
                            use=use & output.get('fourier_potential_dft_mask',use)
                            positive=k2[use]
                            if positive.numel():
                                shell=use&(k2<=4*positive.min())
                                low.append([float(diff[shell].square().sum()),float(diff[use].square().sum())])
                rows[label]={"graphs":len(charges),"charge_span":[min(charges),max(charges)],
                    "step_comparison":drifts,"dEF_dQ_and_dWF_dQ_eV_per_e":charge_responses,
                    "force_RMSE_by_atomic_number_eV_A":{z:float(torch.cat(v).mean().sqrt()) for z,v in species_errors.items()}}
                observed_forces=torch.cat(force_squares)
                if observed_forces.numel():
                    rows[label]['force_RMSE_eV_A']=float(observed_forces.mean().sqrt())
                if max(charges)-min(charges)<1.e-8:
                    rows[label]['charge_response_identifiability']='Single-charge data do not determine capacitance; slopes above are predictions, not validation.'
                if errors:
                    e=torch.stack(errors)
                    center=e-e.mean(0)
                    rows[label].update(EF_vac_WF_RMSE=(e.square().mean(0).sqrt()).tolist(),
                        EF_vac_WF_bias=e.mean(0).tolist(),EF_vac_covariance=float((center[:,0]*center[:,1]).mean()),
                        WF_relative_RMSE_eV=float(center[:,2].square().mean().sqrt()),
                        WF_error_variance_from_EF_vac_covariance=float(center[:,:2].square().mean(0).sum()-2*(center[:,0]*center[:,1]).mean()),
                        reference_WF_identity_max_error_eV=float((e[:,2]-(e[:,1]-e[:,0])).abs().max()))
                    feature_rows[label]=torch.stack(features)
                    residual_rows[label]=torch.stack(targets)
                if reference_gaps:
                    gap=torch.tensor(reference_gaps)
                    rows[label]['spectral_vacuum_minus_EF_plus_WF_RMSE_eV']=float(gap.square().mean().sqrt())
                if low:
                    low=torch.tensor(low).sum(0)
                    rows[label]['low_k_potential_error_power_fraction']=float(low[0]/low[1].clamp_min(1.e-30))
            if all(k in feature_rows for k in ('train','validation')):
                rows['model_field_readout_probe']=_ridge_report(feature_rows['train'],residual_rows['train'],feature_rows['validation'],residual_rows['validation'])
            rows['epoch']=epoch
            rows['scope']='Deterministic panels, not whole-dataset metrics. Validation is development data, never used for fitting or selecting the probe.'
            logging.info("Electronic diagnostic %s",json.dumps(rows,allow_nan=False))
    finally:
        for p,flag in zip(model.parameters(),flags):
            p.requires_grad_(flag)
        model.train(was_training)
        torch.random.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)



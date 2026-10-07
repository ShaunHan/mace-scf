"""Optional inverse-design and training probes; safe to remove for deployment."""
import logging
import numpy as np
import torch
from mace.tools import torch_geometric


class ValidationAudit:
    """Whole-split error tails from existing validation outputs, without reruns.

    Panels are useful for expensive derivatives, but cannot establish a
    generalization gap or exclude rare failure cohorts. All indices below
    refer to the validation loader order, not a fitted or filtered split.
    """
    def __init__(self, density_width=1.5):
        self.rows = []
        self.species = {}
        self.density_width=float(density_width)

    @torch.no_grad()
    def update(self, batch, output):
        from mace_scf.electrostatics.loss import vacuum_reference_weights
        ef = output.get('fermi_level')
        vac = output.get('vacuum_potential')
        vacuum_target, weight = vacuum_reference_weights(batch, vac) if vac is not None else (None,None)
        for i in range(batch.num_graphs):
            start, stop = int(batch.ptr[i]), int(batch.ptr[i+1])
            attrs = batch.node_attrs[start:stop]
            composition = tuple(int(x) for x in attrs.sum(0))
            row = {'index':len(self.rows), 'atoms':stop-start,
                   'composition':composition, 'charge':float(batch.total_charge[i])}
            if vac is not None and ef is not None and weight[i]>0:
                row.update(weight=float(weight[i]), ef=float(ef[i]-batch.fermi_level[i]),
                           vac=float(vac[i]-vacuum_target[i]),
                           wf=float(output['workfunction'][i]-batch.workfunction[i]))
                if 'vacuum_potential_dft' in output and batch.fourier_potential_weight[i]*batch.fourier_proto_potential_weight[i]>0:
                    row['vacuum_reference_gap'] = float(output['vacuum_potential_dft'][i]-vacuum_target[i])
            if output.get('forces') is not None and batch.forces_weight[i]*batch.weight[i]>0:
                square=(output['forces'][start:stop]-batch.forces[start:stop]).square()
                row['force_rmse']=float(square.mean().sqrt())
                for z in range(attrs.shape[1]):
                    values=square[attrs[:,z]>0]
                    previous=self.species.setdefault(str(z),[0.,0])
                    previous[0]+=float(values.sum());previous[1]+=values.numel()
            if 'fourier_total_potential_dft' in output and batch.fourier_potential_weight[i]*batch.fourier_proto_potential_weight[i]>0:
                delta=(output['fourier_total_potential'][i]-output['fourier_total_potential_dft'][i])/output['k_vectors_grid_shape'].prod()
                k2=output['k_vectors'][i].square().sum(-1)
                use=output['k_vectors_mask'][i] & output['fourier_total_potential_dft_mask'][i] & (k2>0)
                power=torch.where(use,delta.square().sum(-1),0.)
                row['potential_error_power_eV2']=float(power.sum())
                row['gaussian_lowpass_error_power_eV2']=float((power*torch.exp(-self.density_width**2*k2)).sum())
                if 'planar_mode_mask' in output:
                    row['planar_error_power_eV2']=float((power*output['planar_mode_mask'][i]).sum())
            self.rows.append(row)

    def summary(self):
        rows=[r for r in self.rows if 'wf' in r]
        report={'graphs':len(self.rows), 'index_convention':'supplied loader order, zero based',
                'force_rmse_by_element_column_eV_A':{z:(v[0]/v[1])**.5
                    for z,v in self.species.items() if v[1]}}
        total=sum(r.get('potential_error_power_eV2',0.) for r in self.rows)
        if total>0:
            report['potential_error_power_fractions']={
                'normal_reciprocal_line':sum(r.get('planar_error_power_eV2',0.) for r in self.rows)/total,
                'Gaussian_lowpass':sum(r.get('gaussian_lowpass_error_power_eV2',0.) for r in self.rows)/total,
                'Gaussian_width_A':self.density_width}
        if not rows:
            forces=sorted((r for r in self.rows if 'force_rmse' in r),key=lambda r:r['force_rmse'],reverse=True)
            report['largest_force_errors']=forces[:8]
            return report
        error=np.array([[r['ef'],r['vac'],r['wf']] for r in rows])
        weights=np.array([r['weight'] for r in rows]);weights/=weights.sum()
        mean=weights@error;center=error-mean
        report.update(observed_EF_vac_WF=len(rows), EF_vac_WF_rmse_eV=np.sqrt(weights@error**2).tolist(),
                      EF_vac_WF_bias_eV=mean.tolist(),EF_vac_covariance_eV2=float(weights@(center[:,0]*center[:,1])),
                      WF_absolute_error_quantiles_eV=dict(zip(('median','p90','p99','max'),np.quantile(abs(error[:,2]),[.5,.9,.99,1.]).tolist())))
        report['WF_identity_max_error_eV']=float(abs(error[:,2]-(error[:,1]-error[:,0])).max())
        report['WF_variance_components_eV2']={
            'EF':float(weights@center[:,0]**2), 'vacuum':float(weights@center[:,1]**2),
            'covariance_contribution':float(-2*(weights@(center[:,0]*center[:,1])))}
        gaps=[r['vacuum_reference_gap'] for r in rows if 'vacuum_reference_gap' in r]
        if gaps:
            report['spectral_vacuum_minus_EF_plus_WF_RMSE_eV']=float(np.sqrt(np.mean(np.square(gaps))))
        power=weights*error[:,2]**2
        order=np.argsort(-power)
        report['WF_squared_error_share_largest_10_percent']=float(power[order[:max(1,int(np.ceil(.1*len(rows))))]].sum()/max(power.sum(),1.e-30))
        report['largest_WF_errors']=[rows[int(i)] for i in order[:8]]
        groups={}
        for row in rows:groups.setdefault(str(row['composition']),[]).append(row)
        report['composition_cohorts']=sorted(({'element_counts':k,'graphs':len(v),
                'wf_rmse_eV':float(np.sqrt(np.average([r['wf']**2 for r in v],weights=[r['weight'] for r in v])))}
                for k,v in groups.items()),key=lambda r:r['wf_rmse_eV'],reverse=True)[:12]
        report['scope']='Full supplied split; diagnostic only, no reference-dependent correction is installed.'
        return report


def runtime_inventory(model, optimizer, device, batch_size):
    """Separate model width, optimizer storage and CUDA memory at stage start."""
    import json
    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
    optimizer_bytes=sum(v.numel()*v.element_size() for state in optimizer.state.values()
                        for v in state.values() if torch.is_tensor(v))
    report={'parameters':sum(p.numel() for p in model.parameters()),
            'parameter_MiB':parameter_bytes/2**20,'optimizer_MiB':optimizer_bytes/2**20,
            'batch_size':batch_size,'layout':getattr(model,'backbone_layout','mul_ir'),
            'blocks':{name:sum(p.numel() for p in child.parameters()) for name,child in model.named_children()},
            'optimizer_groups':[{'name':g.get('name','unnamed'),'lr':g['lr'],
                'weight_decay':g.get('weight_decay',0.),
                'parameters':sum(p.numel() for p in g['params'])} for g in optimizer.param_groups]}
    if torch.device(device).type=='cuda':
        props=torch.cuda.get_device_properties(device)
        report.update(gpu=props.name,device_MiB=props.total_memory/2**20)
    logging.info('Runtime inventory %s',json.dumps(report))


def optimizer_budget(optimizer, ema, updates):
    """Report update-dependent time scales and the actual AdamW shrink factor.

    These are diagnostics, not automatic learning-rate or decay rescaling.
    Averaging more graphs into one gradient is not more optimizer updates.
    """
    import json
    groups=[]
    for group in optimizer.param_groups:
        beta1,beta2=group.get('betas',(0.,0.))
        steps=[float(optimizer.state[p].get('step',0)) for p in group['params'] if p in optimizer.state]
        rate=group['lr']*group.get('weight_decay',0.)
        groups.append({'name':group.get('name','unnamed'), 'lr':group['lr'],
            'parameter_update_range':[min(steps,default=0.),max(steps,default=0.)],
            'moment_memory_epochs':[1./max(1.e-30,(1-beta)*updates) for beta in (beta1,beta2)],
            'AdamW_decay_only_retention_per_epoch':(1.-rate)**updates if isinstance(optimizer,torch.optim.AdamW) else None})
    report={'updates_per_epoch':updates,'groups':groups}
    if ema is not None:
        report['EMA_asymptotic_memory_epochs']=1./max(1.e-30,(1-float(ema.decay))*updates)
    logging.info('Optimizer time scales %s',json.dumps(report,allow_nan=False))


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


def audit_training(model, wrapper, loss, ema, train_loader, valid_loader, device, epoch, validation_metrics=None):
    """Optional whole-split gap metrics and bounded physical response probes."""
    import json
    from contextlib import nullcontext
    from mace_scf.electrostatics.potential import evaluate_electronic
    from mace_scf.electrostatics.coupled import SCFConvergenceError, SCFNumericalError
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
        if validation_metrics is not None:
            # The same EMA model, 50-step observer and complete splits. Small
            # panels missed the hydrated-slab tail and understated the gap.
            from .train import evaluate
            import time
            start=time.perf_counter()
            loader=torch_geometric.dataloader.DataLoader(train_loader.dataset,
                batch_size=valid_loader.batch_size,shuffle=False,drop_last=False)
            _, train_metrics=evaluate(model,wrapper,loss,ema,loader,device,split='training')
            keys=('rmse_e_per_atom','rmse_f','rmse_mu_per_atom','rmse_fermi_level','rmse_rho',
                  'rmse_esp','rmse_esp_vac','rmse_wf_abs','rmse_wf_rel')
            report={'epoch':epoch,'training_graphs':len(loader.dataset),'validation_graphs':len(valid_loader.dataset),
                'train':{k:train_metrics[k] for k in keys if k in train_metrics},
                'validation':{k:validation_metrics[k] for k in keys if k in validation_metrics},
                'weighted_training_objectives':train_metrics.get('weighted_loss_terms',{}),
                'weighted_validation_objectives':validation_metrics.get('weighted_loss_terms',{}),
                'seconds':time.perf_counter()-start,
                'scope':'Whole splits at the same EMA parameters and deployment step count; read-only, native units.'}
            logging.info('Full-split generalization %s',json.dumps(report,allow_nan=False))
        model.eval()
        model.requires_grad_(False)
        with ema.average_parameters() if ema is not None else nullcontext():
            for label,loader in (("train",train_loader),("validation",valid_loader)):
                errors=[]
                features=[]
                targets=[]
                low=[]
                low_modes=[]
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
                    output={key:value.detach() if torch.is_tensor(value) else value
                            for key,value in output.items()}
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
                                low_modes.append([int(shell.sum()),int(use.sum())])
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
                    mode_counts=torch.tensor(low_modes).sum(0)
                    rows[label]['low_k_mode_fraction']=float(mode_counts[0]/mode_counts[1])
                    rows[label]['low_k_error_concentration']=float((low[0]/low[1].clamp_min(1.e-30))/(mode_counts[0]/mode_counts[1]))
            if all(k in feature_rows for k in ('train','validation')):
                rows['model_field_readout_probe']=_ridge_report(feature_rows['train'],residual_rows['train'],feature_rows['validation'],residual_rows['validation'])
            rows['epoch']=epoch
            if all('WF_relative_RMSE_eV' in rows[k] for k in ('train','validation')):
                train,valid=rows['train'],rows['validation']
                rows['WF_generalization']={
                    'panel_relative_gap_eV':valid['WF_relative_RMSE_eV']-train['WF_relative_RMSE_eV'],
                    'validation_covariance_amplification_eV2':-2*valid['EF_vac_covariance'],
                    'bias_only_correction_is_insufficient':valid['WF_relative_RMSE_eV']>abs(valid['EF_vac_WF_bias'][2])}
            response=model.field_dependent_charges_map
            rows['response_weight_norms']={name:float(p.detach().norm()) for name,p in response.named_parameters()
                                          if name.endswith('weight') and name.split('.')[0] in
                                          ('common_level','scalar_out','vector_out','state_scalar','state_vector','neighbor_scalar','neighbor_vector')}
            groups=getattr(getattr(wrapper,'optimizer',None),'param_groups',[])
            rows['regularization']={'field_weight_decay':next((g['weight_decay'] for g in groups
                if g.get('name')=='field_dependent_charges_map'),None)}
            rows['scope']='Deterministic panels, not whole-dataset metrics. Validation is development data, never used for fitting or selecting the probe.'
            logging.info("Electronic diagnostic %s",json.dumps(rows,allow_nan=False))
    finally:
        for p,flag in zip(model.parameters(),flags):
            p.requires_grad_(flag)
        model.train(was_training)
        torch.random.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)



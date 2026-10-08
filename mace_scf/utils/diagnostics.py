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
        self.density_sums = np.zeros(3)

    @torch.no_grad()
    def update(self, batch, output):
        from mace_scf.electrostatics.loss import vacuum_reference_weights
        ef = output.get('fermi_level')
        vac = output.get('vacuum_potential')
        vacuum_target, weight = vacuum_reference_weights(batch, vac) if vac is not None else (None,None)
        if 'fourier_farfield_density_dft' in output:
            from mace_scf.electrostatics.loss import spectral_errors
            weights = batch.weight*batch.fourier_density_weight
            self.density_sums += [float((spectral_errors(batch,output,key)*weights).sum())
                                 for key in ('fourier_density','fourier_farfield_density')]+[float(weights.sum())]
        for i in range(batch.num_graphs):
            start, stop = int(batch.ptr[i]), int(batch.ptr[i+1])
            attrs = batch.node_attrs[start:stop]
            composition = tuple(int(x) for x in attrs.sum(0))
            row = {'index':len(self.rows), 'atoms':stop-start,
                   'composition':composition, 'charge':float(batch.total_charge[i])}
            if vac is not None and ef is not None and weight[i]>0 and batch.fermi_level_weight[i]>0:
                row.update(weight=float(weight[i]), ef=float(ef[i]-batch.fermi_level[i]),
                           vac=float(vac[i]-vacuum_target[i]),
                           wf=float(output['workfunction'][i]-(vacuum_target[i]-batch.fermi_level[i])))
                if 'vacuum_potential_dft' in output and batch.fourier_potential_weight[i]*batch.fourier_proto_potential_weight[i]>0:
                    row['vacuum_reference_gap'] = float(output['vacuum_potential_dft'][i]-vacuum_target[i])
            if output.get('forces') is not None and batch.forces_weight[i]*batch.weight[i]>0:
                square=(output['forces'][start:stop]-batch.forces[start:stop]).square()
                row['force_rmse']=float(square.mean().sqrt())
                for z in range(attrs.shape[1]):
                    values=square[attrs[:,z]>0]
                    previous=self.species.setdefault(str(z),[0.,0,0.])
                    previous[0]+=float(values.sum());previous[1]+=values.numel()
                    previous[2]+=float(batch.forces[start:stop][attrs[:,z]>0].square().sum())
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
                    for z,v in self.species.items() if v[1]},
                'force_target_rms_by_element_column_eV_A':{z:(v[2]/v[1])**.5
                    for z,v in self.species.items() if v[1]},
                'force_rmse_over_target_rms_by_element_column':{z:(v[0]/v[2])**.5 if v[2]>0 else None
                    for z,v in self.species.items() if v[1]}}
        if self.density_sums[2]>0:
            report['density_observers'] = {
                'density_RMSE_e_A3':float(np.sqrt(self.density_sums[0]/self.density_sums[2])),
                'farfield_RMSE_e_A3':float(np.sqrt(self.density_sums[1]/self.density_sums[2])),
                'scope':'Different Gaussian resolutions of the same error; smaller farfield RMSE is not an accuracy gain.'}
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
                      EF_vac_WF_relative_rmse_eV=np.sqrt(weights@center**2).tolist(),
                      EF_vac_WF_bias_eV=mean.tolist(),EF_vac_covariance_eV2=float(weights@(center[:,0]*center[:,1])),
                      WF_absolute_error_quantiles_eV=dict(zip(('median','p90','p99','max'),np.quantile(abs(error[:,2]),[.5,.9,.99,1.]).tolist())))
        report['WF_identity_max_error_eV']=float(abs(error[:,2]-(error[:,1]-error[:,0])).max())
        report['WF_variance_components_eV2']={
            'EF':float(weights@center[:,0]**2), 'vacuum':float(weights@center[:,1]**2),
            'covariance_contribution':float(-2*(weights@(center[:,0]*center[:,1])))}
        relative_power=weights*center[:,2]**2
        relative_order=np.argsort(-relative_power)
        report['WF_centered_squared_error_share_largest_10_percent']=float(
            relative_power[relative_order[:max(1,int(np.ceil(.1*len(rows))))]].sum()/max(relative_power.sum(),1.e-30))
        # A centered population RMSE is not the error after one DFT anchor.
        # This expectation uses independent anchors from the same weighted
        # population; an actual MD trajectory can have correlated errors.
        report['independent_single_anchor_EF_vac_WF_RMSE_eV']=np.sqrt(2*(weights@center**2)).tolist()
        report['relative_error_scope']='One whole-split offset for RMSE_rel; no fitted offset is applied to predictions. Single-anchor expectation assumes independent configurations.'
        gaps=[r['vacuum_reference_gap'] for r in rows if 'vacuum_reference_gap' in r]
        if gaps:
            report['spectral_minus_measured_vacuum_RMSE_eV']=float(np.sqrt(np.mean(np.square(gaps))))
            report['spectral_minus_measured_vacuum_bias_eV']=float(np.mean(gaps))
            report['spectral_minus_measured_vacuum_relative_RMSE_eV']=float(np.std(gaps))
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


def objective_gradients(model, wrapper, loss, loader, device, epoch=0):
    """Read-only parameter-gradient alignment on a deterministic training pair.

    Report actual weighted gradients before clipping. No parameter, optimizer,
    EMA, random state or .grad buffer is updated. This small panel can expose a
    conflict; it cannot establish its prevalence across the training split.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        return {'skipped':'no trainable parameters'}
    index = ((epoch//50)*max(1,len(loader.dataset)//4)) % len(loader.dataset)
    indices = [index]
    if loss.relative_scalar_losses and len(loader.dataset)>1:
        indices.append((index+max(1,len(loader.dataset)//2)) % len(loader.dataset))
    batch = torch_geometric.Batch.from_data_list([loader.dataset[i] for i in indices]).to(device)
    with torch.enable_grad():
        output = wrapper(model,batch.to_dict(),training=True)
        gradients, report = {}, {}

        def record(name, value):
            raw = (torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                   if value.requires_grad else (None,)*len(parameters))
            # Keep only one objective's gradients on the accelerator. Joining
            # on CPU avoids a second model-sized GPU allocation for fine-tuning.
            gradient = torch.cat([torch.zeros(p.numel(),dtype=p.dtype,device='cpu') if g is None else
                                  g.detach().cpu().flatten() for p,g in zip(parameters,raw)])
            gradients[name] = gradient
            report[name] = {'weighted_loss':float(value.detach()),
                            'gradient_norm':float(gradient.norm())}

        for name,function in loss.loss_fns.items():
            if loss.loss_weights[name]:
                record(name,loss.loss_weights[name]*function(batch,output))
                if hasattr(function,'relative'):
                    weights = function.weights(batch)
                    report[name].update(relative=function.relative, observed_labels=int((weights>0).sum()),
                                        pair_weight=float(weights.sum()-weights.square().sum()/weights.sum().clamp_min(1.e-30)))
        if 'reference_response' in output:
            record('reference_conditioning',loss.reference_loss(batch,output)+output['energy'].sum()*0.)
        if 'fourier_density' in gradients and 'fourier_farfield_density_dft' in output:
            from mace_scf.electrostatics.loss import WeightedFourierDensity
            reference = getattr(loss.loss_fns['fourier_density'],'reference','density')
            other = 'density' if reference == 'farfield' else 'farfield'
            value = loss.loss_weights['fourier_density']*WeightedFourierDensity(other)(batch,output)
            record('density_observer_'+other,value)
        alignment = {}
        names = list(gradients)
        for i,left in enumerate(names):
            for right in names[i+1:]:
                a,b = gradients[left],gradients[right]
                denominator = a.norm()*b.norm()
                alignment[left+' / '+right] = float(a@b/denominator) if denominator>0 else None
    return {'training_indices':indices,'atoms':len(batch.positions),'terms':report,'cosines':alignment,
            'scope':'Training pair for relative losses, otherwise one graph; before clipping, diagnostic only.'}


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


def _ridge_report(train_x, train_y, valid_x, valid_y, relative=False):
    """Select regularization on a training-only inner split; report development."""
    if len(train_x)<12 or len(valid_x)<2:
        return {"skipped":"need >=12 observed training and >=2 validation graphs"}
    x,y,v,z=(a.double() for a in (train_x,train_y,valid_x,valid_y))
    inner=torch.arange(len(x))%4==0
    means=x[~inner].mean(0)
    scale=x[~inner].std(0).clamp_min(1.e-6)
    if not relative:
        means[0]=0.
    scale[0]=1.
    a,b=(x-means)/scale,(v-means)/scale
    def mse(error):
        if relative:
            error=error-error.mean()
        return float(error.square().mean())
    target=y[~inner]-y[~inner].mean() if relative else y[~inner]
    baseline=mse(y[inner])
    candidates=[(baseline,None)]
    for ridge in (1.e-3,.1,10.,1000.):
        gram=a[~inner].T@a[~inner]+ridge*torch.eye(a.shape[1],dtype=a.dtype)
        coefficient=torch.linalg.solve(gram,a[~inner].T@target)
        candidates.append((mse(y[inner]-a[inner]@coefficient),ridge))
    score,ridge=min(candidates,key=lambda item:item[0])
    if ridge is None:
        return {"selected":"no correction", "relative":relative, "inner_RMSE":baseline**.5,
                "development_RMSE":mse(z)**.5}
    if relative:
        a,b=(x-x.mean(0))/scale,(v-x.mean(0))/scale
    target=y-y.mean() if relative else y
    coefficient=torch.linalg.solve(a.T@a+ridge*torch.eye(a.shape[1],dtype=a.dtype),a.T@target)
    return {"selected_ridge":ridge,"relative":relative,"training_before":mse(y)**.5,
            "training_after":mse(y-a@coefficient)**.5,
            "development_before":mse(z)**.5,
            "development_after":mse(z-b@coefficient)**.5,
            "development_absolute_after":float((z-b@coefficient).square().mean().sqrt()),
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
            keys=('rmse_e_per_atom','rmse_f','rmse_mu_per_atom','rmse_fermi_level','rmse_fermi_level_rel','rmse_rho',
                  'rmse_esp','rmse_esp_vac','rmse_esp_vac_rel','rmse_wf_abs','rmse_wf_rel')
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
                drifts=[]
                charges=[]
                force_squares=[]
                species_errors={}
                charge_responses=[]
                reference_gaps=[]
                conditional_rows=[]
                conditional_stiffness=[]
                conditional_failures=[]
                for index,batch in enumerate(_panel(loader,32 if label=="train" else 16)):
                    batch=batch.to(device)
                    data=batch.to_dict()
                    try:
                        output=evaluate_electronic(model,data,steps=50,compute_force=True,
                            **({'reference_conditioning':True} if getattr(model.field_dependent_charges_map,'coupled',False) else {}))
                    except RuntimeError as exc:
                        if 'Reference-conditioned chemical-potential response is singular' not in str(exc):
                            raise
                        conditional_failures.append({'panel_index':index,'reason':str(exc)})
                        output=evaluate_electronic(model,data,steps=50,compute_force=True)
                    output={key:value.detach() if torch.is_tensor(value) else value
                            for key,value in output.items()}
                    reference = output.get('reference_response')
                    if reference is not None:
                        conditional_stiffness.append(float(reference['reference_charge_stiffness'][0]))
                        row = {'charge_error_e':float(reference['reference_charge_error'][0])}
                        if reference['reference_masks'].get('fermi_level',torch.zeros(1))[0]:
                            row['EF_given_DFT_field_eV'] = float(reference['fermi_level'][0]-batch.fermi_level[0])
                        if batch.vacuum_potential_weight[0]>0:
                            row['vacuum_given_DFT_EF_eV'] = float(reference['vacuum_potential'][0]-batch.vacuum_potential[0])
                        conditional_rows.append(row)
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
                        if (float(batch.vacuum_potential_weight[0])>0 and float(batch.fermi_level_weight[0])>0
                                and int(batch.pbc.sum())==2):
                            vacuum_target = batch.vacuum_potential[0]
                            error=torch.stack((output['fermi_level'][0]-batch.fermi_level[0],
                                output['vacuum_potential'][0]-vacuum_target,
                                output['workfunction'][0]-(vacuum_target-batch.fermi_level[0])))
                            if 'vacuum_potential_dft' in output and float(batch.fourier_proto_potential_weight[0]*batch.fourier_potential_weight[0])>0:
                                reference_gaps.append(float(output['vacuum_potential_dft'][0]-vacuum_target))
                            errors.append(error.cpu())
                            features.append(_probe_features(model,data,output).cpu())
                            targets.append((-error[0]).cpu())
                        if 'fourier_potential_dft' in output and float(batch.fourier_potential_weight[0])>0:
                            diff=(output['fourier_potential']-output['fourier_potential_dft'])/output['k_vectors_grid_shape'].prod()
                            k2=output['k_vectors'].square().sum(-1)
                            use=output['k_vectors_mask']&(k2>0)
                            use=use & output.get('fourier_potential_dft_mask',use)
                            power=torch.where(use,diff.square().sum(-1),0.)
                            low.append([float((power*torch.exp(-float(model.coulomb_energy.density_smearing_width)**2*k2)).sum()),
                                        float(power.sum())])
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
                    rows[label]['spectral_minus_measured_vacuum_RMSE_eV']=float(gap.square().mean().sqrt())
                if low:
                    low=torch.tensor(low).sum(0)
                    rows[label]['Gaussian_lowpass_potential_error_power_fraction']=float(low[0]/low[1].clamp_min(1.e-30))
                if conditional_rows:
                    names = set().union(*(row.keys() for row in conditional_rows))
                    rows[label]['reference_conditioning'] = {
                        name+'_RMSE':float(np.sqrt(np.mean([row[name]**2 for row in conditional_rows if name in row])))
                        for name in names}
                    rows[label]['reference_conditioning']['scope'] = 'Oracle conditional errors, not deployment metrics; charge error measures the departure from the prescribed fixed-Q ensemble.'
                    rows[label]['reference_conditioning']['dmu_dQ_eV_per_e_range'] = [min(conditional_stiffness),max(conditional_stiffness)]
                if conditional_failures:
                    rows[label]['reference_conditioning_failures'] = conditional_failures
            if all(k in feature_rows for k in ('train','validation')):
                rows['model_field_EF_readout_probe']=_ridge_report(feature_rows['train'],residual_rows['train'],feature_rows['validation'],residual_rows['validation'],
                    relative=getattr(loss.loss_fns.get('fermi_level'),'relative',False))
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
            for parameter,flag in zip(model.parameters(),flags):
                parameter.requires_grad_(flag)
            rows['objective_gradients'] = objective_gradients(model,wrapper,loss,train_loader,device,epoch)
            rows['objective_gradients']['parameter_state'] = 'EMA' if ema is not None else 'current model'
            rows['scope']='Deterministic panels, not whole-dataset metrics. Validation is development data, never used for fitting or selecting the probe.'
            logging.info("Electronic diagnostic %s",json.dumps(rows,allow_nan=False))
    finally:
        for p,flag in zip(model.parameters(),flags):
            p.requires_grad_(flag)
        model.train(was_training)
        torch.random.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)



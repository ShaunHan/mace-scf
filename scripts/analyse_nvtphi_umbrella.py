#!/usr/bin/env python3
"""Conditional 1D WHAM profiles, overlap and block uncertainty for ONE replica.

python scripts/analyse_nvtphi_umbrella.py umbrella_volt/umbrella_8/replica_0

Forward/reverse profiles are kept separate. Different pairs/surface compositions
are never pooled. A PMF along O-O distance is not automatically an OER barrier.
Requires NumPy and SciPy; Matplotlib is optional for the preview figure.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

KB_EV_K = 8.617333262145e-5


def inefficiency(values):
    x = np.asarray(values, dtype=float)
    x = x-x.mean()
    if np.dot(x,x) < 1e-24:
        return 1.
    size = 1 << (2*len(x)-1).bit_length()
    f = np.fft.rfft(x,n=size)
    ac = np.fft.irfft(f*f.conj(),n=size)[:len(x)]
    ac /= ac[0]
    stop = np.flatnonzero(ac[1:] <= 0)
    end = int(stop[0]+1) if len(stop) else max(2,len(x)//2)
    return float(np.clip(1+2*ac[1:end].sum(),1,len(x)))


def wham(histograms, reduced_bias):
    """Maximize the binned WHAM likelihood; free offsets f_k have f_0=0."""
    h = np.asarray(histograms,dtype=float)
    totals, counts = h.sum(axis=1), h.sum(axis=0)
    if np.any(totals<=0):
        raise ValueError('Every window needs production samples')
    occupied = counts>0
    u = reduced_bias[:,occupied]
    logn = np.log(totals)
    normalized_counts, normalized_totals = counts[occupied]/counts.sum(), totals/totals.sum()

    def objective(x):
        f = np.r_[0.,x]
        terms = logn[:,None]+f[:,None]-u
        denominator = logsumexp(terms,axis=0)
        value = np.dot(normalized_counts,denominator)-np.dot(normalized_totals,f)
        gradient = np.exp(terms-denominator).dot(normalized_counts)-normalized_totals
        return float(value), gradient[1:]

    solution = minimize(objective,np.zeros(len(h)-1),jac=True,method='L-BFGS-B',
                        options={'maxiter':20000,'ftol':1e-14,'gtol':1e-9,'maxls':50})
    if np.max(np.abs(objective(solution.x)[1]))>1e-6:
        raise RuntimeError(f'WHAM did not converge: {solution.message}')
    f = np.r_[0.,solution.x]
    logp = np.full(len(counts),-np.inf)
    logp[occupied] = np.log(counts[occupied])-logsumexp(logn[:,None]+f[:,None]-u,axis=0)
    logp -= logsumexp(logp)
    return logp


def resample_blocks(values, block, rng):
    n = len(values)
    starts = rng.integers(n,size=int(np.ceil(n/block)))
    ids = (starts[:,None]+np.arange(block)[None,:])%n
    return values[ids.ravel()[:n]]


def analyse(folder, bin_width=.02, bootstraps=50):
    folder = Path(folder)
    cfg = json.loads((folder/'run.json').read_text())
    if cfg.get('method') != 'fixed_window_OO':
        raise ValueError('Use output from the accompanying fixed-window driver, in a fresh folder')
    with (folder/'md.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['phase']=='sample' and int(r['phase_step'])>0]
    if not rows:
        raise ValueError('No production samples yet')
    kbt = KB_EV_K*cfg['settings']['TEMPERATURE_K']
    target = cfg['settings']['TARGET_U_SHE']
    report = dict(source_index=cfg['initial']['info']['source_index'],
                  oxygen_pair=cfg['initial']['info']['oxygen_pair'],
                  meaning='Conditional O-O distance profile for this fixed composition/pair and the implemented controller.',
                  limitations=['No proton reservoir or pH control.',
                               'Thermodynamic constant-potential interpretation requires validated energy/voltage consistency and timestep sampling.',
                               'Projected barriers need equilibrated orthogonal coordinates and defined reactant/product basins.',
                               'Bootstrap errors omit model error, hidden unsampled states and systematic controller error.'],
                  directions={})
    curves = {}
    for direction in dict.fromkeys(r['direction'] for r in rows):
        subset = [r for r in rows if r['direction']==direction]
        window_ids = sorted({int(r['window']) for r in subset})
        planned = sum(s['phase']=='sample' and s['direction']==direction for s in cfg['stages'])
        groups = [[r for r in subset if int(r['window'])==w] for w in window_ids]
        series, scales, stats, centers, stiffnesses = [], [], [], [], []
        for w, rr in zip(window_ids,groups):
            values = lambda key:np.array([float(r[key]) for r in rr])
            distance = values('distance_A')
            if len(distance)<10:
                raise ValueError(f'{direction} window {w}: fewer than 10 stored production samples')
            center, stiffness = values('center_A'), values('k_eV_A2')
            if np.ptp(center)>1e-10 or np.ptp(stiffness)>1e-10:
                raise ValueError('A production window contains a changing bias')
            centers.append(center[0]); stiffnesses.append(stiffness[0])
            tracked = ('distance_A','total_charge_e','energy_model_eV','O1_H_neighbors',
                       'O2_H_neighbors','O1_Ir_neighbors','O2_Ir_neighbors')
            g = max(inefficiency(values(key)) for key in tracked)
            scales.append(g); series.append(distance)
            half = len(rr)//2
            stats.append(dict(window=w, center_A=float(center[0]), mean_distance_A=float(distance.mean()),
                              distance_shift_from_center_A=float(distance.mean()-center[0]),
                              production_ps=float(values('phase_step').max()*cfg['settings']['TIMESTEP_FS']/1000),
                              effective_samples=float(len(rr)/g), correlation_stride=float(g),
                              mean_U_SHE=float(values('voltage_V_SHE').mean()),
                              mean_temperature_K=float(values('temperature_K').mean()),
                              distance_half_drift_A=float(distance[half:].mean()-distance[:half].mean()),
                              charge_half_drift_e=float(values('total_charge_e')[half:].mean()-values('total_charge_e')[:half].mean()),
                              coordination={key:float(values(key).mean()) for key in tracked[3:]}))
        all_distances = np.concatenate(series)
        edges = np.arange(np.floor(all_distances.min()/bin_width)*bin_width,
                          np.ceil(all_distances.max()/bin_width)*bin_width+1.01*bin_width,bin_width)
        x = (edges[:-1]+edges[1:])/2
        raw = np.array([np.histogram(a,edges)[0] for a in series])
        density = raw/raw.sum(axis=1)[:,None]
        overlap = np.minimum(density[:,None,:],density[None,:,:]).sum(axis=2)
        connected = {0}
        for _ in range(len(raw)):
            connected.update(np.flatnonzero(np.any(overlap[list(connected)]>0,axis=0)).tolist())
        flags = []
        if len(groups)!=planned or any(s['production_ps']+1e-10<cfg['settings']['SAMPLE_STEPS']*cfg['settings']['TIMESTEP_FS']/1000 for s in stats):
            flags.append('Incomplete planned production; profile is provisional.')
        if min(s['effective_samples'] for s in stats)<50:
            flags.append('Some windows have fewer than 50 estimated independent samples.')
        if max(abs(s['mean_U_SHE']-target) for s in stats)>.15:
            flags.append('Some window mean voltages differ from target by >0.15 V; inspect stationarity and uncertainty.')
        if max(abs(s['mean_temperature_K']-cfg['settings']['TEMPERATURE_K']) for s in stats)>20:
            flags.append('Some window mean temperatures differ from target by >20 K.')
        short = min(stats,key=lambda s:s['center_A'])
        if short['mean_distance_A'] > 1.35:
            flags.append('The shortest-center window has mean O-O distance >1.35 A; it has not established a detached O2 basin.')
        np.savetxt(folder/f'overlap_{direction}.csv',overlap,delimiter=',')
        report['directions'][direction] = dict(windows=stats,flags=flags)
        if len(connected)!=len(raw):
            flags.append('Disconnected sampled overlap graph: no combined PMF was calculated.')
            continue
        if len(raw)<2:
            flags.append('At least two overlapping windows are required for this analysis.')
            continue
        reduced_bias = .5*np.asarray(stiffnesses)[:,None]*(x-np.asarray(centers)[:,None])**2/kbt
        h = raw/np.asarray(scales)[:,None]
        logp = wham(h,reduced_bias)
        # A fixed populated high-distance reference bin; this does not identify a reactant basin.
        reference = int(np.argmax(h.sum(axis=0)*np.exp(-((x-max(centers))/.15)**2)))
        profile = -kbt*(logp-logp[reference])
        rng = np.random.default_rng(917)
        draws = []
        for _ in range(bootstraps):
            sampled = [resample_blocks(a,min(len(a),max(1,int(np.ceil(2*g)))),rng) for a,g in zip(series,scales)]
            hb = np.array([np.histogram(a,edges)[0] for a in sampled])/np.asarray(scales)[:,None]
            try:
                lp = wham(hb,reduced_bias)
            except RuntimeError:
                continue
            if np.isfinite(lp[reference]):
                draws.append(np.where(np.isfinite(lp),-kbt*(lp-lp[reference]),np.nan))
        error = np.full(len(x),np.nan)
        if len(draws)>1:
            draws = np.asarray(draws)
            valid = np.isfinite(draws).sum(axis=0)>=max(2,int(.8*len(draws)))
            error[valid] = np.nanstd(draws[:,valid],axis=0,ddof=1)
        output = np.column_stack((x,np.where(np.isfinite(profile),profile,np.nan),error,raw.sum(axis=0),h.sum(axis=0)))
        np.savetxt(folder/f'pmf_{direction}.csv',output,delimiter=',',comments='',
                   header='distance_A,conditional_PMF_eV,block_bootstrap_sd_eV,samples,effective_samples')
        report['directions'][direction].update(reference_distance_A=float(x[reference]),bootstrap_draws=len(draws))
        if bootstraps and len(draws)<max(2,.8*bootstraps):
            flags.append('Too few usable bootstrap fits for a reliable error estimate.')
        curves[direction]=(x,profile,error)
    (folder/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        if curves:
            fig,ax=plt.subplots(figsize=(6.4,4.2),layout='constrained')
            for name,(x,f,e) in curves.items():
                ax.plot(x,f,label=name); ax.fill_between(x,f-e,f+e,alpha=.18)
            ax.set(xlabel='O-O distance (A)',ylabel='Conditional PMF (eV)',
                   title='Fixed composition; inspect overlap and stationarity')
            ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False)
            fig.savefig(folder/'pmf.png',dpi=300); plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--bin-width',type=float,default=.02)
    p.add_argument('--bootstrap',type=int,default=50)
    a=p.parse_args()
    if a.bin_width<=0 or a.bootstrap<0:
        p.error('Positive bin width and nonnegative bootstrap count required')
    analyse(a.folder,a.bin_width,a.bootstrap)

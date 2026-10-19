#!/usr/bin/env python3
"""Numerical checks accompany algebraic proofs; they do not replace proofs."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import networkx as nx
from scipy.special import expit
from src.sai import metrics, distances, from_matrix, add_edge_distances
from scripts.run_experiments import write_csv, CFG, OUT


def direct_T(w):
    p=w/w.sum(); return float(np.dot(p,np.log(len(w)*p)))


def main():
    rng=np.random.default_rng(20261014)
    errors={k:0. for k in ['grouped_kl','support_decomposition','finite_update','derivative','unit_rescaling']}
    for _ in range(200):
        cost=np.arange(1,9,dtype=float); count=rng.integers(1,30,8); a=rng.uniform(.1,5); h=rng.uniform(1,8)
        z=metrics(cost,count,a,h); w=expit(-a*(cost-h)); q=count/count.sum(); P=q*w/np.dot(q,w)
        errors['grouped_kl']=max(errors['grouped_kl'],abs(z['theil']-np.dot(P,np.log(P/q))))
        zz=metrics(np.r_[cost,np.inf],np.r_[count,31],a,h)
        errors['support_decomposition']=max(errors['support_decomposition'],abs(zz['sai']-zz['support_term']-zz['within_term']))
        z2=metrics(cost*1000,count,a/1000,h*1000)
        errors['unit_rescaling']=max(errors['unit_rescaling'],abs(z['sai']-z2['sai']))
        w=rng.uniform(.05,.9,30); v=w+rng.uniform(0,.05,30)
        W=w.sum(); V=v.sum(); S=np.dot(w,np.log(w)); U=np.dot(v,np.log(v))
        errors['finite_update']=max(errors['finite_update'],abs(direct_T(v)-direct_T(w)-(U/V-S/W-np.log(V/W))))
        j=int(rng.integers(30)); eps=1e-6; plus=w.copy(); minus=w.copy(); plus[j]+=eps; minus[j]-=eps
        fd=(direct_T(plus)-direct_T(minus))/(2*eps); exact=(np.log(w[j])-S/W)/W
        errors['derivative']=max(errors['derivative'],abs(fd-exact))
    assert max(errors.values())<1e-8,errors
    rows=[]
    for idx,g in enumerate(nx.graph_atlas_g()):
        if not 3<=len(g)<=6 or not nx.is_connected(g): continue
        d=distances(g); c,n=from_matrix(d)
        for profile,(a,h) in CFG['profiles'].items():
            before=metrics(c,n,a,h)
            for u,v in nx.non_edges(g):
                after=metrics(*from_matrix(add_edge_distances(d,u,v)),a,h)
                rows.append(dict(atlas_index=idx,n=len(g),graph6=nx.to_graph6_bytes(g,header=False).decode().strip(),u=u,v=v,profile=profile,a=a,h0=h,delta_sai=after['sai']-before['sai'],delta_mean_distance=after['mean_distance_reachable']-before['mean_distance_reachable'],delta_mean=after['mean']-before['mean'],sai_before=before['sai'],sai_after=after['sai']))
    write_csv('exhaustive_small_edge_audit.csv',rows)
    adverse=[r for r in rows if r['delta_sai']>1e-10 and r['delta_mean_distance']<0]
    witness=max(adverse,key=lambda r:r['delta_sai']) if adverse else None
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows: groups[(r['atlas_index'],r['profile'])].append(r)
    disagreements=[]
    for key,rr in groups.items():
        min_i=min(r['delta_sai'] for r in rr); min_l=min(r['delta_mean_distance'] for r in rr)
        opt_i={(r['u'],r['v']) for r in rr if r['delta_sai']<=min_i+1e-12}
        opt_l={(r['u'],r['v']) for r in rr if r['delta_mean_distance']<=min_l+1e-12}
        if not opt_i.intersection(opt_l):disagreements.append({'atlas_index':key[0],'profile':key[1],'graph6':rr[0]['graph6'],'sai_optimal_edges':sorted(opt_i),'distance_optimal_edges':sorted(opt_l)})
    result={'identity_trials':200,'max_absolute_errors':errors,'enumeration':'all connected unlabeled simple graphs with 3..6 vertices; all nonedges; four submitted profiles','edge_profile_cases':len(rows),'edge_cases_increasing_sai':len(adverse),'max_increase_witness':witness,'graph_profile_decisions':len(groups),'disjoint_optimum_decisions':len(disagreements),'disjoint_optimum_witnesses':disagreements,'interpretation':'finite exhaustive domain only; neither prevalence on general networks nor historical novelty'}
    (OUT/'artifacts/property_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

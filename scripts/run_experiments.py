#!/usr/bin/env python3
"""Scope-preserving reconstruction, full configuration; no original numbers hardcoded."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import multiprocessing

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import networkx as nx
import scipy
from src.sai import metrics, distances, histogram, from_matrix, gini, calibrate, add_edge_distances

OUT=ROOT/'results/issue-1'
CFG=json.loads((ROOT/'configs/full.json').read_text())


def write_csv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'artifacts'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def graph_meta(g):
    return {'n':len(g),'edges':g.number_of_edges(),'degree_gini':gini([v for _,v in g.degree()])}


def evaluate(g,profiles,meta,d=None):
    d=distances(g) if d is None else d
    costs,counts=from_matrix(d)
    return [dict(meta,**graph_meta(g),profile=name,a=a,h0=h,**metrics(costs,counts,a,h)) for name,(a,h) in profiles.items()]


def dumbbell(left,right,internal):
    # Two complete clusters; internal path vertices are explicitly counted.
    g=nx.disjoint_union(nx.complete_graph(left),nx.complete_graph(right))
    bridge=[left-1]+list(range(left+right,left+right+internal))+[left]
    g.add_edges_from(zip(bridge[:-1],bridge[1:]))
    return nx.convert_node_labels_to_integers(g,ordering='sorted')


def synthetic():
    rows=[]; n=CFG['n']; trials=CFG['trials']; base=CFG['seed']
    profiles=CFG['profiles']
    canonical={'complete':nx.complete_graph(n),'path':nx.path_graph(n),'ring':nx.cycle_graph(n),'star':nx.star_graph(n-1)}
    for name,g in canonical.items():
        rows+=evaluate(g,dict(profiles,zoo=[3,4]),{'family':name,'parameter':0,'trial':0,'seed':0})
    for family,grid in [('ER',CFG['er_p']),('BA',CFG['ba_m']),('WS',CFG['ws_beta'])]:
        for j,param in enumerate(grid):
            for t in range(trials):
                seed=base+{'ER':100000,'BA':200000,'WS':300000}[family]+100*j+t
                if family=='ER': g=nx.gnp_random_graph(n,param,seed=seed)
                elif family=='BA': g=nx.barabasi_albert_graph(n,param,seed=seed)
                else: g=nx.watts_strogatz_graph(n,4,param,seed=seed)
                rows+=evaluate(g,dict(profiles,zoo=[3,4]),{'family':family,'parameter':param,'trial':t,'seed':seed})
        print('finished',family,flush=True)
    write_csv('canonical_random.csv',rows)
    rows=[]
    for t in range(trials):
        seed=base+400000+t; g=nx.barabasi_albert_graph(n,2,seed=seed); d=distances(g)
        for a in CFG['sensitivity_a']:
            for h in CFG['sensitivity_h0']:
                rows+=evaluate(g,{'sweep':[a,h]},{'family':'BA','trial':t,'seed':seed},d)
    write_csv('sensitivity.csv',rows); print('finished sensitivity',flush=True)
    rows=[]
    for j,p in enumerate(CFG['weighted_p']):
        for t in range(trials):
            seed=base+500000+100*j+t; g=nx.gnp_random_graph(n,p,seed=seed)
            rng=np.random.default_rng(seed)
            for u,v in g.edges(): g[u][v]['weight']=float(rng.uniform(1,10))
            for mode,pr,d in [('hops',{'moderate':[1,5]},distances(g)),('synthetic-cost',{'weighted':[.8,12]},distances(g,True))]:
                rows+=evaluate(g,pr,{'family':'ER','parameter':p,'trial':t,'seed':seed,'cost_mode':mode,'cost_unit':'synthetic cost units' if mode!='hops' else 'hop'},d)
    write_csv('weighted.csv',rows)
    edge_rows=[]; initial=[]; selected=[]
    for name,g,pr in [('two_cliques_25',dumbbell(25,25,0),{'A':[2,3],'B':[.5,6]}),('heterogeneous_15_30_path5',dumbbell(15,30,5),{'tradeoff':[10,6],'A':[2,3],'B':[.5,6]})]:
        d=distances(g); initial+=evaluate(g,pr,{'topology':name,'state':'initial'},d)
        nx.write_edgelist(g,OUT/'artifacts'/f'{name}.edgelist',data=False)
        edges=list(nx.non_edges(g)); edges.sort()
        for u,v in edges:
            dd=add_edge_distances(d,u,v); g.add_edge(u,v)
            edge_rows+=evaluate(g,pr,{'topology':name,'u':u,'v':v,'candidates':len(edges)},dd)
            g.remove_edge(u,v)
        primary='A' if name=='two_cliques_25' else 'tradeoff'
        subset=[r for r in edge_rows if r['topology']==name and r['profile']==primary]
        for objective,key,maximize in [('SAI','sai',False),('mean_distance','mean_distance_reachable',False),('Jain','jain',True),('satisfaction_Gini','satisfaction_gini',False),('minimum','minimum',True)]:
            best=sorted(subset,key=lambda r:((-r[key] if maximize else r[key]),r['u'],r['v']))[0]
            selected.append(dict(best,objective=objective))
        print('finished candidates',name,len(edges),flush=True)
    write_csv('edge_candidates.csv',edge_rows); write_csv('dumbbell_initial.csv',initial); write_csv('edge_selected.csv',selected)
    # Calibration is an explicit synthetic anchor demonstration, not a measured service.
    rows=[]; g=nx.barabasi_albert_graph(n,2,seed=base+600000); d=distances(g)
    for c1,c2 in [(2,5),(3,7),(4,8)]:
        for f1 in [.9,1.,1.1]:
            for f2 in [.9,1.,1.1]:
                a,h=calibrate(c1*f1,.9,c2*f2,.1)
                rows+=evaluate(g,{'two-anchor':[a,h]},{'c1_nominal':c1,'c2_nominal':c2,'c1_factor':f1,'c2_factor':f2,'s1':.9,'s2':.1,'anchor_type':'assumed synthetic hop acceptability'},d)
    write_csv('calibration.csv',rows)
    print('finished synthetic',flush=True)


def as_worker(task):
    family,t=task; seed=CFG['seed']+700000+t
    if family=='AS':
        with gzip.open(ROOT/'data/raw/oregon1_010331.txt.gz','rt') as f:
            g=nx.read_edgelist(f,comments='#',nodetype=int)
        g.remove_edges_from(nx.selfloop_edges(g))
        if (len(g),g.number_of_edges())!=(10670,22002): raise ValueError('SNAP size mismatch')
    elif family=='ER': g=nx.fast_gnp_random_graph(10670,44004/(10670*10669),seed=seed)
    else: g=nx.watts_strogatz_graph(10670,4,.1,seed=seed)
    start=time.perf_counter(); costs,counts=histogram(g,CFG['as_batch_size']); path_s=time.perf_counter()-start
    start=time.perf_counter(); z=metrics(costs,counts,1,4); aggregate_s=time.perf_counter()-start
    hfile=OUT/'artifacts'/f'as_histogram_{family}_{t}.json'
    hfile.write_text(json.dumps({'costs':[float(x) if np.isfinite(x) else 'unreachable' for x in costs],'counts':counts.tolist()}))
    return dict(family=family,trial=t,seed=seed,**graph_meta(g),ws_k=4 if family=='WS' else '',ws_beta=.1 if family=='WS' else '',path_seconds=path_s,aggregation_seconds=aggregate_s,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**z)


def as_study(workers):
    tasks=[('AS',0)]+[(f,t) for f in ['ER','WS'] for t in range(CFG['as_trials'])]
    rows=[]
    with multiprocessing.get_context('spawn').Pool(processes=workers,maxtasksperchild=1) as pool:
        for row in pool.imap(as_worker,tasks):
            rows.append(row); write_csv('as_results.csv',rows)
            print('AS study',row['family'],row['trial'],'I',row['sai'],'seconds',row['path_seconds'],flush=True)


def bench_one(n,backend,t):
    g=nx.barabasi_albert_graph(n,2,seed=CFG['seed']+800000+t)
    before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    start=time.perf_counter()
    if backend=='dense': costs,counts=from_matrix(distances(g))
    else: costs,counts=histogram(g,CFG['as_batch_size'])
    elapsed=time.perf_counter()-start
    start=time.perf_counter()
    for _ in range(100): z=metrics(costs,counts,1,5)
    aggregation=(time.perf_counter()-start)/100
    return dict(n=n,edges=g.number_of_edges(),backend=backend,trial=t,path_and_histogram_seconds=elapsed,aggregation_seconds=aggregation,total_seconds=elapsed+aggregation,baseline_rss_mib=before,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**z)


def benchmark():
    rows=[]
    # Sequential fresh interpreters: peak RSS is not inherited from earlier sizes.
    for n in CFG['benchmark_n']:
        for backend in ['dense','batched']:
            for t in range(CFG['benchmark_repeats']):
                p=subprocess.run([sys.executable,__file__,'--stage','bench-one','--n',str(n),'--backend',backend,'--trial',str(t)],capture_output=True,text=True,check=True)
                row=json.loads(p.stdout); rows.append(row); write_csv('runtime_memory.csv',rows)
            print('benchmark',n,backend,flush=True)


def metadata():
    import psutil
    data={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'host':platform.node(),'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'networkx':nx.__version__,'logical_cpus':os.cpu_count(),'cpu_affinity':psutil.Process().cpu_affinity(),'memory_bytes':psutil.virtual_memory().total,'config_sha256':hashlib.sha256((ROOT/'configs/full.json').read_bytes()).hexdigest(),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'src').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))},'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}
    (OUT/'artifacts/environment.json').write_text(json.dumps(data,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['synthetic','as','benchmark','bench-one'],required=True); p.add_argument('--workers',type=int,default=4); p.add_argument('--n',type=int); p.add_argument('--backend'); p.add_argument('--trial',type=int,default=0); args=p.parse_args()
    for part in ['artifacts','figures','logs']: (OUT/part).mkdir(parents=True,exist_ok=True)
    if args.stage=='bench-one': print(json.dumps(bench_one(args.n,args.backend,args.trial)))
    else:
        metadata()
        if args.stage=='synthetic': synthetic()
        elif args.stage=='as': as_study(args.workers)
        else: benchmark()

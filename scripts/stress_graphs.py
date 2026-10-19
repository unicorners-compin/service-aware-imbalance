#!/usr/bin/env python3
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import networkx as nx
from src.sai import distances,from_matrix,metrics,add_edge_distances
from scripts.run_experiments import CFG,OUT,evaluate,write_csv,dumbbell


def modules(sizes,seed,connect):
    rng=np.random.default_rng(seed);g=nx.Graph(); groups=[]; offset=0
    for n in sizes:
        h=nx.random_regular_graph(4,n,seed=int(rng.integers(2**31)))
        h=nx.relabel_nodes(h,{i:i+offset for i in h});g=nx.compose(g,h)
        groups.append(set(h));offset+=n
    if connect:
        for left,right in zip(groups[:-1],groups[1:]):
            u,v=min((u,v) for u,v in g.edges() if u in left and v in left)
            x,y=min((u,v) for u,v in g.edges() if u in right and v in right)
            g.remove_edge(u,v);g.remove_edge(x,y);g.add_edges_from([(u,x),(v,y)])
    assert set(dict(g.degree()).values())=={4}
    return nx.convert_node_labels_to_integers(g,ordering='sorted')


def graphs():
    fixed={}
    for step in [2,7,12,24]:fixed[f'circulant_1_{step}']=nx.circulant_graph(50,[1,step])
    fixed['circular_ladder']=nx.circular_ladder_graph(25)
    g=nx.cycle_graph(50);g.add_edges_from((i,i+25) for i in range(25));fixed['mobius_ladder']=g
    for core in [5,10,20,25]:fixed[f'lollipop_{core}']=nx.lollipop_graph(core,50-core)
    for core in [5,10,20,25]:fixed[f'barbell_{core}']=nx.barbell_graph(core,50-2*core)
    g=nx.disjoint_union_all([nx.complete_graph(10) for _ in range(5)]);g.add_edges_from((i*10+9,(i+1)*10) for i in range(4));fixed['clique_chain']=g
    g=g.copy();g.add_edge(49,0);fixed['clique_ring']=g
    g=nx.balanced_tree(2,5);g.remove_nodes_from(range(50,63));fixed['binary_tree']=g
    fixed['star']=nx.star_graph(49);fixed['wheel']=nx.wheel_graph(50)
    fixed['grid_5_10']=nx.convert_node_labels_to_integers(nx.grid_2d_graph(5,10))
    h=nx.convert_node_labels_to_integers(nx.grid_2d_graph(5,5));g=nx.disjoint_union(h,h);g.add_edge(24,25);fixed['grid_bridge']=g
    g=nx.complete_graph(10);g.add_edges_from((0,i) for i in range(10,50));fixed['core_single_leaf']=g
    for name,g in fixed.items():yield name,0,0,g
    for t in range(20):
        seed=CFG['seed']+900000+t
        yield 'random_regular4',t,seed,nx.random_regular_graph(4,50,seed=seed)
        yield 'random_regular3',t,seed,nx.random_regular_graph(3,50,seed=seed)
        yield 'modules2_regular4',t,seed,modules([25,25],seed,True)
        yield 'modules4_regular4',t,seed,modules([12,12,13,13],seed,True)
        yield 'disconnected2_regular4',t,seed,modules([25,25],seed,False)
        g=nx.complete_graph(10);rng=np.random.default_rng(seed)
        for leaf in range(10,50):g.add_edges_from((int(u),leaf) for u in rng.choice(10,size=2,replace=False))
        yield 'core_periphery',t,seed,g


def main():
    profiles=dict(CFG['profiles']);profiles.update({f'threshold_{h}':[1,h] for h in [1,2,3,4,5,6,8,12]})
    rows=[];edge_rows=[];decisions=[];manifest=[]
    folder=OUT/'artifacts/stress_graphs';folder.mkdir(exist_ok=True)
    for family,t,seed,g in graphs():
        assert len(g)==50 and not g.is_multigraph()
        if 'regular4' in family or family.startswith('circulant'):assert g.number_of_edges()==100 and set(dict(g.degree()).values())=={4}
        name=f'{family}_{t}';nx.write_edgelist(g,folder/(name+'.edgelist'),data=False)
        manifest.append({'id':name,'family':family,'trial':t,'seed':seed,'n':len(g),'edges':g.number_of_edges(),'components':nx.number_connected_components(g)})
        d=distances(g);rows+=evaluate(g,profiles,{'family':family,'trial':t,'seed':seed,'graph_id':name},d)
        if family not in ['lollipop_10','clique_chain','grid_bridge','modules4_regular4','core_periphery']:continue
        candidates=sorted(nx.non_edges(g));pr={'strict':[1,1],'moderate':[1,5],'tradeoff':[10,6]}
        baseline={p:metrics(*from_matrix(d),a,h) for p,(a,h) in pr.items()};subset=[]
        for u,v in candidates:
            costs,counts=from_matrix(add_edge_distances(d,u,v))
            for p,(a,h) in pr.items():
                z=metrics(costs,counts,a,h)
                subset.append(dict(graph_id=name,family=family,trial=t,seed=seed,u=u,v=v,profile=p,a=a,h0=h,delta_sai=z['sai']-baseline[p]['sai'],delta_mean=z['mean']-baseline[p]['mean'],delta_distance=z['mean_distance_reachable']-baseline[p]['mean_distance_reachable'],**z))
        edge_rows+=subset
        for p in pr:
            rr=[r for r in subset if r['profile']==p];optimal={};best={}
            for objective,key,sign in [('SAI','sai',1),('distance','mean_distance_reachable',1),('Jain','jain',-1),('Gini','satisfaction_gini',1)]:
                val=min(sign*r[key] for r in rr);opt=[r for r in rr if sign*r[key]<=val+1e-12]
                optimal[objective]={(r['u'],r['v']) for r in opt};best[objective]=opt[0]
            decisions.append({'graph_id':name,'family':family,'trial':t,'profile':p,'candidate_count':len(candidates),'sai_before':baseline[p]['sai'],'sai_best':best['SAI']['sai'],'mean_before':baseline[p]['mean'],'mean_after_sai_choice':best['SAI']['mean'],'minimum_before':baseline[p]['minimum'],'minimum_after_sai_choice':best['SAI']['minimum'],'sai_edge':str((best['SAI']['u'],best['SAI']['v'])),'distance_edge':str((best['distance']['u'],best['distance']['v'])),'disjoint_distance':not bool(optimal['SAI']&optimal['distance']),'disjoint_jain':not bool(optimal['SAI']&optimal['Jain']),'disjoint_gini':not bool(optimal['SAI']&optimal['Gini']),'sai_regret_distance':best['distance']['sai']-best['SAI']['sai'],'sai_optimal_count':len(optimal['SAI']),'distance_optimal_count':len(optimal['distance'])})
        print('completed all candidates',name,len(candidates),flush=True)
    write_csv('stress_metrics.csv',rows);write_csv('stress_edges.csv',edge_rows);write_csv('stress_decisions.csv',decisions)
    (OUT/'artifacts/stress_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('complete',len(manifest),'graphs',len(rows),'graph-profile rows',len(edge_rows),'candidate-profile rows',flush=True)

if __name__=='__main__':main()

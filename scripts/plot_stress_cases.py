#!/usr/bin/env python3
import ast
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from src.sai import distances
from scripts.build_report import OUT,FIG,read,save


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    data=read('stress_metrics.csv')
    families=['circulant_1_2','circulant_1_7','circulant_1_12','circulant_1_24','random_regular4','modules2_regular4','modules4_regular4']
    labels=['Circulant (1,2)','Circulant (1,7)','Circulant (1,12)','Circulant (1,24)','Random 4-regular','Two 4-regular modules','Four 4-regular modules']
    fig,axs=plt.subplots(1,3,figsize=(12,4.4),sharey=True)
    for j,family in enumerate(families):
        rr=[r for r in data if r['family']==family and r['profile']=='moderate']
        for ax,key in zip(axs,['degree_gini','sai','mean']):
            vals=np.array([r[key] for r in rr]);ax.scatter(vals,np.full(len(vals),j),s=16,alpha=.45,color='#285c91');ax.scatter(vals.mean(),j,marker='|',s=180,color='#c64e35')
    for ax,key in zip(axs,['Degree Gini','SAI','Mean satisfaction']):ax.set_xlabel(key);ax.grid(axis='x',alpha=.2)
    axs[0].set_yticks(range(len(families)),labels);axs[0].invert_yaxis();axs[0].set_xlim(-.01,.01)
    fig.suptitle('Controlled comparison: N=50, E=100, every degree=4; a=1, h0=5\nIndividual graph points; random families have 20 seeds');fig.tight_layout();save('13_matched_regular_controls')
    # Plot all profiles for these same graphs, not just the favorable profile.
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for family,label in zip(families,labels):
        rr=[r for r in data if r['family']==family and str(r['profile']).startswith('threshold_')]
        xx=sorted({r['h0'] for r in rr})
        for ax,key in zip(axs,['sai','mean']):
            yy=[np.mean([r[key] for r in rr if r['h0']==h]) for h in xx];ax.plot(xx,yy,marker='.',label=label);ax.set(xlabel='h0 (hops), a=1',ylabel=key)
    axs[0].legend(fontsize=6);save('14_matched_profile_dependence')
    selected=summary['representative']
    if selected:
        edges=read('stress_edges.csv');rr=[r for r in edges if r['graph_id']==selected['graph_id'] and r['profile']==selected['profile']]
        chosen={label:ast.literal_eval(selected[key]) for label,key in [('SAI','sai_edge'),('Mean distance','distance_edge')]}
        fig,ax=plt.subplots(figsize=(6.5,4));ax.scatter([-r['delta_distance'] for r in rr],[r['delta_sai'] for r in rr],s=9,alpha=.25,color='#315d83')
        for (label,(u,v)),color,marker in zip(chosen.items(),['#c33d29','#00947f'],['*','X']):
            point=next(r for r in rr if int(r['u'])==u and int(r['v'])==v)
            ax.scatter(-point['delta_distance'],point['delta_sai'],s=150,color=color,marker=marker,label=f'{label}: ({u},{v})')
        ax.set(xlabel='Average shortest-path reduction',ylabel='SAI change',title=f"Selected exploratory case: {selected['graph_id']}\nAll {len(rr)} candidate edges; a=10, h0=6");ax.legend();save('15_selected_decision_tradeoff')
        g=nx.read_edgelist(OUT/'artifacts/stress_graphs'/(selected['graph_id']+'.edgelist'),nodetype=int)
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        pos=nx.spring_layout(g,seed=20261014)
        nx.draw_networkx(g,pos,ax=axs[0],node_size=24,with_labels=False,edge_color='#c5c5c5',width=.8)
        comparison=[]
        for label,edge in [('Initial',None),*chosen.items()]:
            gg=g.copy()
            if edge:gg.add_edge(*edge)
            d=distances(gg);cost=d[~np.eye(len(g),dtype=bool)];w=expit(-10*(cost-6));ww=np.sort(w)
            axs[1].step(ww,np.arange(1,len(ww)+1)/len(ww),where='post',label=label)
            comparison.append({'label':label,'mean':float(w.mean()),'minimum':float(w.min()),'q05':float(np.quantile(w,.05,method='inverted_cdf')),'pass_fraction':float((cost<=6).mean())})
        for (label,edge),color in zip(chosen.items(),['#c33d29','#00947f']):
            nx.draw_networkx_edges(g,pos,edgelist=[edge],edge_color=color,width=2.5,ax=axs[0]);nx.draw_networkx_labels(g,pos,labels={i:str(i) for i in edge},font_size=8,ax=axs[0])
        axs[0].set(title='Initial graph with alternative added edges');axs[0].axis('off');axs[1].set(xlabel='Pair satisfaction (proxy)',ylabel='Fraction of ordered pairs <= value',title='Complete distributions, including the lower tail');axs[1].legend(fontsize=8);save('16_selected_graph_and_satisfaction')
        (OUT/'artifacts/representative_quality.json').write_text(json.dumps(comparison,indent=2))
    print('Stress figures complete; main-text examples must retain declared selection and scope.')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Generate all charts from stored per-trial CSVs, never hand-enter results."""
import csv
import json
from pathlib import Path
import sys
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
OUT=ROOT/'results/issue-1'; FIG=OUT/'figures'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})


def read(name):
    out=[]
    for r in csv.DictReader((OUT/'artifacts'/name).open()):
        for k,v in r.items():
            try:r[k]=float(v)
            except (ValueError,TypeError):pass
        out.append(r)
    return out


def save(name):
    plt.savefig(FIG/(name+'.pdf'));plt.savefig(FIG/(name+'.png'),dpi=160);plt.close()


def curve(ax,rows,x,y,label):
    groups=defaultdict(list)
    for r in rows:
        if np.isfinite(r[y]):groups[r[x]].append(r[y])
    xx=np.array(sorted(groups)); means=np.array([np.mean(groups[t]) for t in xx]); sd=np.array([np.std(groups[t],ddof=1) if len(groups[t])>1 else 0 for t in xx])
    ax.plot(xx,means,label=label);ax.fill_between(xx,means-sd,means+sd,alpha=.13)


def main():
    FIG.mkdir(exist_ok=True)
    random=read('canonical_random.csv'); sens=read('sensitivity.csv'); weighted=read('weighted.csv'); edges=read('edge_candidates.csv'); initial=read('dumbbell_initial.csv'); selected=read('edge_selected.csv')
    profiles=['direct-sensitive','strict','moderate','tolerant']
    for family in ['ER','BA','WS']:
        fig,ax=plt.subplots(figsize=(6,3.7))
        for profile in profiles:
            curve(ax,[r for r in random if r['family']==family and r['profile']==profile],'parameter','sai',profile)
        ax.set(xlabel={'ER':'Edge probability p','BA':'Attachment edges m','WS':'Rewiring probability beta'}[family],ylabel='SAI (all ordered pairs)',title=f'{family}: N=50; 20 independent trials; mean +/- SD');ax.legend(fontsize=8);save(f'01_{family}')
    er=[r for r in random if r['family']=='ER' and r['profile']=='moderate']
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for key,label in [('sai','Total SAI'),('support_term','Unreachability term'),('within_term','Within-support term')]:curve(axs[0],er,'parameter',key,label)
    for key,label in [('reachability','Reachable fraction'),('mean','Mean, all pairs'),('mean_reachable','Mean, reachable pairs')]:curve(axs[1],er,'parameter',key,label)
    for ax in axs:ax.set_xlabel('Edge probability p');ax.legend(fontsize=8)
    axs[0].set_ylabel('Normalized imbalance');axs[1].set_ylabel('Fraction / satisfaction');save('02_ER_decomposition')
    aa=sorted({r['a'] for r in sens});hh=sorted({r['h0'] for r in sens})
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for ax,key in zip(axs,['sai','mean']):
        zz=np.array([[np.mean([r[key] for r in sens if r['a']==a and r['h0']==h]) for h in hh] for a in aa])
        im=ax.imshow(zz,origin='lower',aspect='auto',extent=[hh[0],hh[-1],0,len(aa)]);ax.set_yticks(np.arange(len(aa))+.5,aa);ax.set(xlabel='h0 (hops)',ylabel='a (per hop)',title=key);fig.colorbar(im,ax=ax)
    save('03_profile_sensitivity')
    fig,ax=plt.subplots(figsize=(6,3.7))
    for mode in ['hops','synthetic-cost']:curve(ax,[r for r in weighted if r['cost_mode']==mode],'parameter','sai',mode)
    ax.set(xlabel='ER edge probability',ylabel='SAI',title='Different cost units AND profiles; not a controlled mapping-only contrast');ax.legend();save('04_weighted_cost')
    for topology,profile in [('two_cliques_25','A'),('heterogeneous_15_30_path5','tradeoff')]:
        before=next(r for r in initial if r['topology']==topology and r['profile']==profile)
        subset=[r for r in edges if r['topology']==topology and r['profile']==profile]
        fig,ax=plt.subplots(figsize=(6,3.7)); ax.scatter([before['mean_distance_reachable']-r['mean_distance_reachable'] for r in subset],[r['sai']-before['sai'] for r in subset],s=10,alpha=.5)
        for r in selected:
            if r['topology']==topology and r['objective'] in ['SAI','mean_distance']:
                ax.scatter(before['mean_distance_reachable']-r['mean_distance_reachable'],r['sai']-before['sai'],s=80,marker='x' if r['objective']=='SAI' else '+',label=r['objective'])
        ax.axhline(0,c='grey',lw=.6);ax.set(xlabel='Average distance reduction',ylabel='SAI change',title=f'{topology}: {len(subset)} candidate edges');ax.legend();save('05_edges_'+topology)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for family in sorted({r['family'] for r in random}):
        rr=[r for r in random if r['family']==family and r['profile']=='zoo' and np.isfinite(r['sai'])]
        axs[0].scatter([r['degree_gini'] for r in rr],[r['sai'] for r in rr],s=12,alpha=.4,label=family)
        axs[1].scatter([r['mean'] for r in rr],[r['sai'] for r in rr],s=12,alpha=.4,label=family)
    axs[0].set(xlabel='Degree Gini',ylabel='SAI');axs[1].set(xlabel='Mean satisfaction',ylabel='SAI');axs[0].legend(fontsize=7);fig.suptitle('a=3, h0=4; each point is one graph, not a trial mean');save('06_structure_function')
    fig,axs=plt.subplots(1,3,figsize=(12,3.4)); compare=[]
    for profile in profiles:
        rr=[r for r in random if r['profile']==profile and np.isfinite(r['sai'])]
        for ax,key in zip(axs,['jain','satisfaction_gini','distance_entropy']):
            ax.scatter([r[key] for r in rr],[r['sai'] for r in rr],s=7,alpha=.3,label=profile);ax.set(xlabel=key,ylabel='SAI')
            compare.append({'profile':profile,'baseline':key,'n':len(rr),'spearman':float(spearmanr([r[key] for r in rr],[r['sai'] for r in rr]).statistic)})
    axs[0].legend(fontsize=7);save('07_same_input_comparisons')
    # A systematic grouping prevents mixing profiles in the reported correlations.
    (OUT/'artifacts/comparison_summary.json').write_text(json.dumps(compare,indent=2))
    fig,ax=plt.subplots(figsize=(11,3));ax.axis('off')
    boxes=[(.09,'Static graph\n+ cost units'),(.31,'All-pair shortest costs\nUnreachable = infinity'),(.55,'Declared anchors / profile\nw = logistic(cost)'),(.81,'Report jointly\nI, coverage, mean, tails')]
    for x,label in boxes:ax.text(x,.62,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#edf3fa',ec='#315d83'),fontsize=10)
    for x,y in zip([.18,.42,.67],[.22,.44,.71]):ax.annotate('',xy=(y,.62),xytext=(x,.62),arrowprops=dict(arrowstyle='->'))
    ax.text(.53,.12,'W = 0: I undefined.  W > 0: normalize shares, compute entropy, separate support contribution.',ha='center',fontsize=10);save('08_workflow')
    if (OUT/'artifacts/as_results.csv').exists() and len(read('as_results.csv'))==41:
        data=read('as_results.csv'); fig,axs=plt.subplots(1,2,figsize=(9,3.7))
        for ax,key in zip(axs,['sai','mean']):
            for j,family in enumerate(['AS','ER','WS']):
                vals=[r[key] for r in data if r['family']==family];ax.scatter(np.full(len(vals),j),vals,s=18,alpha=.5);ax.scatter(j,np.mean(vals),marker='_',s=250,c='black')
            ax.set_xticks(range(3),['Oregon-1','ER','WS k=4']);ax.set_ylabel(key)
        fig.suptitle('All pairs, a=1, h0=4; AS one snapshot, ER/WS 20 seeds');save('09_AS')
    if (OUT/'artifacts/runtime_memory.csv').exists():
        data=read('runtime_memory.csv');fig,axs=plt.subplots(1,3,figsize=(12,3.6))
        for backend in ['dense','batched']:
            rr=[r for r in data if r['backend']==backend]
            for ax,key in zip(axs,['path_and_histogram_seconds','aggregation_seconds','peak_rss_mib']):
                curve(ax,rr,'n',key,backend);ax.set(xlabel='N',ylabel=key,xscale='log',yscale='log')
        axs[0].legend();save('10_runtime_memory')
    import networkx as nx
    g=nx.read_edgelist(OUT/'artifacts/two_cliques_25.edgelist',nodetype=int)
    # Fixed cluster layout, reused for both panels.
    pos={i:(np.cos(2*np.pi*i/25)-1.4,np.sin(2*np.pi*i/25)) for i in range(25)}
    pos.update({i:(np.cos(2*np.pi*(i-25)/25)+1.4,np.sin(2*np.pi*(i-25)/25)) for i in range(25,50)})
    best=next(r for r in selected if r['topology']=='two_cliques_25' and r['objective']=='SAI');edge=(int(best['u']),int(best['v']))
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for ax,title in zip(axs,['Initial: two K25 and one bridge','One added edge; all mean/SAI/Jain/Gini optima agree']):
        nx.draw_networkx(g,pos,ax=ax,node_size=18,with_labels=False,width=.3,edge_color='#bbbbbb');ax.set_title(title,fontsize=9);ax.axis('off')
    nx.draw_networkx_edges(g,pos,edgelist=[edge],edge_color='#d33d28',width=2,ax=axs[1]);save('11_dumbbell_topology')
    cal=read('calibration.csv');fig,axs=plt.subplots(1,2,figsize=(9,3.7))
    for c1,c2 in sorted({(r['c1_nominal'],r['c2_nominal']) for r in cal}):
        rr=[r for r in cal if r['c1_nominal']==c1 and r['c2_nominal']==c2]
        for ax,key in zip(axs,['sai','mean']):ax.scatter([r['h0'] for r in rr],[r[key] for r in rr],label=f'anchors {c1:g},{c2:g} hops');ax.set(xlabel='Calibrated h0 (hops)',ylabel=key)
    axs[0].legend(fontsize=8);fig.suptitle('Assumed 0.9/0.1 acceptability anchors; independent +/-10% cost perturbations');save('12_calibration')
    print('generated',len(list(FIG.glob('*.pdf'))),'figures')

if __name__=='__main__': main()

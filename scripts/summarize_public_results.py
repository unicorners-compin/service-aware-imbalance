#!/usr/bin/env python3
"""Validate retained full-run outputs without internal storage or manuscript files."""
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/issue-1'
EXPECTED={'canonical_random.csv':3820,'sensitivity.csv':2700,'weighted.csv':360,
'edge_candidates.csv':3285,'dumbbell_initial.csv':5,'edge_selected.csv':10,
'calibration.csv':27,'exhaustive_small_edge_audit.csv':3284,'as_results.csv':41,
'runtime_memory.csv':48,'stress_metrics.csv':1704,'stress_decisions.csv':129,'stress_edges.csv':143340}
def main():
    observed={};support_error=0.;mean_error=0.;decisions=[];bench={}
    for name,expected in EXPECTED.items():
        with (OUT/'artifacts'/name).open() as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==expected,(name,len(rows),expected)
        observed[name]=len(rows)
        for row in rows:
            if 'sai' not in row:continue
            value=float(row['sai']);q=float(row['reachability'])
            if not math.isfinite(value):
                assert q==0;continue
            assert -1e-12<=value<=1+1e-12
            support_error=max(support_error,abs(value-float(row['support_term'])-float(row['within_term'])))
            mean_error=max(mean_error,abs(float(row['mean'])-q*float(row['mean_reachable'])))
        if name=='stress_decisions.csv':decisions=rows
        if name=='runtime_memory.csv':
            for row in rows:bench.setdefault((row['n'],row['trial']),{})[row['backend']]=float(row['sai'])
    backend_error=max(abs(v['dense']-v['batched']) for v in bench.values())
    assert max(support_error,mean_error,backend_error)<1e-10
    candidates=[r for r in decisions if r['disjoint_distance']=='True']
    representative=max(candidates,key=lambda r:float(r['sai_regret_distance'])) if candidates else None
    summary={'complete_configured_rows':observed,'support_identity_max_error':support_error,
    'mean_identity_max_error':mean_error,'dense_batched_sai_max_error':backend_error,
    'stress_decision_count':len(decisions),'stress_disjoint_optima':{k:sum(r[k]=='True' for r in decisions) for k in ['disjoint_distance','disjoint_jain','disjoint_gini']},
    'representative_selection_rule':'Maximum SAI regret of the lexicographic mean-distance optimum among cases with disjoint SAI/distance optima; exploratory existence case.',
    'representative':representative}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Validated',sum(observed.values()),'retained records; output:',OUT/'summary.json')
if __name__=='__main__':main()

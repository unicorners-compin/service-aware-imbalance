"""Exact SAI and same-input baselines. Natural logs; base-independent SAI."""
import numpy as np
import networkx as nx
from scipy.special import log_expit, logsumexp
from scipy.sparse.csgraph import shortest_path


def calibrate(c1, s1, c2, s2):
    if not (c1 < c2 and 0 < s2 < s1 < 1):
        raise ValueError('Need c1<c2 and 0<s2<s1<1')
    l1, l2 = np.log(1/s1-1), np.log(1/s2-1)
    a = (l2-l1)/(c2-c1)
    return float(a), float(c1-l1/a)


def sparse_graph(g, weighted=False):
    a = nx.to_scipy_sparse_array(g, nodelist=list(g), weight='weight' if weighted else None, format='csr', dtype=float)
    # SciPy csgraph requires int32 sparse indices on this pinned release.
    a.indices = a.indices.astype(np.int32)
    a.indptr = a.indptr.astype(np.int32)
    return a


def distances(g, weighted=False):
    return shortest_path(sparse_graph(g, weighted), directed=False, unweighted=not weighted, method='D')


def histogram(g, batch_size=128):
    """Exact unweighted all-pair histogram, O(batch_size*N) distance memory."""
    n = len(g)
    a = sparse_graph(g)
    hist = np.zeros(n, dtype=np.int64)
    for start in range(0, n, batch_size):
        ids = np.arange(start, min(start+batch_size, n))
        d = shortest_path(a, directed=False, unweighted=True, method='D', indices=ids)
        d[np.arange(len(ids)), ids] = np.inf
        v = d[np.isfinite(d)].astype(np.int64)
        h = np.bincount(v, minlength=n)
        hist += h
    active = np.flatnonzero(hist)
    costs, counts = active.astype(float), hist[active]
    missing = n*(n-1)-int(counts.sum())
    if missing:
        costs, counts = np.r_[costs, np.inf], np.r_[counts, missing]
    return costs, counts


def from_matrix(d):
    return np.unique(d[~np.eye(len(d), dtype=bool)], return_counts=True)


def gini(values, counts=None):
    x = np.asarray(values, dtype=float)
    c = np.ones(len(x)) if counts is None else np.asarray(counts, dtype=float)
    idx = np.argsort(x)
    x, c = x[idx], c[idx]
    total, n = np.dot(x, c), c.sum()
    if total <= 0:
        return np.nan
    return float(np.sum(c*x*(2*np.cumsum(c)-c-n))/(n*total))


def metrics(costs, counts, a, h0):
    if a <= 0 or not np.isfinite(a) or not np.isfinite(h0):
        raise ValueError('a must be positive and h0 finite')
    costs, counts = np.asarray(costs, float), np.asarray(counts, float)
    if costs.shape != counts.shape or np.any(counts <= 0) or np.any(costs < 0) or np.any(np.isnan(costs)):
        raise ValueError('invalid cost histogram')
    m = float(counts.sum())
    if m <= 1:
        raise ValueError('need at least two ordered pairs')
    finite = np.isfinite(costs)
    r = counts[finite].sum()
    q = r/m
    qc = counts/m
    out = {'pairs':int(m), 'reachable_pairs':int(r), 'reachability':float(q),
           'distance_entropy':float(-np.dot(qc,np.log(qc))),
           'mean_distance_reachable':float(np.dot(costs[finite],counts[finite])/r) if r else np.nan,
           'efficiency':float(np.dot(1/costs[finite],counts[finite])/m) if r else 0.0}
    keys = ['sai','theil','jain','satisfaction_gini','mean','mean_reachable','log_mean',
            'support_term','within_term','sai_reachable','effective_pair_fraction','minimum','q05','pass_fraction']
    if r == 0:
        out.update({k:np.nan for k in keys})
        out.update(mean=0., minimum=0., q05=0., pass_fraction=0., effective_pair_fraction=0.)
        return out
    lw = log_expit(-a*(costs[finite]-h0))
    c = counts[finite]
    logW = logsumexp(np.log(c)+lw)
    mass = np.exp(np.log(c)+lw-logW)
    T = float(np.dot(mass, lw-logW+np.log(m)))
    T = max(T, 0.)  # clip only roundoff below zero
    support = -np.log(q)
    within = max(T-support, 0.)
    # Scaled values avoid artificial zero-total failures for strict profiles.
    scaled = np.zeros(len(costs)); scaled[finite] = np.exp(lw-lw.max())
    w = np.zeros(len(costs)); w[finite] = np.exp(lw)
    sort = np.argsort(w)
    quant = w[sort][np.searchsorted(np.cumsum(counts[sort]), 0.05*m, side='left')]
    out.update(sai=T/np.log(m), theil=T,
        jain=float(np.exp(2*logW-np.log(m)-logsumexp(np.log(c)+2*lw))),
        satisfaction_gini=gini(scaled,counts), mean=float(np.exp(logW-np.log(m))),
        mean_reachable=float(np.exp(logW-np.log(r))),log_mean=float(logW-np.log(m)),
        support_term=float(support/np.log(m)),within_term=float(within/np.log(m)),
        sai_reachable=float(within/np.log(r)) if r>1 else 0.,
        effective_pair_fraction=float(np.exp(-T)),minimum=float(w.min()),q05=float(quant),
        pass_fraction=float(counts[costs<=h0].sum()/m))
    return out


def add_edge_distances(d, u, v, weight=1.):
    """Exact one-edge update for undirected nonnegative static path costs."""
    return np.minimum(d, np.minimum(d[:,u,None]+weight+d[None,v,:], d[:,v,None]+weight+d[None,u,:]))

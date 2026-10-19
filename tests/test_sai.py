import unittest
import numpy as np
import networkx as nx
from scipy.special import expit
from src.sai import metrics, distances, histogram, from_matrix, calibrate, add_edge_distances


class MetricTests(unittest.TestCase):
    def test_independent_direct_formula(self):
        for g in [nx.path_graph(9), nx.star_graph(8), nx.empty_graph(9), nx.disjoint_union(nx.complete_graph(4), nx.path_graph(5))]:
            d=distances(g); x=d[~np.eye(len(g),dtype=bool)]
            for a,h in [(1,3),(5,1),(.5,6)]:
                z=metrics(*from_matrix(d),a,h); w=expit(-a*(x-h))
                if w.sum()==0:
                    self.assertTrue(np.isnan(z['sai'])); continue
                p=w[w>0]/w.sum(); H=-np.dot(p,np.log(p))
                self.assertAlmostEqual(z['sai'],1-H/np.log(len(x)),places=12)
                self.assertAlmostEqual(z['jain'],w.sum()**2/(len(w)*np.dot(w,w)),places=12)
                self.assertAlmostEqual(z['satisfaction_gini'],np.abs(w[:,None]-w).sum()/(2*len(w)*w.sum()),places=12)
                self.assertAlmostEqual(z['sai'],z['support_term']+z['within_term'],places=12)

    def test_histogram_and_update(self):
        g=nx.gnp_random_graph(17,.15,seed=123)
        d=distances(g)
        c,n=histogram(g,4); c2,n2=from_matrix(d)
        np.testing.assert_equal(c,c2); np.testing.assert_equal(n,n2)
        for u,v in list(nx.non_edges(g))[:12]:
            h=g.copy(); h.add_edge(u,v)
            np.testing.assert_allclose(add_edge_distances(d,u,v),distances(h))

    def test_calibration(self):
        a,h=calibrate(2,.9,8,.1)
        np.testing.assert_allclose(expit(-a*(np.array([2,8])-h)),[.9,.1])
        with self.assertRaises(ValueError): calibrate(2,.1,8,.9)

    def test_limits_and_replication(self):
        z=metrics([1,3,7],[4,8,8],100,4)
        self.assertAlmostEqual(z['sai'],np.log(20/12)/np.log(20))
        self.assertAlmostEqual(z['jain'],12/20)
        a=metrics([1,3],[2,4],1,2); b=metrics([1,3],[4,8],1,2)
        self.assertAlmostEqual(a['theil'],b['theil'])
        self.assertNotAlmostEqual(a['sai'],b['sai'])

    def test_underflow_is_not_zero_total(self):
        z=metrics([1000,1001],[2,4],100,1)
        self.assertTrue(np.isfinite(z['sai']))
        self.assertTrue(np.isfinite(z['log_mean']))

    def test_complete_and_uniform_disconnected(self):
        self.assertAlmostEqual(metrics([1],[30],2,3)['sai'],0)
        z=metrics([1,np.inf],[12,18],2,3)
        self.assertAlmostEqual(z['sai'],np.log(30/12)/np.log(30))
        self.assertAlmostEqual(z['sai_reachable'],0)

    def test_distance_class_identity(self):
        costs = np.array([1., 2., 5., np.inf])
        counts = np.array([8, 12, 16, 6])
        w = expit(-(costs - 3))
        mass = counts*w / np.dot(counts, w)
        active = mass > 0
        class_kl = np.dot(mass[active], np.log(mass[active]/(counts/counts.sum())[active]))
        self.assertAlmostEqual(metrics(costs, counts, 1, 3)['theil'], class_kl, places=12)

    def test_cost_unit_invariance(self):
        costs, counts = np.array([1., 4., 8., np.inf]), [4, 8, 8, 2]
        original = metrics(costs, counts, 0.8, 5)
        converted = metrics(costs*1000, counts, 0.8/1000, 5000)
        for key in ['sai', 'mean', 'jain', 'satisfaction_gini']:
            self.assertAlmostEqual(original[key], converted[key], places=12)

    def test_flat_profile_variance_limit(self):
        costs = np.array([1., 2., 5., 7.])
        counts = np.array([4, 8, 6, 2])
        expanded = np.repeat(costs, counts)
        a = 1e-4
        coefficient = metrics(costs, counts, a, 3)['theil'] / a**2
        self.assertAlmostEqual(coefficient, np.var(expanded)/8, delta=1e-4)

if __name__=='__main__': unittest.main()

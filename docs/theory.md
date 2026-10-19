# Mathematical interpretation of Service-Aware Imbalance

All logarithms below are natural. M=N(N−1), w_i≥0, W=Σw_i>0,
p_i=w_i/W, H=−Σp_i log p_i. Changing log base leaves I unchanged.
These are proofs/interpretations of the submitted construction, not claims
of new general-purpose inequality theory.

## 1. Established Theil identity and physical interpretation

T=(1/M)Σ(w_i/mean(w))log(w_i/mean(w))=Σp_i log(Mp_i)=log M−H.
Therefore I=T/log M is exactly normalized Theil T. The effective number
of equally weighted pairs is exp(H)=M exp(−T). This is concentration of
acceptability mass, not probability that a packet succeeds, throughput,
or a causal resource allocation. Multiplying every positive w_i by the
same constant leaves I unchanged: mean and tail outcomes are essential.

J=(Σw_i)^2/(MΣw_i²)=1/(1+CV²) uses the same input but a second moment.
Satisfaction Gini uses pairwise absolute differences. Degree Gini acts
on a different population and is only a structural comparator.
Agreement with Jain is an empirical outcome, not a universal theorem.

## 2. Exact coverage decomposition

Let R={i:w_i>0}, r=|R|, q=r/M. For the finite-parameter logistic, R is
exactly the reachable set. Define T_R=Σ_{i∈R}p_i log(rp_i).
Substitution gives T=log(M/r)+T_R and

I=−log(q)/log(M)+[log(r)/log(M)] I_R, I_R=T_R/log(r).

For r=1 define T_R=I_R=0. W=0 is undefined. Keep the submitted I as an
all-pair coverage-and-concentration convention and display both terms;
I_R alone would discard failures. mean_all=q mean_reachable. This
directly distinguishes support growth from within-support equality at
percolation. It does not require inventing a different metric.

## 3. Pair entropy versus distance-class entropy

For distance class d let n_d be its multiplicity, q_d=n_d/M (including
the unreachable class), f_d=1/(1+exp(a(d−h0))), and
P_d=n_df_d/Σ_k n_k f_k. Grouping identical terms gives

T=Σ_d P_d log(P_d/q_d),
H_pair=H(P)+Σ_d P_d log n_d.

Thus SAI is determined by the distance histogram and profile, but is
not the plain entropy H(q) of that histogram. Networks with the same
histogram and N are indistinguishable under every fixed profile. Even
the same mean plus I cannot identify which endpoints are underserved.
Conditional equality within a distance class is identically perfect
under w=f(d); it cannot test heterogeneous real service in this model.

## 4. Interpretable limiting cases

For a connected finite graph, as a→0, w_i=1/2−a(d_i−h0)/4+O(a³),
and Taylor expansion of T about uniform shares gives
T=a² Var(d)/8+O(a³). A large h0 at fixed a also makes I→0 through
saturation, irrespective of engineering quality. Low I is not proof
of hub causality or of good real service.

For a→∞, choose h0 not equal to a distance and suppose k>0 pairs
have d<h0. Their weights converge to 1, all others to 0, so
I→log(M/k)/log M and J→k/M. In this limit SAI is a monotone transform
of the accepted-pair fraction, not a uniquely informative new fairness
notion. Boundary ties have weight 1/2 and need separate accounting.

## 5. Why an improving edge can increase inequality

Let S=Σw_i log w_i. Then T=S/W−log(W/M). For any finite edge-induced
change with W'>0,

T'−T=S'/W'−S/W−log(W'/W).

For positive weights, ∂T/∂w_i=[log(w_i)−S/W]/W. Improving a weight
below the satisfaction-mass-weighted geometric mean exp(S/W) reduces
T locally; improving one above it increases T locally. An edge improves
multiple pairs, so its finite effect is evaluated by the exact identity,
not inferred from average path shortening. This is an application of
the established entropy functional, not a new routing guarantee.

## 6. Calibration, units, conventions and complexity

Given two external acceptability anchors (c1,s1),(c2,s2), with c1<c2
and 1>s1>s2>0, let L_j=log(1/s_j−1). Then
a=(L2−L1)/(c2−c1), h0=c1−L1/a. a has inverse cost units. If costs
are multiplied by b>0, a'=a/b and h0'=b h0 preserve all weights.
An SLA supplying only one threshold does not identify both parameters.
The included examples are assumed anchors, not empirical calibration.

Duplicating each observation b times preserves T but changes
I to T/log(bM). In an undirected graph using ordered versus unordered
pairs therefore changes normalized I while preserving T. Report the
convention and raw T; avoid unqualified cross-size comparisons.

Unweighted all-pairs BFS takes O(N(N+E)); repeated nonnegative Dijkstra
typically O(N(E+N)log N) with binary heaps. Full storage costs O(N²).
Exact streaming retains O(E+N+BN) memory for batch size B, plus the
unweighted distance histogram O(N); it reduces memory, not worst-case
all-pair path work. Aggregation over D occupied histogram bins costs
O(D) for Theil/Jain and O(D log D) for sorted Gini. Runtime tests use
SciPy's pinned csgraph implementation, not a claim of optimal BFS.

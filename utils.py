"""
Functions 
"""
import random
from copy import deepcopy
from itertools import combinations

import networkx as nx
import numpy as np
import xgi

__all__ = [
    "compute_eigenvalues",
    "compute_eigenvalues_multi",
    "shuffle_hyperedges",
    "node_swap",
    "find_triangles",
    "flag_complex_d2",
    "random_flag_complex_d2",
    "degree_corr",
    "deg_hetero_ratio",
]


def compute_eigenvalues(H, order, weight, rescale_per_node=True):
    """Returns the Lyapunov exponents of corresponding to the Laplacian of order d.

    Parameters
    ----------
    HG : xgi.HyperGraph
        Hypergraph
    order : int
        Order to consider.
    weight: float
        Weight, i.e coupling strenght gamma in [1]_.
    rescale_per_node: bool, (default=True)
        Whether to rescale each Laplacian of order d by d (per node).

    Returns
    -------
    lyap : array
        Array of dim (N,) with unsorted Lyapunov exponents
    """

    # compute Laplacian
    L = xgi.laplacian(H, order, rescale_per_node=rescale_per_node)
    K = xgi.degree_matrix(H, order)

    # compute eigenvalues
    eivals, _ = np.linalg.eigh(L)
    lyap = -(weight / np.mean(K)) * eivals
    return lyap


def compute_eigenvalues_multi(H, orders, weights, rescale_per_node=True):
    """Returns the Lyapunov exponents of corresponding to the muliotder Laplacian.

    Parameters
    ----------
    HG : xgi.HyperGraph
        Hypergraph
    orders : list of int
        Orders of interactions to consider.
    weights: list of float
        Weight of each order, i.e coupling strenghts gamma_i in [1]_.
    rescale_per_node: bool, (default=True)
        Whether to rescale each Laplacian of order d by d (per node).

    Returns
    -------
    lyap : array
        Array of dim (N,) with unsorted Lyapunov exponents
    """

    # compute multiorder Laplacian
    L_multi = xgi.multiorder_laplacian(
        H, orders, weights, rescale_per_node=rescale_per_node
    )

    # compute eigenvalues
    eivals_multi, _ = np.linalg.eigh(L_multi)
    lyap_multi = -eivals_multi
    return lyap_multi


def shuffle_hyperedges(S, order, p):
    """Shuffle existing hyperdeges of order d with probablity p

    Parameters
    ----------
        S: xgi.HyperGraph
                Hypergraph
        order: int
                Order of hyperedges to shuffle
        p: float
                Probability of shuffling each hyperedge

        Returns
        -------
        H: xgi.HyperGraph
                Hypergraph with edges of order d shuffled

    """

    nodes = S.nodes
    H = xgi.Hypergraph(S)

    d_hyperedges = H.edges.filterby("order", order).members(dtype=dict)

    for id_, members in d_hyperedges.items():
        if random.random() <= p:
            H.remove_edge(id_)
            new_hyperedge = tuple(random.sample(nodes, order + 1))
            while new_hyperedge in H._edge.values():
                new_hyperedge = tuple(random.sample(nodes, order + 1))
            H.add_edge(new_hyperedge)

    assert H.num_nodes == S.num_nodes
    assert xgi.num_edges_order(H, 1) == xgi.num_edges_order(S, 1)
    assert xgi.num_edges_order(H, 2) == xgi.num_edges_order(S, 2)

    return H


def node_swap(H, nid1, nid2, id_temp=-1, order=None):
    """Swap node nid1 and node nid2 in all edges of order order that contain them

    Parameters
    ----------
    H: HyperGraph
        Hypergraph to consider
    nid1: node ID
        ID of first node to swap
    nid2: node ID
        ID of second node to swap
    id_temp: node ID
        Temporary ID given to nodes when swapping
    order: {int, None}, default: None
        If None, consider all orders. If an integer,
        consider edges of that order.

    Returns
    -------
    HH: HyperGraph

    """

    # make sure id_temps does not exist yet
    while id_temp in H.edges:
        id_temp -= 1

    if order:
        edge_dict = H.edges.filterby("order", order).members(dtype=dict).copy()
    else:
        edge_dict = H.edges.members(dtype=dict).copy()

    new_edge_dict = deepcopy(edge_dict)
    HH = H.copy()

    for key, members in edge_dict.items():

        if nid1 in members:
            members.remove(nid1)
            members.add(id_temp)
        new_edge_dict[key] = members

    for key, members in new_edge_dict.items():

        if nid2 in members:
            members.remove(nid2)
            members.add(nid1)
        new_edge_dict[key] = members

    for key, members in new_edge_dict.items():

        if id_temp in members:
            members.remove(id_temp)
            members.add(nid2)
        new_edge_dict[key] = members

    HH.remove_edges_from(edge_dict)
    HH.add_edges_from(new_edge_dict)

    return HH


def find_triangles(G):
    """Returns list of 3-node cliques present in a graph

    Parameters
    ----------
    G : networkx Graph
        Graph to consider

    Returns
    -------
    list of triangles
    """

    triangles = set(
        frozenset((n, nbr, nbr2))
        for n in G
        for nbr, nbr2 in combinations(G[n], 2)
        if nbr in G[nbr2]
    )
    return [set(tri) for tri in triangles]


def flag_complex_d2(G, p2=None):
    """Returns list of 3-node cliques present in a graph

    Parameters
    ----------
    G : networkx Graph
        Graph to consider
    p2: float
        Probability (between 0 and 1) of filling empty triangles in graph G

    Returns
    -------
    S : xgi.SimplicialComplex

    """
    nodes = G.nodes()
    edges = G.edges()

    S = xgi.SimplicialComplex()
    S.add_nodes_from(nodes)
    S.add_simplices_from(edges)

    triangles_empty = find_triangles(G)

    if p2:
        triangles = [el for el in triangles_empty if random.random() <= p2]
    else:
        triangles = triangles_empty

    S.add_simplices_from(triangles)

    return S


def random_flag_complex_d2(N, p, seed=None):
    """Generate a maximal simplicial complex (up to order 2) from a
    :math:`G_{N,p}` Erdős-Rényi random graph by filling all empty triangles with 2-simplices.

    Parameters
    ----------
    N : int
        Number of nodes
    p : float
        Probabilities (between 0 and 1) to create an edge
        between any 2 nodes
    seed : int or None (default)
        The seed for the random number generator

    Returns
    -------
    SimplicialComplex

    Notes
    -----
    Computing all cliques quickly becomes heavy for large networks.
    """
    if seed is not None:
        random.seed(seed)

    if (p < 0) or (p > 1):
        raise ValueError("p must be between 0 and 1 included.")

    G = nx.fast_gnp_random_graph(N, p, seed=seed)

    return flag_complex_d2(G)


def degree_corr(H):
    """Return the cross-order degree correlation of hypergraph H

    Parameters
    ----------
    H: xgi.Hypergraph
        Hypergraph to consider

    Returns
    -------
    float

    """
    K1 = xgi.degree_matrix(H, order=1)
    K2 = xgi.degree_matrix(H, order=2)
    return np.corrcoef(K1, K2, rowvar=False)[0, 1]


def deg_hetero_ratio(HG):
    """Return the degree heterogeneity ratio of hypergraph H

    Parameters
    ----------
    H: xgi.Hypergraph
        Hypergraph to consider

    Returns
    -------
    float

    """
    k1_max = HG.nodes.degree(order=1).max()
    k1_mean = HG.nodes.degree(order=1).mean()

    k2_max = HG.nodes.degree(order=2).max()
    k2_mean = HG.nodes.degree(order=2).mean()

    h1 = (k1_max - k1_mean) / k1_mean
    h2 = (k2_max - k2_mean) / k2_mean
    r2 = h2 / h1  # eq 12
    return r2

"""
Useful functions to generate hypergraphs 
and compute the multiorder Laplacian 
author: Maxime LUCAS (ml.maximelucas@gmail.com)
"""

import random
from itertools import combinations, permutations
from math import factorial

import networkx as nx
import numpy as np

__all__ = [
    'sort_hyperedges', 
    'hyperedges_of_order', 
    'to_simplicial_complex_from_hypergraph',
    'random_hypergraph',
    'fully_connected_hypergraph',
    'adj_tensor_of_order', 
    'adj_matrix_of_order',
    'degree_of_order',
    'laplacian_of_order',
    'compute_laplacians_resource_constrained',
    'compute_eigenvalues_resource_constrained',
    'compute_eigenvalues_resource_constrained_alphas'
]
    

#=======
# UTILS
#=======

def sort_hyperedges(hyperedges, directed=False) : 
    """Returns list of hyperedges sorted by length and then alphabetically. 
    If not directed, pre-sort nodes in each hyperedge alphabetically
    """
    
    if not directed : 
        hyperedges = [tuple(sorted(he)) for he in hyperedges]
    
    return sorted(hyperedges, key=lambda x: (len(x), x))
    
    
def hyperedges_of_order(hyperedges, d) :
    """Returns list of all d-hyperedges"""
    
    return [hyperedge for hyperedge in hyperedges if len(hyperedge)==d+1]
    
    
def to_simplicial_complex_from_hypergraph(hyperedges, verbose=False) : 
    """Converts a hypergraph to a simplicial complex
    by adding all missing subfaces.
    
    Parameters
    ----------
    hyperedges : list of tuples 
        List of hyperedges in the hypergraph to fill
    verbose : bool
        If True, print all added hyperedges 
        
    Returns
    -------
    hyperedges_simplicial : list of tuples 
    
    """
    
    hyperedges_simplicial = _add_all_subfaces(hyperedges, verbose=verbose)
    
    return hyperedges_simplicial   

            
def _add_all_subfaces(hyperedges, verbose=False) : 
    """Adds all missing subfaces to hypergraph
    
    Goes through all hyperedges, from larger to smaller,
    and adds their subfaces if they do not exist.
    
    Parameters
    ----------
    hyperedges : list of tuples 
        List of hyperedges in the hypergraph to fill
    verbose : bool
        If True, print all added hyperedges 
        
    Returns
    -------
    hyperedges : list of tuples     
    
    """
    
    hyperedges_to_add = []
    
    # check that all subfaces of each hyperedge exist
    for hedge in hyperedges[::-1] : # loop over hyperedges, from larger to smaller 

        size = len(hedge) # number of node, i.e. order+1
        d = size - 1 # order
        if size>=3 : # nodes are all there already, so no need to check size<=2
            for face in combinations(hedge, size-1) : # check if all subfaces are present

                face = tuple(sorted(face))
                if face not in hyperedges : 
                    hyperedges_to_add.append(face)
    
        
    if verbose : 
        print(f"Info: the following hyperedges were added")
        print(hyperedges_to_add)
        
    hyperedges_final = hyperedges + list(set(hyperedges_to_add))
    hyperedges_final = sort_hyperedges(hyperedges_final)
        
    return hyperedges_final
    
#=======================
# HYPERGRAPH GENERATORS
#=======================

def random_hypergraph(N, ps) : 
    """Generates a random hypergraph
    
    Generate N nodes, and connect any d+1 nodes 
    by a hyperedge with probability ps[d].
        
    Parameters
    ----------
    N : int 
        Number of nodes 
    ps : list of floats
        Probabilities (between 0 and 1) to create a hyperedge
        at each order d between any d+1 nodes. ps[0] is edges, 
        ps[1] for triangles, etc.
        
    Returns
    -------
    List of tuples
        List of hypergraphs
    """
    
    #I first generate a standard ER graph with edges connected with probability p1
    G = nx.fast_gnp_random_graph(N, ps[0], seed=None)
    
    nodes = G.nodes()
    hyperedges = list(G.edges()) 
    
    for i,p in enumerate(ps[1:]) :
        d = i + 2 # order (+2 because we started with [1:])
        for hyperedge in combinations(nodes, d+1) : 
            if random.random() <= p :
                hyperedges.append(hyperedge)
                
    return sort_hyperedges(hyperedges)

def random_simplicial_complex_d2(N, p1, p2) : 
    """Generates a random hypergraph
    
    Generate N nodes, and connect any d+1 nodes 
    by a hyperedge with probability ps[d].
        
    Parameters
    ----------
    N : int 
        Number of nodes 
    ps : list of floats
        Probabilities (between 0 and 1) to create a hyperedge
        at each order d between any d+1 nodes. ps[0] is edges, 
        ps[1] for triangles, etc.
        
    Returns
    -------
    List of tuples
        List of hypergraphs
    """
    
    #I first generate a standard ER graph with edges connected with probability p1
    G = nx.fast_gnp_random_graph(N, p1, seed=None)
    
    nodes = G.nodes()
    hyperedges = list(G.edges()) 
    
    for hyperedge in combinations(nodes, 3) : 
        if random.random() <= p2 :
            hyperedges.append(hyperedge)
            
            # add each edge too
            hyperedges.append((hyperedge[0], hyperedge[1]))
            hyperedges.append((hyperedge[0], hyperedge[2]))
            hyperedges.append((hyperedge[1], hyperedge[2]))

    # get rid of duplicates 
    hyperedges = set(sort_hyperedges(hyperedges))

    return sort_hyperedges(hyperedges)

def random_maximal_simplicial_complex_d2(N,p) : 
    """Generate maximal simplicial complex from graph,
    up to order 2, by filling all empty triangles.
    
    Parameters
    ----------
    N : int 
        Number of nodes 
    p : list of floats
        Probabilities (between 0 and 1) to create an edge 
        between any 2 nodes
        
    Returns
    -------
    hyperedges_final : list of tuples
        List of hyperedges, i.e. tuples of length 2 and 3.
    
    Notes
    -----
    Computing all cliques quickly becomes heavy for large networks.
    
    """
    
    G = nx.fast_gnp_random_graph(N, p, seed=None)
    
    nodes = G.nodes()
    edges = list(G.edges()) 
    
    # compute all triangles to fill
    all_cliques = list(nx.enumerate_all_cliques(G))
    triad_cliques = [tuple(x) for x in all_cliques if len(x)==3 ]
    triad_cliques = sort_hyperedges(triad_cliques)
    
#     # make simplicial complex by filling only some empty triangles
#     if p2 < 1 :
#         randoms = np.random.random((len(triad_cliques)))
#         idx = np.where(randoms <= p2)[0]
#         triad_cliques = [triad_cliques[i] for i in idx]
    
    hyperedges_final = sort_hyperedges(edges + triad_cliques)
    return hyperedges_final


def fully_connected_hypergraph(N, d_max) : 
    """Generates a fully connected hypergraphs
    
    Generate N nodes and connect any d+1 nodes
    by a hyperedge, up to order d_max. 
    
    Parameters
    ----------
    N : int 
        Number of nodes  
    d_max : int 
        Highest order of interactions. For example, 
        d_max=2 means we go up to triangles.    
    """
    
    nodes = range(N)
    
    hyperedges = [] 
    
    for d in range(1, d_max+1) :
        for hyperedge in combinations(nodes, d+1) : 
            hyperedges.append(hyperedge)
    
    return sort_hyperedges(hyperedges)
    
    
#======================
# Multiorder Laplacian
#======================

def adj_tensor_of_order(d, N, hyperedges) : 
    """Returns the adjacency tensor of order d
    
    Parameters
    ----------
    d : int 
        Order of the adjacency matrix 
    d_simplices : list of tuples 
        Sorted list of hyperedges of order d
        
    Returns
    -------
    M : numpy array
        Adjacency tensor of order d
        
    """
    
    d_hyperedges = hyperedges_of_order(hyperedges, d)
    
    assert len(d_hyperedges[0]) == d+1
    
    dims = (N,) * (d+1)
    M = np.zeros(dims)
    
    for d_hyperedge in d_hyperedges :
        for d_hyperedge_perm in permutations(d_hyperedge) : 
    
            M[d_hyperedge_perm] = 1
    return M

def adj_matrix_of_order(d, M) : 
    """Returns the adjacency matrix of order d
    
    Parameters
    ----------
    d : int 
        Order of the adjacency matrix 
    M : numpy array
        Adjacency tensor of order d
        
    Returns
    -------
    adj_d : numpy array
        Matrix of dim (N, N)
    
    """
    
    adj_d = 1 / factorial(d-1) * np.sum(M, axis=tuple(range(d+1)[2:])) # sum over all axes except first 2 (i,j)
    
    return adj_d

def degree_of_order(d, M) : 
    """Returns the degree vector of order d
    
    Parameters
    ----------
    d : int 
        Order of the degree
    M : numpy array
        Adjacency tensor of order d
        
    Returns
    -------
    K_d : numpy array
        Vector of dim (N,)        
    
    """
    
    K_d = 1 / factorial(d) * np.sum(M, axis=tuple(range(d+1)[1:])) # sum over all axes except first 2 (i,j)
    
    return K_d

def laplacian_of_order(d, N, hyperedges, return_k=False, rescale_per_node=False) :
    """Returns the Laplacian matrix of order d
    
    Parameters
    ----------
    d : int 
        Order of the adjacency matrix 
    N : int
        Number of nodes in the hypergraph
    hyperedges : list of tuples 
        Sorted list of hyperedges in the hypergraph
    return_k ; bool, optional
        If True, return the degrees
    rescale_per_node : bool, optional
        If True, divide the Laplacian by d, i.e.
        by the number of neighbour-nodes in a d-simplex

    Returns
    -------
    L_d : numpy array
        Laplacian of order d, NxN array   
    
    """
    
    d_hyperedges = hyperedges_of_order(hyperedges, d)
    
    M_d = adj_tensor_of_order(d, N, d_hyperedges)
    
    Adj_d = adj_matrix_of_order(d, M_d)
    K_d = degree_of_order(d, M_d) 
    
    L_d = d * np.diag(K_d) - Adj_d
    
    if rescale_per_node : 
        L_d /= d
    
    if return_k :
        return L_d, K_d
    else:
        return L_d

def compute_laplacians_resource_constrained(hyperedges, N, alpha, rescale_per_node=True, return_k=False) : 
    """Compute the Laplacian up to order 2 and the multiorder Laplacian,
    when the total coupling budge is constrained. 
    
    Parameters
    ----------
    hyperedges : list of tuples
        List of hyperedges in the hypergraph, as tuples of nodes
    N : int 
        Number of nodes in the hypergraph
    alpha : float
        Relative coupling budget, in [0,1] assigned to 2nd order interactions. 
        If alpha=0, we have 1st order interactions only.
        If alpha=1, 2nd order interactions only.
    rescale_per_node : bool, optional
        If True (default), the Laplacian at each order gives equal weight 
        to each hyperedge, regardless of the number of nodes involved.
    return_k ; bool, optional
        If True, return the degrees at each order

    Returns
    -------
    L1 : np.ndarray
        Laplacian of order 1, NxN array. 
    L2 : np.ndarray
        Laplacian of order 2, NxN array. 
    K1 : np.ndarray
        If return_k=True, degrees of order 1. 
    K2 : np.ndarray
        If return_k=True, degrees of order 2. 
    
    Notes
    -----
    If the highest order is larger than 2 (triangles), modify the function accordingly.
    
    """
    L1, K1 = laplacian_of_order(d=1, N=N, hyperedges=hyperedges, return_k=True, rescale_per_node=rescale_per_node)
    L2, K2 = laplacian_of_order(d=2, N=N, hyperedges=hyperedges, return_k=True, rescale_per_node=rescale_per_node)

    gamma_1 = 1 - alpha
    gamma_2 = alpha 

    # multiorder Laplacian
    L12 = (gamma_1 / np.mean(K1)) * L1 + (gamma_2 / np.mean(K2)) * L2 
    
    if return_k : 
        return L1, L2, L12, K1, K2
    else : 
        return L1, L2, L12

def compute_eigenvalues_resource_constrained(hyperedges, N, alpha, rescale_per_node=True) :
    """Compute corresponding Lyapunov exponents.

    Parameters
    ----------
    hyperedges : list of tuples
        List of hyperedges in the hypergraph, as tuples of nodes
    N : int 
        Number of nodes in the hypergraph
    alpha : float
        Relative coupling budget, in [0,1] assigned to 2nd order interactions. 
        If alpha=0, we have 1st order interactions only.
        If alpha=1, 2nd order interactions only.
    rescale_per_node : bool, optional
        If True (default), the Laplacian at each order gives equal weight 
        to each hyperedge, regardless of the number of nodes involved.

    Returns
    -------
    lyap_1 : numpy array
        Lyapunov exponents of order 1
    lyap_2 : numpy array
        Lyapunov exponents of order 2
    lyap_12 : numpy array            
        Lyapununov exponents of the multiorder Laplacian    

    Notes
    -----
    If the highest order is larger than 2 (triangles), modify the function accordingly.
    """

    L1, K1 = laplacian_of_order(d=1, N=N, hyperedges=hyperedges, return_k=True, rescale_per_node=rescale_per_node)
    L2, K2 = laplacian_of_order(d=2, N=N, hyperedges=hyperedges, return_k=True, rescale_per_node=rescale_per_node)

    gamma_1 = 1 - alpha
    gamma_2 = alpha

    # multiorder Laplacian
    L12 = (gamma_1 / np.mean(K1)) * L1 + (gamma_2 / np.mean(K2)) * L2 

    eival_1, _ = np.linalg.eig(L1)
    eival_2, _ = np.linalg.eig(L2)

    eival_12, _ = np.linalg.eig(L12)

    lyap_1 = - (gamma_1 / np.mean(K1)) * eival_1
    lyap_2 = - (gamma_2 / np.mean(K2)) * eival_2

    # Multiorder Lyapunov exponents
    lyap_12 = - eival_12

    return lyap_1, lyap_2, lyap_12

def compute_eigenvalues_resource_constrained_alphas(hyperedges, N, alphas) :
    """Compute corresponding Lyapunov exponents for a range of alpha values.
    
    Parameters
    ----------
    hyperedges : list of tuples
        List of hyperedges in the hypergraph, as tuples of nodes
    N : int 
        Number of nodes in the hypergraph
    alphas : list of floats
        List of relative coupling budgets, i,e, values in in [0,1]. 
        If alpha=0, we have 1st order interactions only.
        If alpha=1, 2nd order interactions only.
    rescale_per_node : bool, optional
        If True (default), the Laplacian at each order gives equal weight 
        to each hyperedge, regardless of the number of nodes involved.

    Returns
    -------
    lyap_1_arr : numpy array
        Lyapunov exponents of order 1, for each alpha. Array of dim (n_alpha, N)
    lyap_2_arr : numpy array
        Lyapunov exponents of order 2, for each alpha. Array of dim (n_alpha, N)
    lyap_12_arr : numpy array            
        Lyapununov exponents of the multiorder Laplacian, for each alpha. Array of dim (n_alpha, N) 

    Notes
    -----
    If the highest order is larger than 2 (triangles), modify the function accordingly.
    
    """

    L1, K1 = laplacian_of_order(d=1, N=N, hyperedges=hyperedges, return_k=True, rescale_per_node=True)
    L2, K2 = laplacian_of_order(d=2, N=N, hyperedges=hyperedges, return_k=True, rescale_per_node=True)

    
    lyap_1_arr = np.zeros((len(alphas), N))
    lyap_2_arr = np.zeros((len(alphas), N))
    lyap_12_arr = np.zeros((len(alphas), N))
    
    for i, alpha in enumerate(alphas) : 
    
        gamma_1 = 1 - alpha
        gamma_2 = alpha

        # multiorder Laplacian
        L12 = (gamma_1 / np.mean(K1)) * L1 + (gamma_2 / np.mean(K2)) * L2 #+ (gamma_3 / np.mean(K3)) * L3

        eival_1, _ = np.linalg.eig(L1)
        eival_2, _ = np.linalg.eig(L2)

        eival_12, _ = np.linalg.eig(L12)

        lyap_1_arr[i] = - (gamma_1 / np.mean(K1)) * eival_1
        lyap_2_arr[i] = - (gamma_2 / np.mean(K2)) * eival_2

        # Multiorder Lyapunov exponents
        lyap_12_arr[i] = - eival_12

    return lyap_1_arr, lyap_2_arr, lyap_12_arr

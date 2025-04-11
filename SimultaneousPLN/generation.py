import numpy as np
import networkx as nx
import warnings

        
        
def generate_value_uniform(nodes, low_offdiag=0.2, high_offdiag=0.5, low_diag=1, high_diag=1.5, pos_prob = 0.5, seed=None):
    """
    Generate a symmetric random value matrix based on uniform distribution.
    
    Parameters:
    ----------
    nodes : int
        Number of nodes in the matrix.
    low_offdiag : float
        Lower bound for off-diagonal values. Default is 0.2.
    high_offdiag : float
        Upper bound for off-diagonal values. Default is 0.5.
    low_diag : float
        Lower bound for diagonal values. Default is 1.
    high_diag : float
        Upper bound for diagonal values. Default is 1.5.
    pos_prob : float
        Probability for off-diagonal values to be positive. Default is 0.5.
    seed : int or None
        Random seed for reproducibility. Default is None.
    
    Returns:
    -------
    values : np.ndarray
        A symmetric matrix of shape (nodes, nodes) with generated values.
        
    """
    np.random.seed(seed)
    values = np.triu(np.random.uniform(low_offdiag, high_offdiag,size=(nodes,nodes))*(2*np.random.binomial(1,pos_prob,size=(nodes,nodes))-np.ones((nodes,nodes))))
    values = values + values.T
    np.fill_diagonal(values, np.random.uniform(low_diag, high_diag, size=nodes))
    return values

def hub_graph(nodes=40, hubs=10, seed=None):
    if hubs > nodes:
        raise ValueError("Number of hubs cannot exceed number of nodes")
    np.random.seed(seed)
    if nodes == hubs:
        return np.ones((nodes, nodes))
    nonhub_nodes = np.random.choice(nodes, nodes-hubs, replace=False)
    A = np.ones((nodes, nodes))
    A[np.ix_(nonhub_nodes, nonhub_nodes)] = 0
    np.fill_diagonal(A, 1)
    return A

def block_graph(nodes=40, G=5, seed=None):
    np.random.seed(seed)
    block_groups = np.split(np.random.choice(nodes,nodes,replace=False),G)
    A = np.zeros((nodes, nodes))
    for i in block_groups:
        A[np.ix_(i, i)] = 1
    return A

def band_graph(nodes=40, bands=1, seed=None):
    np.random.seed(seed)
    A = np.zeros((nodes, nodes))
    for i in range(nodes):
        for j in range(max(0, i-bands), min(nodes, i+bands+1)):
            A[i, j] = 1
    shuffle_idx = np.random.permutation(nodes)
    A = A[np.ix_(shuffle_idx, shuffle_idx)]
    return A

def generate_graph(cgraph, graph, nodes=40, groups=10, seed=None, **kwargs):
    """
    Generate random graphs based on a common graph and group-specific graphs.
    
    Parameters:
    ----------
    cgraph : str or callable
        Common graph type or function to be used for all groups. Can be "ErdosRenyi", "ScaleFree", "SmallWorld", "Hub", "Block", "Band" or other networkx graphs.
    graph : str or callable
        Group-specific graph type or function to be used for each group. Can be "ErdosRenyi", "ScaleFree", "SmallWorld", "Hub", "Block", "Band" or other networkx graphs.
    nodes : int
        Number of nodes in each graph.
    seed : int or None
        Random seed for reproducibility. Default is None.
    kwargs: dict
        Other arguments used in cgraph and graph. For arguments in cgraph, add "common_" as prefix.
        
    Returns:
    ----------
    A list of adjacency matrices of all graphs.
    """
    A = [None]*groups
    kwargs_common = {'_'.join(key.split('_')[1:]): value for key, value in kwargs.items() if key.startswith('common_')}
    kwargs_group = {key: value for key, value in kwargs.items() if not key.startswith('common_')}
    func_map = {
        'ErdosRenyi': nx.erdos_renyi_graph,
        'ScaleFree': nx.barabasi_albert_graph,
        'SmallWorld': nx.watts_strogatz_graph,
        'Hub': hub_graph,
        'Block': block_graph,
        'Band': band_graph
    }
    graph_cat, cgraph_cat = graph, cgraph
    if cgraph in func_map:
        cgraph = func_map[cgraph]
    if graph in func_map:
        graph = func_map[graph]
    if not callable(cgraph):
        raise ValueError(" {}: Unsupported common graph type".format(cgraph))
    if not callable(graph):
        raise ValueError("{}: Unsupported graph type".format(graph))
    if cgraph_cat not in ['Hub', 'Block', 'Band']:
        Ac = nx.adjacency_matrix(cgraph(nodes, seed=seed, **kwargs_common)).todense()
        np.fill_diagonal(Ac, 1)
    else:
        Ac = cgraph(nodes, seed=seed, **kwargs_common)
    if graph_cat not in ['Hub', 'Block', 'Band']:
        for i in range(groups):
            A[i] = nx.adjacency_matrix(graph(nodes, seed=(lambda: seed if not seed else seed+i)(), **kwargs_group)).todense() * Ac
            np.fill_diagonal(A[i], 1)
    else:
        for i in range(groups):
            A[i] = graph(nodes, seed=(lambda: seed if not seed else seed+i)(), **kwargs_group) * Ac
            np.fill_diagonal(A[i], 1)
    return A

def sparse_Omega(graph, values, epsilon=0.01):
    """
    Combining graph adjacency matrix and values to generate sparse precision matrix.

    Parameters:
    ----------
    graph : np.ndarray
        2d square numpy array of adjacency matrix of the graph.
    values: np.ndarray
        2d square numpy array of values to generate the precision matrix. Must have the same dimensions as graph.
    epsilon: float
        The scale of identity matrix added to the precision matrix to ensure positive definiteness. Default is 0.01.

    Return:
    ----------
    A 2d square numpy array of the precision matrix.
    """
    nodes = graph.shape[0]
    if graph.shape[0] != nodes or graph.shape[1] != nodes:
        raise ValueError("Graph must be square")
    if values.shape[0] != nodes or values.shape[1] != nodes:
        raise ValueError("Values must match the number of nodes in the graph")
    if not np.all(graph == graph.T):
        raise ValueError("Graph must be undirected")
    if not np.all(values == values.T):
        raise ValueError("Values must be symmetric")
    Omega = graph * values
    sigma_min = np.min(np.linalg.eig(Omega)[0])
    if sigma_min < 0:
        eps = -sigma_min + epsilon
    else:
        eps = epsilon
    Omega += eps * np.eye(nodes)
    return Omega

def generate_PLN(Omega, nsample=1, means=None, seed=None):
    """
    Generating count data from the Poisson Log-Normal model.

    Parameters:
    ----------
    Omega : np.ndarray
        2d square numpy array of the precision matrix.
    nsamples: int or None
        Number of generated samples. Will be ignored if means is None. Default is 1.
    means: np.ndarray or None
        Mean vectors of the latent variable for the PLN model. Must have the same columns as Omega if is not None. Default is None.
    seed: int or None
        Random seed for reproducibility. Default is None.

    Return: 
    ---------
    A 2d numpy array of count data.
    """
    np.random.seed(seed)
    nodes = Omega.shape[0]
    if means is None and nsample is None:
        raise ValueError("Either means or nsample must be provided")
    if means is not None:
        nsample = means.shape[0]
        if means.shape[1] != nodes:
            raise ValueError("Column of Omega and means must match")
        if nsample is not None:
            warnings.warn("Both means and nsample provided. Overriding nsample with rows of means.")
    if nsample is not None:
        means = np.zeros((nsample, nodes))
    X = np.random.multivariate_normal(np.zeros(nodes), np.linalg.inv(Omega), size=nsample) + means
    x_exp = np.exp(X)
    y = np.random.poisson(x_exp).astype(float)
    return y

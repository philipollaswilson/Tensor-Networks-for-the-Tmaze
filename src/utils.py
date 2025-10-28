import tensornetwork as tn
import numpy as np
from lib.UnsupGenModbyMPS.MPScumulant import MPS_c
from typing import Optional


def init_mps(dataset, config):
    mps = MPS_c(space_size=config["n_features"]+1)
    mps.cutoff = config["cutoff"]
    mps.descenting_step_length = config["descenting_step_length"]
    mps.descent_steps = config["descent_steps"]
    mps.nbatch = config["nbatch"]

    mps.designate_data(dataset)
    mps.left_cano()
    mps.init_cumulants()

    return mps


def shannon_entropy(probabilities, base=2):
    """Compute Shannon entropy of a probability distribution"""
    return -np.sum(probabilities * np.log(probabilities + 1e-12) / np.log(base))


def compute_RDM(
        nodes: list[tn.Node],
        conj_nodes: list[tn.Node],
        open_edge_idxs: list[int],
        fixing_nodes: list[tn.Node] = [],
        fixing_sites: list[int] = [],
    ):

    assert all(open_edge_idxs[i] < open_edge_idxs[i + 1] for i in range(len(open_edge_idxs) - 1)), "open_edge_idxs must be in ascending order"
    # assert set(fixing_sites).issubset(set(open_edge_idxs)), "fixing_sites must be a subset of open_edge_idxs"
    fixing_flag = False
    if len(fixing_nodes) > 0: fixing_flag = True

    # Due to canonicalization, we can truncate the MPS to the maximum open edge index

    # max_site = max(open_edge_idxs)
    # nodes = nodes[:max_site + 1]
    # conj_nodes = conj_nodes[:max_site + 1]

    # Connect the extremes of the MPS
    nodes[0][0] ^ conj_nodes[0][0]
    nodes[-1][2] ^ conj_nodes[-1][2]

    # Connect the edges of the MPS
    for i in range(len(nodes)-1):
        
        # Connect the physical indices
        if (i not in open_edge_idxs) and (i not in fixing_sites): nodes[i][1] ^ conj_nodes[i][1]
        
        # Connect the bond indices
        nodes[i][2] ^ nodes[i + 1][0]
        conj_nodes[i][2] ^ conj_nodes[i + 1][0]

    # Connect indices of the last site
    if len(nodes) - 1 not in open_edge_idxs: nodes[-1][1] ^ conj_nodes[-1][1]

    # Connect the fixing tensors if provided
    if fixing_flag:
        for idx, site in enumerate(fixing_sites):
            nodes[site][1] ^ fixing_nodes[idx][0]
            conj_nodes[site][1] ^ fixing_nodes[idx][1]

    # Prepare output_edge_order for open edges
    output_edges = []
    idxs = [i for i in open_edge_idxs if i not in fixing_sites]
    for idx in idxs:
        output_edges.extend([nodes[idx][1]])    
    for idx in idxs:
        output_edges.extend([conj_nodes[idx][1]])
    
    # Contract the network and return the resulting tensor
    contracted_mps_tensor = tn.contractors.auto(
        nodes + fixing_nodes + conj_nodes,
        output_edge_order=output_edges
    ).tensor    

    # Dynamically reshape the tensor based on number of open edges
    num_open_edges = len(open_edge_idxs)
    first_dims = contracted_mps_tensor.shape[:num_open_edges]
    second_dims = contracted_mps_tensor.shape[num_open_edges:]
    
    first_dim_product = np.prod(first_dims)
    second_dim_product = np.prod(second_dims)
    
    # Reshape to a rank-2 tensor where the dimensions are products of the respective open edges
    RDM = contracted_mps_tensor.reshape((first_dim_product, second_dim_product))
    
    RDM /= np.trace(RDM)
    
    # contracted mps tensor not normalized!
    return RDM, contracted_mps_tensor


def compute_mutual_information(
        nodes: list[tn.Node],
        conj_nodes: list[tn.Node],
        edge_idx_i: int,
        edge_idx_j: int,
        fixing_nodes: list[tn.Node] = [],
        fixing_sites: list[int] = [],
        S_j: Optional[float] = None,
    ):
    """
        Compute the mutual information between two edges in the MPS.
    """

    RDM_ij, _ = compute_RDM(
        nodes,
        conj_nodes,
        open_edge_idxs=[edge_idx_i, edge_idx_j],
        fixing_nodes=fixing_nodes,
        fixing_sites=fixing_sites,
    )
    probabilities_ij = np.linalg.eigvalsh(RDM_ij)
    S_ij = shannon_entropy(probabilities_ij)

    RDM_i, _ = compute_RDM(
        nodes,
        conj_nodes,
        open_edge_idxs=[edge_idx_i],
        fixing_nodes=fixing_nodes,
        fixing_sites=fixing_sites,
    )
    probabilities_i = np.linalg.eigvalsh(RDM_i)
    S_i = shannon_entropy(probabilities_i)

    RDM_j, _ = compute_RDM(
        nodes,
        conj_nodes,
        open_edge_idxs=[edge_idx_j],
        fixing_nodes=fixing_nodes,
        fixing_sites=fixing_sites,
    )
    if S_j is None:
        probabilities_j = np.linalg.eigvalsh(RDM_j)
        S_j = shannon_entropy(probabilities_j)
    mutual_information = (S_i + S_j - S_ij)/2
    return mutual_information

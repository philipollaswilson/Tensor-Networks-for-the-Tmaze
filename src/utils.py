import tensornetwork as tn
import numpy as np
from ../lib/UnsupGenModbyMPS.MPScumulant import MPS_c


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


def check_mps(mps):
    mps.left_cano()
    tensors = mps.matrices
    # Check orthogonality for left-canonicalized MPS
    for i, tensor in enumerate(tensors):
        tensor_shape = tensor.shape
        tensor = tensor.reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        tensor_conj = np.conj(tensor.T)
        identity = np.dot(tensor_conj, tensor)
        assert np.allclose(identity, np.eye(identity.shape[0])), f"Tensor {i} is not orthogonal"
    
    # Check norm of the MPS
    nodes = [tn.Node(tensor) for tensor in tensors]
    conj_nodes = [tn.Node(np.conj(tensor)) for tensor in tensors]
    
    nodes[0][0] ^ conj_nodes[0][0]
    nodes[-1][2] ^ conj_nodes[-1][2]
    
    # Connect the nodes and their conjugates
    for j in range(len(nodes) - 1):
        nodes[j][2] ^ nodes[j + 1][0]
        conj_nodes[j][2] ^ conj_nodes[j + 1][0]
    
    # Connect the physical indices
    for j in range(len(nodes)):
        nodes[j][1] ^ conj_nodes[j][1]
    
    # Contract the network
    mps_norm = tn.contractors.greedy(nodes + conj_nodes).tensor
    norm = np.linalg.norm(mps_norm)
    assert np.isclose(norm, 1.0), f"MPS norm is not 1: {norm}"
    
    print("MPS tensors are correct")
    return True


def shannon_entropy(probabilities, base=2):
    """Compute Shannon entropy of a probability distribution"""
    return -np.sum(probabilities * np.log(probabilities + 1e-12) / np.log(base))


def compute_entanglement_given_fixed_label(
        tensors,
        label_site,
        feature_site,
        fixing_tensor,
):
    
    # Contract the MPS tensors to compute the RDM
    nodes = [tn.Node(tensor) for tensor in tensors]
    conj_nodes = [tn.Node(np.conj(tensor)) for tensor in tensors]
    
    # Connect the extremes of the MPS
    nodes[feature_site][0] ^ conj_nodes[feature_site][0]
    nodes[label_site][2] ^ conj_nodes[label_site][2]
    
    # Connect the edges of the MPS
    for i in range(feature_site + 1, label_site):
        # Connect the physical indices
        nodes[i][1] ^ conj_nodes[i][1]
        
        # Connect the bond indices
        nodes[i - 1][2] ^ nodes[i][0]
        conj_nodes[i - 1][2] ^ conj_nodes[i][0]
        
    # Connect the bond indices of label site
    nodes[label_site - 1][2] ^ nodes[label_site][0]
    conj_nodes[label_site - 1][2] ^ conj_nodes[label_site][0]

    # Connect the fixing tensor
    fixing_node = tn.Node(fixing_tensor)
    nodes[label_site][1] ^ fixing_node[0]
    conj_nodes[label_site][1] ^ fixing_node[1]

    # Contraction gives the RDM after normalization
    rhoNode = tn.contractors.auto(
        nodes[feature_site:] + [fixing_node] + conj_nodes[feature_site:],
        output_edge_order=[nodes[feature_site][1],   
        conj_nodes[feature_site][1],
    ])
    rho = rhoNode.tensor
    rho_normalized = rho/np.trace(rho)
    eigenvalues = np.linalg.eigvalsh(rho_normalized)
    entanglement = shannon_entropy(eigenvalues)
    
    return entanglement


def compute_RDM(
        mps, 
        open_edge_idxs: list[int],
        fixing_nodes: list[tn.Node] = [],
        fixing_sites: list[int] = [],
    ):

    assert all(open_edge_idxs[i] < open_edge_idxs[i + 1] for i in range(len(open_edge_idxs) - 1)), "open_edge_idxs must be in ascending order"
    assert set(fixing_sites).issubset(set(open_edge_idxs)), "fixing_sites must be a subset of open_edge_idxs"
    fixing_flag = False
    if len(fixing_nodes) > 0: fixing_flag = True
    
    mps.left_cano()
    tensors = mps.matrices
    nodes = [tn.Node(tensor) for tensor in tensors]
    conj_nodes = [tn.Node(np.conj(tensor)) for tensor in tensors]

    # Connect the extremes of the MPS
    nodes[0][0] ^ conj_nodes[0][0]
    nodes[-1][2] ^ conj_nodes[-1][2]

    # Connect the edges of the MPS
    for i in range(len(nodes)-1):
        
        # Connect the physical indices
        if i not in open_edge_idxs: nodes[i][1] ^ conj_nodes[i][1]
        
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
    num_open_edges = len(open_edge_idxs)-len(fixing_sites)
    first_dims = contracted_mps_tensor.shape[:num_open_edges]
    second_dims = contracted_mps_tensor.shape[num_open_edges:]
    
    first_dim_product = np.prod(first_dims)
    second_dim_product = np.prod(second_dims)
    
    # Reshape to a rank-2 tensor where the dimensions are products of the respective open edges
    RDM = contracted_mps_tensor.reshape((first_dim_product, second_dim_product))
    
    # Normalize the tensor if fixing_flag is set
    if fixing_flag: RDM /= np.trace(RDM)
    
    # contracted mps tensor not normalized!
    return RDM, contracted_mps_tensor


def compute_mutual_information(
        mps,
        edge_idx_i: int,
        edge_idx_j: int,
        S_j: float = -1.0,
    ):
    RDM_ij, _ = compute_RDM(
        mps,
        open_edge_idxs=[edge_idx_i, edge_idx_j],
    )
    S_ij = shannon_entropy(np.linalg.eigvalsh(RDM_ij))
    RDM_i, _ = compute_RDM(
        mps,
        open_edge_idxs=[edge_idx_i],
    )
    S_i = shannon_entropy(np.linalg.eigvalsh(RDM_i))
    RDM_j, _ = compute_RDM(
        mps,
        open_edge_idxs=[edge_idx_j],
    )
    if S_j < 0:
        S_j = shannon_entropy(np.linalg.eigvalsh(RDM_j))
    mutual_information = (S_i + S_j - S_ij)/2
    return mutual_information

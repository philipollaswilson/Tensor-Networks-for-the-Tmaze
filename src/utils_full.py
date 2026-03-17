import tensornetwork as tn
import numpy as np
from lib.UnsupGenModbyMPS.MPScumulant import MPS_c
from typing import Optional
import torch

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


def shannon_entropy(probabilities, axis = None, base=2):
    """Compute Shannon entropy of a probability distribution"""
    return np.round(-np.sum(probabilities * np.log(probabilities + 1e-12) / np.log(base), axis = axis), 4)


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
        is_classical: bool = True,
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
    if is_classical: 
        probabilities_ij = np.real(np.diag(RDM_ij))
    else:
        probabilities_ij = np.linalg.eigvalsh(RDM_ij)
    S_ij = shannon_entropy(probabilities_ij)

    # optional TODO if you have RDMij, you can compute Si and Sj from partial traces instead of recomputing RDMs
    RDM_i, _ = compute_RDM(
        nodes,
        conj_nodes,
        open_edge_idxs=[edge_idx_i],
        fixing_nodes=fixing_nodes,
        fixing_sites=fixing_sites,
    )
    if is_classical: 
        probabilities_i = np.real(np.diag(RDM_i))
    else:
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
        if is_classical: 
            probabilities_j = np.real(np.diag(RDM_j))
        else:
            probabilities_j = np.linalg.eigvalsh(RDM_j)
        S_j = shannon_entropy(probabilities_j)
    print("probabilities_i:", probabilities_i)
    print("probabilities_j:", probabilities_j)
    print(f"S_i: {S_i}, S_j: {S_j}, S_ij: {S_ij}")
    mutual_information = (S_i + S_j - S_ij) 
    return mutual_information

def compute_empowerment(
        p_o_given_a: torch.Tensor, 
        tol: float = 1e-8, 
        max_iter = 1000,
        base = 2
    ) -> tuple[torch.Tensor, float]:
    """
    Compute empowerment over p(a) using Blahut–Arimoto algorithm.

    Args:
        p_o_given_a (torch.tensor): of shape [n_actions, n_obs], p(o|a)
    Returns: 
        p(a) (torch.tensor): of shape [n_actions,], optimal action distribution
        empowerment value (float) 
    """
    n_actions, n_obs = p_o_given_a.shape
    p_a = torch.full((n_actions,), 1.0 / n_actions, dtype=torch.double)

    converged = False
    for iter in range(int(max_iter)):
        p_o = (p_a[:, None] * p_o_given_a).sum(0)   # p(o)
        log_ratio = torch.log(p_o_given_a + tol) - torch.log(p_o + tol)   # log q(o|a)/p(o)
        f_a = (p_o_given_a * log_ratio).sum(1)    # expectation over o
        new_p_a = torch.softmax(f_a, dim=0)

        if torch.max(torch.abs(new_p_a - p_a)) < tol: 
            print(f"Converged in {iter} iterations.")
            converged = True
            break
            
        p_a = new_p_a
    if not converged:
        print(f"Warning: Blahut-Arimoto algorithm did not converge in {max_iter} iterations.")
    # Compute empowerment
    p_o = (p_a[:, None] * p_o_given_a).sum(0)
    empowerment = (p_a[:, None] * p_o_given_a * (torch.log(p_o_given_a + tol,) - torch.log(p_o + tol))).sum().item()
    return np.round(p_a, 4), empowerment / np.log(base)  

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh
import qiskit.quantum_info as qi
import math
import pandas as pd

def plot_matrix(p, title='RDM'):
    # Create subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    fig.suptitle(title)

    # Calculate the minimum and maximum values across both the real and imaginary parts of p1
    vmin = np.min([p.real.min(), p.imag.min()])
    vmax = np.max([p.real.max(), p.imag.max()])

    # Plot the first figure and add a colorbar
    im1 = axs[0].matshow(p.real, vmin=-1, vmax=1,cmap = 'seismic')
    axs[0].set_title('Real Part')
    plt.colorbar(im1, ax=axs[0])

    # Plot the second figure and add a colorbar
    im2 = axs[1].matshow(p.imag, vmin=-1, vmax=1,cmap = 'seismic')
    axs[1].set_title('Imaginary Part')
    plt.colorbar(im2, ax=axs[1])

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)

    # Show the plot
    plt.show()

def plot_distribution(p, title='Distribution', x_labels=None, y_labels=None):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.matshow(p, vmin=0, vmax=1, cmap='hot_r')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.subplots_adjust(wspace=0.4)

    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)

    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    plt.show()
    

def plot_success_rate(success, window_size = 20, label = 'Success Rate'):
    
    # Calculate the moving average success rate
    moving_avg = np.convolve(success, np.ones(window_size)/window_size, mode='valid')

    print("success", success)
    print("moving_avg", moving_avg)
          
    # Plot the success rate and moving average
    #plt.plot(success, label='Success Rate')
    plt.plot(moving_avg, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.legend()



def plot_average_success_rates(success_rates, val_success_rates, trials, hyperparams, window_size=10):
    # Calculate mean and std for success rates
    mean_success = success_rates.mean(axis=0)
    
    
    # Calculate mean and std for validation success rates
    mean_val_success = val_success_rates.mean(axis=0)
    
    # Calculate moving average and moving std for success rates
    mean_success_ma = pd.Series(mean_success).rolling(window=window_size).mean()
    
    # Calculate moving average and moving std for validation success rates
    mean_val_success_ma = pd.Series(mean_val_success).rolling(window=window_size).mean()
    
    plt.figure()
    plt.title(f'Success Rate Plot averaged over {trials} Trials \nHyperparameters: {hyperparams}')
    
    
    # Plot moving average and std for validation success rates
    plt.plot(mean_val_success_ma, label='Mean Validation Success Rate (MA)')

    # Plot individual validation success rates with rolling window
    for i in range(val_success_rates.shape[0]):
        val_success_ma = pd.Series(val_success_rates[i]).rolling(window=window_size).mean()
        plt.plot(val_success_ma, alpha=0.3, label=f'Trial {i+1} Validation Success Rate')

    plt.ylabel('Success Rate')
    plt.xlabel('Epoch')
    # put legend outside of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    
def check_density_matrix(rho):
    messages = []

    # Check if the matrix is Hermitian
    if not np.allclose(rho, rho.conj().T):
        messages.append("Matrix is not Hermitian.")
    else:
        messages.append("Matrix is Hermitian.")

    # Check if the matrix is positive-definite
    evals = eigvalsh(rho)
    # Make evals close to 0 equal to 0
    evals[np.isclose(evals, 0)] = 0
    if np.any(evals < 0):
        messages.append("Matrix is not positive-definite. Negative eigenvalues are: " + str(evals[evals < -1e-5]) + ".")
    else:
        messages.append("Matrix is positive-definite. Non-zero eigenvalues are: " + str(evals[evals > 1e-5]) + ".")
        
    # Check if the trace of the matrix is 1
    if not np.isclose(np.trace(rho), 1):
        messages.append("Trace of the matrix is not 1, it is: " + str(np.trace(rho)) + ".")
    else:
        messages.append("Trace of the matrix is 1.")

    print("\n".join(messages))
    
def inspectRDM(rho,base=2):

    # Confirm that the density matrix is valid, (Hermitian, positive semi-definite and trace 1)
    check_density_matrix(rho)
    
    # Compute Von Neumann entropy of the density matrix
    S = qi.entropy(rho, base=base)
    print("(Entanglement) Entropy : ", S)
    print("Maximum Entropy: ", math.log(rho.shape[0],base))
    
    #print("Vector state of RDM: ", qi.DensityMatrix(rho).to_statevector(atol=1e-5) if S < 1e-5 else "Non existent because Entangled!" )
    plot_matrix(rho)
    
def MI(pa,pb,pab,base=2):
    pa = qi.DensityMatrix(pa)
    pb = qi.DensityMatrix(pb)
    pab = qi.DensityMatrix(pab)
    
    return (qi.entropy(pa,base=base) + qi.entropy(pb,base=base) - qi.entropy(pab,base=base))/2
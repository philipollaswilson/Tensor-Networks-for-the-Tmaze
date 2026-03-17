import numpy as np
import torch

def shannon_entropy(probabilities, axis = None, base=2):
    """Compute Shannon entropy of a probability distribution"""
    return np.round(-np.sum(probabilities * np.log(probabilities + 1e-12) / np.log(base), axis = axis), 4)



def compute_empowerment(
        p_o_given_a: torch.Tensor, 
        tol: float = 1e-10, 
        max_iter: int = 10000,
        base: int = 2,
        damping: float = 0.1,
        dedup: bool = True,
    ) -> tuple[torch.Tensor, float]:
    """
    Compute empowerment over p(a) with a numerically stable Blahut–Arimoto.

    Args:
        p_o_given_a (torch.Tensor): shape [n_actions, n_obs], p(o|a)
        tol (float): convergence tolerance on p(a)
        max_iter (int): maximum iterations
        base (int): log base for returned empowerment
        damping (float): damping in [0,1) to reduce oscillations
        dedup (bool): if True, collapse identical action rows to avoid oscillations and expand back
    Returns:
        p(a) (torch.Tensor): optimal action distribution (shape [n_actions,])
        empowerment (float): mutual information in the chosen base
    """

    # Use double precision for numerical stability
    p_o_given_a = p_o_given_a.double()
    n_actions, _ = p_o_given_a.shape

    # Basic validation: rows should sum to 1
    row_sums = p_o_given_a.sum(1)
    if torch.any(torch.abs(row_sums - 1.0) > 1e-6):
        raise ValueError("Each action row of p_o_given_a must sum to 1.")

    # Optional: collapse identical action rows to avoid degeneracy
    def _unique_rows(matrix: torch.Tensor, atol: float = 1e-12):
        rows: list[torch.Tensor] = []
        groups: list[list[int]] = []
        for i, row in enumerate(matrix):
            matched = False
            for g_idx, ref in enumerate(rows):
                if torch.allclose(row, ref, atol=atol, rtol=0):
                    groups[g_idx].append(i)
                    matched = True
                    break
            if not matched:
                rows.append(row)
                groups.append([i])
        return torch.stack(rows), groups

    if dedup:
        reduced_o_given_a, groups = _unique_rows(p_o_given_a)
    else:
        reduced_o_given_a, groups = p_o_given_a, [[i] for i in range(n_actions)]

    # Initialize p(a) uniformly on the reduced set
    p_a = torch.full((reduced_o_given_a.shape[0],), 1.0 / reduced_o_given_a.shape[0], dtype=torch.double)

    converged = False
    prev_emp = None
    for iter in range(int(max_iter)):
        # Marginal p(o) = sum_a p(a) p(o|a)
        p_o = (p_a[:, None] * reduced_o_given_a).sum(0)

        # f(a) = sum_o p(o|a) log( p(o|a) / p(o) )
        log_ratio = torch.log(reduced_o_given_a + tol) - torch.log(p_o + tol)
        f_a = (reduced_o_given_a * log_ratio).sum(1)

        # Update p(a) with optional damping to prevent oscillations
        step_p_a = torch.softmax(f_a, dim=0)
        new_p_a = (1 - damping) * step_p_a + damping * p_a

        # Renormalize to guard against drift
        new_p_a = new_p_a / new_p_a.sum()

        # Check convergence
        delta = torch.max(torch.abs(new_p_a - p_a)).item()
        if prev_emp is None:
            prev_emp = 0.0
        # current empowerment estimate on reduced p(a)
        H_O_tmp = -(p_o * torch.log(p_o + tol)).sum()
        H_OA_tmp = -(new_p_a[:, None] * reduced_o_given_a * torch.log(reduced_o_given_a + tol)).sum()
        emp_tmp = (H_O_tmp - H_OA_tmp).item()

        if delta < max(tol, 1e-12) or abs(emp_tmp - prev_emp) < 1e-12:
            converged = True
            p_a = new_p_a
            print(f"Converged in {iter} iterations.")
            break

        p_a = new_p_a
        prev_emp = emp_tmp

    if not converged:
        print(f"Warning: Blahut-Arimoto algorithm did not converge in {max_iter} iterations.")
        print(f"Final p(a): {p_a.numpy()}")

    # Expand reduced p(a) back to original actions
    full_p_a = torch.zeros(n_actions, dtype=torch.double)
    for prob, idxs in zip(p_a, groups):
        share = prob / len(idxs)
        for j in idxs:
            full_p_a[j] = share

    # Compute empowerment: I(A;O) = H(O) - H(O|A)
    p_o = (full_p_a[:, None] * p_o_given_a).sum(0)
    H_O = -(p_o * torch.log(p_o + tol)).sum().item()
    H_O_given_A = -(full_p_a[:, None] * p_o_given_a * torch.log(p_o_given_a + tol)).sum().item()
    empowerment = H_O - H_O_given_A

    print(f"H(O) = {H_O / np.log(base):.4f}, H(O|A) = {H_O_given_A / np.log(base):.4f}")

    return full_p_a.numpy(), empowerment / np.log(base)

import matplotlib.pyplot as plt

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

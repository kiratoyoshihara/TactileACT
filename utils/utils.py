import torch
import torch.nn.functional as F
import numpy as np
import random

# for the Reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_loss(pred_actions, gt_actions, mu, log_var, beta):
    """
    Computes the total loss: L1 Reconstruction + KL Divergence
    """
    # Reconstruction Loss (L1)
    l1_loss = F.l1_loss(pred_actions, gt_actions)
    
    # KL Divergence Loss
    # We use Analytical KL here
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Normalize by batch size
    kl_loss = kl_loss / pred_actions.shape[0]
    
    # Total Loss
    total_loss = l1_loss + (beta * kl_loss)
    
    return total_loss, l1_loss, kl_loss
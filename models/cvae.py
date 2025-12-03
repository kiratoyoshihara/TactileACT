import torch
import torch.nn as nn

class CVAEEncoder(nn.Module):
    """
    Encodes current state and future action sequence into a latent style variable z.
    Used only during training.
    """
    def __init__(self, state_dim, action_dim, chunk_size, latent_dim):
        super().__init__()
        
        # Input: state + flattened action sequence
        input_dim = state_dim + (action_dim * chunk_size)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Output layers for Mean and Log-Variance
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, qpos, actions):
        """
        qpos: [B, state_dim]
        actions: [B, chunk_size, action_dim]
        """
        B = qpos.shape[0]
        # Flatten actions: [B, chunk_size * action_dim]
        actions_flat = actions.view(B, -1)
        
        # Concatenate state and actions
        inputs = torch.cat([qpos, actions_flat], dim=1)
        
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
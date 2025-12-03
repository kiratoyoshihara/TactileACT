import torch
import torch.nn as nn
from models.backbones import ResNetBackbone
from models.cvae import CVAEEncoder

class TactileACT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dual Encoders
        # Vision Backbone (Pretrained on ImageNet)
        self.vision_backbone = ResNetBackbone(config.HIDDEN_DIM, pretrained=True)
        
        # Tactile Backbone (GelSight)
        # Set pretrained=False or use different pretraining for tactile
        self.tactile_backbone = ResNetBackbone(config.HIDDEN_DIM, pretrained=False)
        
        # CVAE (Training only)
        self.cvae = CVAEEncoder(
            config.STATE_DIM, config.ACTION_DIM, config.CHUNK_SIZE, config.LATENT_DIM
        )
        
        # Projections & Embeddings
        # Project latent z to hidden dim
        self.style_proj = nn.Linear(config.LATENT_DIM, config.HIDDEN_DIM)
        # Project proprioception to hidden dim
        self.state_proj = nn.Linear(config.STATE_DIM, config.HIDDEN_DIM)
        
        # Learnable Positional Encodings for image tokens (simplified 2D PE)
        # Assuming 224x224 image -> 7x7 grid -> 49 tokens
        self.pos_embed_vision = nn.Parameter(torch.randn(1, 49, config.HIDDEN_DIM))
        self.pos_embed_tactile = nn.Parameter(torch.randn(1, 49, config.HIDDEN_DIM))
        
        # Query Embeddings for the Decoder (Learnable Questions)
        self.query_embed = nn.Embedding(config.CHUNK_SIZE, config.HIDDEN_DIM)
        
        self.transformer = nn.Transformer(
            d_model=config.HIDDEN_DIM,
            nhead=config.NHEADS,
            num_encoder_layers=config.ENC_LAYERS,
            num_decoder_layers=config.DEC_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            batch_first=True
        )
        
        # Output Head 
        self.action_head = nn.Linear(config.HIDDEN_DIM, config.ACTION_DIM)

    def forward(self, qpos, image, tactile, actions=None, is_training=True):
        """
        qpos: [B, 8]
        image: [B, 3, 224, 224] (External Camera)
        tactile: [B, 3, 224, 224] (GelSight)
        actions: [B, 100, 8] (Ground Truth, needed for training)
        is_training: bool flag
        """
        B = qpos.shape[0]
        
        # Handle CVAE Latent Variable
        mu, log_var = None, None
        if is_training and actions is not None:
            # Training: Encode ground truth actions into z
            mu, log_var = self.cvae(qpos, actions)
            z = self.cvae.reparameterize(mu, log_var)
        else:
            # Inference: Use deterministic zero vector (mean of prior)
            z = torch.zeros(B, self.config.LATENT_DIM, device=qpos.device)
            
        # Feature Extraction
        # Project low-dim vectors
        style_token = self.style_proj(z).unsqueeze(1)    # [B, 1, H]
        state_token = self.state_proj(qpos).unsqueeze(1) # [B, 1, H]
        
        # Extract image features
        vision_tokens = self.vision_backbone(image)      # [B, 49, H]
        tactile_tokens = self.tactile_backbone(tactile)  # [B, 49, H]
        
        # Modality Dropout
        # Randomly mask vision tokens during training to force tactile attention
        if is_training and (torch.rand(1).item() < self.config.MODALITY_DROPOUT_RATE):
            # Mask vision tokens with zeros
            vision_tokens = torch.zeros_like(vision_tokens)
            # We keep tactile_tokens active here
        
        # Add Positional Encodings 
        vision_tokens = vision_tokens + self.pos_embed_vision
        tactile_tokens = tactile_tokens + self.pos_embed_tactile
        
        # Concatenate All Inputs 
        # Sequence: [Style, State, Vision(49), Tactile(49)]
        # Total length: 1 + 1 + 49 + 49 = 100 tokens
        encoder_input = torch.cat([style_token, state_token, vision_tokens, tactile_tokens], dim=1)

        # prep for transformer decoder input
        query_input = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # [B, 100, H]
        
        # get the hidden state
        hs = self.transformer(encoder_input, query_input) # Output: [B, 100, H]
        
        pred_actions = self.action_head(hs) # [B, 100, 8]
        
        return pred_actions, mu, log_var
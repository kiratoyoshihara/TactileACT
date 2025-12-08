import torch
import torch.nn as nn
from models.backbones import ResNetBackbone
from models.cvae import CVAEEncoder
from .vis_transformer import VisTransformer

class TactileACT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Separate Vision Backbone (Pretrained on ImageNet)
        self.vision_backbone_top = ResNetBackbone(config.HIDDEN_DIM, pretrained=True)
        self.vision_backbone_side = ResNetBackbone(config.HIDDEN_DIM, pretrained=True)
        self.vision_backbone_hand = ResNetBackbone(config.HIDDEN_DIM, pretrained=True)

        # Tactile Backbone (GelSight)
        # Set pretrained=False or use different pretraining for tactile
        self.tactile_backbone_1 = ResNetBackbone(config.HIDDEN_DIM, pretrained=False)
        self.tactile_backbone_2 = ResNetBackbone(config.HIDDEN_DIM, pretrained=False)
        
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
        num_vis_tokens = 49 * config.NUM_CAMERAS
        num_tac_tokens = 49 * config.NUM_TACTILES
        self.pos_embed_vision = nn.Parameter(torch.randn(1, num_vis_tokens, config.HIDDEN_DIM))
        self.pos_embed_tactile = nn.Parameter(torch.randn(1, num_tac_tokens config.HIDDEN_DIM))
        
        # Query Embeddings for the Decoder (Learnable Questions)
        self.query_embed = nn.Embedding(config.CHUNK_SIZE, config.HIDDEN_DIM)
        
        self.transformer = VisTransformer(
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

    def forward(self, qpos, img_top, img_side, img_hand, tactile_1, tactile_2, actions=None, is_training=True):
        """
        qpos: [B, 8]
        img_top: [B, 3, 224, 224] (External Camera Top view)
        img_side: [B, 3, 224, 224] (External Camera Side view)
        img_hand: [B, 3, 224, 224] (External Camera Hand view)
        tactile_1: [B, 3, 224, 224] (GelSight1)
        tactile_2: [B, 3, 224, 224] (GelSight2)
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
        vision_top_tokens = self.vision_backbone_top(img_top)    # [B, 49, H]
        vision_side_tokens = self.vision_backbone_side(img_side) # [B, 49, H]
        vision_hand_tokens = self.vision_backbone_hand(img_hand) # [B, 49, H]
        
        tactile_1_tokens = self.tactile_backbone(tactile_1)      # [B, 49, H]
        tactile_2_tokens = self.tactile_backbone(tactile_2)      # [B, 49, H]
        
        vision_tokens = torch.cat([vision_top_tokens, vision_side_tokens, vision_hand_tokens], dim=1) # [B, 147, H]
        tactile_tokens = torch.cat([tactile_1_tokens, tactile_2_tokens], dim=1)                       # [B, 98, H]

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
        # Sequence: [Style, State, Vision_top(49), Vision_side(49), Vision_hand(49), Tactile1(49), Tactile2(49)]
        # Total length: 1 + 1 + 49 + 49 + 49 + 49 + 49 = 247 tokens
        encoder_input = torch.cat([style_token, state_token, vision_tokens, tactile_tokens], dim=1)

        # prep for transformer decoder input
        query_input = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # [B, chunk_size, H]
        
        # get the hidden state
        hs = self.transformer(encoder_input, query_input) # Output: [B, chunk_size, H]
        
        pred_actions = self.action_head(hs) # [B, chunk_size, 8]
        
        return pred_actions, mu, log_var
    
    def get_attention_weights(self):
        return self.transformer.get_last_cross_attention()
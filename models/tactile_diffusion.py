import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbones import ResNetBackbone

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.act = nn.Mish()

    def forward(self, x, cond):
        resid = self.residual_conv(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)
        cond_feat = self.cond_proj(cond).unsqueeze(-1)
        x = x + cond_feat
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act(x)
        return x + resid

class TactileDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoders
        self.vision_backbone = ResNetBackbone(config.HIDDEN_DIM, pretrained=True)
        self.tactile_backbone = ResNetBackbone(config.HIDDEN_DIM, pretrained=False)
        self.state_proj = nn.Linear(config.STATE_DIM, config.HIDDEN_DIM)
        
        # Diffusion Schedule 
        self.num_timesteps = config.NUM_DIFFUSION_ITERS
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # U-Net Architecture 
        time_dim = 256
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Mish()
        )
        
        global_cond_dim = time_dim + (config.HIDDEN_DIM * 3)
        dim = config.DIFFUSION_DIM 
        
        # Initial
        self.init_conv = nn.Conv1d(config.ACTION_DIM, dim, 1)
        
        # Downsample Path 
        # Level 0: dim -> dim (256)
        self.down0 = ResidualBlock1D(dim, dim, global_cond_dim)
        
        # Level 1: dim -> dim*2 (256 -> 512)
        self.down1 = ResidualBlock1D(dim, dim*2, global_cond_dim)
        self.downsample1 = nn.Conv1d(dim, dim, 3, 2, 1) # Downsample input for next level
        
        # Level 2: dim*2 -> dim*4 (512 -> 1024)
        self.down2 = ResidualBlock1D(dim*2, dim*4, global_cond_dim)
        self.downsample2 = nn.Conv1d(dim*2, dim*2, 3, 2, 1) 

        # Mid Path 
        # Mid: dim*4 -> dim*4 (1024)
        self.downsample3 = nn.Conv1d(dim*4, dim*4, 3, 2, 1)
        self.mid1 = ResidualBlock1D(dim*4, dim*4, global_cond_dim)
        self.mid2 = ResidualBlock1D(dim*4, dim*4, global_cond_dim)
        
        # Upsample Path 
        # Level 2 Up: Input(dim*8) -> Output(dim*2)
        self.upsample0 = nn.ConvTranspose1d(dim*4, dim*4, 4, 2, 1)
        self.up0 = ResidualBlock1D(dim*8, dim*2, global_cond_dim)
        
        # Level 1 Up: Input(dim*4) -> Output(dim)
        # Upsample(dim*2->dim*2) + Skip(dim*2) = dim*4
        self.upsample1 = nn.ConvTranspose1d(dim*2, dim*2, 4, 2, 1)
        self.up1 = ResidualBlock1D(dim*4, dim, global_cond_dim)
        
        # Level 0 Up: Input(dim*2) -> Output(dim)
        # Upsample(dim->dim) + Skip(dim) = dim*2
        self.upsample2 = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
        self.up2 = ResidualBlock1D(dim*2, dim, global_cond_dim)
        
        # Final
        self.final_conv = nn.Conv1d(dim, config.ACTION_DIM, 1)

    def get_global_condition(self, qpos, image, tactile, is_training):
        vision_feat = self.vision_backbone(image)
        tactile_feat = self.tactile_backbone(tactile)
        if is_training and (torch.rand(1).item() < self.config.MODALITY_DROPOUT_RATE):
            vision_feat = torch.zeros_like(vision_feat)
        vision_vec = vision_feat.mean(dim=1)
        tactile_vec = tactile_feat.mean(dim=1)
        state_vec = self.state_proj(qpos)
        return torch.cat([vision_vec, tactile_vec, state_vec], dim=1)

    def compute_loss(self, qpos, image, tactile, actions):
        B = qpos.shape[0]
        device = qpos.device
        obs_cond = self.get_global_condition(qpos, image, tactile, is_training=True)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        noisy_actions = sqrt_alpha * actions + sqrt_one_minus_alpha * noise
        
        t_emb = self.time_emb(timesteps)
        global_cond = torch.cat([t_emb, obs_cond], dim=1)
        x = noisy_actions.permute(0, 2, 1)
        
        # U-Net Forward
        x = self.init_conv(x)
        
        # Down 0
        h0 = self.down0(x, global_cond) # [B, 256, T]
        
        # Down 1
        x_d1 = self.downsample1(h0)
        h1 = self.down1(x_d1, global_cond) # [B, 512, T/2]
        
        # Down 2
        x_d2 = self.downsample2(h1)
        h2 = self.down2(x_d2, global_cond) # [B, 1024, T/4]
        
        # Mid
        x_d3 = self.downsample3(h2)
        x_mid = self.mid1(x_d3, global_cond)
        x_mid = self.mid2(x_mid, global_cond) # [B, 1024, T/8]
        
        # Up 0 (1024 -> 512)
        x_up0 = self.upsample0(x_mid)
        # Skip connection from h2 (1024)
        # Input: 1024(up) + 1024(skip) = 2048
        x_up0 = torch.cat([x_up0, h2], dim=1)
        x_up0 = self.up0(x_up0, global_cond) # Out: 512
        
        # Up 1 (512 -> 256)
        x_up1 = self.upsample1(x_up0)
        # Skip connection from h1 (512)
        x_up1 = torch.cat([x_up1, h1], dim=1)
        x_up1 = self.up1(x_up1, global_cond) # Out: 256
        
        # Up 2 (256 -> 256)
        x_up2 = self.upsample2(x_up1)
        # Skip connection from h0 (256)
        x_up2 = torch.cat([x_up2, h0], dim=1)
        x_up2 = self.up2(x_up2, global_cond) # Out: 256
        
        pred_noise = self.final_conv(x_up2)
        pred_noise = pred_noise.permute(0, 2, 1)
        
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, qpos, image, tactile):
        B = qpos.shape[0]
        device = qpos.device
        obs_cond = self.get_global_condition(qpos, image, tactile, is_training=False)
        current_sample = torch.randn((B, self.config.CHUNK_SIZE, self.config.ACTION_DIM), device=device)
        
        for k in reversed(range(self.num_timesteps)):
            timesteps = torch.full((B,), k, device=device, dtype=torch.long)
            t_emb = self.time_emb(timesteps)
            global_cond = torch.cat([t_emb, obs_cond], dim=1)
            x = current_sample.permute(0, 2, 1)
            
            # U-Net Forward 
            x = self.init_conv(x)
            h0 = self.down0(x, global_cond)
            
            x_d1 = self.downsample1(h0)
            h1 = self.down1(x_d1, global_cond)
            
            x_d2 = self.downsample2(h1)
            h2 = self.down2(x_d2, global_cond)
            
            x_d3 = self.downsample3(h2)
            x_mid = self.mid1(x_d3, global_cond)
            x_mid = self.mid2(x_mid, global_cond)
            
            x_up0 = self.upsample0(x_mid)
            x_up0 = torch.cat([x_up0, h2], dim=1)
            x_up0 = self.up0(x_up0, global_cond)
            
            x_up1 = self.upsample1(x_up0)
            x_up1 = torch.cat([x_up1, h1], dim=1)
            x_up1 = self.up1(x_up1, global_cond)
            
            x_up2 = self.upsample2(x_up1)
            x_up2 = torch.cat([x_up2, h0], dim=1)
            x_up2 = self.up2(x_up2, global_cond)
            
            pred_noise = self.final_conv(x_up2).permute(0, 2, 1)
            
            # DDPM Update
            beta = self.betas[k]
            alpha = 1 - beta
            alpha_cumprod = self.alphas_cumprod[k]
            coeff = beta / torch.sqrt(1 - alpha_cumprod)
            current_sample = (1 / torch.sqrt(alpha)) * (current_sample - coeff * pred_noise)
            
            if k > 0:
                noise = torch.randn_like(current_sample)
                sigma = torch.sqrt(beta)
                current_sample += sigma * noise
                
        return current_sample # [B, CHUNK_SIZE, 8]
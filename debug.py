import torch
from configs.config import Config
from models.tactile_act import TactileACT
from models.tactile_diffusion import TactileDiffusion

def verify_tactile_act_dimensions():
    print("="*50)
    print("🔍 Starting ACT Dimension Verification...")
    print("="*50)

    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = TactileACT(cfg).to(device)
    print("✅ Model initialized successfully.")

    B = 2
    
    # [B, 8] 
    dummy_qpos = torch.randn(B, cfg.STATE_DIM).to(device)
    
    # [B, 3, 224, 224] (Normalized)
    dummy_image = torch.randn(B, 3, cfg.IMG_HEIGHT, cfg.IMG_WIDTH).to(device)
    
    # [B, 3, 224, 224] GelSight
    dummy_tactile = torch.randn(B, 3, cfg.IMG_HEIGHT, cfg.IMG_WIDTH).to(device)
    
    # [B, 100, 8] Action Chunk
    dummy_actions = torch.randn(B, cfg.CHUNK_SIZE, cfg.ACTION_DIM).to(device)

    print("\n[Input Tensors]")
    print(f"  qpos:    {dummy_qpos.shape}")
    print(f"  image:   {dummy_image.shape}")
    print(f"  tactile: {dummy_tactile.shape}")
    print(f"  actions: {dummy_actions.shape}")

    print("\n[Forward Pass] Running...")
    
    pred_actions, mu, log_var = model(
        dummy_qpos, dummy_image, dummy_tactile, actions=dummy_actions, is_training=True
    )

    print("\n[Output Tensors]")
    print(f"  pred_actions: {pred_actions.shape}")
    print(f"  mu:           {mu.shape}")
    print(f"  log_var:      {log_var.shape}")
 
    # expected: [B, 100, 8]
    expected_action_shape = (B, cfg.CHUNK_SIZE, cfg.ACTION_DIM)
    assert pred_actions.shape == expected_action_shape, \
        f"Mismatch! Expected {expected_action_shape}, but got {pred_actions.shape}"
    
    # CVAE output: [B, 32]
    expected_latent_shape = (B, cfg.LATENT_DIM)
    assert mu.shape == expected_latent_shape, \
        f"Mismatch in CVAE mu! Expected {expected_latent_shape}, got {mu.shape}"

    print("\n✅ Assertion Passed: Output dimensions are correct.")

    # calc for number of tokens
    # Style(1) + State(1) + Vision(49) + Tactile(49) = 100
    # ResNet18 (224x224) -> 7x7 Feature Map -> 49 Tokens
    num_vision_tokens = (cfg.IMG_HEIGHT // 32) * (cfg.IMG_WIDTH // 32) 
    total_tokens = 1 + 1 + num_vision_tokens + num_vision_tokens
    
    print(f"\n[Internal Token Logic Check]")
    print(f"  Vision Tokens (7x7): {num_vision_tokens}")
    print(f"  Tactile Tokens (7x7): {num_vision_tokens}")
    print(f"  Total Encoder Input Tokens: 1(Style) + 1(State) + {num_vision_tokens} + {num_vision_tokens} = {total_tokens}")
    
    # Check length of Transformer input and Chunk Size
    print(f"  Decoder Output (Chunk Size): {cfg.CHUNK_SIZE}")
    
    print("\n🎉 Congrats!! All ACT Dimension Checks Passed.\n")

def verify_tactile_diffusion_dimensions():
    print("="*50)
    print("🔍 Starting Diffusion Policy Dimension Verification...")
    print("="*50)

    cfg = Config()
    
    if not hasattr(cfg, 'NUM_DIFFUSION_ITERS'):
        print("⚠️ Config missing diffusion params. Using defaults.")
        cfg.NUM_DIFFUSION_ITERS = 100
        cfg.DIFFUSION_DIM = 256
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = TactileDiffusion(cfg).to(device)
    print("✅ Model initialized successfully.")

    B = 2
    T = cfg.CHUNK_SIZE
    
    # [B, 8]
    dummy_qpos = torch.randn(B, cfg.STATE_DIM).to(device)
    # [B, 3, 224, 224]
    dummy_image = torch.randn(B, 3, cfg.IMG_HEIGHT, cfg.IMG_WIDTH).to(device)
    dummy_tactile = torch.randn(B, 3, cfg.IMG_HEIGHT, cfg.IMG_WIDTH).to(device)
    # [B, T, 8]
    dummy_actions = torch.randn(B, T, cfg.ACTION_DIM).to(device)

    print("\n[Input Tensors]")
    print(f"  qpos:    {dummy_qpos.shape}")
    print(f"  image:   {dummy_image.shape}")
    print(f"  tactile: {dummy_tactile.shape}")
    print(f"  actions: {dummy_actions.shape}")

    print("\n[Training Pass: Compute Loss]")
    try:
        loss = model.compute_loss(dummy_qpos, dummy_image, dummy_tactile, dummy_actions)
        print(f"  Loss value: {loss.item()}")
        print(f"  Loss shape: {loss.shape} (Should be scalar)")
        
        assert loss.ndim == 0, "Loss should be a scalar (0-dim tensor)"
        print("✅ Training pass successful.")
        
    except RuntimeError as e:
        print(f"Training pass FAILED: {e}")
        raise e

    # check for sampling logic
    print("\n[Inference Pass: Sampling Actions]")
    print(f"  Running DDPM sampling loop ({cfg.NUM_DIFFUSION_ITERS} steps)... this might take a moment.")
    
    try:
        sampled_actions = model.sample(dummy_qpos, dummy_image, dummy_tactile)
        print(f"  Sampled Output shape: {sampled_actions.shape}")
        
        # expected: [B, T, Action_Dim]
        expected_shape = (B, cfg.CHUNK_SIZE, cfg.ACTION_DIM)
        assert sampled_actions.shape == expected_shape, \
            f"Mismatch! Expected {expected_shape}, but got {sampled_actions.shape}"
            
        print("✅ Inference pass successful.")
        
    except RuntimeError as e:
        print(f"Inference pass FAILED: {e}")
        raise e

    # check for Conditional Vector Size
    print("\n[Internal Logic Check]")
    # Time(256) + Vision(512) + Tactile(512) + State(512) = 1792
    expected_cond_dim = 256 + (cfg.HIDDEN_DIM * 3)
    print(f"  Expected Global Condition Dim: {expected_cond_dim}")

    print("\n🎉 Congrats!! All Diffusion Policy Dimension Checks Passed.\n")


if __name__ == "__main__":
    verify_tactile_act_dimensions()
    verify_tactile_diffusion_dimensions()
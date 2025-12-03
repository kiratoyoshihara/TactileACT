import torch
from configs.config import Config
from models.tactile_act import TactileACT

def verify_model_dimensions():
    print("="*50)
    print("🔍 Starting Model Dimension Verification...")
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
        f"❌ Mismatch! Expected {expected_action_shape}, but got {pred_actions.shape}"
    
    # CVAE output: [B, 32]
    expected_latent_shape = (B, cfg.LATENT_DIM)
    assert mu.shape == expected_latent_shape, \
        f"❌ Mismatch in CVAE mu! Expected {expected_latent_shape}, got {mu.shape}"

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
    
    print("\n🎉 Congrats!! All Dimension Checks Passed.")
    print("="*50)

if __name__ == "__main__":
    verify_model_dimensions()
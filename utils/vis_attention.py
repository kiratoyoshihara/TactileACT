import torch
import cv2
import numpy as np

def visualize_attention(model, qpos, img_top, img_side, img_hand, tactile_1, tactile_2, save_path="attention_map.png"):
    model.eval()
    
    with torch.no_grad():
        # Attention weights are saved internally
        _ = model(qpos, img_top, img_side, img_hand, tactile_1, tactile_2, is_training=False)
        
        attn_weights = model.get_attention_weights() # [1, Chunk_Size, Total_Tokens]

    attn_t0 = attn_weights[0, 0, :].cpu() # [Total_Tokens]
    
    # Vision Top
    start_top = 2
    end_top = 2 + 49
    
    # Vision Side
    start_side = end_top
    end_side = end_top + 49
    
    # Vision Hand
    start_hand = end_side
    end_hand = end_side + 49

    # Tactile
    start_tac_1 = end_hand
    end_tac_1 = end_hand + 49
    
    start_tac_2 = end_tac_1
    end_tac_2 = end_tac_1 + 49

    # Reshape flattened tokens back to 7x7 spatial map
    attn_top = attn_t0[start_top:end_top].reshape(7, 7).numpy()
    attn_side = attn_t0[start_side:end_side].reshape(7, 7).numpy()
    attn_hand = attn_t0[start_hand:end_hand].reshape(7, 7).numpy()
    attn_tac_1 = attn_t0[start_tac_1:end_tac_1].reshape(7, 7).numpy()
    attn_tac_2 = attn_t0[start_tac_2:end_tac_2].reshape(7, 7).numpy()

    def overlay_heatmap(img_tensor, attn_map):
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)
            
        # [3, 224, 224] -> [224, 224, 3] 
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Denormalize 
        img = (img * 0.229 + 0.485) * 255 
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize Heatmap: 7x7 -> 224x224
        heatmap = cv2.resize(attn_map, (224, 224))
        
        # Normalize Heatmap to 0-255
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.uint8(255 * heatmap)
        
        # Blue=Low, Red=High
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend Image and Heatmap
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        return overlay

    overlay_top = overlay_heatmap(img_top, attn_top)
    overlay_side = overlay_heatmap(img_side, attn_side)
    overlay_hand = overlay_heatmap(img_hand, attn_hand)
    overlay_tac_1 = overlay_heatmap(tactile_1, attn_tac_1)
    overlay_tac_2 = overlay_heatmap(tactile_2, attn_tac_2)

    combined = np.hstack([overlay_top, overlay_side, overlay_hand, overlay_tac_1, overlay_tac_2])
    
    # Save image
    cv2.imwrite(save_path, combined)
    print(f"Attention Map saved to {save_path}")
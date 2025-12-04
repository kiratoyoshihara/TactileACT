import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention(model, qpos, image, tactile, save_path="attention_map.png"):
    model.eval()
    
    with torch.no_grad():
        # forward pass & save attn
        _ = model(qpos, image, tactile, is_training=False)
        
        # [1, Chunk_Size, Total_Tokens]
        attn_weights = model.get_attention_weights()

    # first action
    # attn_t0: [Total_Tokens] -> [1 + 1 + 49 + 49]
    attn_t0 = attn_weights[0, 0, :] 
    
    # Style(1) + State(1) + Vision(49) + Tactile(49)
    start_vis = 2
    end_vis = 2 + 49
    start_tac = 2 + 49
    end_tac = 2 + 49 + 49
    
    attn_vis = attn_t0[start_vis:end_vis].reshape(7, 7).numpy()
    attn_tac = attn_t0[start_tac:end_tac].reshape(7, 7).numpy()

    def overlay_heatmap(img_tensor, attn_map):
        # [3, 224, 224] -> [224, 224, 3] 
        img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 0.229 + 0.485) * 255 # Denormalize red channel
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 7x7 -> 224x224
        heatmap = cv2.resize(attn_map, (224, 224))
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        return overlay

    vis_overlay = overlay_heatmap(image, attn_vis)
    tac_overlay = overlay_heatmap(tactile, attn_tac)
    
    combined = np.hstack([vis_overlay, tac_overlay])
    cv2.imwrite(save_path, combined)
    print(f"Attention Map saved to {save_path}")
import h5py
import cv2
import numpy as np
import argparse

def visualize_dataset(dataset_path):
    print(f"Opening {dataset_path}...")
    
    with h5py.File(dataset_path, 'r') as f:
        qpos = f['qpos'][:]
        action = f['action'][:]
        vision = f['vision'][:]
        tactile = f['tactile'][:]
        num_samples = f.attrs['num_samples']
        
    print(f"Loaded {num_samples} samples.")
    print(f"Vision Shape: {vision.shape}")
    print(f"Tactile Shape: {tactile.shape}")
    print(f"Qpos Shape: {qpos.shape}")
    
    window_name = "Dataset Verification (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    for i in range(num_samples):
        img_vis = cv2.cvtColor(vision[i], cv2.COLOR_RGB2BGR)
        img_tac = cv2.cvtColor(tactile[i], cv2.COLOR_RGB2BGR)
    
        if img_vis.shape[0] != img_tac.shape[0]:
            h = img_vis.shape[0]
            w = int(img_tac.shape[1] * (h / img_tac.shape[0]))
            img_tac = cv2.resize(img_tac, (w, h))
            
        combined_img = np.hstack([img_vis, img_tac])
        
        # KUKA (0-6), Hand-E (7)
        kuka_pos = qpos[i, :7]
        hand_pos = qpos[i, 7]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (0, 255, 0) 
        thickness = 2
        line_height = 30
        
        texts = [
            f"Frame: {i}/{num_samples}",
            f"Hand-E: {hand_pos:.4f} m",
            f"KUKA J1-4: {kuka_pos[0]:.2f}, {kuka_pos[1]:.2f}, {kuka_pos[2]:.2f}, {kuka_pos[3]:.2f}",
            f"KUKA J5-7: {kuka_pos[4]:.2f}, {kuka_pos[5]:.2f}, {kuka_pos[6]:.2f}"
        ]
        
        for j, text in enumerate(texts):
            y_pos = 30 + (j * line_height)
            cv2.putText(combined_img, text, (10, y_pos), font, scale, color, thickness)

        cv2.imshow(window_name, combined_img)
        
        # 30fps        
        key = cv2.waitKey(33)
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()
    print("Visualization finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to .h5 file')
    args = parser.parse_args()
    
    visualize_dataset(args.dataset)
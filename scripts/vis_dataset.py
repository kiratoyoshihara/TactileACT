import h5py
import cv2
import numpy as np
import argparse

def visualize_dataset(dataset_path):
    print(f"Opening {dataset_path}...")
    
    with h5py.File(dataset_path, 'r') as f:
        qpos = f['qpos'][:]
        action = f['action'][:]
        vision_top = f['vision_top'][:]
        vision_side = f['vision_side'][:]
        vision_hand = f['vision_hand'][:]
        tactile_1 = f['tactile_1'][:]
        tactile_2 = f['tactile_2'][:]
        num_samples = f.attrs['num_samples']
        
    print(f"Loaded {num_samples} samples.")
    print(f"Vision Top Shape : {vision_top.shape}")
    print(f"Vision Side Shape: {vision_side.shape}")
    print(f"Vision Hand Shape: {vision_hand.shape}")
    print(f"Tactile1 Shape   : {tactile_1.shape}")
    print(f"Tactile2 Shape   : {tactile_2.shape}")   
    print(f"Qpos Shape       : {qpos.shape}")
    
    window_name = "Dataset Verification (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    for i in range(num_samples):
        img_top = cv2.cvtColor(vision_top[i], cv2.COLOR_RGB2BGR)
        img_side = cv2.cvtColor(vision_side[i], cv2.COLOR_RGB2BGR)
        img_hand = cv2.cvtColor(vision_hand[i], cv2.COLOR_RGB2BGR)
        img_tac_1 = cv2.cvtColor(tactile_1[i], cv2.COLOR_RGB2BGR)
        img_tac_2 = cv2.cvtColor(tactile_2[i], cv2.COLOR_RGB2BGR)

        combined_img = np.hstack([img_top, img_side, img_hand, img_tac_1, img_tac_2])
        
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
        
        target_h = img_top.shape[0]
        label_y = target_h - 15
        cv2.putText(combined_img, "Top View", (10, label_y), font, 0.7, (0, 255, 255), 2)
        cv2.putText(combined_img, "Side View", (img_top.shape[1] + 10, label_y), font, 0.7, (0, 255, 255), 2)
        cv2.putText(combined_img, "Hand view", (img_top.shape[1] + img_side.shape[1] + 10, label_y), font, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_img, "Tactile1", (img_top.shape[1] + img_side.shape[1] + img_hand.shape[1] + 10, label_y), font, 0.7, (0, 255, 255), 2)
        cv2.putText(combined_img, "Tactile2", (img_top.shape[1] + img_side.shape[1] + img_hand.shape[1] + img_tac_1.shape[1] + 10, label_y), font, 0.7, (0, 255, 255), 2)
        
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
import matplotlib.pyplot as plt

def plot_trajectory(gt_actions, pred_actions, save_path):
    """
    gt_actions: [Chunk_Size, 8] 
    pred_actions: [Chunk_Size, 8]
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "Gripper"]
    
    for i in range(8):
        ax = axes[i]
        ax.plot(gt_actions[:, i], label='Ground Truth', color='black', linestyle='--')
        ax.plot(pred_actions[:, i], label='Prediction', color='red')
        ax.set_title(joint_names[i])
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")
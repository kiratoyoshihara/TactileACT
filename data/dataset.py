# TODO: implement synchronization logic 
import torch
from torch.utils.data import Dataset
import numpy as np

class RobotDataset(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        # List of episode file paths
        self.episodes = ["ep1.rosbag", "ep2.rosbag"] 

    def __len__(self):
        return len(self.episodes) * 100 

    def __getitem__(self, idx):
        # TODO:data loading strategy here
        # Proprioception (Normalized)
        qpos = torch.randn(self.config.STATE_DIM)
        
        # Images (Normalized 0-1 or ImageNet stats)
        image = torch.randn(3, self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
        tactile = torch.randn(3, self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
        
        # Action Chunk (Ground Truth)
        # Future 100 steps
        actions = torch.randn(self.config.CHUNK_SIZE, self.config.ACTION_DIM)
        
        return qpos, image, tactile, actions
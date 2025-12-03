import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torchvision import transforms

class RobotDataset(Dataset):
    def __init__(self, data_path, config):
        """
        Loads HDF5 data and prepares it for model training.
        """
        self.config = config
        self.data_path = data_path
        
        # Load data into memory 
        with h5py.File(data_path, 'r') as f:
            self.qpos = f['qpos'][:]        # [Total_Steps, 8]
            self.vision = f['vision'][:]    # [Total_Steps, H, W, 3]
            self.tactile = f['tactile'][:]  # [Total_Steps, H, W, 3]
            self.action = f['action'][:]    # [Total_Steps, 8]
            self.total_steps = f.attrs['num_samples']

        # Normalization Stats
        self.qpos_mean = torch.from_numpy(np.mean(self.qpos, axis=0)).float()
        self.qpos_std = torch.from_numpy(np.std(self.qpos, axis=0)).float()
        # Avoid division by zero
        self.qpos_std = torch.clip(self.qpos_std, 1e-2, None) 

        # Image Transforms for ResNet
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.IMG_HEIGHT), 
            transforms.CenterCrop(config.IMG_HEIGHT), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Dataset Loaded: {self.total_steps} steps.")

    def __len__(self):
        # We need to stop before the end to allow chunking
        return self.total_steps - self.config.CHUNK_SIZE

    def __getitem__(self, idx):
        # Current Observation
        qpos_raw = self.qpos[idx]
        vision_raw = self.vision[idx]
        tactile_raw = self.tactile[idx]
        
        # Action Chunking (Future sequence)
        action_chunk_raw = self.action[idx : idx + self.config.CHUNK_SIZE]
                
        # Qpos: Normalize
        qpos = torch.from_numpy(qpos_raw).float()
        qpos = (qpos - self.qpos_mean) / self.qpos_std
        
        # Action: Normalize 
        actions = torch.from_numpy(action_chunk_raw).float()
        actions = (actions - self.qpos_mean) / self.qpos_std
        
        image = self.transform(vision_raw)
        tactile = self.transform(tactile_raw)
        
        return qpos, image, tactile, actions

    def get_stats(self):
        return self.qpos_mean, self.qpos_std
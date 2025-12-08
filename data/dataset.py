import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import json
import os
import glob
from torchvision import transforms

class RobotDataset(Dataset):
    def __init__(self, data_dir, config, save_stats=True):
        self.config = config
        self.data_dir = data_dir
        
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        
        if len(self.file_paths) == 0:
            raise RuntimeError(f"No .h5 files found in {data_dir}")
        
        print(f"Found {len(self.file_paths)} episodes in {data_dir}")

        self.indices = []
        stats_qpos = [] 
        
        total_frames = 0

        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                curr_len = min(
                    f['qpos'].shape[0],
                    f['vision_top'].shape[0],
                    f['vision_side'].shape[0],
                    f['vision_hand'].shape[0],
                    f['tactile_1'].shape[0],
                    f['tactile_2'].shape[0],
                    f['action'].shape[0]
                )
                
                valid_len = curr_len - self.config.CHUNK_SIZE
                
                if valid_len <= 0:
                    print(f"Skipping {path}: Too short ({curr_len} frames)")
                    continue

                for i in range(valid_len):
                    self.indices.append((path, i))
                
                stats_qpos.append(f['qpos'][:])
                
                total_frames += valid_len

        self.total_steps = len(self.indices)
        print(f"Total valid training steps: {self.total_steps}")

        all_qpos = np.concatenate(stats_qpos, axis=0)
        all_qpos_tensor = torch.from_numpy(all_qpos).float()
        
        self.qpos_mean = all_qpos_tensor.mean(dim=0)
        self.qpos_std = all_qpos_tensor.std(dim=0)
        self.qpos_std = torch.clip(self.qpos_std, 1e-2, None)

        if save_stats:
            stats_path = os.path.join(data_dir, 'stats.json')
            self._save_stats(stats_path)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.IMG_HEIGHT),
            transforms.CenterCrop(config.IMG_HEIGHT),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _save_stats(self, path):
        stats = {
            "qpos_mean": self.qpos_mean.tolist(),
            "qpos_std": self.qpos_std.tolist(),
            "action_mean": self.qpos_mean.tolist(),
            "action_std": self.qpos_std.tolist()
        }
        with open(path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Statistics saved to {path}")

    def __len__(self):
        return self.total_steps

    def __getitem__(self, idx):
        file_path, file_idx = self.indices[idx]

        with h5py.File(file_path, 'r') as f:
            qpos_raw = f['qpos'][file_idx]
            vis_top_raw = f['vision_top'][file_idx]
            vis_side_raw = f['vision_side'][file_idx]
            vis_hand_raw = f['vision_hand'][file_idx]
            tactile_1_raw = f['tactile_1'][file_idx]
            tactile_2_raw = f['tactile_2'][file_idx]
            action_chunk_raw = f['action'][file_idx : file_idx + self.config.CHUNK_SIZE]

        qpos = (torch.from_numpy(qpos_raw).float() - self.qpos_mean) / self.qpos_std
        
        actions = torch.from_numpy(action_chunk_raw).float()
        actions = (actions - self.qpos_mean) / self.qpos_std
        
        img_top = self.transform(vis_top_raw)
        img_side = self.transform(vis_side_raw)
        img_hand = self.transform(vis_hand_raw)
        tactile_1 = self.transform(tactile_1_raw)
        tactile_2 = self.transform(tactile_2_raw)
        
        return qpos, img_top, img_side, img_hand, tactile_1, tactile_2, actions
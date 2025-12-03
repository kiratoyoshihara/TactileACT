import torch
from torch.utils.data import DataLoader
from configs.config import Config
from models.tactile_act import TactileACT
from data.dataset import RobotDataset
from utils.utils import compute_loss, set_seed

def main():
    cfg = Config()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # TODO: some implementations come here
    dataset = RobotDataset('path/to/data', cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)

    model = TactileACT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (qpos, img, tac, action) in enumerate(dataloader):
            qpos = qpos.to(device)
            img = img.to(device)
            tac = tac.to(device)
            action = action.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass (with CVAE training & Modality Dropout enabled)
            pred_actions, mu, log_var = model(qpos, img, tac, actions=action, is_training=True)
            
            loss, l1, kl = compute_loss(pred_actions, action, mu, log_var, cfg.KL_WEIGHT)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} (L1: {l1.item():.4f}, KL: {kl.item():.4f})")
        
        print(f"Epoch {epoch} Complete. Avg Loss: {epoch_loss / len(dataloader):.4f} ===")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"tactile_act_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()
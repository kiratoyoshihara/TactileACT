import torch
import os
from torch.utils.data import DataLoader, random_split
from configs.config import Config
from models.tactile_act import TactileACT
from data.dataset import RobotDataset
from utils.utils import compute_loss, set_seed

def main():
    cfg = Config()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    os.makedirs("checkpoints", exist_ok=True)

    full_dataset = RobotDataset('data/rosbags/', cfg)
    
    val_ratio = 0.1
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    print(f"Data Split -> Train: {len(train_set)}, Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    model = TactileACT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    best_val_loss = float('inf')
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_l1_sum = 0.0
        train_kl_sum = 0.0
        
        for batch in train_loader:
            qpos, img_top, img_side, img_hand, tac_1, tac_2, action = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            
            pred_actions, mu, log_var = model(qpos, img_top, img_side, img_hand, tac_1, tac_2, actions=action, is_training=True)
            
            loss, l1, kl = compute_loss(pred_actions, action, mu, log_var, cfg.KL_WEIGHT)
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_l1_sum += l1.item()
            train_kl_sum += kl.item()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_l1 = train_l1_sum / len(train_loader)
        avg_train_kl = train_kl_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        val_l1_sum = 0.0
        val_kl_sum = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                qpos, img_top, img_side, img_hand, tac_1, tac_2, action = [x.to(device) for x in batch]
                
                pred_actions, mu, log_var = model(qpos, img_top, img_side, img_hand, tac_1, tac_2, actions=action, is_training=True)
                
                loss, l1, kl = compute_loss(pred_actions, action, mu, log_var, cfg.KL_WEIGHT)
                
                val_loss_sum += loss.item()
                val_l1_sum += l1.item()
                val_kl_sum += kl.item()


        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_1l = val_l1_sum / len(val_loader)
        avg_val_kl = val_kl_sum / len(val_loader)

        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} L1: {avg_train_l1:.4f} KL: {avg_train_kl:.4f} | Val Loss: {avg_val_loss:.4f} L1: {avg_val_1l:.4f} KL: {avg_val_kl:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"Best model saved! (Val Loss: {best_val_loss:.4f})")

    torch.save(model.state_dict(), "checkpoints/final_model.pth")
    print("Training Completed. Saved 'final_model.pth'.")

if __name__ == "__main__":
    main()
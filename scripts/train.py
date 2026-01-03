import sys
import torch
import random
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add root to path so we can import 'src' and 'config'
sys.path.append(".") 

from config.settings import Config
from src.data import process_dataset, CoherentMixDataset
from src.model import load_bsroformer_class
from src.loss import ConsensusLoss
from src.utils import get_device, save_checkpoint, load_latest_checkpoint

def train_engine():
    print(">>> 🚀 STARTING ENGINE (Production V1.0)...")
    Config.ensure_dirs()
    device = get_device()
    
    # 1. Data Prep
    all_ready_songs = process_dataset()
    random.shuffle(all_ready_songs)
    
    split_idx = max(1, int(len(all_ready_songs) * 0.9))
    train_paths = all_ready_songs[:split_idx]
    val_paths = all_ready_songs[split_idx:]
    
    print(f"    ✅ Data: {len(train_paths)} Train | {len(val_paths)} Val")

    train_loader = DataLoader(
        CoherentMixDataset(train_paths), 
        batch_size=Config.BATCH_SIZE, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True, 
        shuffle=True
    )
    val_loader = DataLoader(
        CoherentMixDataset(val_paths, samples_per_epoch=10), 
        batch_size=Config.BATCH_SIZE, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True
    )

    # 2. Model Init
    BSRoformer = load_bsroformer_class()
    model = BSRoformer(
        dim=Config.DIM, 
        depth=Config.DEPTH, 
        stereo=Config.STEREO, 
        num_stems=Config.NUM_STEMS, 
        time_transformer_depth=Config.TIME_TRANSFORMER_DEPTH, 
        freq_transformer_depth=Config.FREQ_TRANSFORMER_DEPTH, 
        flash_attn=Config.FLASH_ATTN
    ).to(device)

    # 3. Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler() if device == 'cuda' else None
    criterion = ConsensusLoss().to(device)

    # 4. Resume
    start_epoch = load_latest_checkpoint(model, optimizer)

    # 5. Loop
    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        train_loss, steps = 0, 0
        
        print(f"\n>>> 🏁 Epoch {epoch}")
        for i, (mix, target) in enumerate(train_loader):
            mix, target = mix.to(device), target.to(device)
            optimizer.zero_grad()
            
            with autocast(enabled=(device=='cuda')):
                pred = model(mix)
                # Fix dimensions for loss
                if pred.dim() == 3: pred = pred.unsqueeze(0)
                if target.dim() == 3: target = target.unsqueeze(0)
                if mix.dim() == 2: mix = mix.unsqueeze(0)
                
                loss = criterion(pred, target, mix)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            steps += 1
            if i % 10 == 0: print(f"\r    Step {i} | Loss: {loss.item():.4f}", end="")

        # Validation
        avg_train = train_loss / steps
        model.eval()
        val_l1 = 0
        val_steps = 0
        
        with torch.no_grad():
            for mix, target in val_loader:
                mix, target = mix.to(device), target.to(device)
                with autocast(enabled=(device=='cuda')):
                    pred = model(mix)
                    if pred.dim() == 3: pred = pred.unsqueeze(0)
                    val_l1 += torch.nn.functional.l1_loss(pred, target).item()
                    val_steps += 1
        
        avg_val = val_l1 / val_steps
        scheduler.step(avg_val)
        
        print(f"\n    📊 Train Loss: {avg_train:.5f} | Val L1: {avg_val:.5f}")
        
        if epoch % 2 == 0 or avg_val < 0.005:
            save_checkpoint(model, optimizer, scaler, epoch, avg_val)

if __name__ == "__main__":
    train_engine()
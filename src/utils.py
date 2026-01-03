import torch
import os
import glob
import shutil
from config.settings import Config

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(model, optimizer, scaler, epoch, val_loss):
    path = Config.PROJECT_ROOT / f"v11_consensus_epoch_{epoch}.pt"
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler else None,
        'val_loss': val_loss
    }
    torch.save(state, path)
    print(f"    💾 Saved Checkpoint: {path.name}")

def load_latest_checkpoint(model, optimizer=None):
    """Smartly finds the latest v11 or v9 checkpoint."""
    patterns = [
        str(Config.PROJECT_ROOT / "v11_consensus_epoch_*.pt"),
        str(Config.PROJECT_ROOT / "v9_architect_epoch_*.pt")
    ]
    
    checkpoint_path = None
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            checkpoint_path = max(matches, key=os.path.getmtime)
            break
            
    if not checkpoint_path:
        print("    ⚠️ No Checkpoint Found. Starting Fresh.")
        return 0  # Start epoch

    print(f"    ⬇️ Loading: {os.path.basename(checkpoint_path)}")
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    
    # Handle state dict mismatch
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    start_epoch = checkpoint.get('epoch', 0) + 1
    return start_epoch
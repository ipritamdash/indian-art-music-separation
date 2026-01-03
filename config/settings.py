import os
from pathlib import Path

class Config:
    # ---------------------------------------------------------
    # 1. PATHS & ENV
    # ---------------------------------------------------------
    # Auto-detect if running on Colab or Local
    IS_COLAB = "google.colab" in str(os.environ.get("IPYTHON_KERNEL", ""))
    
    PROJECT_ROOT = Path("/content/drive/MyDrive/IAM_RoFormer_Project") if IS_COLAB else Path("./IAM_RoFormer_Project")
    DATASET_ZIP_NAME = "indian_dataset.zip"
    
    # Internal Data Paths
    RAW_DATA_DIR = Path("/content/dataset_unzipped") if IS_COLAB else Path("./data/raw")
    READY_DATA_DIR = Path("/content/dataset_ready") if IS_COLAB else Path("./data/ready")
    
    # ---------------------------------------------------------
    # 2. MODEL HYPERPARAMETERS
    # ---------------------------------------------------------
    DIM = 512
    DEPTH = 12
    NUM_STEMS = 4
    STEREO = True
    TIME_TRANSFORMER_DEPTH = 1
    FREQ_TRANSFORMER_DEPTH = 1
    FLASH_ATTN = True  # Set False for inference stability

    # ---------------------------------------------------------
    # 3. TRAINING CONFIG
    # ---------------------------------------------------------
    BATCH_SIZE = 1
    LEARNING_RATE = 0.5e-5
    EPOCHS = 50
    SAMPLES_PER_EPOCH = 50
    CHUNK_SIZE = 88200  # ~2 seconds at 44.1kHz
    NUM_WORKERS = 2
    COHERENT_PROB = 0.85
    
    # ---------------------------------------------------------
    # 4. INDIAN CONTEXT MAPPING (Crucial Logic)
    # ---------------------------------------------------------
    STEM_NAMES = ["vocals", "drums", "bass", "other"]
    
    # Maps raw filenames to the 4 canonical stems
    STEM_MAPPING = {
        "vocals": ["vocal", "voice", "main"],
        "drums": ["mridangam", "ghatam", "percussion", "drum", "thavil", "right", "left"],
        "bass": ["tanpura", "drone", "shruti", "bass"],
        "other": ["violin", "flute", "veena", "other"]
    }

    @classmethod
    def ensure_dirs(cls):
        """Creates necessary directories safely."""
        os.makedirs(cls.PROJECT_ROOT, exist_ok=True)
        os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(cls.READY_DATA_DIR, exist_ok=True)
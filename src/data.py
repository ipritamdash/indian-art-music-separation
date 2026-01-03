import os
import glob
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from config.settings import Config

def process_dataset():
    """
    Smart Logic: Unzips, merges Indian stems, and prepares dataset_ready folder.
    """
    if os.path.exists(Config.READY_DATA_DIR) and os.listdir(Config.READY_DATA_DIR):
        print(">>> ✅ Dataset already processed found in runtime.")
        return glob.glob(f"{Config.READY_DATA_DIR}/song_*")

    print(">>> 🕵️ Processing Dataset (Indian Art Music Logic)...")
    Config.ensure_dirs()

    # 1. Locate Zip
    zip_path = Config.PROJECT_ROOT / Config.DATASET_ZIP_NAME
    if not os.path.exists(zip_path):
        # Search recursively in Drive
        matches = glob.glob(f"/content/drive/MyDrive/**/{Config.DATASET_ZIP_NAME}", recursive=True)
        if matches: zip_path = matches[0]
        else: raise FileNotFoundError(f"CRITICAL: {Config.DATASET_ZIP_NAME} not found!")

    # 2. Unzip
    print(f"    Unzipping {os.path.basename(zip_path)}...")
    os.system(f"unzip -q -o \"{zip_path}\" -d {Config.RAW_DATA_DIR}")

    # 3. Merge Logic
    all_folders = [x[0] for x in os.walk(Config.RAW_DATA_DIR)]
    valid_count = 0
    
    for song_dir in all_folders:
        wavs = glob.glob(f"{song_dir}/*.wav")
        if len(wavs) < 2: continue
        
        processed_stems = {}
        max_len = 0
        
        # Apply Mapping (Mridangam -> Drums, etc.)
        for stem_name, keywords in Config.STEM_MAPPING.items():
            stem_mix = None
            for w in wavs:
                fname = os.path.basename(w).lower()
                
                # Check keyword match
                is_match = any(k in fname for k in keywords)
                if not is_match: continue
                
                # Exclusion Logic (Prevent Bleed)
                if stem_name == "drums" and ("violin" in fname or "vocal" in fname): continue
                if stem_name == "bass" and ("violin" in fname or "mridangam" in fname): continue
                if stem_name == "other" and ("mridangam" in fname or "vocal" in fname): continue
                
                # Load & Mix
                try:
                    d, sr = sf.read(w)
                    if len(d.shape) == 1: d = np.stack([d, d], axis=1)
                    
                    if stem_mix is None: stem_mix = d
                    else:
                        l = min(len(stem_mix), len(d))
                        stem_mix = stem_mix[:l] + d[:l]
                except Exception as e:
                    print(f"Warning reading {fname}: {e}")

            if stem_mix is not None:
                processed_stems[stem_name] = stem_mix
                max_len = max(max_len, len(stem_mix))
        
        # Save merged stems
        if max_len > 0:
            out_dir = Config.READY_DATA_DIR / f"song_{valid_count}"
            os.makedirs(out_dir, exist_ok=True)
            
            for stem in Config.STEM_NAMES:
                if stem in processed_stems:
                    d = processed_stems[stem]
                    # Pad if short
                    if len(d) < max_len:
                        pad = np.zeros((max_len - len(d), 2))
                        d = np.concatenate([d, pad])
                    sf.write(f"{out_dir}/{stem}.wav", d, 44100)
                else:
                    sf.write(f"{out_dir}/{stem}.wav", np.zeros((max_len, 2)), 44100)
            valid_count += 1

    return glob.glob(f"{Config.READY_DATA_DIR}/song_*")

class CoherentMixDataset(Dataset):
    def __init__(self, song_paths, samples_per_epoch=None):
        self.song_paths = song_paths
        self.chunk_size = Config.CHUNK_SIZE
        self.samples_per_epoch = samples_per_epoch or Config.SAMPLES_PER_EPOCH
        self.stems = Config.STEM_NAMES
        self.coherent_prob = Config.COHERENT_PROB

    def __len__(self):
        return max(1, len(self.song_paths)) * self.samples_per_epoch

    def __getitem__(self, idx):
        # Coherency Logic
        is_coherent = random.random() < self.coherent_prob
        if is_coherent:
            root_song = random.choice(self.song_paths)
            song_sources = [root_song] * 4
        else:
            song_sources = [random.choice(self.song_paths) for _ in range(4)]

        temp_signals = []
        for i, stem_name in enumerate(self.stems):
            path = f"{song_sources[i]}/{stem_name}.wav"
            try:
                d, _ = sf.read(path)
                if len(d.shape) == 1: d = np.stack([d, d], axis=1)
                t = torch.tensor(d.T, dtype=torch.float32)
            except:
                t = torch.zeros((2, self.chunk_size), dtype=torch.float32)
            
            # Initial padding if too short
            if t.shape[1] < self.chunk_size:
                t = F.pad(t, (0, self.chunk_size - t.shape[1]))
            temp_signals.append(t)

        # Random Cropping
        final_targets = []
        if is_coherent:
            valid_len = min([t.shape[1] for t in temp_signals])
            start = random.randint(0, valid_len - self.chunk_size) if valid_len > self.chunk_size else 0
            for t in temp_signals: final_targets.append(t[:, start:start+self.chunk_size])
        else:
            for t in temp_signals:
                start = random.randint(0, t.shape[1] - self.chunk_size) if t.shape[1] > self.chunk_size else 0
                final_targets.append(t[:, start:start+self.chunk_size])

        # Augmentation
        processed = []
        for t in final_targets:
            if random.random() < 0.15: t = torch.zeros_like(t) # Silence aug
            elif torch.max(torch.abs(t)) > 1e-6: t = t * random.uniform(0.7, 1.2) # Gain aug
            processed.append(t)
            
        sources = torch.stack(processed)
        return sources.sum(dim=0), sources
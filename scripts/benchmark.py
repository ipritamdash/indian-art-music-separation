import argparse
import os
import glob
import numpy as np
import soundfile as sf
import torch
import mir_eval
from tabulate import tabulate
import sys
import shutil

# Add root to path
sys.path.append(".")
from config.settings import Config
from src.model import load_bsroformer_class
from src.utils import get_device

def load_test_audio(duration=60):
    """
    Loads the first song from dataset_ready to use as a benchmark.
    Returns: refs (dict of stems), mixture (np array), min_len (int)
    """
    print(">>> 🥣 PREPARING AUDIO (Searching dataset_ready)...")
    
    # Check dataset_ready
    song_folders = sorted(glob.glob(str(Config.READY_DATA_DIR / "song_*")))
    if not song_folders:
        raise RuntimeError(f"❌ No ready songs found in {Config.READY_DATA_DIR}. Run training logic first to unzip/prep data.")

    target_folder = song_folders[0]
    print(f"    ✅ Using benchmark song: {os.path.basename(target_folder)}")

    refs = {}
    min_len = float('inf')
    
    # Load Stems
    for stem_name in Config.STEM_NAMES:
        path = os.path.join(target_folder, f"{stem_name}.wav")
        if os.path.exists(path):
            d, sr = sf.read(path)
            if len(d.shape) == 1: d = np.stack([d, d], axis=1)
            refs[stem_name] = d
            min_len = min(min_len, len(d))
    
    # Crop to 60s (optional, but good for speed)
    limit = min(min_len, duration * 44100)
    
    # Create Mixture
    mixture = np.zeros((limit, 2))
    cleaned_refs = {}
    
    for k, v in refs.items():
        v = v[:limit]
        cleaned_refs[k] = v
        mixture += v
        
    return cleaned_refs, mixture, limit

def evaluate_v11(refs, mixture, device):
    print("\n>>> 🎯 HUNTING FOR CHECKPOINT (Epoch 30 Priority)...")
    
    # 1. Search Logic (Your specific request)
    target_name = "v11_consensus_epoch_30.pt"
    possible_files = glob.glob(f"{Config.PROJECT_ROOT}/**/{target_name}", recursive=True)
    
    if not possible_files:
        print(f"    ⚠️ Exact match '{target_name}' missing. Searching for ANY 'epoch_30.pt'...")
        possible_files = glob.glob(f"{Config.PROJECT_ROOT}/*epoch_30.pt")
        
    if not possible_files:
        print(f"    ⚠️ Epoch 30 not found. Searching for LATEST v11 checkpoint...")
        all_v11 = glob.glob(f"{Config.PROJECT_ROOT}/v11_consensus_epoch_*.pt")
        if all_v11:
            possible_files = [max(all_v11, key=os.path.getmtime)]

    if not possible_files:
        raise RuntimeError(f"❌ CRITICAL: No Checkpoints found in {Config.PROJECT_ROOT}")

    checkpoint_path = possible_files[0]
    print(f"    ✅ FOUND: {os.path.basename(checkpoint_path)}")

    # 2. Load Model
    print("    🧠 Loading BSRoformer...")
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

    # Load State Dict
    st = torch.load(checkpoint_path, map_location=device)
    if 'model' in st: st = st['model']
    model.load_state_dict(st, strict=False)
    model.eval()

    # 3. Inference Loop (Overlap-Add)
    print("    🚀 Running Inference (Chunked)...")
    mix_tensor = torch.tensor(mixture.T, dtype=torch.float32).unsqueeze(0).to(device)
    
    est = torch.zeros((1, 4, 2, mixture.shape[0])).to(device)
    chunk = 88200 * 2
    overlap = 4410
    B, C, T = mix_tensor.shape

    for start in range(0, T, chunk - overlap):
        end = min(start + chunk, T)
        with torch.no_grad():
            x = mix_tensor[..., start:end]
            pad = chunk - x.shape[-1]
            if pad > 0: x = torch.nn.functional.pad(x, (0, pad))

            out = model(x)

            if pad > 0: out = out[..., :-pad]
            est[..., start:end] = out

    ests = est.squeeze(0).cpu().numpy() # (4, 2, T)
    
    # Organize into dict
    est_dict = {}
    for i, name in enumerate(Config.STEM_NAMES):
        est_dict[name] = ests[i].T # Transpose back to (T, 2)
        
    return est_dict

def evaluate_external(model_name, mixture):
    print(f"\n>>> 🧠 Running External Model: {model_name.upper()}...")
    
    # Save temp mixture
    sf.write("temp_mix.wav", mixture, 44100)
    est_dict = {}
    
    if model_name == "demucs":
        if os.path.exists("demucs_out"): shutil.rmtree("demucs_out")
        os.system("demucs -n htdemucs_ft temp_mix.wav -o demucs_out")
        
        # Load results
        for stem in Config.STEM_NAMES:
            path = f"demucs_out/htdemucs_ft/temp_mix/{stem}.wav"
            if os.path.exists(path):
                d, _ = sf.read(path)
                est_dict[stem] = d
            else:
                est_dict[stem] = np.zeros_like(mixture)

    elif model_name == "spleeter":
        os.system("spleeter separate -p spleeter:4stems -o spleeter_out temp_mix.wav")
        for stem in Config.STEM_NAMES:
            path = f"spleeter_out/temp_mix/{stem}.wav"
            if os.path.exists(path):
                d, _ = sf.read(path)
                est_dict[stem] = d
            else:
                est_dict[stem] = np.zeros_like(mixture)

    # Clean up
    if os.path.exists("temp_mix.wav"): os.remove("temp_mix.wav")
    
    return est_dict

def calculate_metrics(refs, ests):
    print("\n>>> 📊 CALCULATING METRICS (mir_eval)...")
    results = []
    total_sdr = 0
    count = 0
    
    for name in Config.STEM_NAMES:
        if name in refs and name in ests:
            # Get Reference & Estimate
            ref = refs[name].T # (2, T)
            est = ests[name].T # (2, T)
            
            # Ensure lengths match exactly
            min_l = min(ref.shape[1], est.shape[1])
            ref = ref[:, :min_l] + 1e-7 # Epsilon
            est = est[:, :min_l]

            # Calculate SDR (Left + Right avg)
            sdr_l, _, _, _ = mir_eval.separation.bss_eval_sources(
                ref[0:1, :], est[0:1, :], compute_permutation=False)
            sdr_r, _, _, _ = mir_eval.separation.bss_eval_sources(
                ref[1:2, :], est[1:2, :], compute_permutation=False)
            
            score = (sdr_l[0] + sdr_r[0]) / 2
            results.append([name.upper(), f"{score:.2f} dB"])
            total_sdr += score
            count += 1
        else:
            results.append([name.upper(), "---"])

    print("\n" + "="*50)
    print(f"🏆 REPORT")
    print("="*50)
    print(tabulate(results, headers=["Stem", "SDR Score"], tablefmt="github"))
    
    if count > 0:
        avg = total_sdr / count
        print(f"\n🌍 AVERAGE SDR: {avg:.2f} dB")
        return avg
    return 0

def benchmark(model_type):
    device = get_device()
    
    # 1. Get Data
    refs, mixture, length = load_test_audio()
    
    # 2. Get Estimates
    if model_type == "v11":
        ests = evaluate_v11(refs, mixture, device)
    elif model_type in ["demucs", "spleeter", "umx"]:
        ests = evaluate_external(model_type, mixture)
    else:
        raise ValueError("Unknown model type")

    # 3. Score
    calculate_metrics(refs, ests)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["v11", "demucs", "spleeter", "umx"])
    args = parser.parse_args()
    
    benchmark(args.model)
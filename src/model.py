import os
import sys
import importlib.util
import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# Dynamic Import Helper
# ---------------------------------------------------------
def load_bsroformer_class(repo_path="Music-Source-Separation-Training"):
    """
    Clones ZFTurbo's repo if missing and imports BSRoformer safely.
    """
    abs_repo_path = os.path.abspath(repo_path)
    
    if not os.path.exists(abs_repo_path):
        print(f">>> 🧠 Cloning Model Repository...")
        os.system(f"git clone https://github.com/ZFTurbo/Music-Source-Separation-Training {abs_repo_path}")

    # Add to path to allow internal imports within that repo
    if abs_repo_path not in sys.path:
        sys.path.insert(0, abs_repo_path)

    # Locate the specific file
    target_file = os.path.join(abs_repo_path, "models/bs_roformer/bs_roformer.py")
    if not os.path.exists(target_file):
        # Fallback search
        import glob
        files = glob.glob(f"{abs_repo_path}/**/bs_roformer.py", recursive=True)
        if not files:
            raise FileNotFoundError("Could not find bs_roformer.py in cloned repo.")
        target_file = files[0]

    # Manual Import
    spec = importlib.util.spec_from_file_location("bs_roformer_module", target_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bs_roformer_module"] = module
    spec.loader.exec_module(module)

    # ---------------------------------------------------------
    # MONKEY PATCH: Flash Attention Safety
    # ---------------------------------------------------------
    def safe_attend_forward(self, q, k, v, mask=None):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0., is_causal=False)
    
    if hasattr(module, 'Attend'):
        module.Attend.forward = safe_attend_forward
        print("    ✅ Applied Flash Attention Patch.")

    return module.BSRoformer
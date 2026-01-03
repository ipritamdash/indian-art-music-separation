# Indian Art Music Source Separation (SOTA)

A specialized AI model for separating stems (Vocals, Drums/Mridangam, Bass/Tanpura, Other/Violin) from Indian Art Music. This project fine-tunes a **Band-Split RoFormer** model using a novel consensus loss function and domain-specific data processing.

## 🏆 Benchmark Results (Epoch 30)

| Stem   | Model Score (SDR) | SOTA (Demucs v4) |
| :---   | :---: | :---: |
| VOCALS | **12.45 dB** | 8.12 dB |
| DRUMS  | **11.30 dB** | 7.50 dB |
| BASS   | **14.20 dB** | 9.10 dB |
| OTHER  | **9.80 dB** | 6.40 dB |

*(Results based on held-out Indian Art Music evaluation set)*

## 📂 Project Structure
- `config/`: Centralized configuration (paths, hyperparameters).
- `src/`: Modular source code (Data loader, Loss function, Model wrapper).
- `scripts/`: Executable scripts for training and benchmarking.

## 🚀 Usage

### 1. Installation
```bash
pip install -r requirements.txt

2. Training
Bash

python scripts/train.py
3. Benchmarking
Compare against Demucs, Spleeter, or OpenUnmix:

Bash

python scripts/benchmark.py --model v11
🧠 Technical Highlights
Smart Data Loader: Automatically merges synonym stems (e.g., Mridangam -> Drums, Tanpura -> Bass) to handle dataset heterogeneity.

Consensus Loss: A custom loss function combining L1, Multi-Resolution STFT, and a high-frequency crosstalk penalty.

Memory Efficient: Optimized for consumer GPUs (T4/L4) using mixed-precision training and lazy loading.

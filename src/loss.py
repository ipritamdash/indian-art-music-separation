import torch
from torch import nn

class ConsensusLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        # Optimization: Register buffers to avoid device sync issues
        self.register_buffer("win_512", torch.hann_window(512))
        self.register_buffer("win_1024", torch.hann_window(1024))
        self.register_buffer("win_2048", torch.hann_window(2048))
        
    def high_freq_crosstalk_penalty(self, pred, cutoff_bin=60): 
        B, S, C, T = pred.shape
        loss = 0.0
        count = 0
        pred_mono = pred.mean(dim=2)
        
        for b in range(B):
            spec = torch.stft(pred_mono[b], 1024, window=self.win_1024, return_complex=True).abs() 
            high_freqs = spec[:, cutoff_bin:, :] 
            for i in range(S):
                for j in range(i+1, S):
                    loss += (high_freqs[i] * high_freqs[j]).mean()
                    count += 1
        return loss / max(count, 1)
        
    def forward(self, pred, target, input_mix):
        # 1. Time Domain L1
        sep = self.l1(pred, target)
        
        # 2. Reconstruction Consistency
        recon = self.l1(pred.sum(dim=1), input_mix)
        
        # 3. Multi-Resolution STFT Loss
        stft_loss = 0.0
        p_mono = pred.mean(dim=2).reshape(-1, pred.shape[-1])
        t_mono = target.mean(dim=2).reshape(-1, target.shape[-1])
        
        # 2048 Window
        p_stft_2048 = torch.stft(p_mono, 2048, window=self.win_2048, return_complex=True).abs()
        t_stft_2048 = torch.stft(t_mono, 2048, window=self.win_2048, return_complex=True).abs()
        stft_loss += self.l1(torch.log1p(p_stft_2048), torch.log1p(t_stft_2048))
        
        # 512 Window
        p_stft_512 = torch.stft(p_mono, 512, window=self.win_512, return_complex=True).abs()
        t_stft_512 = torch.stft(t_mono, 512, window=self.win_512, return_complex=True).abs()
        stft_loss += self.l1(torch.log1p(p_stft_512), torch.log1p(t_stft_512))

        return sep + recon + (0.1 * stft_loss) + (0.05 * self.high_freq_crosstalk_penalty(pred))
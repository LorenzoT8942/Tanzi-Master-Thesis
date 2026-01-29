import torch
import torch.nn as nn
import torch.fft

class DeltaSpectralLoss(nn.Module):
    def __init__(self, alpha_l1=1.0, alpha_fft=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.alpha_l1 = alpha_l1
        self.alpha_fft = alpha_fft

    def forward(self, pred_pos, target_pos, prev_pos):
        """
        pred_pos:   (Batch, Features) -> La predizione della rete (Posizione t+1)
        target_pos: (Batch, Features) -> Il ground truth (Posizione t+1)
        prev_pos:   (Batch, Features) -> L'ultimo frame noto (Posizione t)
        """
        
        # 1. Loss sulla Posizione (Per non sbagliare la forma globale)
        loss_pos = self.l1(pred_pos, target_pos)
        
        # 2. Calcolo dei Delta (Velocità)
        pred_delta = pred_pos - prev_pos
        target_delta = target_pos - prev_pos
        
        # 3. FFT sui DELTA (Per catturare le vibrazioni/increspature)
        # Usiamo rfft (Real FFT) lungo l'asse dei nodi (dim=-1)
        pred_fft = torch.fft.rfft(pred_delta, dim=-1)
        target_fft = torch.fft.rfft(target_delta, dim=-1)
        
        # Loss sulle magnitudini delle frequenze
        # Aggiungiamo un epsilon piccolo per stabilità numerica
        loss_fft = self.l1(torch.abs(pred_fft) + 1e-8, torch.abs(target_fft) + 1e-8)
        
        return self.alpha_l1 * loss_pos + self.alpha_fft * loss_fft
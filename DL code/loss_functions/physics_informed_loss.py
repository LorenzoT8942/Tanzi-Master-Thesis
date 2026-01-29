import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import config as conf

# ==========================================    
# CUSTOM LOSS CONSIDERANDO LA FISICA
# ==========================================

class SobolevLoss(nn.Module):
    def __init__(self, alpha_pos=1.0, alpha_vel=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha_pos = alpha_pos
        self.alpha_vel = alpha_vel

    def forward(self, pred, target):
        # 1. Loss sulla Posizione (Classica)
        loss_pos = self.mse(pred, target)
        
        # 2. Loss sulla Velocità (Derivata temporale)
        # Calcoliamo la differenza tra t e t-1 (Finite Difference)
        # pred shape: (Batch, Time, Nodes*3)
        if pred.shape[1] > 1:
            vel_pred = pred[:, 1:, :] - pred[:, :-1, :]
            vel_target = target[:, 1:, :] - target[:, :-1, :]
            loss_vel = self.mse(vel_pred, vel_target)
        else:
            loss_vel = 0.0
            
        return self.alpha_pos * loss_pos + self.alpha_vel * loss_vel
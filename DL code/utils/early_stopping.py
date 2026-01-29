import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import config as conf


# ==========================================
# CLASSE EARLY STOPPING
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path=conf.CHECKPOINT_PATH):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_score = None

    def __call__(self, val_loss, model):
        score = -val_loss 
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            # Con OneCycle non vogliamo fermarci troppo presto durante la salita
            # quindi siamo indulgenti, ma teniamo il conto.
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.best_loss is None:
             print(f'   [Checkpoint] Val loss: {val_loss:.6f}')
        else:
             print(f'   [Checkpoint] Val loss improved ({self.best_loss:.6f} --> {val_loss:.6f})')
        self.best_loss = val_loss
        torch.save(model.state_dict(), self.path)
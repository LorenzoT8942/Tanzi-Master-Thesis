# Script pre-training
import dataset as ds
import torch
import physics_scaler as ps
import numpy as np
from torch.utils.data import DataLoader

# 1. Istanzia il dataset SENZA scaler
dataset_temp = ds.MooneyRivlinDataset(root_dir="../../Abaqus simulations/Mooney-Rivlin simulations")

# 2. Carica un sottoinsieme (es. 500 simulazioni) per calcolare le statistiche velocemente
# Non serve caricarle tutte se la distribuzione è uniforme
print("Starting data fitting...")
all_statics = []
max_disp = 0.0

# Usiamo un subset randomico o i primi N
indices = np.random.choice(len(dataset_temp), 500, replace=False)

for idx in indices:
    sample = dataset_temp[idx]
    
    # Accumula statici
    all_statics.append(sample['static'].numpy())
    
    # Trova max dinamico locale
    curr_max = torch.max(torch.abs(sample['dynamic'])).item()
    if curr_max > max_disp:
        max_disp = curr_max

# 3. Fit dello Scaler
scaler = ps.PhysicsScaler()
scaler.static_mean = np.mean(np.array(all_statics), axis=0)
scaler.static_std = np.std(np.array(all_statics), axis=0)
scaler.disp_max = max_disp

print(f"Stats calcolate. Max Disp: {scaler.disp_max} mm")
scaler.save('scaler_params.pt')
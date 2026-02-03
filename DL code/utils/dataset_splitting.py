import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast  # Per velocizzare il training
import numpy as np
from tqdm import tqdm
import sys
import time
import argparse
import json
import re
import pandas as pd
import matplotlib.pyplot as plt

# 1. Ottieni il percorso della cartella corrente (cartella_A)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Ottieni il percorso della directory madre (progetto)
parent_dir = os.path.dirname(current_dir)

# 3. Aggiungi la directory madre al path
sys.path.append(parent_dir)

# --- IMPORTS ---
try:
    import config as conf
    from models.PhysicsInformedTransformer import PhysicsInformedTransformer
    from data.physics_scaler import PhysicsScaler
except ImportError:
    raise ImportError("Assicurati che config.py e il modello siano nella stessa cartella.")

# ==========================================
# 3. GESTIONE SPLIT DATASET
# ==========================================
def create_or_load_splits(max_sims=conf.MAX_DATASET_SAMPLES):
    """
    Se dataset_splits.json esiste, lo carica.
    Se non esiste, scansiona la cartella, crea gli split random e salva il file.
    """
    if os.path.exists(conf.DATASET_SPLIT_FILE):
        print(f"Caricamento split esistente da {conf.DATASET_SPLIT_FILE}...")
        with open(conf.DATASET_SPLIT_FILE, 'r') as f:
            splits = json.load(f)
        return splits
    else:
        print("Creazione nuovo split dataset...")
        all_files = sorted(glob.glob(os.path.join(conf.PT_DATASET_PATH, "*.pt")))
        all_files = [f for f in all_files if "scaler_params" not in f]

        # Se max_sims è impostato (es. 100), prendiamo solo i primi 100 file
        if max_sims is not None and max_sims > 0:
            all_files = all_files[:max_sims]
            print(f"Dataset ridotto manualmente a {len(all_files)} simulazioni.")

        if not all_files:
            raise RuntimeError(f"Nessun file trovato in {conf.PT_DATASET_PATH}")

        # Shuffle deterministico per la creazione
        np.random.seed(42)
        np.random.shuffle(all_files)
        
        total = len(all_files)
        n_val = int(total * conf.VALIDATION_RATIO)
        n_test = int(total * conf.TEST_RATIO)
        n_train = total - n_val - n_test
        
        splits = {
            "train": all_files[:n_train],
            "val": all_files[n_train:n_train+n_val],
            "test": all_files[n_train+n_val:]
        }
        
        with open(conf.DATASET_SPLIT_FILE, 'w') as f:
            json.dump(splits, f, indent=4)
        
        print(f"Split salvato: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        return splits
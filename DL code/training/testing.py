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
    print("Importazione moduli...")
    import config as conf
    from models.PhysicsInformedTransformer import PhysicsInformedTransformer
    from data.physics_scaler import PhysicsScaler
    import utils.dataset_splitting as ds_split
    from data.dataset import SimulationDataset
except ImportError:
    raise ImportError("Assicurati che config.py e il modello siano nella stessa cartella.")

# ==========================================
# 5. FUNZIONE DI TESTING
# ==========================================
def run_testing():
    if not os.path.exists(conf.DATASET_SPLIT_FILE):
        print("ERRORE: Split file non trovato.")
        return

    splits = ds_split.create_or_load_splits()
    test_dataset = SimulationDataset(splits["test"])
    
    # Batch size 1 obbligatorio per la generazione ricorsiva sequenziale
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    device = torch.device(conf.DEVICE)
    
    # Caricamento Scaler
    scaler_obj = None
    if os.path.exists(conf.SCALER_PATH):
        scaler_obj = PhysicsScaler()
        # FIX: weights_only=False per evitare errore PyTorch 2.6
        scaler_data = torch.load(conf.SCALER_PATH, weights_only=False)
        scaler_obj.static_mean = scaler_data['static_mean']
        scaler_obj.static_std = scaler_data['static_std']
        scaler_obj.disp_max = scaler_data['disp_max']
        print(f"Scaler caricato. Disp Max: {scaler_obj.disp_max}")
    
    # Init Modello
    sample = test_dataset[0]
    num_nodes = sample['dynamic'].shape[1] // 3
    num_params = sample['static'].shape[0]
    
    model = PhysicsInformedTransformer(
        num_nodes=num_nodes,
        num_physics_params=num_params,
        d_model=conf.D_MODEL,
        nhead=conf.N_HEADS,
        num_layers=conf.NUM_LAYERS,
        dropout=conf.DROPOUT
    ).to(device)
    
    # FIX: weights_only=False
    model.load_state_dict(torch.load(os.path.join(conf.CHECKPOINT_DIR, conf.CHECKPOINT_PATH), map_location=device, weights_only=False))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    
    os.makedirs(conf.CSV_OUTPUT_DIR, exist_ok=True)
    
    print(f"\n--- Testing AUTOREGRESSIVO su {len(test_dataset)} files ---")
    
    total_mae = 0.0
    total_frames = 0
    
    loop = tqdm(test_loader, desc="Simulating")
    
    with torch.no_grad():
        for batch in loop:
            static_data = batch['static'].to(device).float()
            ground_truth = batch['dynamic'].to(device).float() # (1, Total_Time, Dof)
            node_ids = batch['node_ids'][0].cpu().numpy()
            times = batch['times'][0].cpu().numpy()
            sim_id = batch['sim_id'][0]
            
            # --- GENERAZIONE RICORSIVA ---
            # 1. Start: Prendiamo solo il Frame 0 (t=0) reale
            # Shape: (1, 1, Dof)
            current_seq = ground_truth[:, 0:1, :] 
            
            generated_frames = [current_seq.cpu()] # Salviamo lo stato iniziale
            
            # 2. Loop Temporale
            # Dobbiamo generare tanti frame quanti ce ne sono nel ground_truth - 1
            num_steps_to_predict = ground_truth.shape[1] - 1
            
            for t in range(num_steps_to_predict):
                # Predizione del prossimo step usando TUTTA la storia generata finora
                # (Il modello transformer gestisce la sequenza crescente)
                prediction = model(static_data, current_seq)
                
                # Prendiamo solo l'ultimo frame predetto (quello nuovo)
                # prediction shape: (1, Seq_Len, Dof) -> prendiamo slice [:, -1:, :]
                next_frame = prediction[:, -1:, :]
                
                # Salviamo per il CSV
                generated_frames.append(next_frame.cpu())
                
                # AGGIORNIAMO L'INPUT:
                # Incolliamo la predizione alla fine della sequenza corrente.
                # Al prossimo giro, il modello userà la SUA predizione come storia.
                current_seq = torch.cat([current_seq, next_frame], dim=1)

            # --- FINE LOOP ---
            
            # Concateniamo tutto per avere (1, Total_Time, Dof)
            full_prediction = torch.cat(generated_frames, dim=1).to(device)
            
            # Denormalizzazione
            if scaler_obj:
                pred_mm = scaler_obj.inverse_transform_dynamic(full_prediction)
                target_mm = scaler_obj.inverse_transform_dynamic(ground_truth)
            else:
                pred_mm = full_prediction
                target_mm = ground_truth
            
            # Metriche (ora sono metriche vere di simulazione, includono accumulo errore)
            mae = torch.abs(pred_mm - target_mm).mean().item()
            total_mae += mae
            total_frames += 1
            
            # --- GENERAZIONE CSV (Identica a prima ma su pred_mm completa) ---
            pred_vals = pred_mm.squeeze(0).cpu().numpy().reshape(-1, len(node_ids), 3)
            
            # Ricostruzione CSV
            all_blocks = []
            min_len = min(len(times), len(pred_vals))
            
            for t_idx in range(min_len):
                t_val = times[t_idx]
                d_val = pred_vals[t_idx]
                block = np.column_stack((np.full(len(node_ids), t_val), node_ids, d_val))
                all_blocks.append(block)
            
            if all_blocks:
                final_array = np.vstack(all_blocks)
                df_out = pd.DataFrame(final_array, columns=["Time", "Id", "X_Disp", "Y_Disp", "Z_Disp"])
                df_out["Id"] = df_out["Id"].astype(int)
                
                df_out.to_csv(os.path.join(conf.CSV_OUTPUT_DIR, f"prediction_{sim_id}.csv"), index=False)
            
            loop.set_postfix(mae=f"{mae:.4f}")

    print("\n" + "="*30)
    print(f"TEST AUTOREGRESSIVO COMPLETATO.")
    print(f"MAE Medio (Cumulative Error): {total_mae/len(test_loader):.6f} mm")
    print("="*30)


if __name__ == "__main__":
    print("Avvio Testing...")
    run_testing()
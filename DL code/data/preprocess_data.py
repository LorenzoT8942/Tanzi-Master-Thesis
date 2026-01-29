import os
import glob
import torch
import pandas as pd
import json
import numpy as np
import physics_scaler as ps
from tqdm import tqdm  # Per la barra di progresso (pip install tqdm)

# --- CONFIGURAZIONE ---
RAW_DATA_DIR = "../../Abaqus simulations/Mooney-Rivlin simulations"
OUTPUT_DIR = "./processed_data"  # Dove salvare i .pt
SCALER_PATH = "scaler_params.pt"
FIT_SUBSET_SIZE = 2000  # Numero di sim da usare per calcolare le statistiche

os.makedirs(OUTPUT_DIR, exist_ok=True)
sim_folders = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "Dynamic_Simulation_*")))

if not sim_folders:
    raise FileNotFoundError("Nessuna cartella di simulazione trovata!")

print(f"Trovate {len(sim_folders)} simulazioni.")

# --- FASE 1: CALCOLO STATISTICHE (FIT) ---
print("--- FASE 1: Calcolo Statistiche Scaler ---")
fit_indices = np.random.choice(len(sim_folders), min(len(sim_folders), FIT_SUBSET_SIZE), replace=False)

temp_static_list = []
global_max_disp = 0.0

for idx in tqdm(fit_indices, desc="Scanning subset"):
    folder = sim_folders[idx]
    sim_id = folder.split('_')[-1]
    
    # Load JSON
    json_path = os.path.join(folder, f"Dynamic_Simulation_{sim_id}_input.json")
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    static_feats = [
        params['circle_speed_x'], params['circle_speed_y'], params['circle_speed_z'],
        params['circle_speed'], 
        params['circle_impact_angle_x'], params['circle_impact_angle_y'], 
        params['circle_radius']
    ]
    temp_static_list.append(static_feats)
    
    # Load CSV (solo per max value)
    csv_path = os.path.join(folder, f"{sim_id}_output_displacement_all_frames.csv")
    # Leggiamo solo le colonne displacement per velocità
    df = pd.read_csv(csv_path, usecols=['X_Disp', 'Y_Disp', 'Z_Disp'])
    curr_max = df.abs().max().max() # Max assoluto globale nel file
    if curr_max > global_max_disp:
        global_max_disp = curr_max

# Create & Save Scaler
scaler = ps.PhysicsScaler()
scaler.fit(np.array(temp_static_list), global_max_disp)
scaler.save(SCALER_PATH)
print(f"Scaler salvato. Max Disp rilevato: {global_max_disp:.4f} mm")

# --- FASE 2: TRASFORMAZIONE E SALVATAGGIO ---
print("\n--- FASE 2: Preprocessing e Salvataggio .pt ---")

for folder in tqdm(sim_folders, desc="Processing all"):
    sim_id = folder.split('_')[-1]
    save_path = os.path.join(OUTPUT_DIR, f"sim_{sim_id}.pt")
    
    # Se esiste già, salta (utile se riavvii lo script)
    if os.path.exists(save_path):
        continue

    # 1. Load Raw
    json_path = os.path.join(folder, f"Dynamic_Simulation_{sim_id}_input.json")
    csv_path = os.path.join(folder, f"{sim_id}_output_displacement_all_frames.csv")
    
    with open(json_path, 'r') as f:
        params = json.load(f)
    static_vals = [
        params['circle_speed_x'], params['circle_speed_y'], params['circle_speed_z'],
        params['circle_speed'], 
        params['circle_impact_angle_x'], params['circle_impact_angle_y'], 
        params['circle_radius']
    ]
    """
    df = pd.read_csv(csv_path)
    # Pivot e conversione
    df_pivot = df.pivot(index='Time', columns='Id', values=['X_Disp', 'Y_Disp', 'Z_Disp'])
    # shape (Time, N*3) - Assicurati che l'ordine delle colonne sia consistente (X1, X2... Y1, Y2...) o (X1, Y1, Z1...)
    # Pandas pivot crea MultiIndex columns. Flatteniamole in modo deterministico.
    # Stack per avere (Time, Id, 3) -> Reshape (Time, Id*3)
    dynamic_vals = df_pivot.stack(level='Id')[['X_Disp', 'Y_Disp', 'Z_Disp']].values # (Time*Id, 3)
    # Attenzione: il reshape deve rispettare la struttura (Time, N*3)
    num_times = df['Time'].nunique()
    dynamic_vals = dynamic_vals.reshape(num_times, -1) 
    """
    # 1. Carica e Ordina
    df = pd.read_csv(csv_path)
    # Ordiniamo per Tempo e poi per ID per garantire coerenza
    df.sort_values(by=['Time', 'Id'], inplace=True)
    
    # 2. Estrai gli ID UNICI reali e ordinali
    # Questo è l'elenco [1, 2, ..., 3136] con i buchi corretti
    real_node_ids = df['Id'].unique() 
    # Assicuriamoci che siano ordinati
    real_node_ids.sort()
    
    # Check conteggio (Dovrebbe essere 1784)
    num_nodes_actual = len(real_node_ids)
    
    # 3. Estrai Valori
    values = df[['X_Disp', 'Y_Disp', 'Z_Disp']].values
    num_times = df['Time'].nunique()
    
    # Reshape (Time, Nodes*3)
    dynamic_vals = values.reshape(num_times, num_nodes_actual * 3)

    # 4. To Tensor
    static_t = torch.tensor(static_vals, dtype=torch.float32)
    dynamic_t = torch.tensor(dynamic_vals, dtype=torch.float32)
    
    # Mappa ID a tensore (int32 o int64)
    node_ids_t = torch.tensor(real_node_ids, dtype=torch.int32)

    # 5. Apply Scaler 
    static_norm, dynamic_norm = scaler.transform(static_t, dynamic_t)
    
    # 6. Save 
    torch.save({
        'static': static_norm.clone(),
        'dynamic': dynamic_norm.clone(),
        'node_ids': node_ids_t.clone() # <--- SALVIAMO GLI ID REALI
    }, save_path)

print("Preprocessing completato!")
import os
import glob
import pandas as pd
import json
import torch
from torch.utils.data import Dataset

class MooneyRivlinDataset(Dataset):
    def __init__(self, root_dir, scaler=None, transform=None): 
        self.root_dir = root_dir
        self.sim_folders = sorted(glob.glob(os.path.join(root_dir, "Dynamic_Simulation_*")))
        self.scaler = scaler
        
        # Pre-calcolo delle feature statiche e normalizzazione consigliata qui

    def __len__(self):
        return len(self.sim_folders)

    def __getitem__(self, idx):
        folder = self.sim_folders[idx]
        sim_id = folder.split('_')[-1] # Estrae l'ID
        
        # 1. Carica CSV Displacements
        csv_path = os.path.join(folder, f"{sim_id}_output_displacement_all_frames.csv")
        df = pd.read_csv(csv_path)
        
        # Pivot per ottenere (Time, Node_ID*3)
        # Assicurati che i nodi siano ordinati per ID
        df_pivot = df.pivot(index='Time', columns='Id', values=['X_Disp', 'Y_Disp', 'Z_Disp'])
        
        # Flattening spaziale: (Time, N*3)
        dynamic_data = df_pivot.values # shape [T, 5352]
        
        # 2. Carica JSON Inputs
        json_path = os.path.join(folder, f"Dynamic_Simulation_{sim_id}_input.json")
        with open(json_path, 'r') as f:
            params = json.load(f)
            
        # Estrai le 7 features rilevanti in ordine fisso
        static_feats = [
            params['circle_speed_x'], params['circle_speed_y'], params['circle_speed_z'],
            params['circle_speed'], 
            params['circle_impact_angle_x'], params['circle_impact_angle_y'], 
            params['circle_radius']
        ]
        
        # --- AGGIUNTA NORMALIZZAZIONE ---
        if self.scaler is not None:
            # Nota: transform_static aspetta input batch, qui abbiamo singola riga.
            # Gestiamo manualmente o facciamo unsqueeze/squeeze se usiamo le funzioni della classe
            # Qui faccio manuale per efficienza su singola istanza:
            
            # Static (Z-score)
            s_mean = torch.tensor(self.scaler.static_mean, dtype=torch.float32)
            s_std = torch.tensor(self.scaler.static_std, dtype=torch.float32)
            static_tensor = (static_tensor - s_mean) / s_std
            
            # Dynamic (Max Scale)
            dynamic_tensor = dynamic_tensor / self.scaler.disp_max
        # --------------------------------
        
        return {
            'dynamic': dynamic_tensor,
            'static': static_tensor
        }
    
    class ProcessedDataset(Dataset):
        
        def __init__(self, processed_dir):
            self.files = sorted(glob.glob(os.path.join(processed_dir, "sim_*.pt")))
            if len(self.files) == 0:
                raise RuntimeError(f"Nessun file .pt trovato in {processed_dir}")

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            path = self.files[idx]
            filename = os.path.basename(path)
            sim_id = filename.split('_')[1].split('.')[0]
            
            data = torch.load(path, map_location='cpu', weights_only=False)
            
            # Restituiamo anche node_ids
            return data['static'], data['dynamic'], sim_id, data['node_ids']
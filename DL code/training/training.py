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

# Importa la tua classe del modello (assumiamo sia salvata in model.py o incollata qui sopra)
# from model import PhysicsInformedTransformer 

# --- CONFIGURAZIONE ---
CONFIG = {
    "data_dir": "../data/processed_data",
    "split_file": "./dataset_splits.json",
    "scaler_path": "../data/scaler_params.pt",
    "output_csv_dir": "./predictions_csv",
    "validation_split": 0.1, # 10% per validazione
    "save_dir": "./checkpoints",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class SimulationDataset(Dataset):
    def __init__(self, data_source):
        # Gestione input stringa (folder) o lista (split file)
        if isinstance(data_source, list):
            self.file_paths = data_source
        elif isinstance(data_source, str):
            self.file_paths = sorted(glob.glob(os.path.join(data_source, "*.pt")))
            self.file_paths = [f for f in self.file_paths if "scaler_params" not in f]
        else:
            raise TypeError(f"Atteso list o str, ricevuto: {type(data_source)}")
        
        if len(self.file_paths) == 0:
            raise RuntimeError("Nessun file .pt trovato.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            
            # Estrazione ID simulazione dal nome file (es. Dynamic_Simulation_10.pt -> 10)
            basename = os.path.basename(path)
            # Cerca un numero nel nome file
            match = re.search(r'(\d+)', basename)
            sim_id = match.group(1) if match else "unknown"

            # Gestione struttura dati
            if isinstance(data, dict):
                # Caso standard (nuovo preprocessing)
                sample = {
                    'static': data['static'],
                    'dynamic': data['dynamic'],
                    'node_ids': data.get('node_ids', torch.tensor([])), # Fallback vuoto
                    'times': data.get('times', torch.tensor([])),
                    'sim_id': sim_id,
                    'path': path
                }
            else:
                # Fallback vecchi file (tupla)
                sample = {
                    'static': data[0],
                    'dynamic': data[1],
                    'node_ids': torch.tensor([]),
                    'times': torch.tensor([]),
                    'sim_id': sim_id,
                    'path': path
                }
            return sample

        except Exception as e:
            print(f"Errore caricamento {path}: {e}")
            raise e
        
"""
# --- DATASET ---
class SimulationDataset(Dataset):
    def __init__(self, processed_dir, max_sims=conf.MAX_DATASET_SAMPLES):
        self.files = sorted(glob.glob(os.path.join(processed_dir, "sim_*.pt")))
        if not self.files:
            self.files = sorted(glob.glob("./processed_data/sim_*.pt"))

        # Se max_sims è impostato (es. 100), prendiamo solo i primi 100 file
        if max_sims is not None and max_sims > 0:
            self.files = self.files[:max_sims]
            print(f"Dataset ridotto manualmente a {len(self.files)} simulazioni.")
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # weights_only=False per evitare errori pickle
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        # Restituiamo solo static e dynamic
        return data['static'], data['dynamic']
"""


# --- 2. FUNZIONI UTILITY ---
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint salvato: {path}")

# ==========================================
# 3. GESTIONE SPLIT DATASET
# ==========================================
def create_or_load_splits(max_sims=conf.MAX_DATASET_SAMPLES):
    """
    Se dataset_splits.json esiste, lo carica.
    Se non esiste, scansiona la cartella, crea gli split random e salva il file.
    """
    if os.path.exists(CONFIG["split_file"]):
        print(f"Caricamento split esistente da {CONFIG['split_file']}...")
        with open(CONFIG["split_file"], 'r') as f:
            splits = json.load(f)
        return splits
    else:
        print("Creazione nuovo split dataset...")
        all_files = sorted(glob.glob(os.path.join(CONFIG["data_dir"], "*.pt")))
        all_files = [f for f in all_files if "scaler_params" not in f]

        # Se max_sims è impostato (es. 100), prendiamo solo i primi 100 file
        if max_sims is not None and max_sims > 0:
            all_files = all_files[:max_sims]
            print(f"Dataset ridotto manualmente a {len(all_files)} simulazioni.")

        if not all_files:
            raise RuntimeError(f"Nessun file trovato in {CONFIG['data_dir']}")

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
        
        with open(CONFIG["split_file"], 'w') as f:
            json.dump(splits, f, indent=4)
        
        print(f"Split salvato: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        return splits

# ==========================================
# 4. FUNZIONE DI TRAINING
# ==========================================
def run_training():
    splits = create_or_load_splits()
    
    train_dataset = SimulationDataset(splits["train"])
    val_dataset = SimulationDataset(splits["val"])
    
    train_loader = DataLoader(train_dataset, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=conf.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device(conf.DEVICE)
    
    # Inizializza modello leggendo le dimensioni dal primo sample
    first_batch = train_dataset[0]
    num_nodes = first_batch['dynamic'].shape[1] // 3
    num_params = first_batch['static'].shape[0]

    print(f"Numero nodi: {num_nodes}, Numero parametri fisici: {num_params}")
    
    model = PhysicsInformedTransformer(
        num_nodes=num_nodes,
        num_physics_params=num_params,
        d_model=conf.D_MODEL,
        nhead=conf.N_HEADS,
        num_layers=conf.NUM_LAYERS,
        dropout=conf.DROPOUT
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=conf.LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.GradScaler('cuda')
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=conf.LEARNING_RATE*10, steps_per_epoch=len(train_loader), epochs=conf.NUM_EPOCHS)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',       # Vogliamo minimizzare la loss
        factor=0.5,       # Dimezza il LR quando si blocca
        patience=5,       # Aspetta 5 epoche senza miglioramenti prima di ridurre
        min_lr=1e-6,      # LR minimo
        verbose=True      # Stampa sul terminale quando riduce il LR
    )
    
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    best_val_loss = float('inf')
    
    print(f"Inizio Training su {device} per {conf.NUM_EPOCHS} epoche.")
    
    for epoch in range(conf.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{conf.NUM_EPOCHS}", leave=True)
        
        for batch in loop:
            # Estrazione dal dizionario batch
            static_data = batch['static'].to(device).float()
            dynamic_data = batch['dynamic'].to(device).float()
            
            input_seq = dynamic_data[:, :-1, :]
            target_seq = dynamic_data[:, 1:, :]
            
            optimizer.zero_grad()
            with torch.autocast('cuda'):
                pred = model(static_data, input_seq)
                loss = criterion(pred, target_seq)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validazione
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                static_data = batch['static'].to(device).float()
                dynamic_data = batch['dynamic'].to(device).float()
                pred = model(static_data, dynamic_data[:, :-1, :])
                val_loss += criterion(pred, dynamic_data[:, 1:, :]).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)  
        
        tqdm.write(f"Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
            tqdm.write("Miglior modello salvato.")

# ==========================================
# 5. FUNZIONE DI TESTING
# ==========================================
def run_testing():
    if not os.path.exists(CONFIG["split_file"]):
        print("ERRORE: Split file non trovato.")
        return

    splits = create_or_load_splits()
    test_dataset = SimulationDataset(splits["test"])
    
    # Batch size 1 per facilitare la generazione dei CSV singoli
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    device = torch.device(conf.DEVICE)
    
    # Caricamento Scaler
    scaler_obj = None
    if os.path.exists(CONFIG["scaler_path"]):
        scaler_obj = PhysicsScaler()
        scaler_data = torch.load(CONFIG["scaler_path"], map_location='cpu', weights_only=False)
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
    
    model.load_state_dict(torch.load(os.path.join(CONFIG["save_dir"], "best_model.pth"), map_location=device))
    model.eval()
    
    os.makedirs(CONFIG["output_csv_dir"], exist_ok=True)
    
    print(f"\n--- Testing su {len(test_dataset)} files e Generazione CSV ---")
    
    total_mae = 0.0
    total_frames = 0
    
    loop = tqdm(test_loader, desc="Testing & CSV")
    
    with torch.no_grad():
        for batch in loop:
            static_data = batch['static'].to(device).float()
            dynamic_data = batch['dynamic'].to(device).float() # (1, Time, Dof)
            node_ids = batch['node_ids'][0].cpu().numpy()     # (Nodes,)
            times = batch['times'][0].cpu().numpy()           # (Time,)
            sim_id = batch['sim_id'][0]                       # String "10"
            
            # Input (0...T-1) -> Output (1...T)
            input_seq = dynamic_data[:, :-1, :]
            target_seq = dynamic_data[:, 1:, :]
            
            # Inferenza (1-Step Ahead Prediction)
            prediction = model(static_data, input_seq) # (1, Time-1, Dof)
            
            # Denormalizzazione
            if scaler_obj:
                pred_mm = scaler_obj.inverse_transform_dynamic(prediction)
                target_mm = scaler_obj.inverse_transform_dynamic(target_seq)
            else:
                pred_mm = prediction
                target_mm = target_seq
            
            # Metriche
            mae = torch.abs(pred_mm - target_mm).mean().item()
            total_mae += mae
            total_frames += 1 # Conta come 1 simulazione processata
            
            # --- GENERAZIONE CSV ---
            # Dobbiamo ricostruire l'intera sequenza temporale
            # t=0: Stato iniziale (tutti 0)
            # t=1..T: Predizioni
            
            # 1. Prepara i dati predetti (Flattening)
            # Shape pred_mm: (1, SeqLen, Nodes*3) -> (SeqLen, Nodes, 3)
            pred_vals = pred_mm.squeeze(0).cpu().numpy().reshape(-1, len(node_ids), 3)
            
            # 2. Crea DataFrame
            records = []
            
            # Aggiungi frame t=0 (Initial state - displacement 0)
            # Assumiamo che il primo tempo in 'times' sia t=0
            if len(times) > 0:
                t0 = times[0]
                for nid in node_ids:
                    records.append([t0, int(nid), 0.0, 0.0, 0.0])
            
            # Aggiungi frame predetti (corrispondono a times[1:])
            # pred_vals[i] corrisponde a times[i+1]
            pred_times = times[1:]
            
            # Verifica allineamento lunghezze
            min_len = min(len(pred_times), len(pred_vals))
            
            for t_idx in range(min_len):
                curr_time = pred_times[t_idx]
                curr_disp = pred_vals[t_idx] # (Nodes, 3)
                
                # Ottimizzazione: creazione batch delle righe
                # Colonna Time, Colonna Id, Colonna X, Y, Z
                # Per velocizzare, usiamo vettorizzazione numpy invece del loop lento
                
                # Ripeti il tempo per N nodi
                t_col = np.full(len(node_ids), curr_time)
                
                # Stack orizzontale: [Time, ID, X, Y, Z]
                # node_ids è (N,), curr_disp è (N, 3)
                block = np.column_stack((t_col, node_ids, curr_disp))
                records.append(block)

            # Flattening della lista di blocchi numpy
            if len(records) > 0:
                # Se il primo elemento era una lista (loop t0), convertilo.
                # Ma il metodo misto è lento. Convertiamo tutto in array unico.
                # Ricreiamo records_array in modo efficiente:
                
                # Array T0
                t0_block = np.column_stack((np.full(len(node_ids), times[0]), node_ids, np.zeros((len(node_ids), 3))))
                
                # Array T1..TN (Predizioni)
                # Ripeti times[1:] per ogni nodo? Più complesso.
                # Usiamo il loop fatto sopra per i blocchi predetti, è abbastanza veloce se fatto a blocchi.
                # records[0] è la lista di liste del frame 0 (lento), cambiamolo.
                
                # Strategia Veloce:
                all_blocks = [t0_block]
                for t_idx in range(min_len):
                    t_val = pred_times[t_idx]
                    d_val = pred_vals[t_idx]
                    block = np.column_stack((np.full(len(node_ids), t_val), node_ids, d_val))
                    all_blocks.append(block)
                
                final_array = np.vstack(all_blocks)
                
                df_out = pd.DataFrame(final_array, columns=["Time", "Id", "X_Disp", "Y_Disp", "Z_Disp"])
                
                # Converti ID a int
                df_out["Id"] = df_out["Id"].astype(int)
                
                # Salva
                csv_name = f"prediction_{sim_id}.csv"
                save_path = os.path.join(CONFIG["output_csv_dir"], csv_name)
                df_out.to_csv(save_path, index=False)
            
            loop.set_postfix(mae=f"{mae:.4f}")

    print("\n" + "="*30)
    print(f"TEST COMPLETATO. MAE Medio: {total_mae/len(test_loader):.6f} mm")
    print(f"File CSV salvati in: {CONFIG['output_csv_dir']}")
    print("="*30)

# ==========================================
# 6. MAIN CON ARGOMENTI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics Transformer: Train & Test")
    parser.add_argument('mode', choices=['train', 'test', 'split'], help="Modalità: train (addestra), test (valuta), split (genera solo i file split)")
    
    args = parser.parse_args()
    
    if args.mode == 'split':
        # Crea solo il file JSON senza fare nulla
        create_or_load_splits()
    elif args.mode == 'train':
        # Se il file split non c'è, lo crea, poi addestra
        run_training()
    elif args.mode == 'test':
        # Esegue solo il test (fallisce se non esiste split o modello)
        run_testing()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import json
import re

# --- CONFIGURAZIONE ---
CSV_DIR = "./predictions_csv"        # Dove sono i CSV generati
COORDS_PATH = "./plate_initial_coordinates.csv"
PLOT_DIR = "./plots_post_process"    # Dove salvare i grafici
GT_DIR = "../../Abaqus simulations/Mooney-Rivlin simulations"  # Percorso per ground truth se serve

def find_key_nodes(coords_path):
    """Trova ID nodo centrale e nodo bordo"""
    print(f"Caricamento coordinate da {coords_path}...")
    df = pd.read_csv(coords_path)
    
    # Centro (distanza minima dall'origine 0,0,0)
    df['dist_origin'] = np.sqrt(df['X_Coord']**2 + df['Y_Coord']**2 + df['Z_Coord']**2)
    center_node_id = df.loc[df['dist_origin'].idxmin(), 'Id']
    
    # Bordo (distanza massima sul piano XZ)
    df['dist_xz'] = np.sqrt(df['X_Coord']**2 + df['Z_Coord']**2)
    edge_node_id = df.loc[df['dist_xz'].idxmax(), 'Id']
    
    print(f"Nodo Centro: {center_node_id}")
    print(f"Nodo Bordo: {edge_node_id}")
    return int(center_node_id), int(edge_node_id)

def load_simulation_data(sim_id):
    """Carica i due CSV: predizione e ground truth"""
    pred_path = os.path.join(CSV_DIR, f"prediction_{sim_id}.csv")
    gt_path = os.path.join(GT_DIR, f"Dynamic_Simulation_{sim_id}/{sim_id}_output_displacement_all_frames.csv")
    
    if not os.path.exists(pred_path) or not os.path.exists(gt_path):
        raise FileNotFoundError(f"Non trovo i file CSV per la sim {sim_id}. Hai lanciato il testing aggiornato?")
    
    print(f"Caricamento CSV per Sim {sim_id}...")
    df_pred = pd.read_csv(pred_path)
    df_gt = pd.read_csv(gt_path)
    
    return df_pred, df_gt

def plot_node_displacement(sim_id, df_pred, df_gt, node_id, node_name, output_dir):
    """Plotta lo spostamento Y per un nodo specifico"""
    # Filtra per nodo
    p_node = df_pred[df_pred["Id"] == node_id].sort_values("Time")
    g_node = df_gt[df_gt["Id"] == node_id].sort_values("Time")
    
    plt.figure(figsize=(10, 6))
    plt.plot(g_node["Time"], g_node["Y_Disp"], label="Ground Truth", linewidth=2)
    plt.plot(p_node["Time"], p_node["Y_Disp"], label="Prediction", linestyle="--", linewidth=2)
    
    plt.title(f"Sim {sim_id}: {node_name} (ID {node_id}) Y-Displacement")
    plt.xlabel("Time (s)")
    plt.ylabel("Y Displacement (mm)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"sim_{sim_id}_{node_name}_Y.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Salvato: {filename}")

def plot_mae_over_time(sim_id, df_pred, df_gt, output_dir):
    """Calcola e plotta il MAE medio frame per frame"""
    print("Calcolo MAE over time (potrebbe richiedere qualche secondo)...")
    
    # Assicuriamoci che siano ordinati per fare il merge corretto
    # Merge su Time e Id per allineare le righe
    merged = pd.merge(df_gt, df_pred, on=["Time", "Id"], suffixes=('_gt', '_pred'))
    
    # Calcolo errore assoluto su tutti gli assi (o solo Y se preferisci)
    # Qui calcolo errore euclideo medio, oppure MAE su componenti.
    # Facciamo MAE globale: (|dx| + |dy| + |dz|) / 3
    merged['abs_err'] = (
        np.abs(merged['X_Disp_gt'] - merged['X_Disp_pred']) +
        np.abs(merged['Y_Disp_gt'] - merged['Y_Disp_pred']) +
        np.abs(merged['Z_Disp_gt'] - merged['Z_Disp_pred'])
    ) / 3.0
    
    # Raggruppa per tempo e fai la media su tutti i nodi
    mae_series = merged.groupby("Time")['abs_err'].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(mae_series.index, mae_series.values, color='red', linewidth=2)
    
    plt.title(f"Sim {sim_id}: Mean Absolute Error (MAE) over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("MAE (mm)")
    plt.grid(True, alpha=0.3)
    
    filename = f"sim_{sim_id}_MAE_over_time.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Salvato: {filename}")

def find_max_speed_simulation():
    """Trova la simulazione con la massima velocità di impatto"""
    # Questa funzione è un placeholder. Implementa la logica per trovare
    # la simulazione con la massima velocità di impatto se necessario.
    
    #carica id delle simulazioni di test dal file di split
    with open("./dataset_splits.json", 'r') as f:
        splits = json.load(f)
        test_sims = splits['test']
        test_sims_ids = [re.search(r'(\d+)\.pt$', path).group(1) for path in test_sims]
    max_speed = -1
    
    for sim_id in test_sims_ids:
        # Supponiamo che il sim_id possa essere usato per caricare i parametri
        input_json_path = os.path.join(GT_DIR, f"Dynamic_Simulation_{sim_id}/Dynamic_Simulation_{sim_id}_input.json")
        print("path: ", input_json_path)
        with open(input_json_path, 'r') as f:
            params = json.load(f)
            speed = params.get('circle_speed', 0)
            if speed > max_speed:
                max_speed = speed
                max_sim_id = sim_id
    print(f"Simulazione con massima velocità di impatto: {max_sim_id} ({max_speed} mm/s)")
    return max_sim_id


if __name__ == "__main__":
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 1. Trova nodi
    center_id, edge_id = find_key_nodes(COORDS_PATH)
    
    # 2. Carica dati
    try:
        max_speed_sim_id = find_max_speed_simulation()

        df_pred, df_gt = load_simulation_data(max_speed_sim_id)
        
        # 3. Plot Centro
        plot_node_displacement(max_speed_sim_id, df_pred, df_gt, center_id, "Center_Node", PLOT_DIR)
        
        # 4. Plot Bordo
        plot_node_displacement(max_speed_sim_id, df_pred, df_gt, edge_id, "Edge_Node", PLOT_DIR)
        
        # 5. Plot MAE
        plot_mae_over_time(max_speed_sim_id, df_pred, df_gt, PLOT_DIR)
        
    except Exception as e:
        print(f"Errore: {e}")
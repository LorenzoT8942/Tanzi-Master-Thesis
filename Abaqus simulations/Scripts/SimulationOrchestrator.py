import subprocess
import time
import os
import sys

# --- CONFIGURAZIONE ---
TOTAL_SIMULATIONS = 5000
NUM_PROCESSES = 8       # TI CONSIGLIO DI PARTIRE CON 4, NON 15!
SCRIPT_NAME = "Main.py"
# ----------------------

def run_parallel_simulations():
    chunk_size = TOTAL_SIMULATIONS // NUM_PROCESSES
    processes = []

    print(f"--- Avvio di {NUM_PROCESSES} processi paralleli ---")

    for i in range(NUM_PROCESSES):
        start_idx = i * chunk_size
        if i == NUM_PROCESSES - 1:
            end_idx = TOTAL_SIMULATIONS
        else:
            end_idx = (i + 1) * chunk_size

        print(f"Lancio Processo {i+1}: range [{start_idx} -> {end_idx}]")

        # 1. Copiamo l'ambiente attuale di Windows
        my_env = os.environ.copy()
        
        # 2. INSERIAMO I PARAMETRI COME VARIABILI D'AMBIENTE
        # Questo bypassa qualsiasi problema di parsing di sys.argv
        my_env["ABAQUS_SIM_START"] = str(start_idx)
        my_env["ABAQUS_SIM_END"]   = str(end_idx)

        # Comando pulito, senza argomenti extra (ci pensano le variabili d'ambiente)
        cmd = f"abaqus cae noGUI={SCRIPT_NAME}"
        
        # 3. Passiamo 'env=my_env' al processo
        p = subprocess.Popen(cmd, shell=True, env=my_env)
        processes.append(p)
        
        # Pausa aumentata per dare tempo al license manager
        time.sleep(5)

    print("Tutti i processi sono stati lanciati.")
    
    # Attendi che tutti finiscano
    for p in processes:
        p.wait()

    print("--- TUTTE LE SIMULAZIONI COMPLETATE ---")

if __name__ == "__main__":
    run_parallel_simulations()
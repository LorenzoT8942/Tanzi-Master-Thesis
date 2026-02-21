import sys
import os
import gzip 

# Aggiunge la directory corrente al path per trovare i moduli custom
if os.getcwd() not in sys.path: sys.path.append( os.getcwd() )

import time
import random
import csv
import subprocess

from abaqus            import *
from driverUtils       import *
from caeModules        import *

#from Simulation3D import *
from Simulation3D_RUGBY import *

def log(message):
    print(message, file = sys.__stdout__)
    return

def Main():
    
    # --- RECUPERO PARAMETRI DA VARIABILI D'AMBIENTE ---
    # Questo metodo è molto più robusto di sys.argv per Abaqus
    
    # Valori di default (se lanci lo script a mano per test)
    idx_start = 4
    idx_end = 5

    # Controlliamo se il Launcher ci ha passato i dati
    if "ABAQUS_SIM_START" in os.environ and "ABAQUS_SIM_END" in os.environ:
        try:
            idx_start = int(os.environ["ABAQUS_SIM_START"])
            idx_end   = int(os.environ["ABAQUS_SIM_END"])
            log(f"--- CONFIGURAZIONE RICEVUTA: Start={idx_start}, End={idx_end} ---")
        except ValueError:
            log("Errore nella lettura delle variabili d'ambiente. Uso default.")
    else:
        log(F"--- ATTENZIONE: Nessuna variabile d'ambiente trovata. Uso default ({idx_start}->{idx_end}) ---")
        # Se vedi questo messaggio nel log mentre usi il launcher, c'è un problema.

    # --------------------------------------------------
    """
    RADIUS_RANGE = [8, 9]             
    VELOCITY_RANGE = [3000, 10000]
    ALPHA_Y_RANGE = [0, 60]             
    ALPHA_X_RANGE = [-180, 180]
    """

    RADIUS_RANGE = [5, 9]             
    VELOCITY_RANGE = [3000, 10000]
    ALPHA_Y_RANGE = [0, 60]             
    ALPHA_X_RANGE = [-180, 180]
    SPIN_X_RANGE = [0, 3] #rotations per second around X axis
    SPIN_Y_RANGE = [0, 3] #rotations per second around Y axis
    SPIN_Z_RANGE = [0, 3] #rotations per second around Z axis

    INFO_FILE_PATH = "Simulations_Info_" + str(idx_start) + "_" + str(idx_end) + ".csv"
    
    # Scrittura header CSV
    with open(INFO_FILE_PATH, 'w', newline='') as info_csv:
        info_csv_writer = csv.writer(info_csv)
        info_csv_writer.writerow(["INDEX", "SIMULATION_TIME", "SIMULATION_LENGTH", "COMPLETED", "INIT_SPEED", "ANGLE_X", "ANGLE_Y", "CIRCLE_RADIUS"])

    # Disabilita journal per evitare conflitti su abaqus.rpy
    try:
        session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)
    except:
        pass

    for idx in range(idx_start, idx_end):

        #controlla se esiste già una cartella per questa simulazione in ../NewSimulations_RUGBY, se sì salta
        sim_folder = os.path.join("..", "NewSimulations_RUGBY", "Dynamic_Simulation_" + str(idx), str(idx) + "_output_displacement_all_frames.csv.gz")

        #controlla che nel file idx_output_displacement_all_frames.csv.gz ci siano almeno 120*3.5 righe, altrimenti considera la simulazione incompleta e rilanciala
        if os.path.exists(sim_folder):
            with gzip.open(sim_folder, 'rt') as f:
                num_lines = sum(1 for line in f)
                if num_lines < 120*3.5*1784: #120 fps * 3.5 secondi * 1784 nodi
                    print(f"Simulazione {idx} incompleta (solo {num_lines} righe). Rilancio.")
                    continue
                else:
                    print(f"Simulazione {idx} già esistente e completa. Salto.")
                    continue

            log("Simulation " + str(idx))
            start = time.time()

            # Seed unico basato su ID
            random.seed(time.time() + idx)

            radius = random.uniform(RADIUS_RANGE[0], RADIUS_RANGE[1])
            velocity = random.uniform(VELOCITY_RANGE[0], VELOCITY_RANGE[1])
            alpha_X = random.uniform(ALPHA_X_RANGE[0], ALPHA_X_RANGE[1])
            alpha_Y = random.uniform(ALPHA_Y_RANGE[0], ALPHA_Y_RANGE[1])
            spin_X = random.uniform(SPIN_X_RANGE[0], SPIN_X_RANGE[1])
            spin_Y = random.uniform(SPIN_Y_RANGE[0], SPIN_Y_RANGE[1])
            spin_Z = random.uniform(SPIN_Z_RANGE[0], SPIN_Z_RANGE[1])

            log(f"Parametri scelti: Radius={radius:.2f}, Velocity={velocity:.2f}, Alpha_X={alpha_X:.2f}, Alpha_Y={alpha_Y:.2f}, Spin_X={spin_X:.2f}, Spin_Y={spin_Y:.2f}, Spin_Z={spin_Z:.2f}")

            sim = Simulation3D()
            
            try:
                (simulation_length, simulation_completed) = sim.runSimulation(
                    CIRCLE_RADIUS   = radius,
                    CIRCLE_VELOCITY = velocity,
                    ALPHA_Y         = alpha_Y,
                    ALPHA_X         = alpha_X,
                    SIMULATION_ID   = idx,
                    SPIN_X          = spin_X,
                    SPIN_Y          = spin_Y,
                    SPIN_Z          = spin_Z,
                    # DISATTIVA L'ESTRAZIONE INTERNA (fondamentale!)
                    SAVEDISPLACEMENT     = False, 
                    SAVECOORDINATES      = True,
                    SAVEPLATECOORDINATES = True,
                    SAVEBALLCOORDINATES  = False
                )
                
                # --- LANCIO ESTRAZIONE IN BACKGROUND ---
                # Estraiamo i dati calcolati che servono allo script esterno
                folder_path = sim.new_path
                ox = str(sim.circle_origin_x)
                oy = str(sim.circle_origin_y)
                oz = str(sim.circle_origin_z)
                
                # Costruiamo il comando da lanciare nel terminale
                # Esempio: abaqus python data_extractor.py 5 "percorso/cartella" 10.0 50.0 0.0
                extractor_cmd = [
                    "abaqus", "python", "data_extraction.py", 
                    str(idx), folder_path, ox, oy, oz
                ]
                
                # Popen lancia il processo e restituisce subito il controllo allo script
                # shell=True è spesso necessario su Windows per far riconoscere il comando "abaqus"
                subprocess.Popen(extractor_cmd, shell=True) 
                
                log(f"Avviata estrazione in background per la simulazione {idx}")
                # ---------------------------------------

            except Exception as e:
                log("ERRORE CRITICO simulazione " + str(idx) + ": " + str(e))
                simulation_length = 0
                simulation_completed = "FAILED"
        
            simulation_time = str(time.time() - start)
            
            # Scrittura risultati (rimane uguale)
            try:
                with open(INFO_FILE_PATH, 'a', newline='') as info_csv:
                    info_csv_append = csv.writer(info_csv)
                    info_csv_append.writerow([idx, simulation_time, simulation_length, simulation_completed, velocity, alpha_X, alpha_Y, radius])
            except IOError:
                time.sleep(1)
                with open(INFO_FILE_PATH, 'a', newline='') as info_csv:
                    info_csv_append = csv.writer(info_csv)
                    info_csv_append.writerow([idx, simulation_time, simulation_length, simulation_completed, velocity, alpha_X, alpha_Y, radius])

if __name__ == "__main__":
    Main()
"""
import sys
import os
if os.getcwd() not in sys.path: sys.path.append( os.getcwd() )
import time
import random
import csv

from abaqus            import *
from driverUtils       import *
from caeModules        import *

from Simulation3D import *


def log(message):
    print(message, file = sys.__stdout__)
    return



def Main():
    
    RADIUS_RANGE = [2, 4.5]             
    VELOCITY_RANGE = [3000, 10000]
    ALPHA_Y_RANGE = [0, 60]             # DEGREE
    ALPHA_X_RANGE = [-180, 180]

    SIMULATIONS_TOT = 5000

    idx_start = 10  # Cambiare questo valore per ripartire da una simulazione specifica

    # NOTA:
    # SIMULATION_TIME = tempo impiegato ad eseguire la simulazione
    # SIMULATION_LENGTH = durata della simulazione, cioè tempo che impiega la palla a fermarsi

    # Creo file per salvare info su tutte le simulazioni: tempo impiegato, se e' terminata, ecc
    INFO_FILE_PATH = "Simulations_Info.csv"
    with open(INFO_FILE_PATH, 'w', newline='') as info_csv:
        info_csv_writer = csv.writer(info_csv)
        info_csv_writer.writerow(["INDEX", "SIMULATION_TIME", "SIMULATION_LENGTH", "COMPLETED", "INIT_SPEED", "ANGLE_X", "ANGLE_Y", "CIRCLE_RADIUS"])


    for idx in range(idx_start, SIMULATIONS_TOT):
            
            log("Simulation " + str(idx))

            start = time.time()

            # Scegli parametri random
            radius = random.uniform(RADIUS_RANGE[0], RADIUS_RANGE[1])
            velocity = random.uniform(VELOCITY_RANGE[0], VELOCITY_RANGE[1])
            alpha_X = random.uniform(ALPHA_X_RANGE[0], ALPHA_X_RANGE[1])
            alpha_Y = random.uniform(ALPHA_Y_RANGE[0], ALPHA_Y_RANGE[1])

            # Esegui la simulazione
            sim = Simulation3D()
            (simulation_length, simulation_completed) = sim.runSimulation(
                CIRCLE_RADIUS   = radius,
                CIRCLE_VELOCITY = velocity,
                ALPHA_Y         = alpha_Y,
                ALPHA_X         = alpha_X,
                SIMULATION_ID   = idx
            )
        
            # Salva info
            simulation_time = str(time.time() - start)
            with open(INFO_FILE_PATH, 'a', newline='') as info_csv:
                info_csv_append = csv.writer(info_csv)
                info_csv_append.writerow([idx, simulation_time, simulation_length, simulation_completed, velocity, alpha_X, alpha_Y, radius])


if __name__ == "__main__":
    Main()
"""




import sys
import os

# Aggiunge la directory corrente al path per trovare i moduli custom
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
    
    # --- RECUPERO PARAMETRI DA VARIABILI D'AMBIENTE ---
    # Questo metodo è molto più robusto di sys.argv per Abaqus
    
    # Valori di default (se lanci lo script a mano per test)
    idx_start = 0
    idx_end = 5000 

    # Controlliamo se il Launcher ci ha passato i dati
    if "ABAQUS_SIM_START" in os.environ and "ABAQUS_SIM_END" in os.environ:
        try:
            idx_start = int(os.environ["ABAQUS_SIM_START"])
            idx_end   = int(os.environ["ABAQUS_SIM_END"])
            log(f"--- CONFIGURAZIONE RICEVUTA: Start={idx_start}, End={idx_end} ---")
        except ValueError:
            log("Errore nella lettura delle variabili d'ambiente. Uso default.")
    else:
        log("--- ATTENZIONE: Nessuna variabile d'ambiente trovata. Uso default (10->5000) ---")
        # Se vedi questo messaggio nel log mentre usi il launcher, c'è un problema.

    # --------------------------------------------------
    """
    RADIUS_RANGE = [8, 9]             
    VELOCITY_RANGE = [3000, 10000]
    ALPHA_Y_RANGE = [0, 60]             
    ALPHA_X_RANGE = [-180, 180]
    """

    RADIUS_RANGE = [9, 15]             
    VELOCITY_RANGE = [3000, 10000]
    ALPHA_Y_RANGE = [0, 60]             
    ALPHA_X_RANGE = [-180, 180]

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
            
            log("Simulation " + str(idx))
            start = time.time()

            # Seed unico basato su ID
            random.seed(time.time() + idx)

            radius = random.uniform(RADIUS_RANGE[0], RADIUS_RANGE[1])
            velocity = random.uniform(VELOCITY_RANGE[0], VELOCITY_RANGE[1])
            alpha_X = random.uniform(ALPHA_X_RANGE[0], ALPHA_X_RANGE[1])
            alpha_Y = random.uniform(ALPHA_Y_RANGE[0], ALPHA_Y_RANGE[1])

            sim = Simulation3D()
            
            try:
                (simulation_length, simulation_completed) = sim.runSimulation(
                    CIRCLE_RADIUS   = radius,
                    CIRCLE_VELOCITY = velocity,
                    ALPHA_Y         = alpha_Y,
                    ALPHA_X         = alpha_X,
                    SIMULATION_ID   = idx
                )
            except Exception as e:
                log("ERRORE CRITICO simulazione " + str(idx) + ": " + str(e))
                simulation_length = 0
                simulation_completed = "FAILED"
        
            simulation_time = str(time.time() - start)
            
            # Scrittura risultati
            # Riapriamo il file in append ogni volta per sicurezza
            try:
                with open(INFO_FILE_PATH, 'a', newline='') as info_csv:
                    info_csv_append = csv.writer(info_csv)
                    info_csv_append.writerow([idx, simulation_time, simulation_length, simulation_completed, velocity, alpha_X, alpha_Y, radius])
            except IOError:
                # Se per caso il file è bloccato (raro qui), aspetta e riprova
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




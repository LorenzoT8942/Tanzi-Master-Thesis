import sys
import os

# Aggiunge la directory corrente al path per trovare i moduli custom
if os.getcwd() not in sys.path: sys.path.append( os.getcwd() )

import time
import random
import csv
import math

from abaqus              import *
from driverUtils         import *
from caeModules          import *
from Simulation3D_BULLET import Simulation3D_BULLET
    
    
def log(message):
    
    print(message, file = sys.__stdout__)
    return
    
    
def Main():
    
    # --- RECUPERO PARAMETRI DA VARIABILI D'AMBIENTE ---
    # Questo metodo è molto più robusto di sys.argv per Abaqus
    
    # Valori di default (se lanci lo script a mano per test)
    idx_start = 7
    idx_end   = 8

    # Controlliamo se il Launcher ci ha passato i dati
    if ( "ABAQUS_SIM_START" in os.environ ) and ( "ABAQUS_SIM_END" in os.environ ):
        
        try:
            
            idx_start = int( os.environ["ABAQUS_SIM_START"] )
            idx_end   = int( os.environ["ABAQUS_SIM_END"] )
            
            log(f"--- CONFIGURAZIONE RICEVUTA: Start = {idx_start}, End = {idx_end} ---")
            
        except ValueError:
            
            log("Errore nella lettura delle variabili d'ambiente. Uso default.")
            
    else:
        
        log(f"--- ATTENZIONE: Nessuna variabile d'ambiente trovata. Uso default ({idx_start}-> {idx_end}) ---")
    
    
    #******************
    # PLATE PARAMETERS
    #******************
    PLATE_WIDTH  = 40.0
    PLATE_HEIGHT = 2.5
    
    
    #******************
    # BULLET PARAMETERS
    #******************
    LENGTH_SIDE_RATIO = 3
    RADIUS_RANGE      = [2,                         3.5]     # [mm]
    SPEED_RANGE       = [8000,                    10000]     # [mm/s]
    BULLET_X_CENTER   = [-PLATE_WIDTH/2, +PLATE_WIDTH/2]
    BULLET_Y_CENTER   = [-PLATE_WIDTH/2, +PLATE_WIDTH/2]
    BULLET_Z_CENTER   = [-PLATE_WIDTH/2, +PLATE_WIDTH/2]
    INFO_FILE_PATH    = "Simulations_Info_" + str(idx_start) + "_" + str(idx_end) + ".csv"
    
    
    #***************************
    # WRITING HEADER IN CSV FILE
    #***************************
    with open(file = INFO_FILE_PATH, mode = 'w', newline = '' ) as info_csv:
        
        info_csv_writer = csv.writer(info_csv)
        info_csv_writer.writerow( [ "INDEX",
                                    "SIMULATION_TIME",
                                    "SIMULATION_LENGTH",
                                    "COMPLETED",
                                    "INIT_SPEED",
                                    "BULLET_RADIUS",
                                    "BULLET_X_CENTER",
                                    "BULLET_Y_CENTER",
                                    "BULLET_Z_CENTER",
                                    "ANGLE_1",
                                    "ANGLE_2" ] )
    
    
    #******************************************************
    # DISABLE JOURNALING TO AVOID CONFLICTS WITH ABAQUS.RPY
    #******************************************************
    try:
        
        session.journalOptions.setValues( replayGeometry  = COORDINATE,
                                          recoverGeometry = COORDINATE )
    
    except:
        
        pass
        
        
    for idx in range(idx_start, idx_end):
            
            log("Simulation " + str(idx))
            start = time.time()
            
            
            #*********************
            # UNIQUE ID-BASED SEED
            #*********************
            random.seed(time.time() + idx)
            
            
            #**************************************
            # RANDOM GENERATION OF INPUT PARAMETERS
            #**************************************
            radius          = random.uniform(RADIUS_RANGE[0], RADIUS_RANGE[1])
            speed           = random.uniform(SPEED_RANGE[0], SPEED_RANGE[1])
            angle1          = random.uniform(0, 30)
            angle2          = random.uniform(0, 30)
            bullet_y_center = (2 / 30) * speed
            bullet_x_center = random.choice([-1, 1]) * bullet_y_center * math.tan(math.radians(angle1))
            bullet_z_center = random.choice([-1, 1]) * bullet_y_center * math.tan(math.radians(angle2))
            
            
            #**********************************
            # INSTANTIATING A SIMULATION OBJECT
            #**********************************
            sim = Simulation3D_BULLET( PLATE_WIDTH  = PLATE_WIDTH,
                                       PLATE_HEIGHT = PLATE_HEIGHT )
            
            
            try:
                
                #*******************
                # RUNNING SIMULATION
                #*******************
                simulation_length, simulation_completed = sim.runSimulation( BULLET_RADIUS     = radius,
                                                                             BULLET_SPEED      = speed,
                                                                             BULLET_X_CENTER   = bullet_x_center,
                                                                             BULLET_Y_CENTER   = bullet_y_center,
                                                                             BULLET_Z_CENTER   = bullet_z_center,
                                                                             SIMULATION_ID     = idx,
                                                                             LENGTH_SIDE_RATIO = LENGTH_SIDE_RATIO )
                                                                               
            except Exception as e:
                
                # log("CRITICAL ERROR simulation " + str(idx) + ": " + str(e))
                print( f"CRITICAL ERROR simulation {str(idx)} - {str(e)} " )
                simulation_length    = 0
                simulation_completed = "FAILED"
            
            
            simulation_time = str(time.time() - start)
            
            # Scrittura risultati
            # Riapriamo il file in append ogni volta per sicurezza
            try:
                
                with open( file = INFO_FILE_PATH, mode = 'a', newline = '' ) as info_csv:
                    
                    info_csv_append = csv.writer(info_csv)
                    info_csv_append.writerow( [ idx,
                                                simulation_time,
                                                simulation_length,
                                                simulation_completed,
                                                speed,
                                                radius,
                                                bullet_x_center,
                                                bullet_y_center,
                                                bullet_z_center,
                                                angle1,
                                                angle2 ] )

            except IOError:
                
                # Se per caso il file è bloccato (raro qui), aspetta e riprova
                time.sleep(1)
                with open( file = INFO_FILE_PATH, mode = 'a', newline = '' ) as info_csv:
                    
                    info_csv_append = csv.writer(info_csv)
                    info_csv_append.writerow( [ idx,
                                                simulation_time,
                                                simulation_length,
                                                simulation_completed,
                                                speed,
                                                radius,
                                                bullet_x_center,
                                                bullet_y_center,
                                                bullet_z_center,
                                                angle1,
                                                angle2 ] )
    
    
if __name__ == "__main__":
    
    Main()
    
    


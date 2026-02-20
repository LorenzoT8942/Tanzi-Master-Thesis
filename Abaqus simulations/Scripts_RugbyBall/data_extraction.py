import sys
import os
import numpy as np
import pandas as pd
import gzip
from odbAccess import openOdb

STRESS_STRAIN_EXTRACTION = False
DELETE_ODB_AFTER_EXTRACTION = False

def log(message):
    print(message)

def main():
    log("Avviata estrazione dati per la simulazione " + str(sys.argv[1]))

    # Verifica che siano stati passati tutti gli argomenti
    if len(sys.argv) < 6:
        log("Errore: argomenti insufficienti per l'estrazione.")
        sys.exit(1)

    # sys.argv[0] è il nome dello script stesso
    index = sys.argv[1]
    folder_path = sys.argv[2]
    circle_origin_x = float(sys.argv[3])
    circle_origin_y = float(sys.argv[4])
    circle_origin_z = float(sys.argv[5])

    os.chdir(folder_path)
    JOB_NAME = "Simulation_Job_" + str(index)
    odb_path = JOB_NAME + ".odb"

    if not os.path.exists(odb_path):
        log("Errore: ODB " + odb_path + " non trovato.")
        sys.exit(1)

    log("Inizio estrazione dati da: " + odb_path)
    
    try:
        odb = openOdb(path=odb_path, readOnly=True)
        step = odb.steps['Step-1']
        
        # Sostituisci qui con la logica esatta di estrazione che usavi in Simulation3D_RUGBY.py
        outputRegionExternal = odb.rootAssembly.instances['PLATE'].nodeSets['SURFACE-ALL']
        rp_node_set = odb.rootAssembly.nodeSets['CIRCLE-RP']

        # ESEMPIO: Estrazione Displacement Piastra
        #disp_filename = str(index) + '_output_displacement_all_frames.csv'
        #with open(disp_filename, 'w') as f:
        #    f.write("Time,Id,X_Disp,Y_Disp,Z_Disp\n")
        #    
        #for frame in step.frames:
        #    frame_time = frame.frameValue
        #    u_field = frame.fieldOutputs['U'].getSubset(region=outputRegionExternal)
        #    
        #    for block in u_field.bulkDataBlocks:
        #        df_chunk = pd.DataFrame({
        #            'Time'  : np.full(len(block.nodeLabels), frame_time),
        #            'Id'    : block.nodeLabels,
        #            'X_Disp': block.data[:, 0],
        #            'Y_Disp': block.data[:, 1],
        #            'Z_Disp': block.data[:, 2]
        #        })
        #        df_chunk['Id'] = df_chunk['Id'].astype(int)
        #        df_chunk.to_csv(disp_filename, mode='a', header=False, index=False)

        # ------------- ESTRAZIONE DISPLACEMENT CON COMPRESSIONE GZIP -------------
        # 1. Aggiungi l'estensione .gz al nome del file
        disp_filename = str(index) + '_output_displacement_all_frames.csv.gz'
        
        # 2. Scrivi l'intestazione specificando la compressione
        # Nota: usiamo pandas per scrivere un dataframe vuoto solo per l'header, 
        # per assicurarci che la compressione sia inizializzata correttamente
        pd.DataFrame(columns=["Time","Id","X_Disp","Y_Disp","Z_Disp"]).to_csv(
            disp_filename, index=False, compression='gzip'
        )
            
        for frame in step.frames:
            frame_time = frame.frameValue
            u_field = frame.fieldOutputs['U'].getSubset(region=outputRegionExternal)
            
            for block in u_field.bulkDataBlocks:
                df_chunk = pd.DataFrame({
                    'Time'  : np.full(len(block.nodeLabels), frame_time),
                    'Id'    : block.nodeLabels,
                    'X_Disp': block.data[:, 0],
                    'Y_Disp': block.data[:, 1],
                    'Z_Disp': block.data[:, 2]
                })
                df_chunk['Id'] = df_chunk['Id'].astype(int)
                
                # 3. Appendi i dati in formato compresso
                df_chunk.to_csv(disp_filename, mode='a', header=False, index=False, compression='gzip')

        # ESEMPIO: Estrazione Palla
        rp_filename = str(index) + '_output_ball_kinematics.csv'
        with open(rp_filename, 'w') as f:
            f.write("Time,X_Abs,Y_Abs,Z_Abs,UR1,UR2,UR3\n")
            
        for frame in step.frames:
            frame_time = frame.frameValue
            u_field = frame.fieldOutputs['U'].getSubset(region=rp_node_set)
            ur_field = frame.fieldOutputs['UR'].getSubset(region=rp_node_set)
            
            if len(u_field.values) > 0:
                u_vals = u_field.values[0].data
                ur_vals = ur_field.values[0].data if len(ur_field.values) > 0 else (0.0, 0.0, 0.0)
                
                abs_x = circle_origin_x + u_vals[0]
                abs_y = circle_origin_y + u_vals[1]
                abs_z = circle_origin_z + u_vals[2]
                
                with open(rp_filename, 'a') as f:
                    f.write(str(frame_time) + "," + str(abs_x) + "," + str(abs_y) + "," + str(abs_z) + "," + str(ur_vals[0]) + "," + str(ur_vals[1]) + "," + str(ur_vals[2]) + "\n")
        
        if STRESS_STRAIN_EXTRACTION:
            # Modifichiamo l'estensione in .csv.gz
            stress_filename = str(index) + '_output_stress_all_frames.csv.gz'
            strain_filename = str(index) + '_output_strain_all_frames.csv.gz'

            try:
                # Recuperiamo il set di elementi della lastra
                plate_elem_set = odb.rootAssembly.instances['PLATE'].elementSets['SET-ALL']

                # Apriamo i file gzip in modalità scrittura testo ('wt')
                with gzip.open(stress_filename, 'wt', encoding='utf-8') as f_stress, \
                        gzip.open(strain_filename, 'wt', encoding='utf-8') as f_strain:

                    # Scriviamo le intestazioni manualmente una sola volta all'inizio del file
                    f_stress.write("Time,Element_Id,Int_Point,S11,S22,S33,S12,S13,S23\n")
                    f_strain.write("Time,Element_Id,Int_Point,E11,E22,E33,E12,E13,E23\n")

                    for frame in odb.steps['Step-1'].frames:
                        frame_time = frame.frameValue

                        # --- ESTRAZIONE STRESS (S) ---
                        if 'S' in frame.fieldOutputs.keys():
                            s_field = frame.fieldOutputs['S'].getSubset(region=plate_elem_set)
                            for block in s_field.bulkDataBlocks:
                                df_s = pd.DataFrame({
                                    'Time'      : frame_time,
                                    'Element_Id': block.elementLabels,
                                    'Int_Point' : block.integrationPoints,
                                    'S11'       : block.data[:, 0],
                                    'S22'       : block.data[:, 1],
                                    'S33'       : block.data[:, 2],
                                    'S12'       : block.data[:, 3],
                                    'S13'       : block.data[:, 4],
                                    'S23'       : block.data[:, 5]
                                })
                                # Passiamo l'handle del file aperto a Pandas. Pandas scriverà direttamente nel flusso compresso.
                                df_s.to_csv(f_stress, header=False, index=False)

                        # --- ESTRAZIONE STRAIN (E / LE) ---
                        strain_key = 'LE' if 'LE' in frame.fieldOutputs.keys() else 'E' if 'E' in frame.fieldOutputs.keys() else None

                        if strain_key:
                            e_field = frame.fieldOutputs[strain_key].getSubset(region=plate_elem_set)
                            for block in e_field.bulkDataBlocks:
                                df_e = pd.DataFrame({
                                    'Time'      : frame_time,
                                    'Element_Id': block.elementLabels,
                                    'Int_Point' : block.integrationPoints,
                                    'E11'       : block.data[:, 0],
                                    'E22'       : block.data[:, 1],
                                    'E33'       : block.data[:, 2],
                                    'E12'       : block.data[:, 3],
                                    'E13'       : block.data[:, 4],
                                    'E23'       : block.data[:, 5]
                                })
                                df_e.to_csv(f_strain, header=False, index=False)

                log("Saved compressed stress and strain data (.csv.gz) successfully.")
            
            except KeyError as e:
                log(f"ERROR: Could not find element set or field output for Stress/Strain. Details: {e}")

        odb.close()
        log("Estrazione completata con successo per l'indice " + str(index))

        if DELETE_ODB_AFTER_EXTRACTION:
            try:
                os.remove(odb_path)
                log("ODB file deleted: " + odb_path)
            except Exception as e:
                log("Warning: Could not delete ODB file. Details: " + str(e))
        
    except Exception as e:
        log("Errore durante l'estrazione per " + str(index) + ": " + str(e))

    
if __name__ == "__main__":
    main()
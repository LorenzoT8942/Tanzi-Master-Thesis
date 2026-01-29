import bpy
import csv
import os

# ==========================================
# CONFIGURAZIONE UTENTE
# ==========================================
# Inserisci i percorsi corretti ai tuoi fil
BASE_PATH = r"D:\Tesi Magistrale\Abaqus simulations\SimVisData"

PATH_COORDS = r"D:\Tesi Magistrale\Abaqus simulations\SimVisData\plate_initial_coordinates.csv"
PATH_DISP   = r"D:\Tesi Magistrale\Abaqus simulations\SimVisData\0_output_displacement_all_frames.csv" 

# Impostazioni Temporali
SIMULATION_FPS = 60  # Sincronizzazione 1:1 con i dati
TOLERANCE = 1e-4     # Tolleranza per raggruppare coordinate

# ==========================================
# FUNZIONI
# ==========================================

def setup_scene():
    """Imposta FPS e pulisce la scena."""
    bpy.context.scene.render.fps = SIMULATION_FPS
    
    # Rimuovi oggetti vecchi
    if "Plate_Skin" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Plate_Skin"], do_unlink=True)
    if "Plate_Mesh_Skin" in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes["Plate_Mesh_Skin"])

def generate_robust_skin_mesh(filepath):
    """
    Genera una mesh 'guscio' (Top, Bottom + Lati) robusta a mesh non uniformi.
    """
    print("Analisi nodi e generazione skin...")
    
    verts = []      # Lista coordinate (x, y, z)
    ids_map = {}    # ID Abaqus -> Indice Blender
    node_data = []  # Cache per calcolo
    
    # 1. Lettura File
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        try:
            next(reader) # Skip header
        except: return None, None, None

        idx = 0
        for row in reader:
            if not row: continue
            try:
                nid = int(row[0])
                x, y, z = float(row[1]), float(row[2]), float(row[3])
                
                verts.append((x, y, z))
                ids_map[nid] = idx
                node_data.append({'x': x, 'y': y, 'z': z, 'idx': idx})
                idx += 1
            except ValueError: continue

    if not verts:
        print("Errore: Nessun vertice trovato.")
        return None, None, None

    # 2. Identifica la Griglia 2D (X, Z)
    # Troviamo le colonne uniche di nodi
    unique_x = sorted(list(set(round(n['x'], 4) for n in node_data)))
    unique_z = sorted(list(set(round(n['z'], 4) for n in node_data)))
    
    nx, nz = len(unique_x), len(unique_z)
    print(f"Griglia superficiale rilevata: {nx} x {nz} ({nx*nz} colonne totali).")

    # Mappe per convertire coordinata -> indice griglia 0..N
    x_map = {val: i for i, val in enumerate(unique_x)}
    z_map = {val: i for i, val in enumerate(unique_z)}

    # Raggruppa i nodi per colonna (ix, iz)
    # grid[(ix, iz)] = lista di nodi in quella colonna
    grid = {}
    
    for n in node_data:
        k_x = round(n['x'], 4)
        k_z = round(n['z'], 4)
        
        if k_x in x_map and k_z in z_map:
            ix = x_map[k_x]
            iz = z_map[k_z]
            
            if (ix, iz) not in grid:
                grid[(ix, iz)] = []
            grid[(ix, iz)].append(n)

    # Per ogni colonna, identifichiamo il nodo TOP (max Y) e BOT (min Y)
    # skin_map[(ix, iz)] = {'top': idx_blender, 'bot': idx_blender}
    skin_map = {}
    
    for key, nodes in grid.items():
        # Ordina per Y
        nodes.sort(key=lambda item: item['y'])
        skin_map[key] = {
            'bot': nodes[0]['idx'], # Il più basso
            'top': nodes[-1]['idx'] # Il più alto
        }

    # 3. Generazione Facce (Solo Guscio)
    faces = []
    
    # Funzione helper per aggiungere quad se esistono i 4 angoli
    def add_quad(p1, p2, p3, p4):
        faces.append((p1, p2, p3, p4))

    # -- TOP e BOTTOM (Iteriamo sulle celle della griglia) --
    for ix in range(nx - 1):
        for iz in range(nz - 1):
            # Indici dei 4 vicini nella griglia
            k00 = (ix,   iz)
            k10 = (ix+1, iz)
            k11 = (ix+1, iz+1)
            k01 = (ix,   iz+1)
            
            if all(k in skin_map for k in [k00, k10, k11, k01]):
                # Faccia Superiore (Top) - Winding CCW
                add_quad(skin_map[k00]['top'], skin_map[k10]['top'], 
                         skin_map[k11]['top'], skin_map[k01]['top'])
                
                # Faccia Inferiore (Bottom) - Winding Inverso (visto da sotto)
                add_quad(skin_map[k00]['bot'], skin_map[k01]['bot'], 
                         skin_map[k11]['bot'], skin_map[k10]['bot'])

    # -- LATI (Chiudiamo il perimetro) --
    
    # Lato SUD (iz = 0)
    iz = 0
    for ix in range(nx - 1):
        k0 = (ix, iz)
        k1 = (ix+1, iz)
        if k0 in skin_map and k1 in skin_map:
            # Top0 -> Top1 -> Bot1 -> Bot0
            add_quad(skin_map[k0]['top'], skin_map[k1]['top'], 
                     skin_map[k1]['bot'], skin_map[k0]['bot'])

    # Lato NORD (iz = max)
    iz = nz - 1
    for ix in range(nx - 1):
        k0 = (ix, iz)
        k1 = (ix+1, iz)
        if k0 in skin_map and k1 in skin_map:
            # Ordine inverso perché siamo "dietro"
            add_quad(skin_map[k1]['top'], skin_map[k0]['top'], 
                     skin_map[k0]['bot'], skin_map[k1]['bot'])

    # Lato OVEST (ix = 0)
    ix = 0
    for iz in range(nz - 1):
        k0 = (ix, iz)
        k1 = (ix, iz+1)
        if k0 in skin_map and k1 in skin_map:
            add_quad(skin_map[k1]['top'], skin_map[k0]['top'], 
                     skin_map[k0]['bot'], skin_map[k1]['bot'])

    # Lato EST (ix = max)
    ix = nx - 1
    for iz in range(nz - 1):
        k0 = (ix, iz)
        k1 = (ix, iz+1)
        if k0 in skin_map and k1 in skin_map:
            add_quad(skin_map[k0]['top'], skin_map[k1]['top'], 
                     skin_map[k1]['bot'], skin_map[k0]['bot'])

    print(f"Generate {len(faces)} facce (Guscio esterno).")
    
    # Crea Mesh
    mesh = bpy.data.meshes.new("Plate_Mesh_Skin")
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new("Plate_Skin", mesh)
    bpy.context.collection.objects.link(obj)
    
    # Materiale
    mat = bpy.data.materials.new("PlateMat")
    mat.diffuse_color = (0.2, 0.6, 1.0, 1.0)
    obj.data.materials.append(mat)
    
    # Smooth shading e Auto Smooth
    for p in mesh.polygons: p.use_smooth = True
    # mesh.use_auto_smooth = True # (Opzionale nelle versioni nuove di Blender)
    
    return obj, ids_map

def apply_displacement_animation(obj, ids_map, disp_path):
    print("Caricamento animazione...")
    
    time_data = {}
    with open(disp_path, 'r') as f:
        reader = csv.reader(f)
        try: next(reader)
        except: return

        for row in reader:
            if not row: continue
            try:
                t = float(row[0])
                nid = int(row[1])
                disp = (float(row[2]), float(row[3]), float(row[4]))
                
                if t not in time_data: time_data[t] = {}
                time_data[t][nid] = disp
            except ValueError: continue

    sorted_times = sorted(time_data.keys())
    print(f"Frame trovati: {len(sorted_times)}")

    # Shape Keys
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis")

    base_coords = [v.co.copy() for v in obj.data.vertices]
    n_verts = len(base_coords)
    idx_to_aba = {v: k for k, v in ids_map.items()}

    obj.animation_data_create()

    for i, t in enumerate(sorted_times):
        frame_number = i + 1
        sk = obj.shape_key_add(name=f"F_{frame_number}")
        
        flat_coords = [0.0] * (n_verts * 3)
        current_disps = time_data[t]
        
        for v_idx in range(n_verts):
            bx, by, bz = base_coords[v_idx]
            aba_id = idx_to_aba.get(v_idx)
            
            if aba_id in current_disps:
                d = current_disps[aba_id]
                bx += d[0]; by += d[1]; bz += d[2]
            
            flat_coords[v_idx*3] = bx
            flat_coords[v_idx*3+1] = by
            flat_coords[v_idx*3+2] = bz
            
        sk.data.foreach_set("co", flat_coords)
        
        # Keyframing
        sk.value = 0.0
        sk.keyframe_insert("value", frame=frame_number - 1)
        sk.value = 1.0
        sk.keyframe_insert("value", frame=frame_number)
        sk.value = 0.0
        sk.keyframe_insert("value", frame=frame_number + 1)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(sorted_times)
    bpy.context.scene.frame_current = 1
    print("Fatto.")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if os.path.exists(PATH_COORDS) and os.path.exists(PATH_DISP):
        setup_scene()
        obj, ids_map = generate_robust_skin_mesh(PATH_COORDS)
        if obj:
            apply_displacement_animation(obj, ids_map, PATH_DISP)
    else:
        print("File non trovati!")
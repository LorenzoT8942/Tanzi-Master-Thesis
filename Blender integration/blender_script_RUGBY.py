import bpy
import bmesh
import csv
import os
import gzip

# ==========================================
# CONFIGURAZIONE UTENTE
# ==========================================
# Inserisci i percorsi corretti ai tuoi file.
# Assicurati che il numero iniziale (es. "0_") corrisponda al tuo SIMULATION_ID
SIM_ID = 2

PATH_COORDS = rf"D:\Tesi Magistrale\Abaqus simulations\NewSimulations_RUGBY\Dynamic_Simulation_{SIM_ID}\plate_initial_coordinates.csv"
PATH_DISP   = rf"D:\Tesi Magistrale\Abaqus simulations\NewSimulations_RUGBY\Dynamic_Simulation_{SIM_ID}\{SIM_ID}_output_displacement_all_frames.csv.gz" 

# Nuovi percorsi per la palla
PATH_BALL_COORDS = rf"D:\Tesi Magistrale\Abaqus simulations\NewSimulations_RUGBY\Dynamic_Simulation_{SIM_ID}\{SIM_ID}_input_coordinates_circle_1.csv"
PATH_BALL_KIN    = rf"D:\Tesi Magistrale\Abaqus simulations\NewSimulations_RUGBY\Dynamic_Simulation_{SIM_ID}\{SIM_ID}_output_ball_kinematics.csv"

# Impostazioni Temporali
SIMULATION_FPS = 60 
TOLERANCE = 1e-4     

# ==========================================
# FUNZIONI PER LA LASTRA
# ==========================================

def setup_scene():
    """Imposta FPS e pulisce la scena da oggetti precedenti."""
    bpy.context.scene.render.fps = SIMULATION_FPS
    
    # Rimuovi vecchia lastra
    if "Plate_Skin" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Plate_Skin"], do_unlink=True)
    if "Plate_Mesh_Skin" in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes["Plate_Mesh_Skin"])
        
    # Rimuovi vecchia palla
    if "RugbyBall" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["RugbyBall"], do_unlink=True)
    if "RugbyBall_Mesh" in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes["RugbyBall_Mesh"])

def generate_robust_skin_mesh(filepath):
    """Genera una mesh 'guscio' robusta per la lastra."""
    print("Analisi nodi e generazione skin lastra...")
    
    verts = []      
    ids_map = {}    
    node_data = []  
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        try: next(reader) 
        except: return None, None

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

    if not verts: return None, None

    unique_x = sorted(list(set(round(n['x'], 4) for n in node_data)))
    unique_z = sorted(list(set(round(n['z'], 4) for n in node_data)))
    nx, nz = len(unique_x), len(unique_z)
    
    x_map = {val: i for i, val in enumerate(unique_x)}
    z_map = {val: i for i, val in enumerate(unique_z)}

    grid = {}
    for n in node_data:
        k_x, k_z = round(n['x'], 4), round(n['z'], 4)
        if k_x in x_map and k_z in z_map:
            ix, iz = x_map[k_x], z_map[k_z]
            if (ix, iz) not in grid: grid[(ix, iz)] = []
            grid[(ix, iz)].append(n)

    skin_map = {}
    for key, nodes in grid.items():
        nodes.sort(key=lambda item: item['y'])
        skin_map[key] = {'bot': nodes[0]['idx'], 'top': nodes[-1]['idx']}

    faces = []
    def add_quad(p1, p2, p3, p4): faces.append((p1, p2, p3, p4))

    for ix in range(nx - 1):
        for iz in range(nz - 1):
            k00, k10, k11, k01 = (ix, iz), (ix+1, iz), (ix+1, iz+1), (ix, iz+1)
            if all(k in skin_map for k in [k00, k10, k11, k01]):
                add_quad(skin_map[k00]['top'], skin_map[k10]['top'], skin_map[k11]['top'], skin_map[k01]['top'])
                add_quad(skin_map[k00]['bot'], skin_map[k01]['bot'], skin_map[k11]['bot'], skin_map[k10]['bot'])

    iz = 0
    for ix in range(nx - 1):
        if (ix, iz) in skin_map and (ix+1, iz) in skin_map:
            add_quad(skin_map[(ix, iz)]['top'], skin_map[(ix+1, iz)]['top'], skin_map[(ix+1, iz)]['bot'], skin_map[(ix, iz)]['bot'])

    iz = nz - 1
    for ix in range(nx - 1):
        if (ix, iz) in skin_map and (ix+1, iz) in skin_map:
            add_quad(skin_map[(ix+1, iz)]['top'], skin_map[(ix, iz)]['top'], skin_map[(ix, iz)]['bot'], skin_map[(ix+1, iz)]['bot'])

    ix = 0
    for iz in range(nz - 1):
        if (ix, iz) in skin_map and (ix, iz+1) in skin_map:
            add_quad(skin_map[(ix, iz+1)]['top'], skin_map[(ix, iz)]['top'], skin_map[(ix, iz)]['bot'], skin_map[(ix, iz+1)]['bot'])

    ix = nx - 1
    for iz in range(nz - 1):
        if (ix, iz) in skin_map and (ix, iz+1) in skin_map:
            add_quad(skin_map[(ix, iz)]['top'], skin_map[(ix, iz+1)]['top'], skin_map[(ix, iz+1)]['bot'], skin_map[(ix, iz)]['bot'])

    mesh = bpy.data.meshes.new("Plate_Mesh_Skin")
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new("Plate_Skin", mesh)
    bpy.context.collection.objects.link(obj)
    
    mat = bpy.data.materials.new(name="SuperMetal")
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Metallic"].default_value = 1.0
    principled.inputs["Roughness"].default_value = 0.02
    principled.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1)

    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)
    for p in mesh.polygons: p.use_smooth = True
    
    return obj, ids_map

def apply_displacement_animation(obj, ids_map, disp_path):
    print("Caricamento animazione deformazione lastra...")
    
    time_data = {}
    if disp_path.endswith('.gz'):
        file_opener = gzip.open(disp_path, 'rt', encoding='utf-8')
    else:
        file_opener = open(disp_path, 'r', encoding='utf-8')

    with file_opener as f:
        reader = csv.reader(f)
        try: next(reader)
        except: return

        for row in reader:
            if not row: continue
            try:
                t, nid = float(row[0]), int(row[1])
                disp = (float(row[2]), float(row[3]), float(row[4]))
                if t not in time_data: time_data[t] = {}
                time_data[t][nid] = disp
            except ValueError: continue

    sorted_times = sorted(time_data.keys())

    if not obj.data.shape_keys: obj.shape_key_add(name="Basis")
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
            
            flat_coords[v_idx*3], flat_coords[v_idx*3+1], flat_coords[v_idx*3+2] = bx, by, bz
            
        sk.data.foreach_set("co", flat_coords)
        sk.value = 0.0
        sk.keyframe_insert("value", frame=frame_number - 1)
        sk.value = 1.0
        sk.keyframe_insert("value", frame=frame_number)
        sk.value = 0.0
        sk.keyframe_insert("value", frame=frame_number + 1)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(sorted_times)
    bpy.context.scene.frame_current = 1

# ==========================================
# FUNZIONI PER LA PALLA DA RUGBY
# ==========================================

def import_and_animate_ball(coords_path, kin_path):
    print("Importazione e generazione palla da rugby...")
    verts = []
    
    # 1. Estrazione nodi iniziali
    with open(coords_path, 'r') as f:
        reader = csv.reader(f)
        try: next(reader) 
        except: return
        for row in reader:
            if not row: continue
            try:
                verts.append((float(row[1]), float(row[2]), float(row[3])))
            except ValueError: continue

    if not verts:
        print("Errore: Nessun vertice trovato per la palla.")
        return

    # 2. Calcolo centroide (per allineare l'origine dell'oggetto su Blender con l'RP di Abaqus)
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    cz = sum(v[2] for v in verts) / len(verts)

    # 3. Centra i vertici intorno all'origine locale
    centered_verts = [(v[0]-cx, v[1]-cy, v[2]-cz) for v in verts]

    # 4. Creazione Mesh base
    mesh = bpy.data.meshes.new("RugbyBall_Mesh")
    mesh.from_pydata(centered_verts, [], [])

    # 5. Convex Hull per creare le facce dal perimetro esterno dei nodi
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bm.to_mesh(mesh)
    bm.free() 

    obj = bpy.data.objects.new("RugbyBall", mesh)
    bpy.context.collection.objects.link(obj)

    # Materiale per la palla (cuoio/gomma scura)
    mat = bpy.data.materials.new(name="RugbyMaterial")
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.2, 0.05, 0.01, 1) # Marrone scuro
    principled.inputs["Roughness"].default_value = 0.8
    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)

    for p in mesh.polygons: p.use_smooth = True

    # 6. Animazione del corpo rigido
    print("Animazione cinematica della palla...")
    obj.animation_data_create()
    obj.rotation_mode = 'XYZ' # Le rotazioni di Abaqus sono angoli di Eulero

    with open(kin_path, 'r') as f:
        reader = csv.reader(f)
        try: next(reader)
        except: return
        
        frame_number = 1
        for row in reader:
            if not row: continue
            try:
                # Time,X_Abs,Y_Abs,Z_Abs,UR1,UR2,UR3
                x, y, z = float(row[1]), float(row[2]), float(row[3])
                rx, ry, rz = float(row[4]), float(row[5]), float(row[6])

                # Assegna posizione e rotazione
                obj.location = (x, y, z)
                obj.rotation_euler = (rx, ry, rz)

                # Inserisci i keyframe
                obj.keyframe_insert(data_path="location", frame=frame_number)
                obj.keyframe_insert(data_path="rotation_euler", frame=frame_number)

                frame_number += 1
            except ValueError: continue

    print("Importazione palla completata.")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    setup_scene()
    
    # Processa Lastra
    if os.path.exists(PATH_COORDS) and os.path.exists(PATH_DISP):
        obj, ids_map = generate_robust_skin_mesh(PATH_COORDS)
        if obj: apply_displacement_animation(obj, ids_map, PATH_DISP)
    else: print("File della lastra non trovati!")
        
    # Processa Palla da Rugby
    if os.path.exists(PATH_BALL_COORDS) and os.path.exists(PATH_BALL_KIN):
        import_and_animate_ball(PATH_BALL_COORDS, PATH_BALL_KIN)
    else: print("File della palla da rugby non trovati!")
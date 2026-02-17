import os
import sys
import time
import abaqusConstants
import mesh
import numpy           as np
import json
import math
import pandas          as pd
import pathlib
   
   
from abaqus      import *
from driverUtils import *
from caeModules  import *
from tkinter     import OFF
    
    
def log(message):
    
    print(message, file = sys.__stdout__)
    return
    
    
class Simulation3D_BULLET():
        
    def __init__( self,
                  PLATE_WIDTH  = 40,
                  PLATE_HEIGHT = 2.5 ):
        
        self.DEBUG = False
        
        
        #***********
        # PARAMETERS
        #***********
        self.index                 = None
        self.bullet_speed          = None
        self.bullet_speed_x        = None
        self.bullet_speed_y        = None
        self.bullet_speed_z        = None
        self.bullet_radius         = None
        self.bullet_impact_angle_x = None
        self.bullet_impact_angle_y = None
        
        
        #*****************
        # OBJECT DIMENSION
        #*****************
        self.plate_width        = PLATE_WIDTH
        self.plate_height       = PLATE_HEIGHT
        self.bullet_base_radius = None   # la seed size la calcolo usando questo come riferimento
        
        
        #******************
        # INITIAL POSITIONS
        #******************
        self.plate_origin_x  = 0
        self.plate_origin_y  = 0
        self.plate_origin_z  = 0
        self.bullet_origin_x = 0
        self.bullet_origin_y = 0
        self.bullet_origin_z = 0
        
        
        #***************
        # MATERIAL: LEAD
        #***************
        self.lead_density = ((1.134e-08,),)
        self.lead_elastic = ((1.4e4, 0.42),)
        
        
        #***************************************
        # MATERIAL: RUBBER (NEO-HOOKEAN MODEL)
        #***************************************
        self.rubber_density = ((1.1e-9,),)         # Densita' tipica della gomma (tonnes/mm^3)
        self.rubber_neo_hooke = ( (0.5, 0.05),)
        
        
        #***************************************
        # MATERIAL: RUBBER (MOONEY-RIVLIN MODEL)
        #***************************************
        self.rubber_mooney_rivlin = ((1.2, 0.1, 0.002),)
        
        
        #***************************
        # MESH PARAMETERS (MODIFIED)
        #***************************
        #self.bullet_seed_size          = 3
        #self.plate_seed_sides_min      = 3
        #self.plate_seed_sides_max      = 5
        #self.plate_seed_top_bottom_min = 3
        #self.plate_seed_top_bottom_max = 5 
        
        
        #***************************
        # MESH PARAMETERS (ORIGINAL)
        #***************************
        self.bullet_seed_size          = 3
        self.plate_seed_sides_min      = 0.5
        self.plate_seed_sides_max      = 1.0
        self.plate_seed_top_bottom_min = 1.0
        self.plate_seed_top_bottom_max = 2
        
        
        #************
        # MISCELLANEA
        #************
        self.time_period     = None        # SIMULATION ELAPSED TIME
        self.output_interval = 1.0 / 60.0  # TIME INTERVAL OUTPUT: SAVES A SNAPSHOT OF THE SIMULATION EVERY 1/30 SECONDS (30 FPS)
        
        
    def _saveInputDataToFile(self):

        
        
        inputData = { "index"                 : self.index,
                      "bullet_speed_x"        : self.bullet_speed_x,
                      "bullet_speed_y"        : self.bullet_speed_y,
                      "bullet_speed_z"        : self.bullet_speed_z,
                      "bullet_speed"          : self.bullet_speed,
                      "bullet_impact_angle_x" : self.bullet_impact_angle_x,
                      "bullet_impact_angle_y" : self.bullet_impact_angle_y,
                      "bullet_radius"         : self.bullet_radius }

        log(f"Saving input data to file: {inputData}")            
        
        #******************************************
        # SAVE TO A FILE NAMED "<INDEX>_INPUT.JSON"
        #******************************************
        filename = os.path.join(self.new_path, self.folder_name + "_input.json")
            
        with open(filename, "w") as outfile:
        
            json.dump(inputData, outfile)
        
        
    def runSimulation( self,
                       BULLET_RADIUS,
                       BULLET_SPEED,
                       #BULLET_X_CENTER,
                       #BULLET_Y_CENTER,
                       #BULLET_Z_CENTER,
                       SIMULATION_ID,
                       LENGTH_SIDE_RATIO,
                       ALPHA_Y,
                       ALPHA_X,
                       SAVEINPUTDATA        = True, 
                       SAVEBULLETSPEED      = True,
                       SAVEDISPLACEMENT     = True, 
                       SAVECOORDINATES      = True,
                       SAVEDATABASE         = True, 
                       SAVEPLATECOORDINATES = True,
                       SAVEJOBINPUT         = True):
        
        
        start_time = time.time()
        
        #**************************
        # RESETTING THE ENVIRONMENT
        #**************************
        Mdb()
        
        log(f"Initialized new Abaqus environment for simulation {SIMULATION_ID}.")
        log(f"Bullet parameters: radius={BULLET_RADIUS:.2f} mm, speed={BULLET_SPEED:.2f} mm/s, angle_y={ALPHA_Y:.2f}°, angle_x={ALPHA_X:.2f}°, length_side_ratio={LENGTH_SIDE_RATIO:.2f}")
        
        #******************
        # SAVING PARAMETERS
        #******************
        self.index             = SIMULATION_ID
        self.bullet_radius     = BULLET_RADIUS
        self.bullet_speed      = BULLET_SPEED
        self.length_side_ratio = LENGTH_SIDE_RATIO
        
        
        #**************************************
        # INITIAL POSITION OF THE BULLET CENTER
        #**************************************
        self.bullet_length   = self.length_side_ratio * self.bullet_radius   # YOU CAN INCREASE/DECREASE (4–8 RECOMMENDED)
        
        
        # print( f"Simulation Id   : {self.index:d}"              )               #---> TO DELETE
        # print( f"Bullet Origin X : {self.bullet_origin_x:8.4f}" )               #---> TO DELETE
        # print( f"Bullet Origin Y : {self.bullet_origin_y:8.4f}" )               #---> TO DELETE
        # print( f"Bullet Origin Z : {self.bullet_origin_z:8.4f}" )               #---> TO DELETE
        
        
        self.simulation_time_perc = 0.1          #PERCENTAGE
        self.TIME_TO_IMPACT       = 2 / 30

        # Time for the bullet to reach the plate, fixed
        self.TIME_TO_IMPACT = 2/30

        # Length of the trajectory that we need for the bullet to take the desired time to reach the plate
        self.trajectory = abs(self.TIME_TO_IMPACT * self.bullet_speed) + self.bullet_length / 2

        # The initial X and Y of the bullet are computed using the desired trajectory length and the angle
        self.bullet_origin_y = self.trajectory * math.cos(math.radians(ALPHA_Y)) + self.bullet_length / 2
        self.bullet_origin_x = - self.trajectory * math.sin(math.radians(ALPHA_Y)) * math.cos(math.radians(ALPHA_X))
        self.bullet_origin_z = - self.trajectory * math.sin(math.radians(ALPHA_Y)) * math.sin(math.radians(ALPHA_X))

        log(f"Calculated bullet origin: x={self.bullet_origin_x:.4f} mm, y={self.bullet_origin_y:.4f} mm, z={self.bullet_origin_z:.4f} mm")
        log(f"Calculated trajectory length: {self.trajectory:.4f} mm, expected time to impact: {self.TIME_TO_IMPACT:.4f} s")

        self.bullet_speed_y = - self.bullet_speed * math.cos(math.radians(ALPHA_Y))
        self.bullet_speed_x = self.bullet_speed * math.sin(math.radians(ALPHA_Y)) * math.cos(math.radians(ALPHA_X))
        self.bullet_speed_z = self.bullet_speed * math.sin(math.radians(ALPHA_Y)) * math.sin(math.radians(ALPHA_X))
        
        self.bullet_impact_angle_y = ALPHA_Y
        self.bullet_impact_angle_x = ALPHA_X
        
        #************************
        # SIMULATION ELAPSED TIME
        #************************

        #self.time_period  = self.TIME_TO_IMPACT
        #self.time_period += abs( distance/self.bullet_speed) #Per interrompere dopo impatto
        self.time_period = 3.5
        
        
        # Crea cartella (se non esiste) con nome <index>
        self.folder_name   = f'Dynamic_Simulation_{self.index}'
        self.previous_path = os.getcwd()
        self.new_path      = os.path.join( self.previous_path, self.folder_name)
        
        # 1. Ottieni il percorso assoluto della directory padre ("..")
        parent_dir = os.path.abspath(os.path.join(self.previous_path, os.pardir))
        
        # 2. Definisci la cartella contenitore "NewSimulations"
        container_dir = os.path.join(parent_dir, 'NewBulletSimulations')
        
        # 3. Il nuovo percorso sarà: ../NewSimulations/Dynamic_Simulation_index
        self.new_path = os.path.join( container_dir, self.folder_name )
        
        os.makedirs( name     = self.new_path,
                     exist_ok = True )
        
        
        #************************
        # CHECKING FOR INPUT DATA
        #************************
        if (self.bullet_origin_y - self.bullet_radius) <= 0:
            
            log('Bullet is too close to the plate.')
            return 0, False
            
            
        #***********************
        #CHANGING WORK DIRECTORY
        #***********************
        os.chdir( self.new_path )
        
        
        if SAVEINPUTDATA:
            
            self._saveInputDataToFile()
    
    
        #****************
        #CREATING A MODEL
        #****************
        MODEL_NAME = f'Simulation_{self.index}'
        model      = mdb.Model( name = MODEL_NAME )
    
    
        #***********************
        #DELETING STANDARD MODEL
        #***********************
        del mdb.models['Model-1']
        
        
        #*******************************
        # CREATING A PLATE PART AND SETS
        #*******************************
        sketch_plate = model.ConstrainedSketch( name      = 'sketch-plate', 
                                                sheetSize = self.plate_width )
        
        
        sketch_plate.rectangle( point1 = ( -self.plate_width/2,                  0 ), 
                                point2 = (  self.plate_width/2, -self.plate_height ) )
        
        
        part_plate = model.Part( name           = 'plate', 
                                 dimensionality = abaqusConstants.THREE_D, 
                                 type           = abaqusConstants.DEFORMABLE_BODY )
        
        
        part_plate.BaseSolidExtrude( sketch = sketch_plate, 
                                     depth  = self.plate_width )
        
        
        # set con tutta la lastra, per il materiale
        part_plate.Set( name  = 'set-all', 
                        cells = part_plate.cells )
        
        
        # crea diversi set per le superfici della plate, per assegnargli una boundary condition (sotto) e per impostare l'interaction (sopra) 
        # e per gestire il seed della mesh
        part_plate.Set( name = 'surface-top',    faces = part_plate.faces.findAt( coordinates = ( (                  0,                    0, self.plate_width/2), ) ) )
        part_plate.Set( name = 'surface-bottom', faces = part_plate.faces.findAt( coordinates = ( (                  0, -self.plate_height/1, self.plate_width/2), ) ) )
        part_plate.Set( name = 'surface-west',   faces = part_plate.faces.findAt( coordinates = ( (-self.plate_width/2, -self.plate_height/2, self.plate_width/2), ) ) )
        part_plate.Set( name = 'surface-east',   faces = part_plate.faces.findAt( coordinates = ( (+self.plate_width/2, -self.plate_height/2, self.plate_width/2), ) ) )
        part_plate.Set( name = 'surface-north',  faces = part_plate.faces.findAt( coordinates = ( (                  0, -self.plate_height/2,                  0), ) ) )
        part_plate.Set( name = 'surface-south',  faces = part_plate.faces.findAt( coordinates = ( (                  0, -self.plate_height/2, self.plate_width/1), ) ) )
        
        
        # superficie per l'interaction
        part_plate.Surface( name       = 'surface-top', 
                            side1Faces = part_plate.faces.findAt(coordinates = ( (0, 0, self.plate_width/2), )  ) )
        
        
        # set tutte  le superfici esterni, per l'output
        part_plate.Set( name  = 'surface-all', 
                        faces = part_plate.faces )
        
        
        #********************************
        # CREATING A BULLET PART AND SETS
        #********************************
        sketch_bullet = model.ConstrainedSketch( name      = 'sketch-bullet',
                                                 sheetSize = 2 * self.bullet_radius )
        
        sketch_bullet.rectangle( point1 = (-self.bullet_radius, -self.bullet_radius),
                                 point2 = (+self.bullet_radius, +self.bullet_radius) )
        
        
        # crea il solido estruso (parallelepipedo)
        part_bullet = model.Part( name           = 'bullet',
                                  dimensionality = abaqusConstants.THREE_D,
                                  type           = abaqusConstants.DEFORMABLE_BODY )
        
        
        part_bullet.BaseSolidExtrude( sketch = sketch_bullet,
                                      depth  = self.bullet_length )
        
        
        # set con tutto il proiettile (materiale)
        part_bullet.Set( name  = 'set-all',
                         cells = part_bullet.cells )
        
        
        # superficie per interaction
        part_bullet.Surface( name       = 'surface-all',
                             side1Faces = part_bullet.faces )
        
        
        # set superfici esterne (output)
        part_bullet.Set( name  = 'surface-all',
                         faces = part_bullet.faces )
        
        
        #*******************
        # DEFINING MATERIALS
        #*******************
        material_plate = model.Material( name = 'material-plate' )
        material_plate.Density( table = self.rubber_density )
        material_plate.Hyperelastic( type     = abaqusConstants.MOONEY_RIVLIN, 
                                     table    = self.rubber_mooney_rivlin, 
                                     testData = OFF )
        
        
        #************************************
        # DEFINING DAMPING (Rayleigh Damping)
        #************************************
        # Alpha: Smorzamento proporzionale alla massa (frena le basse frequenze/moti corpo rigido)
        # Beta: Smorzamento proporzionale alla rigidezza (frena le vibrazioni rapide/interne)
        # Valori tipici per gomma generica: Alpha ~ 5-10, Beta ~ 1e-4 o 1e-5
        material_plate.Damping( alpha = 1.5,
                                beta  = 0.00005 )
        
        
        material_bullet = model.Material( name = 'material-bullet' ) 
        material_bullet.Density( table = self.lead_density )
        material_bullet.Elastic( table = self.lead_elastic )



        #******************************************
        # CREATING SECTIONS AND ASSIGNING MATERIALS
        #******************************************
        
        #**********************
        # SECTION FOR THE PLATE
        #**********************
        model.HomogeneousSolidSection( name      = 'section-plate',
                                       material  = 'material-plate',
                                       thickness = None )
        
        part_plate.SectionAssignment( region      = part_plate.sets['set-all'],
                                      sectionName = 'section-plate' )
        
        
        #***********************
        # SECTION FOR THE BULLET
        #***********************
        model.HomogeneousSolidSection( name      = 'section-bullet',
                                       material  = 'material-bullet',
                                       thickness = None    )
        
        
        #******************************
        # ASSIGN MATERIAL TO THE BULLET
        #******************************
        part_bullet.SectionAssignment( region      = part_bullet.sets['set-all'],
                                       sectionName = 'section-bullet' )
        
        
        #******************
        # CREATING ASSEMBLY
        #******************
        model.rootAssembly.DatumCsysByDefault(abaqusConstants.CARTESIAN)
        
        
        #************************
        # INSTANTIATING THE PLATE
        #************************
        model.rootAssembly.Instance( name      = 'plate',
                                     part      = part_plate,
                                     dependent = abaqusConstants.ON ).translate( vector = (0, 0, -self.plate_width / 2) )
        
        
        #*************************
        # INSTANTIATING THE BULLET
        #*************************
        instance_bullet = model.rootAssembly.Instance( name      = 'bullet',
                                                       part      = part_bullet,
                                                       dependent = abaqusConstants.ON )
        
        #********************
        # ROTATING THE BULLET
        #********************
        instance_bullet.rotateAboutAxis( axisPoint     = (0.0, 0.0, 0.0), 
                                         axisDirection = (  1,   0,   0), 
                                         angle         = -90 )
                                         
        #***********************
        # TRANSLATING THE BULLET
        #***********************
        delta_x = self.bullet_origin_x
        delta_y = ( self.bullet_origin_y - self.bullet_length / 2 )
        delta_z = self.bullet_origin_z
        
        instance_bullet.translate( vector = (delta_x, delta_y, delta_z) )
        
        
        #******************
        # GETTING INSTANCES
        #******************
        bullet = model.rootAssembly.instances['bullet']
        plate  = model.rootAssembly.instances['plate']
        
        
        #******************
        # INSTANCES CENTERS
        #******************
        A = np.array( object = [self.plate_origin_x,   self.plate_origin_y,  self.plate_origin_z], dtype = float )
        B = np.array( object = [self.bullet_origin_x, self.bullet_origin_y, self.bullet_origin_z], dtype = float )
        
        
        
        # Vettore direzione B->A (da proiettile a piastra)
        v    = A - B
        norm = np.linalg.norm(v)
        if norm == 0:
            
            print("Il centro del proiettile e della piastra coincidono")
            
        v /= norm  # vettore target normalizzato
        
        # Vettore iniziale: normale uscente dalla parte corta = +Y
        v0 = np.array( object = [0, 1, 0] )
        
        # Calcolo asse di rotazione (perpendicolare a entrambi i vettori)
        rotation_axis = np.cross(v0, v)
        axis_norm     = np.linalg.norm(rotation_axis)
        
        # Caso particolare: v0 e v sono paralleli o antiparalleli
        if axis_norm < 1e-10:
            
            # Calcolo prodotto scalare per determinare se sono paralleli o antiparalleli
            dot = np.dot(v0, v)
            if dot > 0:
                
                # Vettori già paralleli, nessuna rotazione necessaria
                print("Nessuna rotazione necessaria: normale già allineata")
                
            else:
                
                # Vettori antiparalleli: rotazione di 180° attorno a un asse perpendicolare
                # Scegliamo un asse perpendicolare a v0 (es. asse X o Z)
                if abs(v0[0]) < 0.9:
                    
                    rotation_axis = np.cross(v0, np.array([1, 0, 0]))
                    
                else:
                    
                    rotation_axis = np.cross(v0, np.array([0, 0, 1]))
                
                rotation_axis /= np.linalg.norm(rotation_axis)
                theta          = 180.0
                
                bullet.rotateAboutAxis( axisPoint     = B.tolist(),
                                        axisDirection = rotation_axis.tolist(),
                                        angle         = theta )
        else:
            
            # Normalizza l'asse di rotazione
            rotation_axis /= axis_norm
            
            # Calcolo angolo di rotazione
            dot       = np.clip(np.dot(v0, v), -1.0, 1.0)
            theta_rad = math.acos(dot)
            theta_deg = math.degrees(theta_rad)
            
            # Applica la rotazione attorno all'asse calcolato
            bullet.rotateAboutAxis( axisPoint     = B.tolist(),
                                    axisDirection = rotation_axis.tolist(),
                                    angle         = theta_deg )
        
        
        #******************************************
        # CALCULATING SPEED MAGNITUDE OF THE BULLET
        #******************************************
        #A                          = np.array( object = [self.plate_origin_x,   self.plate_origin_y,  self.plate_origin_z], dtype = float )
        #B                          = np.array( object = [self.bullet_origin_x, self.bullet_origin_y, self.bullet_origin_z], dtype = float )
        #direction                  = A - B
        #distance                   = np.linalg.norm(direction)
        #unit_direction             = direction / distance
        #velocity                   = self.bullet_speed * unit_direction
        #cosines                    = velocity / self.bullet_speed
        #angles                     = np.arccos(cosines)

        #self.bullet_speed_y = - self.bullet_speed * math.cos(math.radians(ALPHA_Y))
        #self.bullet_speed_x = self.bullet_speed * math.sin(math.radians(ALPHA_Y)) * math.cos(math.radians(ALPHA_X))
        #self.bullet_speed_z = self.bullet_speed * math.sin(math.radians(ALPHA_Y)) * math.sin(math.radians(ALPHA_X))

        #self.bullet_impact_angle_x = angles[0]
        #self.bullet_impact_angle_y = angles[1]
        #self.bullet_impact_angle_z = angles[2]

        
        
        
        #**************************************************
        # DEFINING A STEP
        #**************************************************
        step_1 = model.ExplicitDynamicsStep( name                     = 'Step-1', 
                                             previous                 = 'Initial', 
                                             description              = '',
                                             timePeriod               = self.time_period, 
                                             # CORREZIONE MASS SCALING:
                                             # La sequenza è rigorosa: (Obiettivo, Regione, Frequenza, Factor, DT, Type, ...)
                                             # 0.0 è il placeholder per "Factor" (indice 3)
                                             # 0.000002 è il Target Time Increment (indice 4)
                                             # BELOW_MIN è il Tipo (indice 5)
                                             massScaling              = ( (abaqusConstants.SEMI_AUTOMATIC, 
                                                                           abaqusConstants.MODEL, 
                                                                           abaqusConstants.AT_BEGINNING, 
                                                                           0.0, 
                                                                           # previously 0.00001, changed to 1e-6 -> as this decreases the
                                                                           # time increment, it should improve accuracy but increase computation time
                                                                           0.000001, 
                                                                           abaqusConstants.BELOW_MIN, 
                                                                           0, 0, 0.0, 0.0, 0, None), ),
                                             timeIncrementationMethod = abaqusConstants.AUTOMATIC_GLOBAL )
        
        
        #************************************************************
        # SPECIFYING WHICH FIELDS WE WANT IN OUTPUT AND THE FREQUENCY
        #************************************************************
        field = model.FieldOutputRequest( name           = 'F-Output-1', 
                                          createStepName = 'Step-1', 
                                          variables      = ('S', 'E', 'U', 'COORD'), 
                                          timeInterval   = self.output_interval )
        
        
        #**************************************************
        # CREATING BULLET RIGID BODY CONSTRAINT 
        #**************************************************
        
        
        #*****************************************************
        # CREATING REFERENCE POINT IN THE CENTER OF THE BULLET
        #*****************************************************
        RP_bullet_id     = model.rootAssembly.ReferencePoint( point = (delta_x, delta_y, delta_z) ).id
        RP_bullet_region = regionToolset.Region( referencePoints = (model.rootAssembly.referencePoints[RP_bullet_id], ) )
        RP_bullet_set    = model.rootAssembly.Set( name            = "projectile-rp",
                                                   referencePoints = (model.rootAssembly.referencePoints[RP_bullet_id],) )
        
        
        #**********************************************
        # ASSIGNING RIGID BODY CONSTRAINT TO THE BULLET
        #**********************************************
        model.RigidBody( name           = 'constraint-projectile-rigid-body',
                         refPointRegion = RP_bullet_region,
                         bodyRegion     = model.rootAssembly.instances['bullet'].sets['set-all'] )
        
        
        #*******************************************************************
        # CREATING A FILTER TO STOP THE RUN WHEN THE BULLET BOUNCES OR STOPS
        #*******************************************************************

        # Definiamo la regione: l'intera istanza della lastra
        # (Il Set 'set-all' viene creato automaticamente nell'assembly dall'omonimo Set della parte)
        plate_region = model.rootAssembly.instances['plate'].sets['set-all']

        # Definiamo un valore di soglia per l'energia cinetica (ALLKE)
        # Sotto questo valore, consideriamo l'oscillazione "irrilevante".
        self.ENERGY_THRESHOLD = 0.01 
        
        
        #*********************************************************************************
        # CREATING A FILTER TO STOP THE ANALYSIS WHEN THE ENERGY DROPS BELOW THE THRESHOLD
        #*********************************************************************************
        filter_plate_energy = model.ButterworthFilter( name            = "Filter-Energy",
                                                       cutoffFrequency = 100, # Frequenza per smorzare picchi veloci
                                                       operation       = abaqusConstants.MIN,
                                                       limit           = self.ENERGY_THRESHOLD, 
                                                       halt            = False ) # Ferma l'analisi 
        
        
        #**********************************************************************************
        # ADDING A HISTORY OUTPUT REQUEST FOR THE KINETIC ENERGY (ALLKE) OF THE ENTIRE SLAB
        #**********************************************************************************
        model.HistoryOutputRequest( name           = 'H-Output-Plate-Energy', 
                                    createStepName = 'Step-1', 
                                    region         = plate_region,
                                    variables      = ('ALLKE',), # Energia Cinetica Totale
                                    frequency      = 200,        # Frequenza di campionamento
                                    filter         = "Filter-Energy" )
        
        
        #***************************************************
        # DEFINING BOUNDARY CONDITIONS
        #***************************************************
        
        
        #*******************************************
        # CREATING A SET WITH ALL SIDES OF THE PLATE
        #*******************************************
        part_plate.SetByBoolean( name = 'surface-sides', 
                                 sets = ( part_plate.sets['surface-north'],  part_plate.sets['surface-south'],
                                          part_plate.sets['surface-west'],  part_plate.sets['surface-east'] ) )
        
        
        #******************************************************************************************************************
        # SETTING ZERO DISPLACEMENT CONSTRAINT ON ALL SIDES OF THE PLATE, ONLY THE TOP AND BOTTOM SURFACES ARE FREE TO MOVE
        #******************************************************************************************************************
        bc_sides = model.DisplacementBC( name           = 'FixedBC', 
                                         createStepName = 'Initial', 
                                         region         = model.rootAssembly.instances['plate'].sets['surface-sides'], 
                                         u1 = 0.0, 
                                         u2 = 0.0, 
                                         u3 = 0.0 )


        #***************************************************
        # PREDEFINED FIELD: INITIAL VELOCITY
        #***************************************************
        
        
        #*****************************************************************************************************
        # INITIAL VELOCITY ASSIGNED TO THE BULLET VIA A PREDEFINED FIELD (ASSOCIATED WITH ITS REFERENCE POINT)
        #*****************************************************************************************************
        velocity = model.Velocity( name      = "Bullet_Velocity",
                                   region    = RP_bullet_region,
                                   velocity1 = self.bullet_speed_x,
                                   velocity2 = self.bullet_speed_y,
                                   velocity3 = self.bullet_speed_z )
        
        
        #***************************************************
        # INTERACTION: SURFACE-TO-SURFACE CONTACT
        #***************************************************

        # interaction properties
        interaction_properties = model.ContactProperty('IntProp-1')
        interaction_properties.TangentialBehavior( formulation        = abaqusConstants.PENALTY, 
                                                   table              = ((0.5, ), ), 
                                                   maximumElasticSlip = abaqusConstants.FRACTION, 
                                                   fraction           = 0.005 )
        
        interaction_properties.NormalBehavior( pressureOverclosure = abaqusConstants.HARD)
        
        
        #***********************************************
        # DEFINITION OF CONTACT BETWEEN BULLET AND PLATE
        #***********************************************
        model.SurfaceToSurfaceContactExp( name                = 'Int-1', 
                                          createStepName      = 'Initial', 
                                          main                = model.rootAssembly.instances['bullet'].surfaces['surface-all'],
                                          secondary           = model.rootAssembly.instances['plate'].surfaces['surface-top'], 
                                          sliding             = abaqusConstants.FINITE,
                                          interactionProperty = 'IntProp-1',
                                          mechanicalConstraint = abaqusConstants.PENALTY
                                          #weightingFactorType  = abaqusConstants.SPECIFIED,
                                          #weightingFactor      = 0.01
                                        ) # Riduci la rigidità al 10%
        
        
        #********************************************
        # CREATING PLATE MESH
        #********************************************
        # LIBRERIA ELEMENTI 3D:
        # https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/usb/default.htm?startat=pt06ch22s01ael03.html#usb-elm-e3delem
        
        edge_top_north    = part_plate.edges.findAt( coordinates = (                  0,                  0,                  0) )
        edge_top_south    = part_plate.edges.findAt( coordinates = (                  0,                  0, self.plate_width/1) )
        edge_top_east     = part_plate.edges.findAt( coordinates = (+self.plate_width/2,                  0, self.plate_width/2) )
        edge_top_west     = part_plate.edges.findAt( coordinates = (-self.plate_width/2,                  0, self.plate_width/2) )
        
        edge_bottom_north = part_plate.edges.findAt( coordinates = (                  0, -self.plate_height,                  0) )
        edge_bottom_south = part_plate.edges.findAt( coordinates = (                  0, -self.plate_height, self.plate_width  ) )
        edge_bottom_east  = part_plate.edges.findAt( coordinates = (+self.plate_width/2, -self.plate_height, self.plate_width/2) )
        edge_bottom_west  = part_plate.edges.findAt( coordinates = (-self.plate_width/2, -self.plate_height, self.plate_width/2) )
        
        edge_ne           = part_plate.edges.findAt( coordinates = (+self.plate_width/2, -self.plate_height/2,                0) )
        edge_nw           = part_plate.edges.findAt( coordinates = (-self.plate_width/2, -self.plate_height/2,                0) )
        edge_se           = part_plate.edges.findAt( coordinates = (+self.plate_width/2, -self.plate_height/2, self.plate_width) )
        edge_sw           = part_plate.edges.findAt( coordinates = (-self.plate_width/2, -self.plate_height/2, self.plate_width) )
        
        
        #*************************************************************************
        # SEED ON DOUBLE-BIAS HORIZONTAL EDGES (I.E. A GRADIENT WITH THREE VALUES)
        #*************************************************************************
        part_plate.seedEdgeByBias( biasMethod  = abaqusConstants.DOUBLE,
                                   centerEdges = ( edge_top_north, 
                                                   edge_top_south, 
                                                   edge_top_east, 
                                                   edge_top_west,
                                                   edge_bottom_north, 
                                                   edge_bottom_south, 
                                                   edge_bottom_east, 
                                                   edge_bottom_west ),
                                   minSize     = self.plate_seed_top_bottom_min, 
                                   maxSize     = self.plate_seed_top_bottom_max )
        
        
        #******************************************************************************
        # SEED ON THE VERTICAL EDGES WITH SINGLE BIAS (I.E. A GRADIENT WITH TWO VALUES)
        #******************************************************************************
        part_plate.seedEdgeByBias( biasMethod = abaqusConstants.SINGLE, 
                                   end2Edges  = (edge_ne, edge_sw), 
                                   end1Edges  = (edge_nw, edge_se), 
                                   minSize    = self.plate_seed_sides_min, 
                                   maxSize    = self.plate_seed_sides_max )
        
        
        part_plate.generateMesh()
        
        # --- ASSEGNA ELEMENTI IBRIDI PER MATERIALE IPERELASTICO ---
        # FONDAMENTALE per evitare il "Volumetric Locking"
        
        # Seleziona tutte le celle (elementi solidi) della piastra
        plate_cells  = part_plate.cells
        plate_region = part_plate.Set( name  = 'plate-all-cells-region', 
                                       cells = plate_cells)
        
        # Definisci il tipo di elemento: C3D8R
        # (C3D8 = 8-node Brick, R = Reduced integration. Standard per Explicit)
        elemType = mesh.ElemType( elemCode          = abaqusConstants.C3D8R,
                                  elemLibrary       = abaqusConstants.EXPLICIT,
                                  hourglassControl  = abaqusConstants.ENHANCED, #or RELAX_STIFFNESS, che dice essere più robusto per la gomma
                                  distortionControl = abaqusConstants.ON )
        
        
        # Assegna il tipo di elemento alla regione della piastra
        part_plate.setElementType( regions   = plate_region, 
                                   elemTypes = (elemType,) )
                                   
                                   
        #********************************************
        # MESH BULLET
        #********************************************
        part_bullet.seedPart( size = self.bullet_seed_size )
        
        # questa parte si puo' meshare solo se gli elementi hanno una forma tetraedrica, e gli va detto esplicitamente
        part_bullet.setMeshControls( regions   = part_bullet.cells, 
                                     elemShape = abaqusConstants.TET )
        
        part_bullet.generateMesh()
        
        
        #***************
        # CREATING A JOB
        #***************
        JOB_NAME = "Simulation_Job_" + str(self.index)
        
        
        # Imposta il numero di core (numCpus). 
        # Metti un numero inferiore ai tuoi core totali (es. se hai 8 core, metti 6).
        # numDomains deve essere uguale a numCpus per Abaqus/Explicit.
        NUM_CORES = 4

        job = mdb.Job( name              = JOB_NAME, 
                       model             = MODEL_NAME,
                       numCpus           = NUM_CORES, 
                       numDomains        = NUM_CORES, 
                       explicitPrecision = abaqusConstants.DOUBLE )
        
        
        #****************************************
        # SAVING INPUT FILE AS ("<NOME JOB>.INP")
        #****************************************
        if SAVEJOBINPUT:
            
            job.writeInput()
        
        
        #*******************
        # SUBMITTING THE JOB:
        #*******************
        job.submit()
        job.waitForCompletion()
        
        
        end_time     = time.time()
        elapsed_time = end_time - start_time
        #Aspetta che il file system rilasci il lock
        
        time.sleep(5)  # Aspetta 5 secondi prima di aprire l'ODB

        try:
            
            odb = session.openOdb(JOB_NAME + '.odb')
        
        except Exception as e:
            
            # Se fallisce ancora, aspetta altri 10 secondi e riprova
            print("ODB file stuck, try again in 10 seconds...")
            time.sleep(10)
            odb = session.openOdb(JOB_NAME + '.odb')
        
        
        #*****************************************************************
        # CHECKING SIMULATION DURATION AND COMPLETION (FROM ENERGY FILTER)
        #*****************************************************************

        # LOADING DATABASE
        # (Nota: Odb viene aperto qui e riutilizzato nel blocco "SAVING OUTPUT IN FILE CSV" sotto)
        odb = session.openOdb(JOB_NAME + '.odb')
        
        simulation_length    = 0.0 # Valore di default
        simulation_completed = False # Valore di default
        
        
        try:
        
            if self.DEBUG:
                
                # Recupera la region corretta (l'intera lastra)
                log("DEBUG: Stampo le chiavi disponibili nel file ODB:")
                
                # 1. Stampa tutte le History Regions disponibili (per controllare il nome della Region)
                log("History Regions disponibili: " + str(odb.steps['Step-1'].historyRegions.keys()))
                
                # 2. Se la region è corretta, stampa le variabili disponibili al suo interno
                if 'ElementSet ASSEMBLY.PLATE.SET-ALL' in odb.steps['Step-1'].historyRegions.keys():
                    
                    debug_region = odb.steps['Step-1'].historyRegions['ElementSet ASSEMBLY.PLATE.SET-ALL']
                    log("Variabili disponibili nella region PLATE: " + str(debug_region.historyOutputs.keys()))
            
            
            # Nota: il nome della HistoryRegion e' basato sul Set della PARTE
            #plate_history_region = odb.steps['Step-1'].historyRegions['ElementSet ASSEMBLY.PLATE.SET-ALL']
            plate_history_region = odb.steps['Step-1'].historyRegions['ElementSet PLATE.SET-ALL']
            
            # Recupera i dati filtrati (Abaqus nomina l'output 'VARIABILE_FILTRO')
            energy_data = plate_history_region.historyOutputs['ALLKE'].data  #TODO: cambiare nome se si cambia il filtro
            
            # L'ultimo punto (tempo, valore)
            simulation_length, final_energy = energy_data[-1]
            
            # Controlla se il filtro ha funzionato (energia <= soglia)
            simulation_completed = (final_energy <= self.ENERGY_THRESHOLD) 

            log(f"--- Simulazione {self.index} completata in {elapsed_time:.2f} secondi. ---")
            #log(f"Durata effettiva della simulazione (secondi): {simulation_length}")
            #log(f"Energia cinetica finale della lastra: {final_energy} (Completata: {simulation_completed})")
            # --------------------------------

        except (KeyError, IndexError):
            
            # KeyError: Il filtro/history output non trovato
            # IndexError: Trovato ma vuoto (simulazione fallita subito)
            simulation_length    = odb.steps['Step-1'].frames[-1].frameValue
            simulation_completed = False # Non sappiamo se si e' fermata correttamente
            
            log(f"--- Simulazione {self.index} completata (fallback time). ---")
            log(f"Durata effettiva della simulazione (secondi): {simulation_length}")
            log("ATTENZIONE: Impossibile leggere il filtro 'ALLKE_FILTER-ENERGY'.")
    
    
        #**************************
        # SAVING OUTPUT IN FILE CSV
        #**************************
        
        #*******************
        # GETTING THE FRAMES
        #*******************
        firstFrame = odb.steps['Step-1'].frames[0]
        lastFrame  = odb.steps['Step-1'].frames[-1]


        # Frame 1/30 secondi prima dell'impatto
        frameOne30BeforeImpact = None

        for frame in odb.steps['Step-1'].frames:
            # visto che la simulazione e' fatta nel dominio del tempo (il campo domain di Step e' AbaqusConstants.TIME),
            # allora frameValue e' il tempo del frame
            if (frame.frameValue >= 1/30):
                #log("found frame one")
                frameOne30BeforeImpact = frame
                break
        
        
        if frameOne30BeforeImpact == None:
            
            log("frame one not found")
        
        
        #********************
        # REGIONS OF INTEREST
        #********************
        outputRegionExternal       = odb.rootAssembly.instances['PLATE'].nodeSets['SURFACE-ALL']
        outputRegionBulletExternal = odb.rootAssembly.instances['BULLET'].nodeSets['SURFACE-ALL']
        
        
        #****************************
        # SAVING COORDINATES OF PLATE
        #****************************
        if SAVEPLATECOORDINATES:
            
            #********************
            # INITIAL COORDINATES
            #********************
            # ****** Initial coordinates ******
            coordinates_plate = firstFrame.fieldOutputs['COORD'].getSubset(region = outputRegionExternal)
            coordinates_plate_df = pd.DataFrame( { 'Id'       : [ values.nodeLabel for values in coordinates_plate.values ],
                                                    'X_Coord' : [ values.data[0]   for values in coordinates_plate.values ],
                                                    'Y_Coord' : [ values.data[1]   for values in coordinates_plate.values ],
                                                    'Z_Coord' : [ values.data[2]   for values in coordinates_plate.values ] } )
            coordinate_plate_filename = os.path.join(self.new_path, 'plate_initial_coordinates.csv')
            coordinates_plate_df.to_csv(coordinate_plate_filename, index = False)
        
        
        #*****************************
        # SAVING COORDINATES OF BULLET
        #*****************************
        if SAVECOORDINATES:
            
            #*************************************************
            # INITIAL COORDINATES = 2/30 SECONDS BEFORE IMPACT
            #*************************************************
            coordinates_bullet_1    = firstFrame.fieldOutputs['COORD'].getSubset( region = outputRegionBulletExternal )
            
            coordinates_bullet_1_df = pd.DataFrame( { 'Id'      : [ values.nodeLabel for values in coordinates_bullet_1.values ],
                                                      'X_Coord' : [ values.data[0]   for values in coordinates_bullet_1.values ],
                                                      'Y_Coord' : [ values.data[1]   for values in coordinates_bullet_1.values ],
                                                      'Z_Coord' : [ values.data[2]   for values in coordinates_bullet_1.values ] } )
                                                      
            coordinate_bullet_1_filename = os.path.join(self.new_path, str(self.index) + '_input_coordinates_bullet_1.csv')
            
            coordinates_bullet_1_df.to_csv( path_or_buf = coordinate_bullet_1_filename, 
                                            index       = False )
            
            
            #***************************************
            # COORDINATES 1/30 SECONDS BEFORE IMPACT
            #***************************************
            if frameOne30BeforeImpact != None:
                
                
                coordinates_bullet_2    = frameOne30BeforeImpact.fieldOutputs['COORD'].getSubset( region = outputRegionBulletExternal )
                
                coordinates_bullet_2_df = pd.DataFrame( { 'Id'      : [ values.nodeLabel for values in coordinates_bullet_2.values ],
                                                          'X_Coord' : [ values.data[0]   for values in coordinates_bullet_2.values ],
                                                          'Y_Coord' : [ values.data[1]   for values in coordinates_bullet_2.values ],
                                                          'Z_Coord' : [ values.data[2]   for values in coordinates_bullet_2.values ] } )
                                                        
                coordinate_bullet_2_filename = os.path.join( self.new_path, str(self.index) + '_input_coordinates_bullet_2.csv')
                
                coordinates_bullet_2_df.to_csv( path_or_buf = coordinate_bullet_2_filename, 
                                                index       = False )
        
        
        #***************************
        # SAVING PLATE DISPLACEMENTS
        #***************************
        if SAVEDISPLACEMENT:
            
            log("Extracting plate displacements using Pandas Chunks (Fast & Clean)...")

            displacement_output_filename = os.path.join(self.new_path, str(self.index) + '_output_displacement_all_frames.csv')
            
            # 1. Scriviamo prima l'header manualmente per sicurezza e pulizia
            with open(displacement_output_filename, 'w') as f:
                f.write("Time,Id,X_Disp,Y_Disp,Z_Disp\n")
            
            # 2. Iteriamo sui frame e usiamo Pandas in modalità "append"
            for frame in odb.steps['Step-1'].frames:
                
                frame_time = frame.frameValue
                
                # Estrazione veloce vettoriale (bulk)
                u_field = frame.fieldOutputs['U'].getSubset(region=outputRegionExternal)
                
                for block in u_field.bulkDataBlocks:
                    
                    # Dati grezzi dal blocco (Numpy array)
                    node_labels = block.nodeLabels
                    data        = block.data # [Ux, Uy, Uz]
                    num_nodes   = len(node_labels)
                    
                    # Creiamo la colonna del tempo
                    times = np.full(num_nodes, frame_time)
                    
                    # 3. Creiamo un DataFrame TEMPORANEO solo per questo blocco
                    # Questo è veloce perché i dati sono già in memoria come array
                    df_chunk = pd.DataFrame({
                        'Time'  : times,
                        'Id'    : node_labels,
                        'X_Disp': data[:, 0],
                        'Y_Disp': data[:, 1],
                        'Z_Disp': data[:, 2]
                    })
                    
                    # Assicuriamoci che l'ID sia scritto come intero (senza decimali)
                    df_chunk['Id'] = df_chunk['Id'].astype(int)
                    
                    # 4. Appendiamo al CSV
                    # float_format=None (default) usa il formatter "smart" di Python:
                    # 0.0 -> "0.0"
                    # 1.5e-9 -> "1.5e-09"
                    df_chunk.to_csv(displacement_output_filename, 
                                    mode='a',        # append
                                    header=False,    # header già scritto
                                    index=False)     # niente indice 0,1,2...

            log(f"Saved all displacements to {displacement_output_filename}")
        
        
        #****************
        # SAVING DATABASE
        #****************
        if SAVEDATABASE:
            
            mdb.saveAs( str(self.index) + '.cae' )
        
        
        #*****************************************
        # DELETING EXTRA FILES GENERATED BY ABAQUS
        #*****************************************
        files_ext = [ 
                    '.jnl',
                    '.sel',
                    '.res', 
                    '.lck',
                    #'.dat',
                    #'.msg', 
                    #'.sta',
                    '.fil',
                    '.sim',
                    '.stt',
                    '.mdl',
                    '.prt', 
                    '.ipm',
                    '.log',
                    '.com', 
                    '.odb_f',
                    '.abq',
                    '.pac',
                    '.rpy' 
                    ]
        
        if not SAVEJOBINPUT:
            
            files_ext.append('.inp')
        
        
        for file_ex in files_ext:
            
            file_path = JOB_NAME + file_ex
        
            if os.path.exists(file_path):
                os.remove(file_path)
        
        if os.path.exists("abq.app_cache"):
            
            os.remove("abq.app_cache")
        
        
        #******************************
        # RETURNING TO PARENT DIRECTORY
        #******************************
        os.chdir( self.previous_path )
    
        return simulation_length, simulation_completed
        
        
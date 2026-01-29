import os
import torch

BASE_DATASET_PATH = "../Abaqus simulations/Mooney-Rivlin simulations"
PT_DATASET_PATH   = "../data/processed_data"
LOG_FILE = "training_log.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "best_model.pth"

#Hardware parameters
NUM_WORKERS = 10

#Training parameters
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

BATCH_SIZE = 2          
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4

MAX_DATASET_SAMPLES = 10000

#Transformer parameters
D_MODEL = 512
DIM_FEEDFORWARD = D_MODEL * 4
N_HEADS = 8
NUM_LAYERS = 12
DROPOUT = 0.0

# Parametri Early Stopping 0,0001
# Nota: Con OneCycle l'early stopping è meno critico perché il LR scende alla fine,
# ma lo teniamo per salvare il checkpoint migliore.
ES_PATIENCE =  5    
ES_MIN_DELTA = 1e-4 

# Parametri Sliding Window
WINDOWS_SIZE = 10  
USE_WINDOWING = False

# Parametri One Cycle LR
ONE_CYCLE_PARAMS = {
    'max_lr': 5e-4,        # Picco massimo del Learning Rate
    'pct_start': 0.3,      # Percentuale di epoca per salire
    'div_factor': 25.0,     # Fattore di divisione iniziale
    'final_div_factor': 1000.0 # Fattore di divisione finale
}

# Parametri Reduce on Plateau
REDUCE_ON_PLATEAU_PARAMS = {
    'mode': 'min',
    'lr_factor': 0.5,
    'patience': 4,
    'min_lr': 1e-6,
    'start_lr': 1e-5,
}

# Pesi per la Loss Sobolev
SOBOLEV_LOSS_WEIGHTS = {
    'alpha_pos': 1.0,
    'alpha_vel': 1.0,
    'alpha_acc': 0.01
}
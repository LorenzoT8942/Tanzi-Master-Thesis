import torch
import numpy as np

class PhysicsScaler:
    def __init__(self):
        # Statistiche per i parametri statici (7)
        self.static_mean = None
        self.static_std = None
        
        # Statistiche per i displacement (Globali X, Y, Z o scalare unico)
        # Qui usiamo un unico fattore di scala per preservare le proporzioni X/Y/Z
        self.disp_max = 1.0 
        
    def fit(self, static_data, dynamic_data):
        """
        Calcola statistiche dal dataset.
        static_data: (M, 7) numpy array
        dynamic_data: (M, T, N*3) o (M*T*N, 3) numpy array di campioni
        """
        # 1. Static: Standard Scaling (Z-score)
        self.static_mean = np.mean(static_data, axis=0)
        self.static_std = np.std(static_data, axis=0)
        # Evita divisione per zero
        self.static_std[self.static_std < 1e-6] = 1.0 
        
        # 2. Dynamic: MaxAbs Scaling (spesso meglio per deformazioni fisiche partendo da 0)
        # Calcoliamo il valore assoluto massimo globale osservato
        self.disp_max = np.max(np.abs(dynamic_data))
        if self.disp_max < 1e-6: self.disp_max = 1.0

    def transform(self, static_tensor, dynamic_tensor):
        # Applica normalizzazione
        device = static_tensor.device
        s_mean = torch.tensor(self.static_mean, device=device, dtype=static_tensor.dtype)
        s_std = torch.tensor(self.static_std, device=device, dtype=static_tensor.dtype)
        
        static_norm = (static_tensor - s_mean) / s_std
        dynamic_norm = dynamic_tensor / self.disp_max
        return static_norm, dynamic_norm

    def transform_static(self, x):
        # x: Tensor (Batch, 7)
        device = x.device
        mean = torch.tensor(self.static_mean, device=device, dtype=x.dtype)
        std = torch.tensor(self.static_std, device=device, dtype=x.dtype)
        return (x - mean) / std

    def transform_dynamic(self, x):
        # x: Tensor (Batch, Seq, N*3)
        # Semplice divisione per il max globale
        return x / self.disp_max

    def inverse_transform_dynamic(self, x):
        # Per tornare in mm durante l'inferenza
        return x * self.disp_max
    
    def save(self, path):
        torch.save({
            'static_mean': self.static_mean,
            'static_std': self.static_std,
            'disp_max': self.disp_max
        }, path)
        
    def load(self, path):
        data = torch.load(path)
        self.static_mean = data['static_mean']
        self.static_std = data['static_std']
        self.disp_max = data['disp_max']
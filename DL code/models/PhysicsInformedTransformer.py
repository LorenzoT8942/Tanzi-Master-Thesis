import torch
import torch.nn as nn
import math

class PhysicsInformedTransformer(nn.Module):
    def __init__(
        self, 
        num_nodes, 
        num_physics_params, 
        d_model=512, 
        nhead=8, 
        num_layers=6, 
        dim_feedforward=2048, 
        dropout=0.1
    ):
        """
        Args:
            num_nodes (int): Numero totale di nodi nella mesh (es. ~3085).
            num_physics_params (int): Numero di parametri di input dal JSON (es. 7).
            d_model (int): Dimensione dello spazio latente (embedding).
            nhead (int): Numero di teste per la Multi-Head Attention.
            num_layers (int): Numero di strati del Transformer.
        """
        super(PhysicsInformedTransformer, self).__init__()
        
        self.d_model = d_model
        self.output_dim = num_nodes * 3  # X, Y, Z per ogni nodo
        
        # 1. Embedding dei Parametri Fisici (Context)
        # Proietta i parametri scalari (velocità, raggio, angoli) nello spazio del modello
        self.physics_embedding = nn.Sequential(
            nn.Linear(num_physics_params, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 2. Embedding dello Stato della Mesh (Input Frames)
        # Comprime lo stato ad alta dimensionalità (N_nodes * 3) in d_model
        self.state_embedding = nn.Linear(self.output_dim, d_model)

        # 3. Positional Encoding
        # Necessario perché il Transformer non ha nozione intrinseca di sequenza temporale
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 4. Core Transformer (Decoder-Only architecture)
        # batch_first=True per input (Batch, Seq, Feature)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # 5. Head di Predizione
        # Proietta l'embedding latente indietro nello spazio fisico (Displacement dei nodi)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.output_dim)
        )

    def _generate_square_subsequent_mask(self, sz):
        """Genera una maschera causale per impedire al modello di vedere il futuro."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, physics_params, past_frames):
        """
        Args:
            physics_params: Tensor (Batch, num_physics_params) - Dati dal JSON
            past_frames: Tensor (Batch, Seq_Len, num_nodes*3) - Dati dal CSV (storia)
        
        Returns:
            prediction: Tensor (Batch, Seq_Len, num_nodes*3) - Predizione del prossimo step
        """
        # A. Preparazione degli Embedding
        # Embed dei parametri fisici e reshape per trattarlo come il primo token (t=0 context)
        phys_emb = self.physics_embedding(physics_params).unsqueeze(1) # (Batch, 1, d_model)
        
        # Embed della sequenza temporale degli spostamenti
        state_emb = self.state_embedding(past_frames) # (Batch, Seq_Len, d_model)
        
        # B. Concatenazione (Prompting Fisico)
        # Prepariamo la sequenza: [Parametri Fisici, Frame_t0, Frame_t1, ... Frame_tn]
        # Il modello userà i parametri fisici come contesto globale per generare la dinamica
        tokens = torch.cat([phys_emb, state_emb], dim=1)
        
        # C. Positional Encoding
        tokens = self.pos_encoder(tokens)
        
        # D. Maschera Causale
        # Assicura che la predizione al tempo t dipenda solo da t e dai tempi precedenti
        seq_len = tokens.size(1)
        mask = self._generate_square_subsequent_mask(seq_len).to(tokens.device)
        
        # E. Passaggio nel Transformer
        output = self.transformer_decoder(tokens, mask=mask)
        
        # F. Predizione
        # Prendiamo l'output corrispondente ai frame (ignorando il token dei parametri fisici nell'output)
        # Shiftiamo di 1 perché l'output alla posizione i predice i+1
        prediction_tokens = output[:, 1:, :] 
        
        reconstructed_displacement = self.output_head(prediction_tokens)
        
        return reconstructed_displacement

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
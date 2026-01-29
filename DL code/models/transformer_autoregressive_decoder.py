import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config as conf

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class PlateDeformationTransformer(nn.Module):
    def __init__(
        self, 
        num_nodes=1784, 
        static_param_dim=7, 
        d_model=conf.D_MODEL, 
        nhead=conf.N_HEADS, 
        num_layers=conf.NUM_LAYERS, 
        dim_feedforward=conf.DIM_FEEDFORWARD, 
        dropout=conf.DROPOUT
    ):
        """
        Args:
            num_nodes: Numero di nodi nella mesh (3136).
            static_param_dim: Numero di parametri di simulazione (7).
            d_model: Dimensione dello spazio latente del Transformer.
        """
        super().__init__()
        self.state_dim = num_nodes * 3
        
        # 1. Encoders
        # Proietta lo stato fisico (5352) nello spazio latente
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Proietta i parametri statici (es. velocità sfera) nello spazio latente
        self.static_encoder = nn.Sequential(
            nn.Linear(static_param_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 2. Transformer Core (Decoder-only per autoregressione)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 3. Output Head
        self.output_head = nn.Linear(d_model, self.state_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, dynamic_seq, static_params):
        """
        Training Forward Pass.
        
        Args:
            dynamic_seq: (Batch, Seq_Len, N*3) - Sequenza storica deformazioni
            static_params: (Batch, Static_Dim) - Parametri simulazione
            
        Returns:
            predicted_seq: (Batch, Seq_Len, N*3) - Predizione next-token
        """
        batch_size, seq_len, _ = dynamic_seq.shape

        # Embedding
        # (Batch, Seq_Len, d_model)
        x_emb = self.state_encoder(dynamic_seq)
        
        # (Batch, 1, d_model)
        static_emb = self.static_encoder(static_params).unsqueeze(1)
        
        # Concateniamo i parametri statici all'inizio della sequenza come contesto
        # Nuova seq: [Static_Token, Frame_0, Frame_1, ..., Frame_T]
        input_seq = torch.cat([static_emb, x_emb], dim=1)
        
        # Add Positional Encoding
        input_seq = self.pos_encoder(input_seq)
        
        # Causal Masking (fondamentale per autoregressione)
        seq_len_total = input_seq.size(1)
        mask = self._generate_square_subsequent_mask(seq_len_total).to(input_seq.device)
        
        # Transformer Pass
        # Poiché usiamo solo il Decoder in modalità GPT, passiamo input_seq sia come tgt che memory 
        # (oppure usiamo TransformerEncoder con maschera causale, qui uso Decoder per flessibilità futura)
        output_latent = self.transformer_decoder(tgt=input_seq, memory=input_seq, tgt_mask=mask)
        
        # Rimuoviamo il token statico dall'output per allineare le predizioni
        # L'output alla posizione 0 (Static Token) predice il Frame 0
        # L'output alla posizione t predice il Frame t
        #output_latent = output_latent[:, :-1, :] # Ignoriamo l'ultima predizione se non abbiamo il target t+1 qui
        
        # Se vogliamo che l'output abbia la stessa lunghezza dell'input dynamic_seq:
        # Il token statico predice frame 0. Frame 0 predice frame 1.
        # Quindi prendiamo tutto tranne l'ultimo elemento se stiamo facendo training shiftato
        # Per semplicità, restituiamo la sequenza completa proiettata
        
        output_phys = self.output_head(output_latent)
        
        return output_phys

    @torch.no_grad()
    def predict_rollout(self, static_params, initial_state, max_steps):
        """
        Autoregressive Inference Loop.
        """
        self.eval()
        batch_size = static_params.size(0)
        
        # Init sequence with static params embedding only (o con stato iniziale 0)
        current_state = initial_state.unsqueeze(1) # (Batch, 1, State_Dim)
        
        predictions = [current_state]
        
        for _ in range(max_steps):
            # Preparazione input corrente (concatenate tutti i frame precedenti)
            # Nota: In produzione si usa KV-caching per efficienza, qui implementazione naive
            full_seq = torch.cat(predictions, dim=1)
            
            # Forward
            # Nota: qui stiamo chiamando forward su tutta la sequenza cresciuta. 
            # Il modello restituirà la predizione per il prossimo step nell'ultimo token.
            out = self.forward(full_seq, static_params)
            
            # L'ultimo token è la predizione per t+1
            next_step = out[:, -1:, :]
            predictions.append(next_step)
            
        return torch.cat(predictions, dim=1)
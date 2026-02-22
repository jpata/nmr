#!/usr/bin/env python

import pandas
from sklearn.model_selection import train_test_split, KFold

df_big = pandas.read_parquet("/home/joosep/17296666/NMRexp_10to24_1_1004_sc_less_than_1.parquet")
df = df_big.head(50000)

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import re

# Vocabulary for multiplicities (1H splitting patterns)
MULT_MAP = {'s': 0, 'd': 1, 't': 2, 'q': 3, 'dd': 4, 'm': 5} # Add others as needed

def parse_1H_peaks(processed_data):
    if isinstance(processed_data, str):
        processed_data = ast.literal_eval(processed_data)

    peak_features = []
    for peak in processed_data:
        mult, j_couplings, integration, shift1, shift2 = peak

        # 1. Chemical Shift
        shift = (float(shift1) + float(shift2)) / 2.0

        # 2. Integration
        integ_val = float(re.sub(r'[^0-9.]', '', integration)) if integration else 1.0

        # 3. Multiplicity
        mult_val = MULT_MAP.get(mult, 5) 

        # 4. J-couplings: Parse, SORT, and PAD to 6 (as per the paper)
        if isinstance(j_couplings, str):
            j_couplings = ast.literal_eval(j_couplings)

        j_vals = [float(re.sub(r'[^0-9.]', '', j)) for j in j_couplings] if j_couplings else []

        # Sort descending to maintain permutation invariance within the J-couplings themselves
        j_vals.sort(reverse=True)

        # Pad to exactly 6 elements
        while len(j_vals) < 6:
            j_vals.append(0.0)

        # Truncate just in case a rare peak has more than 6 couplings
        j_vals = j_vals[:6]

        # Feature vector: shift(1) + int(1) + mult(1) + j_vals(6) = 9 features
        feature_vector = [shift, integ_val, mult_val] + j_vals
        peak_features.append(feature_vector)

    return torch.tensor(peak_features, dtype=torch.float32)

def parse_13C_peaks(processed_data):
    if isinstance(processed_data, str):
        processed_data = ast.literal_eval(processed_data)

    peak_features = []
    for peak in processed_data:
        shift, _, _ = peak
        # 13C is fully decoupled, so we generally only care about the chemical shift
        peak_features.append([float(shift)])

    return torch.tensor(peak_features, dtype=torch.float32)


class NMRDataset(Dataset):
    def __init__(self, df):
        # Group by SMILES to collect all available modalities for a single molecule
        self.grouped_df = df.groupby('SMILES')
        self.smiles_list = list(self.grouped_df.groups.keys())

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol_data = self.grouped_df.get_group(smiles)

        # Initialize empty tensors in case a modality is missing
        h1_tensor = torch.empty(0, 9) # 9 features: shift, int, mult, j1, j2, j3, j4, j5, j6
        c13_tensor = torch.empty(0, 1) # 1 feature: shift

        # Extract features depending on the NMR type available for this SMILES
        for _, row in mol_data.iterrows():
            if row['NMR_type'] == '1H NMR':
                h1_tensor = parse_1H_peaks(row['NMR_processed'])
            elif row['NMR_type'] == '13C NMR':
                c13_tensor = parse_13C_peaks(row['NMR_processed'])

        return {
            '1H': h1_tensor,
            '13C': c13_tensor,
            'SMILES': smiles
        }


import re
import torch
from torch.nn.utils.rnn import pad_sequence

class SMILESTokenizer:
    def __init__(self):
        # Define the special tokens required by the paper [cite: 248]
        self.PAD_TOKEN = "[PAD]"
        self.BOS_TOKEN = "[BOS]"
        self.EOS_TOKEN = "[EOS]"
        self.UNK_TOKEN = "[UNK]"

        self.special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]

        # Regex pattern to capture chemically meaningful units [cite: 245, 246]
        # 1. Bracketed expressions: \[.*?\] (e.g., [C@@H], [nH], [O-])
        # 2. Multi-character atoms: Br|Cl|Si|Se (add others if your dataset requires them)
        # 3. Single alphabetic characters (atoms like C, N, O, c, n, o)
        # 4. Ring numbers (0-9)
        # 5. Bonds and structural tokens: =, #, -, \, /, (, )
        self.pattern = re.compile(r"(\[.*?\]|Br|Cl|Si|Se|[A-Za-z]|[0-9]|[\(\)\-\+=\#\\/])")

        # In a real scenario, you would build this vocabulary by scanning your entire dataset.
        # For this example, we'll initialize a dynamic vocab that learns as it encodes.
        self.vocab2id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id2vocab = {idx: token for token, idx in self.vocab2id.items()}

    def tokenize(self, smiles):
        """Splits a SMILES string into a list of chemical tokens."""
        # Find all matches based on the regex pattern
        tokens = self.pattern.findall(smiles)
        return tokens

    def encode(self, smiles, update_vocab=False):
        """Converts a SMILES string into a tensor of token IDs, adding BOS and EOS."""
        tokens = self.tokenize(smiles)

        if update_vocab:
            for token in tokens:
                if token not in self.vocab2id:
                    new_id = len(self.vocab2id)
                    self.vocab2id[token] = new_id
                    self.id2vocab[new_id] = token

        # Map tokens to IDs, falling back to UNK if not found
        token_ids = [self.vocab2id.get(tok, self.vocab2id[self.UNK_TOKEN]) for tok in tokens]

        # Wrap with [BOS] and [EOS]
        final_ids = [self.vocab2id[self.BOS_TOKEN]] + token_ids + [self.vocab2id[self.EOS_TOKEN]]
        return torch.tensor(final_ids, dtype=torch.long)

    def decode(self, token_ids):
        """Converts token IDs back into a SMILES string (exact reversibility)[cite: 247]."""
        tokens = []
        for idx in token_ids:
            # Convert tensor item to int if necessary
            idx = idx.item() if torch.is_tensor(idx) else idx
            token = self.id2vocab.get(idx, self.UNK_TOKEN)

            # Skip special padding/control tokens during string reconstruction
            if token not in self.special_tokens:
                tokens.append(token)

        return "".join(tokens)


from torch.nn.utils.rnn import pad_sequence

tokenizer = SMILESTokenizer()

def nmr_collate_fn(batch):
    h1_list = []
    c13_list = []
    smiles_tensors = []

    for item in batch:
        h1_list.append(item['1H'])
        c13_list.append(item['13C'])

        # Encode the SMILES string into token IDs
        smiles_encoded = tokenizer.encode(item['SMILES'], update_vocab=False)
        smiles_tensors.append(smiles_encoded)

    # Pad spectral features (as we did before)
    h1_padded = pad_sequence(h1_list, batch_first=True, padding_value=0.0)
    c13_padded = pad_sequence(c13_list, batch_first=True, padding_value=0.0)

    h1_mask = (h1_padded.sum(dim=-1) != 0).bool()
    c13_mask = (c13_padded.sum(dim=-1) != 0).bool()

    # Pad SMILES sequences with the tokenizer's [PAD] ID (which is 0)
    smiles_padded = pad_sequence(
        smiles_tensors, 
        batch_first=True, 
        padding_value=tokenizer.vocab2id[tokenizer.PAD_TOKEN]
    )

    return {
        '1H': h1_padded,
        '1H_mask': h1_mask,
        '13C': c13_padded,
        '13C_mask': c13_mask,
        'SMILES_TOKENS': smiles_padded
    }


def get_dataloader(df):
    dataset = NMRDataset(df)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=nmr_collate_fn,
        #num_workers=4
    )
    return dataloader


import torch
import torch.nn as nn

class MAB(nn.Module):
    """Multihead Attention Block: Generalizes attention to interactions between two sets."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, X, Y):
        # X maps to Query; Y maps to Key and Value
        # Eq 1 & 2: H = LayerNorm(X + MHA(X, Y, Y)) -> Z = LayerNorm(H + FFN(H))
        attn_out, _ = self.attn(query=X, key=Y, value=Y)
        H = self.ln1(X + attn_out)
        out = self.ln2(H + self.ffn(H))
        return out

class ISAB(nn.Module):
    """Induced Set Attention Block: Reduces O(n^2) complexity to O(nm) using inducing points."""
    def __init__(self, d_model, num_heads, m_inducing_points):
        super().__init__()
        # Learnable inducing points I (m x d)
        self.I = nn.Parameter(torch.Tensor(1, m_inducing_points, d_model))
        nn.init.xavier_uniform_(self.I)

        self.mab1 = MAB(d_model, num_heads)
        self.mab2 = MAB(d_model, num_heads)

    def forward(self, X):
        batch_size = X.size(0)
        I_batch = self.I.expand(batch_size, -1, -1)

        # Eq 3: H = MAB(I, X) - Inducing points summarize global features
        H = self.mab1(I_batch, X)

        # Eq 4: Z = MAB(X, H) - Input peaks attend only to summarized features
        Z = self.mab2(X, H)
        return Z

class PMA(nn.Module):
    """Pooling by Multihead Attention: Aggregates a variable-sized set into a fixed-size representation."""
    def __init__(self, d_model, num_heads, k_seeds):
        super().__init__()
        # Learnable seed matrix S (k x d)
        self.S = nn.Parameter(torch.Tensor(1, k_seeds, d_model))
        nn.init.xavier_uniform_(self.S)

        self.mab = MAB(d_model, num_heads)

    def forward(self, Z):
        batch_size = Z.size(0)
        S_batch = self.S.expand(batch_size, -1, -1)

        # PMA_k(Z) = MAB(S, Z)
        G = self.mab(S_batch, Z)
        return G

class NMRTransPretextEncoder(nn.Module):
    """
    Unified Set Transformer Encoder outputting both equivariant and invariant representations 
    for different self-supervised pretext tasks.
    """
    def __init__(self, d_model=256, num_heads=8, num_layers=6, m_inducing_points=32, k_seeds=4):
        super().__init__()
        # Stack of L ISAB layers
        self.isab_layers = nn.ModuleList([
            ISAB(d_model, num_heads, m_inducing_points) for _ in range(num_layers)
        ])

        # Final pooling layer
        self.pma = PMA(d_model, num_heads, k_seeds)

    def forward(self, X):
        """
        X: Unordered set of peak embeddings of shape (Batch, N_peaks, d_model)
        """
        Z = X

        # 1. Pass through ISAB stack
        for isab in self.isab_layers:
            Z = isab(Z)

        # 2. Extract global context via PMA
        G = self.pma(Z)

        # RETURN EXPLANATION FOR PRETEXT TASKS:
        # Z: (Batch, N_peaks, d_model) -> Permutation-equivariant peak-level features. 
        #    Use this for Masked Peak Modeling or Spectral Denoising where you need to 
        #    reconstruct individual peak parameters.
        #
        # G: (Batch, k_seeds, d_model) -> Permutation-invariant global spectrum features.
        #    Use this (flattened or pooled) for Contrastive Learning to match 1H vs 13C 
        #    or Spectrum vs SMILES representations.

        return Z, G

class NMRModalEncoder(nn.Module):
    """Encodes a single NMR modality into peak-level (Z) and global (G) representations."""
    def __init__(self, d_input, d_model=256, num_heads=8, num_layers=6, m_inducing=32, k_seeds=4):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model) # Project raw features to d_model

        self.isab_layers = nn.ModuleList([
            ISAB(d_model, num_heads, m_inducing) for _ in range(num_layers)
        ])
        self.pma = PMA(d_model, num_heads, k_seeds)

    def forward(self, X):
        Z = self.input_proj(X)
        for isab in self.isab_layers:
            Z = isab(Z)
        G = self.pma(Z)
        return Z, G # Z: Peak-level (Equivariant), G: Global (Invariant)

class NMRTrans(nn.Module):
    """Full NMRTrans Model: Dual Set Encoders + Fusion + Autoregressive Decoder."""
    def __init__(self, d_model=256, vocab_size=500, max_smiles_len=128, k_seeds=4):
        super().__init__()

        # 1. Independent Set Encoders for 1H and 13C
        # 1H inputs: shift, int, mult, plus 6 padded J-couplings = 9 features
        self.encoder_1H = NMRModalEncoder(d_input=9, d_model=d_model, k_seeds=k_seeds)
        # 13C inputs: shift only = 1 feature
        self.encoder_13C = NMRModalEncoder(d_input=1, d_model=d_model, k_seeds=k_seeds)

        # 2. SMILES Target Embeddings
        self.smiles_embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding ONLY for the SMILES target sequence (since SMILES is ordered)
        self.target_pos_enc = nn.Embedding(max_smiles_len, d_model) 

        # 3. Autoregressive Decoder
        # The paper modifies a T5 architecture, removing cross-attention positional bias.
        # In PyTorch, standard TransformerDecoder achieves this if we don't pass memory PE.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, h1_x, h1_mask, c13_x, c13_mask, smiles_tgt, smiles_tgt_mask):
        """
        h1_x: (Batch, N_H, 9)
        c13_x: (Batch, N_C, 1)
        smiles_tgt: (Batch, L) - shifted right for teacher forcing
        """
        batch_size = h1_x.size(0)

        # --- ENCODING ---
        Z_H, G_H = self.encoder_1H(h1_x)
        Z_C, G_C = self.encoder_13C(c13_x)

        # --- FUSION ---
        # Eq 12: Concatenate global and peak-level cues along the sequence dimension
        # H_enc = [G_C, Z_C, G_H, Z_H]
        H_enc = torch.cat([G_C, Z_C, G_H, Z_H], dim=1)

        # Combine masks for cross-attention. 
        # The 'G' tokens (k_seeds) are always valid (unmasked = False).
        # The 'Z' tokens use the original padding masks from the dataloader.
        g_mask = torch.zeros((batch_size, G_C.size(1)), dtype=torch.bool, device=h1_x.device)
        memory_mask = torch.cat([g_mask, c13_mask, g_mask, h1_mask], dim=1)

        # --- DECODING ---
        # Embed the SMILES sequence and add positional ordering
        seq_length = smiles_tgt.size(1)
        positions = torch.arange(0, seq_length, device=smiles_tgt.device).unsqueeze(0)
        tgt_emb = self.smiles_embedding(smiles_tgt) + self.target_pos_enc(positions)

        # Cross-Attention Interface:
        # Notice we do NOT add any positional encodings to H_enc (the memory).
        # This explicitly removes positional bias, ensuring attention weights 
        # depend solely on chemical content compatibility, making it permutation invariant.
        out = self.decoder(
            tgt=tgt_emb, 
            memory=H_enc, 
            tgt_mask=smiles_tgt_mask,           # Causal mask to prevent looking ahead in SMILES
            memory_key_padding_mask=memory_mask # Ignores padded spectral zeros
        )

        logits = self.fc_out(out)
        return logits


import torch.optim as optim

# Assume NMRTrans and DataLoader are defined as we discussed previously
# from model import NMRTrans
# from data import get_dataloader

def train_nmrtrans():
    # --- 1. Setup & Hyperparameters ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vocabulary setup (based on the paper's custom tokenizer) [cite: 244, 245, 248]
    VOCAB_SIZE = 500  
    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2

    # Cross-entropy loss, ignoring the padding tokens [cite: 254]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # --- Cross-Validation Setup ---
    # Split unique SMILES to avoid leakage between modalities of the same molecule
    smiles_unique = df["SMILES"].unique()
    
    # Optional: Initial train/validation split using sklearn
    # train_val_smiles, test_smiles = train_test_split(smiles_unique, test_size=0.1, random_state=42)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # --- 2. Causal Masking Helper ---
    def generate_square_subsequent_mask(sz):
        """Prevents the decoder from looking ahead at future SMILES tokens."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    # --- 3. Cross-Validation Loop ---
    for fold, (train_idx, val_idx) in enumerate(kf.split(smiles_unique)):
        print(f"\n--- Starting Fold {fold + 1}/5 ---")
        
        train_smiles = smiles_unique[train_idx]
        val_smiles = smiles_unique[val_idx]

        train_df = df[df["SMILES"].isin(train_smiles)]
        val_df = df[df["SMILES"].isin(val_smiles)]

        train_loader = get_dataloader(train_df)
        val_loader = get_dataloader(val_df)

        # Initialize model and optimizer for each fold
        model = NMRTrans(vocab_size=VOCAB_SIZE).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        num_epochs = 10 # Reduced for demonstration, change as needed

        for epoch in range(num_epochs):
            # --- Training Phase ---
            model.train()
            epoch_train_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                h1_x = batch["1H"].to(device)
                c13_x = batch["13C"].to(device)
                h1_mask = (h1_x.sum(dim=-1) == 0).to(device)
                c13_mask = (c13_x.sum(dim=-1) == 0).to(device)
                smiles_tokens = batch["SMILES_TOKENS"].to(device)

                tgt_input = smiles_tokens[:, :-1]
                tgt_expected = smiles_tokens[:, 1:]
                tgt_seq_len = tgt_input.size(1)
                tgt_causal_mask = generate_square_subsequent_mask(tgt_seq_len)

                optimizer.zero_grad()
                logits = model(
                    h1_x=h1_x,
                    h1_mask=h1_mask,
                    c13_x=c13_x,
                    c13_mask=c13_mask,
                    smiles_tgt=tgt_input,
                    smiles_tgt_mask=tgt_causal_mask
                )

                logits_flat = logits.reshape(-1, logits.size(-1))
                tgt_expected_flat = tgt_expected.reshape(-1)
                loss = criterion(logits_flat, tgt_expected_flat)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += loss.item()

            # --- Validation Phase ---
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    h1_x = batch["1H"].to(device)
                    c13_x = batch["13C"].to(device)
                    h1_mask = (h1_x.sum(dim=-1) == 0).to(device)
                    c13_mask = (c13_x.sum(dim=-1) == 0).to(device)
                    smiles_tokens = batch["SMILES_TOKENS"].to(device)

                    tgt_input = smiles_tokens[:, :-1]
                    tgt_expected = smiles_tokens[:, 1:]
                    tgt_seq_len = tgt_input.size(1)
                    tgt_causal_mask = generate_square_subsequent_mask(tgt_seq_len)

                    logits = model(
                        h1_x=h1_x,
                        h1_mask=h1_mask,
                        c13_x=c13_x,
                        c13_mask=c13_mask,
                        smiles_tgt=tgt_input,
                        smiles_tgt_mask=tgt_causal_mask
                    )

                    logits_flat = logits.reshape(-1, logits.size(-1))
                    tgt_expected_flat = tgt_expected.reshape(-1)
                    loss = criterion(logits_flat, tgt_expected_flat)
                    epoch_val_loss += loss.item()

            print(f"Fold {fold+1} | Epoch {epoch+1} | Train Loss: {epoch_train_loss/len(train_loader):.4f} | Val Loss: {epoch_val_loss/len(val_loader):.4f}")

    print("\n--- Cross-Validation Completed ---")

if __name__ == "__main__":
    train_nmrtrans()

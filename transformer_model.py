import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
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


class FeatureGroupEmbedding(nn.Module):
    """Domain-aware feature embeddings for weather variables."""
    def __init__(self, feature_groups, input_dim, embed_dim):
        super().__init__()
        self.feature_groups = feature_groups
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Separate embedding for each feature group
        self.embeddings = nn.ModuleDict()
        for group_name, feature_indices in feature_groups.items():
            group_size = len(feature_indices)
            self.embeddings[group_name] = nn.Linear(group_size, embed_dim)

        # Combine all group embeddings
        total_embed_dim = len(feature_groups) * embed_dim
        self.combiner = nn.Linear(total_embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        group_embeds = []
        for group_name, feature_indices in self.feature_groups.items():
            # Extract features for this group
            group_features = x[:, :, feature_indices]  # (batch, seq_len, group_size)
            # Embed
            group_embed = self.embeddings[group_name](group_features)  # (batch, seq_len, embed_dim)
            group_embeds.append(group_embed)

        # Concatenate all group embeddings
        combined = torch.cat(group_embeds, dim=-1)  # (batch, seq_len, total_embed_dim)

        # Project to final embedding dimension
        output = self.combiner(combined)  # (batch, seq_len, embed_dim)
        return self.layer_norm(output)


class TransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer with pre-layer normalization."""
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Pre-norm architecture (more stable)
        # Self-attention block
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)

        # Feed-forward block
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output

        return x


class PrecipitationTransformer(nn.Module):
    """Advanced transformer model for precipitation forecasting."""
    def __init__(self, input_dim, seq_len, d_model=256, nhead=8, num_layers=4,
                 embed_dim=64, dropout=0.1, num_locations=5, feature_groups=None):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.embed_dim = embed_dim

        # Feature group embeddings (domain-aware)
        if feature_groups is not None:
            self.feature_embedding = FeatureGroupEmbedding(feature_groups, input_dim, embed_dim)
        else:
            # Fallback: simple linear projection
            self.feature_embedding = nn.Linear(input_dim, embed_dim)

        # Project from embed_dim to d_model
        self.input_projection = nn.Linear(embed_dim, d_model)

        # Location embeddings
        self.location_embedding = nn.Embedding(num_locations, d_model)

        # Positional encoding for temporal patterns
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Temporal pooling: learn to weight different timesteps
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        # Final prediction head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x, loc_idx):
        # x: (batch, seq_len, input_dim)
        # loc_idx: (batch,)
        batch_size = x.size(0)

        # Feature embeddings (domain-aware)
        x = self.feature_embedding(x)  # (batch, seq_len, embed_dim)

        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add location embeddings (broadcast across sequence)
        loc_emb = self.location_embedding(loc_idx)  # (batch, d_model)
        loc_emb = loc_emb.unsqueeze(1)  # (batch, 1, d_model)
        x = x + loc_emb  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # Pass through transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)  # (batch, seq_len, d_model)

        x = self.norm(x)  # (batch, seq_len, d_model)

        # Temporal pooling with learned attention weights
        attn_weights = self.temporal_attention(x)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)

        # Weighted sum over time dimension
        x = (x * attn_weights).sum(dim=1)  # (batch, d_model)

        # Final prediction
        output = self.output_head(x)  # (batch, 1)

        return output.squeeze(-1)  # (batch,)


class MultiTaskPrecipitationTransformer(nn.Module):
    """Multi-task transformer: predicts both rainfall amount AND rain/no-rain classification."""
    def __init__(self, input_dim, seq_len, d_model=256, nhead=8, num_layers=4,
                 embed_dim=64, dropout=0.1, num_locations=5, feature_groups=None):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.embed_dim = embed_dim

        # Feature group embeddings (domain-aware)
        if feature_groups is not None:
            self.feature_embedding = FeatureGroupEmbedding(feature_groups, input_dim, embed_dim)
        else:
            self.feature_embedding = nn.Linear(input_dim, embed_dim)

        # Project from embed_dim to d_model
        self.input_projection = nn.Linear(embed_dim, d_model)

        # Location embeddings
        self.location_embedding = nn.Embedding(num_locations, d_model)

        # Positional encoding for temporal patterns
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Transformer encoder layers (SHARED)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Temporal pooling: learn to weight different timesteps (SHARED)
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        # REGRESSION HEAD: Predicts continuous precipitation amount
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )

        # CLASSIFICATION HEAD: Predicts rain/no-rain (binary)
        # NOTE: Outputs LOGITS (not probabilities) for numerical stability
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
            # NO Sigmoid here - use logits with BCEWithLogitsLoss
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x, loc_idx):
        # x: (batch, seq_len, input_dim)
        # loc_idx: (batch,)
        batch_size = x.size(0)

        # Feature embeddings (domain-aware)
        x = self.feature_embedding(x)  # (batch, seq_len, embed_dim)

        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add location embeddings (broadcast across sequence)
        loc_emb = self.location_embedding(loc_idx)  # (batch, d_model)
        loc_emb = loc_emb.unsqueeze(1)  # (batch, 1, d_model)
        x = x + loc_emb  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # Pass through transformer encoder layers (SHARED)
        for layer in self.encoder_layers:
            x = layer(x)  # (batch, seq_len, d_model)

        x = self.norm(x)  # (batch, seq_len, d_model)

        # Temporal pooling with learned attention weights (SHARED)
        attn_weights = self.temporal_attention(x)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)

        # Weighted sum over time dimension
        pooled = (x * attn_weights).sum(dim=1)  # (batch, d_model)

        # TWO OUTPUTS
        regression_output = self.regression_head(pooled).squeeze(-1)  # (batch,) - precipitation amount
        classification_logits = self.classification_head(pooled).squeeze(-1)  # (batch,) - rain LOGITS (not probabilities)

        return regression_output, classification_logits


class LSTMBaseline(nn.Module):
    """LSTM baseline model for comparison."""
    def __init__(self, input_dim, seq_len, hidden_dim=128, num_layers=2,
                 dropout=0.1, num_locations=5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Location embeddings
        self.location_embedding = nn.Embedding(num_locations, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for location concat
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, loc_idx):
        # x: (batch, seq_len, input_dim)
        # loc_idx: (batch,)

        # LSTM encoding
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim)

        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # Location embedding
        loc_emb = self.location_embedding(loc_idx)  # (batch, hidden_dim)

        # Concatenate
        combined = torch.cat([last_hidden, loc_emb], dim=-1)  # (batch, hidden_dim*2)

        # Predict
        output = self.output_head(combined)  # (batch, 1)

        return output.squeeze(-1)  # (batch,)


def get_model(model_type, **kwargs):
    """Factory function to create models."""
    if model_type == 'transformer':
        return PrecipitationTransformer(**kwargs)
    elif model_type == 'multitask':
        return MultiTaskPrecipitationTransformer(**kwargs)
    elif model_type == 'lstm':
        return LSTMBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test model instantiation
    print("Testing PrecipitationTransformer...")

    # Sample config
    batch_size = 16
    seq_len = 24
    input_dim = 5
    num_locations = 5

    feature_groups = {
        'thermo': [0, 1],
        'hydro': [2, 3],
        'dynamic': [4]
    }

    model = get_model(
        'transformer',
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=256,
        nhead=8,
        num_layers=4,
        embed_dim=64,
        dropout=0.1,
        num_locations=num_locations,
        feature_groups=feature_groups
    )

    # Create dummy inputs
    x = torch.randn(batch_size, seq_len, input_dim)
    loc_idx = torch.randint(0, num_locations, (batch_size,))

    # Forward pass
    output = model(x, loc_idx)

    print(f"Input shape: {x.shape}")
    print(f"Location indices shape: {loc_idx.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nModel test passed successfully!")

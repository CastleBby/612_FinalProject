import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_feature_groups(feature_groups: Dict[str, Iterable[int]]) -> Dict[str, torch.Tensor]:
    """
    Normalise feature group definitions so downstream modules can iterate in a stable order.
    """
    ordered = {}
    for name in sorted(feature_groups.keys()):
        ordered[name] = torch.tensor(list(feature_groups[name]), dtype=torch.long)
    return ordered


class FeatureGroupEncoder(nn.Module):
    """
    Learns a domain-aware embedding per feature group and fuses them into the transformer input space.
    """

    def __init__(
        self,
        input_dim: int,
        feature_groups: Dict[str, Iterable[int]],
        embed_dim: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.register_buffer("_all_indices", torch.arange(input_dim), persistent=False)
        self.groups = _build_feature_groups(feature_groups)

        group_projections = {}
        for name, indices in self.groups.items():
            group_projections[name] = nn.Linear(indices.numel(), embed_dim)
        self.group_projections = nn.ModuleDict(group_projections)

        fused_dim = embed_dim * len(self.groups)
        self.fusion = nn.Linear(fused_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, d_model) domain-informed representation
        """
        batch, seq_len, _ = x.shape
        embeddings = []
        for name, indices in self.groups.items():
            idx = indices.to(x.device)
            group_slice = torch.index_select(x, -1, idx)
            emb = self.group_projections[name](group_slice)
            embeddings.append(emb)
        fused = torch.cat(embeddings, dim=-1)
        return self.fusion(fused)


class MultiScaleAttentionBlock(nn.Module):
    """
    Two-stage attention block that keeps the first attention module accessible for visualisation.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.attn1 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn2 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First attention at native temporal resolution
        attn_out1, _ = self.attn1(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + attn_out1)

        # Down-sampled attention to capture broader temporal context
        pooled = F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2, ceil_mode=True).transpose(1, 2)
        attn_out2, _ = self.attn2(pooled, pooled, pooled, need_weights=False)
        upsampled = F.interpolate(
            attn_out2.transpose(1, 2),
            size=x.size(1),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        x = self.norm2(x + upsampled)
        x = self.norm3(x + self.ff(x))
        return x


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class PrecipitationTransformer(nn.Module):
    """
    Sequence model for precipitation nowcasting with location conditioning.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        embed_dim: int,
        dropout: float,
        num_locations: int,
        feature_groups: Dict[str, Iterable[int]],
        return_cls: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.return_cls_default = return_cls

        self.feature_proj = nn.Linear(input_dim, d_model)
        self.group_encoder = FeatureGroupEncoder(
            input_dim=input_dim,
            feature_groups=feature_groups,
            embed_dim=embed_dim,
            d_model=d_model,
        )
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len + 32)
        self.location_embedding = nn.Embedding(num_locations, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.multi_scale = MultiScaleAttentionBlock(d_model=d_model, nhead=nhead, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        # Auxiliary rain/no-rain classifier; trains alongside regression
        self.class_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        loc_idx: Optional[torch.Tensor] = None,
        return_cls: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            loc_idx: (batch,) location indices
            return_cls: if True, also return rain/no-rain probability
        Returns:
            regression prediction, or (regression, classification) if requested
        """
        if return_cls is None:
            return_cls = self.return_cls_default

        base = self.feature_proj(x)
        group_features = self.group_encoder(x)
        h = base + group_features
        h = self.positional_encoding(h)

        if loc_idx is not None:
            loc_emb = self.location_embedding(loc_idx)
            h = h + loc_emb.unsqueeze(1)

        h = self.multi_scale(h)
        h = self.transformer(h)
        h = h[:, -1, :]  # last timestep focus
        reg_out = self.head(h).squeeze(-1)
        cls_out = torch.sigmoid(self.class_head(h)).squeeze(-1)
        if return_cls:
            return reg_out, cls_out
        return reg_out


def get_model(
    model_type: str,
    input_dim: int,
    seq_len: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    embed_dim: int,
    dropout: float,
    num_locations: int,
    feature_groups: Dict[str, Iterable[int]],
    return_cls: bool = False,
) -> nn.Module:
    if model_type != "transformer":
        raise ValueError(f"Unsupported model_type: {model_type}")
    return PrecipitationTransformer(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        embed_dim=embed_dim,
        dropout=dropout,
        num_locations=num_locations,
        feature_groups=feature_groups,
        return_cls=return_cls,
    )

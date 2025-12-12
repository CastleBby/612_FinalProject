"""
NowcastingGPT-style VQ-VAE for Time Series
Adapted from: https://github.com/Cmeo97/NowcastingGPT

Stage 1: VQ-VAE Tokenizer
- Encodes temporal sequences into discrete tokens
- Codebook for quantization
- Decoder reconstructs sequences

Original uses spatial (image) data, this version uses temporal (time series) data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer - the "codebook" in VQ-VAE.
    Maps continuous embeddings to discrete codebook entries.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings  # Size of codebook
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook: learnable discrete token embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, z):
        """
        Args:
            z: (batch, seq_len, embedding_dim) - continuous embeddings from encoder
        Returns:
            quantized: (batch, seq_len, embedding_dim) - quantized embeddings
            loss: VQ loss
            perplexity: codebook usage metric
        """
        # Flatten for quantization
        z_flattened = z.reshape(-1, self.embedding_dim)  # (batch*seq_len, dim)
        
        # Calculate distances to all codebook entries
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        
        # Find nearest codebook entry for each input
        encoding_indices = torch.argmin(distances, dim=1)  # (batch*seq_len,)
        
        # Quantize by looking up codebook entries
        quantized = self.embedding(encoding_indices)
        
        # Reshape back to sequence
        quantized = quantized.reshape_as(z)
        
        # VQ loss: commitment loss + codebook loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)  # Encoder commits to codebook
        q_latent_loss = F.mse_loss(quantized, z.detach())  # Codebook learns from encoder
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator: copy gradients from decoder to encoder
        quantized = z + (quantized - z).detach()
        
        # Calculate perplexity (measure of codebook usage)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices.reshape(z.shape[0], z.shape[1])


class TemporalEncoder(nn.Module):
    """
    Encoder: Maps continuous time series to continuous latent representations.
    Uses 1D convolutions (temporal convolutions) similar to spatial convolutions in original.
    """
    def __init__(self, input_dim=1, hidden_dims=[64, 128, 256], embedding_dim=512):
        super().__init__()
        
        layers = []
        in_channels = input_dim
        
        # Stack of 1D convolutions with downsampling
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim
        
        # Final projection to embedding dimension
        layers.append(nn.Conv1d(in_channels, embedding_dim, kernel_size=1))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - input time series
        Returns:
            z: (batch, latent_seq_len, embedding_dim) - latent representations
        """
        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        z = self.encoder(x)  # (batch, embedding_dim, latent_seq_len)
        z = z.transpose(1, 2)  # (batch, latent_seq_len, embedding_dim)
        return z


class TemporalDecoder(nn.Module):
    """
    Decoder: Reconstructs time series from quantized latent representations.
    Uses transposed 1D convolutions for upsampling.
    """
    def __init__(self, embedding_dim=512, hidden_dims=[256, 128, 64], output_dim=1):
        super().__init__()
        
        layers = []
        in_channels = embedding_dim
        
        # Stack of transposed 1D convolutions with upsampling
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim
        
        # Final projection to output dimension
        layers.append(nn.ConvTranspose1d(in_channels, output_dim, kernel_size=1))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z_q):
        """
        Args:
            z_q: (batch, latent_seq_len, embedding_dim) - quantized latents
        Returns:
            x_recon: (batch, seq_len, output_dim) - reconstructed time series
        """
        # Conv1d expects (batch, channels, length)
        z_q = z_q.transpose(1, 2)  # (batch, embedding_dim, latent_seq_len)
        x_recon = self.decoder(z_q)  # (batch, output_dim, seq_len)
        x_recon = x_recon.transpose(1, 2)  # (batch, seq_len, output_dim)
        return x_recon


class VQVAE(nn.Module):
    """
    Complete VQ-VAE model (Stage 1 of NowcastingGPT).
    
    Architecture matches original NowcastingGPT structure:
    Input → Encoder → Quantization (Codebook) → Decoder → Reconstruction
    
    Adapted for time series instead of images.
    """
    def __init__(
        self,
        input_dim=1,
        hidden_dims=[64, 128, 256],
        embedding_dim=512,
        num_embeddings=512,  # Codebook size
        commitment_cost=0.25
    ):
        super().__init__()
        
        self.encoder = TemporalEncoder(input_dim, hidden_dims, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = TemporalDecoder(embedding_dim, list(reversed(hidden_dims)), input_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - input time series
        Returns:
            x_recon: reconstructed input
            vq_loss: vector quantization loss
            perplexity: codebook usage
            encoding_indices: discrete token indices
        """
        # Encode to continuous latent
        z = self.encoder(x)
        
        # Quantize to discrete tokens
        z_q, vq_loss, perplexity, encoding_indices = self.vq_layer(z)
        
        # Decode from quantized latent
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss, perplexity, encoding_indices
    
    def encode(self, x):
        """Encode input to discrete token indices."""
        z = self.encoder(x)
        _, _, _, encoding_indices = self.vq_layer(z)
        return encoding_indices
    
    def decode_tokens(self, token_indices):
        """Decode from discrete token indices to continuous output."""
        # Look up embeddings for tokens
        z_q = self.vq_layer.embedding(token_indices)
        # Decode
        x_recon = self.decoder(z_q)
        return x_recon


class GPTDecoder(nn.Module):
    """
    GPT-style Autoregressive Transformer (Stage 2 of NowcastingGPT).
    
    Predicts future tokens autoregressively given past tokens.
    Uses causal masking to prevent looking at future tokens.
    """
    def __init__(
        self,
        vocab_size,  # Size of codebook
        d_model=512,
        nhead=8,
        num_layers=12,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=1024
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer decoder layers with causal masking
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection to vocab
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_indices, causal_mask=True):
        """
        Args:
            token_indices: (batch, seq_len) - discrete token indices
            causal_mask: whether to use causal masking (for training)
        Returns:
            logits: (batch, seq_len, vocab_size) - next token predictions
        """
        batch_size, seq_len = token_indices.shape
        
        # Token + positional embeddings
        positions = torch.arange(seq_len, device=token_indices.device).unsqueeze(0)
        x = self.token_embedding(token_indices) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Create causal mask (prevent attending to future)
        if causal_mask:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(token_indices.device)
        else:
            mask = None
        
        # Self-attention with causal masking
        # For decoder-only, we use the same input as both tgt and memory
        x = self.transformer(x, x, tgt_mask=mask)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, context_tokens, num_steps, temperature=1.0):
        """
        Autoregressive generation of future tokens.
        
        Args:
            context_tokens: (batch, context_len) - conditioning tokens
            num_steps: number of future steps to generate
            temperature: sampling temperature
        Returns:
            generated_tokens: (batch, context_len + num_steps)
        """
        generated = context_tokens
        
        for _ in range(num_steps):
            # Get predictions for next token
            logits = self.forward(generated, causal_mask=True)
            
            # Take logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class NowcastingGPT(nn.Module):
    """
    Complete NowcastingGPT model: VQ-VAE + GPT Transformer.
    
    Stage 1 (VQ-VAE): Continuous → Discrete tokens
    Stage 2 (GPT): Autoregressive token prediction
    Stage 1 (VQ-VAE Decoder): Discrete tokens → Continuous output
    
    Adapted for time series precipitation nowcasting.
    """
    def __init__(
        self,
        # VQ-VAE params
        input_dim=1,
        hidden_dims=[64, 128, 256],
        embedding_dim=512,
        num_embeddings=512,
        # GPT params
        d_model=512,
        nhead=8,
        num_layers=12,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()
        
        self.vqvae = VQVAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings
        )
        
        self.gpt = GPTDecoder(
            vocab_size=num_embeddings,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
    def forward(self, x, predict_future_steps=0):
        """
        Full forward pass through both stages.
        
        Args:
            x: (batch, seq_len, input_dim) - input sequences
            predict_future_steps: number of future steps to predict
        Returns:
            x_recon: reconstruction of input
            x_future: predicted future (if predict_future_steps > 0)
            vq_loss: VQ-VAE loss
            perplexity: codebook usage
        """
        # Stage 1: Encode to discrete tokens
        x_recon, vq_loss, perplexity, encoding_indices = self.vqvae(x)
        
        if predict_future_steps > 0:
            # Stage 2: Predict future tokens with GPT
            future_tokens = self.gpt.generate(
                encoding_indices,
                num_steps=predict_future_steps
            )
            
            # Stage 3: Decode future tokens
            all_tokens = future_tokens
            x_with_future = self.vqvae.decode_tokens(all_tokens)
            
            # Split into past (reconstruction) and future (prediction)
            x_future = x_with_future[:, x.shape[1]:, :]
            
            return x_recon, x_future, vq_loss, perplexity
        else:
            return x_recon, None, vq_loss, perplexity


if __name__ == '__main__':
    # Test NowcastingGPT architecture
    print("="*80)
    print("Testing NowcastingGPT Architecture for Time Series")
    print("="*80)
    
    # Model configuration (scaled down for 2-hour training on H100)
    model = NowcastingGPT(
        input_dim=1,
        hidden_dims=[64, 128, 256],
        embedding_dim=256,  # Reduced from 512
        num_embeddings=512,  # Codebook size
        d_model=256,  # Reduced from 512
        nhead=8,
        num_layers=8,  # Reduced from 12
        dim_feedforward=1024,  # Reduced from 2048
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    vqvae_params = sum(p.numel() for p in model.vqvae.parameters())
    gpt_params = sum(p.numel() for p in model.gpt.parameters())
    
    print(f"\nModel Parameters:")
    print(f"  VQ-VAE: {vqvae_params/1e6:.2f}M")
    print(f"  GPT Transformer: {gpt_params/1e6:.2f}M")
    print(f"  Total: {total_params/1e6:.2f}M")
    
    # Test forward pass
    batch_size = 4
    seq_len = 24
    x = torch.randn(batch_size, seq_len, 1)
    
    print(f"\nInput shape: {x.shape}")
    
    # Test VQ-VAE only
    x_recon, vq_loss, perplexity, tokens = model.vqvae(x)
    print(f"\nVQ-VAE:")
    print(f"  Reconstruction shape: {x_recon.shape}")
    print(f"  Token indices shape: {tokens.shape}")
    print(f"  Perplexity: {perplexity.item():.2f} (codebook usage)")
    
    # Test full model with prediction
    x_recon, x_future, vq_loss, perplexity = model(x, predict_future_steps=6)
    print(f"\nFull NowcastingGPT:")
    print(f"  Reconstruction: {x_recon.shape}")
    print(f"  Future prediction: {x_future.shape}")
    
    print("\n" + "="*80)
    print("✓ Architecture test passed!")
    print("="*80)


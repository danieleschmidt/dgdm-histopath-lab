"""
Dynamic Graph Diffusion Models - Core diffusion mechanism implementation.

This module implements the core diffusion process for graph-based representations
of histopathological tissue, including noise schedules and reverse processes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable
from torch import Tensor


class DiffusionScheduler:
    """Noise scheduler for diffusion process with multiple schedule types."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "cosine"
    ):
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        elif schedule == "sigmoid":
            self.betas = self._sigmoid_beta_schedule(num_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
            
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> Tensor:
        """Cosine schedule for better performance."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
        
    def _sigmoid_beta_schedule(
        self, timesteps: int, start: float = -3, end: float = 3
    ) -> Tensor:
        """Sigmoid schedule for controlled noise injection."""
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (end - start) + start


class DiffusionLayer(nn.Module):
    """
    Core diffusion layer that applies noise and learns reverse process.
    Integrates with graph structure for tissue-aware diffusion.
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_timesteps: int = 1000,
        schedule: str = "cosine",
        conditioning_dim: Optional[int] = None
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        # Initialize noise scheduler
        self.scheduler = DiffusionScheduler(num_timesteps, schedule=schedule)
        
        # Time embedding for timestep conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main denoising network
        self.denoise_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim * 2),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Optional conditioning network
        if conditioning_dim is not None:
            self.condition_net = nn.Linear(conditioning_dim, hidden_dim)
        else:
            self.condition_net = None
            
    def get_timestep_embedding(self, timesteps: Tensor, dim: int = 128) -> Tensor:
        """Sinusoidal timestep embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb
        
    def add_noise(
        self, 
        x_start: Tensor, 
        noise: Tensor, 
        timesteps: Tensor
    ) -> Tensor:
        """Add noise to clean samples according to diffusion schedule."""
        sqrt_alphas_cumprod = torch.sqrt(
            self.scheduler.alphas_cumprod.to(x_start.device)[timesteps]
        )
        sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.scheduler.alphas_cumprod.to(x_start.device)[timesteps]
        )
        
        # Expand dimensions for broadcasting
        while len(sqrt_alphas_cumprod.shape) < len(x_start.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)
            
        return (
            sqrt_alphas_cumprod * x_start + 
            sqrt_one_minus_alphas_cumprod * noise
        )
        
    def predict_noise(
        self,
        x_noisy: Tensor,
        timesteps: Tensor,
        condition: Optional[Tensor] = None
    ) -> Tensor:
        """Predict noise to be removed from noisy input."""
        # Get timestep embeddings
        t_emb = self.get_timestep_embedding(timesteps)
        t_emb = self.time_embed(t_emb)
        
        # Add conditioning if provided
        if condition is not None and self.condition_net is not None:
            cond_emb = self.condition_net(condition)
            t_emb = t_emb + cond_emb
            
        # Prepare input for denoising network
        # Expand t_emb to match x_noisy dimensions
        while len(t_emb.shape) < len(x_noisy.shape):
            t_emb = t_emb.unsqueeze(-2)
        t_emb = t_emb.expand(*x_noisy.shape[:-1], -1)
        
        # Concatenate noisy input with time embedding
        net_input = torch.cat([x_noisy, t_emb], dim=-1)
        
        return self.denoise_net(net_input)
        
    def forward(
        self,
        x_start: Tensor,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        condition: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward diffusion process.
        
        Args:
            x_start: Clean input tensor [batch_size, num_nodes, node_dim]
            timesteps: Timesteps to sample [batch_size]
            noise: Optional noise tensor
            condition: Optional conditioning tensor
            
        Returns:
            Tuple of (noisy_x, predicted_noise)
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(
                0, self.num_timesteps, (batch_size,), device=device
            )
            
        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Add noise to clean samples
        x_noisy = self.add_noise(x_start, noise, timesteps)
        
        # Predict the noise
        predicted_noise = self.predict_noise(x_noisy, timesteps, condition)
        
        return x_noisy, predicted_noise
        
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        condition: Optional[Tensor] = None,
        num_inference_steps: int = 50
    ) -> Tensor:
        """
        Sample from the diffusion model using DDPM sampling.
        
        Args:
            shape: Shape of samples to generate
            device: Device to generate samples on
            condition: Optional conditioning tensor
            num_inference_steps: Number of denoising steps
            
        Returns:
            Generated samples
        """
        # Start from pure noise
        sample = torch.randn(shape, device=device)
        
        # Create sampling schedule
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long
        )
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.predict_noise(sample, t_batch, condition)
            
            # Compute denoising step
            alpha = self.scheduler.alphas[t].to(device)
            alpha_cumprod = self.scheduler.alphas_cumprod[t].to(device)
            
            # Expand for broadcasting
            while len(alpha.shape) < len(sample.shape):
                alpha = alpha.unsqueeze(-1)
                alpha_cumprod = alpha_cumprod.unsqueeze(-1)
                
            # DDPM sampling formula
            pred_original_sample = (
                sample - torch.sqrt(1 - alpha_cumprod) * predicted_noise
            ) / torch.sqrt(alpha_cumprod)
            
            # Add noise for non-final steps
            if i < len(timesteps) - 1:
                noise = torch.randn_like(sample)
                variance = self.scheduler.posterior_variance[t].to(device)
                while len(variance.shape) < len(sample.shape):
                    variance = variance.unsqueeze(-1)
                sample = (
                    torch.sqrt(alpha) * pred_original_sample +
                    torch.sqrt(variance) * noise
                )
            else:
                sample = pred_original_sample
                
        return sample
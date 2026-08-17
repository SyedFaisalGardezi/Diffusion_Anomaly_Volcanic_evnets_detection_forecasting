from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from opensimplex import OpenSimplex
from cfg_ddim.utils import gather
# from cfg_ddim.experiment_cfg_ddim import Configs

class DenoiseDiffusion:
    """
    ## Denoise Diffusion with Time-Step Guidance (TSG) and DDIM Sampling
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device, guidance_scale: float = 3.5, lambda_max: int = 1000 , T_MIN: int = 50 , T_MAX: int = 900,  S: float = 1.0,  ALPHA: float = 900,):
        """
        * `eps_model` is the denoising model \( \epsilon_\theta(x_t, t) \)
        * `n_steps` is the number of timesteps \( T \)
        * `device` is the device to place constants on
        * `guidance_scale` controls the TSG guidance intensity
        """
        super().__init__()
        self.eps_model = eps_model
        self.guidance_scale = guidance_scale
        self.time_embedding = 256
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.device = device
        self.sigma2 = self.beta
        
        self.n_steps = n_steps
        self.lambda_max = lambda_max
        
        self.T_MIN = T_MIN
        self.T_MAX = self.n_steps - self.T_MIN
        
        self.S = S
        self.ALPHA = ALPHA
    
    def generate_simplex_noise(shape, base_freq=2**-6, octaves=6, decay=0.8, device='cpu'):
        """
        Generates multi-octave simplex noise.

        Parameters:
        - shape: Tuple[int, int, int] representing (B, T, N)
        - base_freq: Base frequency for the noise
        - octaves: Number of noise layers
        - decay: Amplitude decay factor
        - device: Device to place the tensor

        Returns:
        - Tensor of shape (B, T, N) with simplex noise
        """
        # dname = "SWAT"
        dname = None
        
        if dname == "SWAT":
            base_freq = 2**-7
            octaves = 5
            decay  = 0.85
            
        
        B, T, N = shape
        noise = torch.zeros(B, T, N, device=device)
        for b in range(B):
            seed = torch.randint(0, 10000, (1,)).item()
            simplex = OpenSimplex(seed)
            for t in range(T):
                for n in range(N):
                    amplitude = 1.0
                    frequency = base_freq
                    value = 0.0
                    for _ in range(octaves):
                        value += amplitude * simplex.noise2d(t * frequency, n * frequency)
                        frequency *= 2
                        amplitude *= decay
                    noise[b, t, n] = value
        return noise
    
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var
    # 
    # def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
    #     if eps is None:
    #         eps = torch.randn_like(x0)
    #     mean, var = self.q_xt_x0(x0, t)
    #     return mean + (var ** 0.5) * eps


    ## 2nd order 
    def q_sample_ode(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        One-shot ODE-based forward noise addition using deterministic DDIM-style formulation.
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        """
        if eps is None:
            # eps = torch.randn_like(x0)
            eps = generate_simplex_noise(x0.shape, device=x0.device)

        # [B] → [B, 1, 1] to broadcast with [B, C, L]
        alpha_bar_t = gather(self.alpha_bar, t)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bar_t).view(-1, 1, 1)

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * eps


    def get_time_embedding(self, t: torch.Tensor, dim: int = 256): ## embeding =256 to match the defination in 1
        
        half_dim = self.time_embedding // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    def get_power_schedule(self, t_emb: torch.Tensor, t: torch.Tensor, std_scaling=True):
        """
        Vectorized version of time-step perturbation (Power schedule)
        Args:
            t_emb: [B, D] time embeddings
            t: [B] timesteps
        Returns:
            [B, D] perturbed embeddings
        """
        T_MIN = self.T_MIN
        T_MAX = self.T_MAX 
        S =self.S 
        ALPHA = self.ALPHA

        # Create mask for valid time steps
        valid_mask = (t >= T_MIN) & (t <= T_MAX)

        # Compute noise scales (only for valid ones)
        noise_scale = S * (t.float() ** ALPHA)
        if std_scaling:
            emb_std = t_emb.std(dim=1, keepdim=True)  # shape [B, 1]
            noise_scale = noise_scale.unsqueeze(1) * emb_std  # shape [B, D]
        else:
            noise_scale = noise_scale.unsqueeze(1)  # shape [B, 1]

        # Generate noise
        noise = torch.randn_like(t_emb)

        # Apply noise only to valid positions
        perturbed = t_emb + noise_scale * noise * valid_mask.unsqueeze(1)

        return perturbed
    
    def time_step_guidance(self, xt: torch.Tensor, t: torch.Tensor):
        embed_dim = self.time_embedding
        t_emb = self.get_time_embedding(t, embed_dim)

        # ✅ Vectorized perturbation
        t_emb_perturbed = self.get_power_schedule(t_emb, t)

        eps_theta_cond = self.eps_model(xt, t, t_emb)
        eps_theta_uncond = self.eps_model(xt, t,  t_emb_perturbed)

        eps_theta = eps_theta_uncond + self.guidance_scale * (eps_theta_cond - eps_theta_uncond)
        return eps_theta



    
    def ddim_sample(self, xt: torch.Tensor, t: torch.Tensor, guidance_scale: float = 3.0, eval: bool = False):
        eps_theta = self.time_step_guidance(xt, t)
    
        # ✅ Use a temporary truncated version for inference
        alpha_bar = self.alpha_bar[:self.lambda_max] if eval else self.alpha_bar
    
        alpha_bar_t = gather(alpha_bar, t)
        if torch.any(t - 1 < 0):
            alpha_bar_t_prev = torch.ones_like(alpha_bar_t)
        else:
            alpha_bar_t_prev = gather(alpha_bar, t - 1)
    
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
        xt_minus_1 = torch.sqrt(alpha_bar_t_prev) * x0_pred + torch.sqrt(1 - alpha_bar_t_prev) * eps_theta
    
        return xt_minus_1
    
    def ddim_sample_heun(self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, eps_prev=None):
        # Get model output at current timestep
        eps_theta_t = self.time_step_guidance(x_t, t)

        # Predict x0
        alpha_bar_t = gather(self.alpha_bar, t)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta_t) / torch.sqrt(alpha_bar_t)

        # Heun method: Estimate x_{t-1}
        alpha_bar_t_prev = gather(self.alpha_bar, t_prev)
        x_t1_euler = torch.sqrt(alpha_bar_t_prev) * x0_pred + torch.sqrt(1 - alpha_bar_t_prev) * eps_theta_t

        # Re-evaluate model at x_t1 (Euler estimate)
        eps_theta_t1 = self.time_step_guidance(x_t1_euler, t_prev)

        # Average the two eps predictions
        eps_theta_avg = 0.5 * (eps_theta_t + eps_theta_t1)

        # Final 2nd-order update
        x_t1_heun = torch.sqrt(alpha_bar_t_prev) * x0_pred + torch.sqrt(1 - alpha_bar_t_prev) * eps_theta_avg
        return x_t1_heun



    def p_x0(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.time_step_guidance(xt, t)
        alpha_bar_t = gather(self.alpha_bar, t)
        return (xt - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
    
    # def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
    #     batch_size = x0.shape[0]
    # 
    #     # Random time step t and its previous t_prev for Heun method
    #     t = torch.randint(1, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)  # start from 1
    #     t_prev = torch.clamp(t - 1, min=0)  # ensure t_prev ≥ 0
    # 
    #     if noise is None:
    #         noise = torch.randn_like(x0)
    # 
    #     # Forward process using 2nd-order ODE (DDIM-style q_sample)
    #     x_t = self.q_sample_ode(x0, t, eps=noise)
    # 
    #     # Reverse using Heun's method to get x_{t-1}
    #     x_t1 = self.ddim_sample_heun(x_t, t, t_prev)
    # 
    #     # Forward sample again from x0 at t_prev
    #     x_t1_target = self.q_sample_ode(x0, t_prev, eps=noise)
    # 
    #     # L2 loss between Heun-denoised x_{t-1} and forward sampled x_{t-1}
    #     return F.mse_loss(x_t1, x_t1_target)


    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        ## in Ho.et.al they generated diff random t for each sample in a btach to cover diff t ranges
        # t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        
        # Set λ to 250 or 500
        # t = torch.randint(0, self.lambda_max, (batch_size,), device=x0.device, dtype=torch.long)
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        
        # print(f"\n time steps for one batch = {t}")
        if noise is None:
            noise = torch.randn_like(x0)

        # xt = self.q_sample(x0, t, eps=noise)
        # print(f"x shape in loss before q_sample_ode {x0.shape}")
        xt = self.q_sample_ode(x0, t, eps=noise)
        # print(f"xt shape in loss {xt.shape}")
        eps_theta = self.time_step_guidance(xt, t)

        return F.mse_loss(noise, eps_theta)

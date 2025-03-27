import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import test_vae
import test_clip
import torch

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PatchLatent(nn.Module):
    def __init__(self, in_channels, embed_dim, image_size, patch_size=2):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_latent = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
    
    def forward(self, x):
        device = x.device  
        x = self.patch_latent(x)          
        x = x.flatten(2).transpose(1, 2)    
        
        pos_embed = self.pos_embed.to(device)
        x = x + pos_embed
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wo = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x):
        batch, seq_len, embed_dim = x.shape
        q = self.wq(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = F.softmax(attention, dim=-1)
        attention = torch.matmul(attention, v) 
        
        attention = attention.transpose(1, 2).reshape(batch, seq_len, embed_dim)
        out = self.wo(attention)
        return out

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class DiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)
        
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim)
        )
        nn.init.constant_(self.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.adaln_modulation[-1].bias, 0)
    
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(c).chunk(6, dim=1)
        
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x

class DiT(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, image_size=32, patch_size=2,n_layers = 12, num_heads=8):
        super().__init__()
        self.imagesize = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.patch_embed = PatchLatent(in_channels, embed_dim, image_size, patch_size)
        self.clip_text = test_clip.get_cliptext_project(target_config=512,device="cuda")
        self.blocks = nn.ModuleList([DiTBlock(embed_dim, num_heads) for _ in range(n_layers)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.final_proj_1 = nn.Linear(embed_dim, embed_dim*4)
        self.final_proj_2 = nn.Linear(embed_dim*4, in_channels * (patch_size ** 2))
    
    def forward(self, x, prompt):
        x = self.patch_embed(x)  
        c = self.clip_text(prompt)
        
        for blk in self.blocks:
            x = blk(x, c)
        x = self.norm(x)
        x = self.final_proj_1(x)
        x = F.gelu(x)
        x = self.final_proj_2(x)  
        
        x = x.reshape(x.shape[0], -1)  
        x = x.reshape(x.shape[0], self.in_channels, self.imagesize, self.imagesize)
        return x

class StableDIT:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = DiT(in_channels=4, embed_dim=512, image_size=32, patch_size=2, n_layers=24, num_heads=8).to(device)
        self.vae_encoder = test_vae.get_vae_projector(device)
        self.clip_text = test_clip.get_cliptext_project(target_config=512, device=device)
        
        self.timesteps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02

    def get_noise_schedule(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=self.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])
        
        return {
            'betas': betas,
            'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'alphas_cumprod_prev': alphas_cumprod_prev,
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod),
            'sqrt_recip_alphas': torch.sqrt(1 / alphas),
            'posterior_variance': betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        }
    
    def add_noise(self, x, t, noise=None):
        noise_schedule = self.get_noise_schedule()
        noise = torch.randn_like(x) if noise is None else noise
        
        sqrt_alphas_cumprod_t = noise_schedule['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = noise_schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)
        x_noisy = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_noisy, noise
    
    def loss_function(self, images, prompts):
        batch_size = images.shape[0]
        
        with torch.inference_mode():
            latents = self.vae_encoder.encode(images)
        
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(latents)
        noisy_latents, target_noise = self.add_noise(latents, t, noise)
        
        predicted = self.model(noisy_latents, prompts)
        
        loss = F.mse_loss(predicted, target_noise)
        return loss
    
    def sample(self, prompt, batch_size=1, guidance_scale=7.5):
        noise_schedule = self.get_noise_schedule()
        shape = (batch_size, 4, 32, 32)
        
        x = torch.randn(shape, device=self.device)
        
        for i in range(self.timesteps - 1, -1, -1):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            betas_t = noise_schedule['betas'][t]
            sqrt_one_minus_alphas_cumprod_t = noise_schedule['sqrt_one_minus_alphas_cumprod'][t]
            sqrt_recip_alphas_t = noise_schedule['sqrt_recip_alphas'][t]
            
            predicted_noise = self.model(x, prompt)
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t) + torch.sqrt(betas_t) * noise)
            
        return x
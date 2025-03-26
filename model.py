import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PatchLatent(nn.Module):
    def __init__(self, in_channels, embed_dim, image_size, patch_size=2):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_latent = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
    
    def forward(self, x):
        x = self.patch_latent(x)          
        x = x.flatten(2).transpose(1, 2)    
        
        print(x.shape)
        print(self.pos_embed.shape)
        
        x = x + self.pos_embed             
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
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
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
        self.patch_embed = PatchLatent(in_channels, embed_dim, image_size, patch_size)

        self.clip_text = test_clip.get_cliptext_project(target_config=512,device="cuda")

        self.blocks = nn.ModuleList([DiTBlock(embed_dim, num_heads) for _ in range(n_layers)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.final_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x,prompt):
        # x: (N, C, H, W)
        x = self.patch_embed(x)  # (N, num_patches, embed_dim)
        c = self.clip_text("prompt")
        print("Embeddings " ,c.shape)
        
        for blk in self.blocks:
            x = blk(x, c)
        x = self.norm(x)
        x = self.final_proj(x)
        return x

import test_vae
import test_clip

if __name__ == '__main__':
    
    dim_embed = 512
    input_img = torch.randn(1, 3, 256, 256).to("cuda")
    vae_encoder = test_vae.get_vae_projector()
    
    model = DiT(in_channels=4, embed_dim=512, image_size=32, patch_size=2, n_layers=24, num_heads=8).to("cuda")
    dummy_input = vae_encoder.encode(input_img)
    print("VAE SIZE",dummy_input.shape)
    out = model(dummy_input,"cat")
    
    print("Output shape:", out.shape)
    # Expected output shape: (N, num_patches, embed_dim)
    
    print("Model size: " , sum(p.numel() for p in model.parameters()))


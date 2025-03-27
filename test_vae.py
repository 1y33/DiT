import torch
from diffusers import AutoencoderKL

class VAEProjector:
    def __init__(self, vae_model_id, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL.from_pretrained(vae_model_id)
        self.vae.to(self.device)
        self.vae.eval()
        
    def encode(self, image):
        # Ensure image is on the correct device
        image = image.to(self.device)
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def decode(self, latents):
        # Ensure latents are on the correct device
        latents = latents.to(self.device)
        with torch.no_grad():
            decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return decoded

def get_vae_projector(device="cuda"):
    vae_model_id = "stabilityai/sd-vae-ft-ema"
    vae_projector = VAEProjector(vae_model_id, device)
    return vae_projector

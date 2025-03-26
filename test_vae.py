import torch
from diffusers import AutoencoderKL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
import torchvision.transforms as transforms


class VAEProjector:
    def __init__(self, vae_model_id):
        self.vae = AutoencoderKL.from_pretrained(vae_model_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vae.to(self.device)
        
        self.vae.eval()
    def encode(self,image):
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def decode(self,latents):
        with torch.no_grad():
            decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return decoded
    


def get_vae_projector():
    vae_model_id = "stabilityai/sd-vae-ft-ema"
    vae_projector = VAEProjector(vae_model_id)
    return vae_projector

# # Get a sample image from a URL
# def load_image_from_url(url):
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content)).convert("RGB")
#     return img

# # Preprocess the image for the VAE
# def preprocess_image(image, target_size=(512, 512)):
#     transform = transforms.Compose([
#         transforms.Resize(target_size),
#         transforms.ToTensor(),  # Converts to [0, 1] range
#         transforms.Normalize([0.5], [0.5])  # Maps to [-1, 1] range
#     ])
#     # Add batch dimension
#     return transform(image).unsqueeze(0).to(vae_projector.device)

# # Function to visualize original and reconstructed images
# def visualize_reconstruction(original_img, reconstructed_tensor):
#     # Convert reconstructed tensor to image
#     reconstructed = (reconstructed_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1) / 2
#     reconstructed = np.clip(reconstructed, 0, 1)
    
#     # Plot images
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     ax[0].imshow(original_img)
#     ax[0].set_title("Original Image")
#     ax[0].axis("off")
    
#     ax[1].imshow(reconstructed)
#     ax[1].set_title("Reconstructed Image")
#     ax[1].axis("off")
    
#     plt.tight_layout()
#     plt.show()

# # Example URL - replace with any image URL
# image_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

# # Load and process image
# original_img = load_image_from_url(image_url)
# processed_img = preprocess_image(original_img)

# # Encode and decode
# latents = vae_projector.encode(processed_img)
# reconstructed_img = vae_projector.decode(latents)

# # Visualize the results
# visualize_reconstruction(original_img, reconstructed_img)

# print(f"Original image shape: {processed_img.shape}")
# print(f"Latent representation shape: {latents.shape}")
# print(f"Reconstructed image shape: {reconstructed_img.shape}")

# # Save the images to disk
# def save_images(original_img, reconstructed_tensor, save_dir="saved_images"):
#     import os
#     # Create the directory if it doesn't exist
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Save the original image
#     original_path = os.path.join(save_dir, "original.png")
#     original_img.save(original_path)
    
#     # Convert reconstructed tensor to PIL Image
#     reconstructed = (reconstructed_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1) / 2
#     reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
#     reconstructed_img_pil = Image.fromarray(reconstructed)
    
#     # Save the reconstructed image
#     reconstructed_path = os.path.join(save_dir, "reconstructed.png")
#     reconstructed_img_pil.save(reconstructed_path)
    
#     print(f"Images saved to {save_dir}/")
#     print(f"Original image: {original_path}")
#     print(f"Reconstructed image: {reconstructed_path}")

# # Save the original and reconstructed images
# save_images(original_img, reconstructed_img)

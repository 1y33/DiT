import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

class CLIPTextProjector(nn.Module):
    def __init__(self, clip_model, processor, target_config=768, device="cuda"):
        super().__init__()
        self.clip_model = clip_model
        self.processor = processor
        self.target_config = target_config
        self.device = device
        
        self.projector = nn.Linear(512, target_config)
        self.projector.to(device)  # Move projector to CUDA
    
    def get_prompt_encodings(self, prompt):
        # Explicitly set max_length to 77 and use truncation to handle long prompts
        text_inputs = self.processor(
            prompt, 
            padding=True, 
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        text_inputs = {key: val.to(self.device) for key, val in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.to(self.device)
        
    def forward(self, prompt):
        return self.projector(self.get_prompt_encodings(prompt))

def get_cliptext_project(target_config, device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map=device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_project = CLIPTextProjector(model, processor, target_config, device)
    text_project.to(device)  # Make sure the whole module is on CUDA

    return text_project
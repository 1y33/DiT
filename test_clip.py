import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

class CLIPTextProjector(nn.Module):
    def __init__(self,clip_model,processor, target_config= 768):
        super().__init__()
        self.clip_model = clip_model
        self.processor = processor
        self.target_config = target_config
                
        
        self.projector = nn.Linear(512, target_config)
    
    def get_prompt_encodings(self,prompt):
        text_inputs = self.processor(prompt, padding=True, return_tensors="pt")
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.to("cuda")
        
    def forward(self,prompt):
        return self.projector(self.get_prompt_encodings(prompt))
    
def get_cliptext_project(target_config,device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",device_map=device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_project = CLIPTextProjector(model,processor,target_config)

    return text_project
    
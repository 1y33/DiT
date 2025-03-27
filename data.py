from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer

class DualImageDataset(Dataset):
    def __init__(self, image_dataset, image_size=256, transform=None, max_length=77):
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        ])
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.max_length = max_length
        
        self.data = []
        for item in image_dataset:
            prompt = item['prompt']
            self.data.append({'prompt': prompt, 'image': item['image1']})
            self.data.append({'prompt': prompt, 'image': item['image2']})
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = self._truncate_prompt(item['prompt'])
        
        image = item['image']
        if self.transform:
            image = self.transform(image)
            
        return {
            'prompt': prompt,
            'image': image
        }
    
    def _truncate_prompt(self, prompt):
        tokens = self.tokenizer(
            prompt, 
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=False
        )
        
        truncated_prompt = self.tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)
        return truncated_prompt

def create_dataloader(image_dataset, batch_size=256, image_size=256, shuffle=True, num_workers=8):
    dataset = DualImageDataset(image_dataset, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  
    )
    
    return dataloader

def get_loader():
    image_dataset = load_dataset("Rapidata/open-image-preferences-v1-more-results")
    image_dataset = image_dataset['train']
    dataloader = create_dataloader(image_dataset, batch_size=32, image_size=256, shuffle=True, num_workers=4)
    
    return dataloader
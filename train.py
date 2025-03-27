import os
import torch
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from accelerate import Accelerator, DistributedDataParallelKwargs

from model import StableDIT
from data import get_loader

def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    device = accelerator.device
    is_main_process = accelerator.is_main_process
    
    if is_main_process:
        print(f"Using device: {device}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Distributed type: {accelerator.distributed_type}")
    
    epochs = 600
    learning_rate = 1e-5
    weight_decay = 0.0
    
    base_dir = "path"
    save_dir = os.path.join(base_dir, "checkpoints")
    samples_dir = os.path.join(base_dir, "samples")
    
    # main process is from accelerator
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        print(f"Saving checkpoints to: {save_dir}")
        print(f"Saving samples to: {samples_dir}")
    
    if is_main_process:
        print("Initializing StableDIT...")
    stable_dit = StableDIT(device=device)
    
    if is_main_process:
        print("Loading data...")
    dataloader = get_loader()
    
    optimizer = torch.optim.AdamW(stable_dit.model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
    
    stable_dit.model, optimizer, dataloader, scheduler = accelerator.prepare(stable_dit.model, optimizer, dataloader, scheduler)
    
    eval_prompts = [
        "a photo of a cat sitting on a windowsill",
        "a beautiful landscape with mountains and a lake"
    ]
    
    if is_main_process:
        print("Starting training...")
    global_step = 0
    
    for epoch in range(epochs):
        stable_dit.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not is_main_process)
        for batch in progress_bar:
            images = batch['image'].to(device)
            prompts = batch['prompt']
            
            with accelerator.accumulate(stable_dit.model):
                loss = stable_dit.loss_function(images, prompts)
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(stable_dit.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            loss_value = loss.detach().float().item()
            epoch_loss += loss_value
            
            if is_main_process:
                progress_bar.set_postfix({"loss": loss_value, "lr": optimizer.param_groups[0]['lr']})
            
            global_step += 1
            
            if global_step % 1000 == 0 and is_main_process:
                original_model = stable_dit.model
                stable_dit.model = accelerator.unwrap_model(stable_dit.model)
                generate_samples(stable_dit, eval_prompts, device, global_step, samples_dir)
                stable_dit.model = original_model
        
        gathered_epoch_loss = accelerator.gather(torch.tensor(epoch_loss, device=device)).sum().item()
        avg_loss = gathered_epoch_loss / (len(dataloader) * accelerator.num_processes)
        
        if is_main_process:
            print(f"Epoch {epoch+1}/{epochs}: Average loss = {avg_loss:.6f}")
        
        if is_main_process and ((epoch + 1) % 5 == 0 or epoch == epochs - 1):
            unwrapped_model = accelerator.unwrap_model(stable_dit.model)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'loss': avg_loss,
            }, 
            f"{save_dir}/stable_dit_model_epoch_{epoch+1}.pt")
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        accelerator.wait_for_everyone()
    
    if is_main_process:
        print("Training completed!")

def generate_samples(stable_dit, prompts, device, step, samples_dir):
    stable_dit.model.eval()
    print(f"Generating samples at step {step}...")
    
    with torch.no_grad():
        all_images = []
        for prompt in prompts:
            latents = stable_dit.sample(prompt, batch_size=1)
            images = stable_dit.vae_encoder.decode(latents)
            all_images.append(images[0])
            save_image(
                images[0].cpu(), 
                os.path.join(samples_dir, f"sample_step_{step}_prompt_{prompts.index(prompt)}.png"),
                normalize=True, 
                value_range=(-1, 1)
            )
        
        grid = make_grid(torch.stack(all_images), nrow=len(prompts), normalize=True, value_range=(-1, 1))
        save_image(grid, os.path.join(samples_dir, f"grid_step_{step}.png"))
    
    stable_dit.model.train()
    print(f"Samples saved at step {step}")

if __name__ == "__main__":
    main()

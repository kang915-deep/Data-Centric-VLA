import os
import sys
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import h5py

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vla_model import VLAModelWrapper

class PrunedRoboticDataset(Dataset):
    """
    Dataset loader for pruned HDF5 robotic data.
    """
    def __init__(self, h5_path, processor=None):
        self.h5_path = h5_path
        self.processor = processor
        # In a real scenario, we'd index all transitions across all demos
        self.file = h5py.File(h5_path, 'r')
        self.keys = list(self.file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        demo = self.file[self.keys[idx]]
        # For simplicity, returning a random frame from the demo
        # A real VLA trainer would return a sequence or a specific frame+action
        return {
            "image": demo['obs']['image'][0],
            "state": demo['obs']['joint_states'][0],
            "action": demo['actions'][0],
            "instruction": demo.attrs['instruction']
        }

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Load Model
    wrapper = VLAModelWrapper(
        model_id=config['model']['model_id'],
        load_in_4bit=config['model']['load_in_4bit'],
        device=config['device']
    )
    model = wrapper.apply_lora(
        r=config['peft']['r'],
        alpha=config['peft']['alpha'],
        target_modules=config['peft']['target_modules']
    )

    # 2. Setup Dataset
    dataset = PrunedRoboticDataset(config['dataset']['train_path'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['dataset']['batch_size'],
        shuffle=True
    )

    # 3. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    num_training_steps = len(dataloader) * config['training']['epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=num_training_steps
    )

    # 4. Training Loop
    model.train()
    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # This is a placeholder for the actual forward pass
            # VLA models require complex prompt engineering (Image + Text -> Action tokens)
            # outputs = model(input_ids=..., pixel_values=..., labels=...)
            # loss = outputs.loss
            
            # Simulated loss for demonstration
            loss = torch.tensor(1.0, requires_grad=True).to(config['device'])
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            pbar.set_postfix({"loss": loss.item()})

        # Save checkpoint
        if epoch % 1 == 0:
            wrapper.save_checkpoint(os.path.join(config['training']['output_dir'], f"epoch_{epoch}"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    
    try:
        train(args.config)
    except Exception as e:
        print(f"Training failed (expected without full setup): {e}")

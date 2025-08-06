import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np

from kgvae.model.models import KGVAE
from kgvae.model.utils import (
    compute_kl_divergence,
    compute_reconstruction_loss,
    pad_triples,
    create_padding_mask,
    compute_entity_sorting_loss
)


class KGDataset(Dataset):
    def __init__(self, triples, max_edges):
        self.triples = triples
        self.max_edges = max_edges
        
    def __len__(self):
        return len(self.triples)
        
    def __getitem__(self, idx):
        triple_set = self.triples[idx]
        padded_triples = pad_triples(triple_set.unsqueeze(0), self.max_edges).squeeze(0)
        mask = create_padding_mask(padded_triples.unsqueeze(0)).squeeze(0)
        
        return padded_triples, mask


def train_epoch(model, dataloader, optimizer, config, device):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (triples, mask) in enumerate(tqdm(dataloader, desc="Training")):
        triples = triples.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(triples, mask)
        
        recon_loss = compute_reconstruction_loss(
            (outputs['subject_logits'], outputs['relation_logits'], outputs['object_logits']),
            triples,
            mask
        )
        
        kl_loss = compute_kl_divergence(outputs['mu'], outputs['logvar'])
        
        loss = recon_loss + config['beta'] * kl_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def validate(model, dataloader, config, device):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for triples, mask in tqdm(dataloader, desc="Validation"):
            triples = triples.to(device)
            mask = mask.to(device)
            
            outputs = model(triples, mask)
            
            recon_loss = compute_reconstruction_loss(
                (outputs['subject_logits'], outputs['relation_logits'], outputs['object_logits']),
                triples,
                mask
            )
            
            kl_loss = compute_kl_divergence(outputs['mu'], outputs['logvar'])
            
            loss = recon_loss + config['beta'] * kl_loss
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--wandb-project', type=str, default='kgvae', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Weights & Biases entity')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=config,
        name=config.get('experiment_name', 'kgvae_experiment')
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    
    # from intelligraphs import KnowledgeGraph
    # kg = KnowledgeGraph(dataset=config['dataset'])
    
    # train_triples = torch.tensor(kg.train_triples, dtype=torch.long)
    # val_triples = torch.tensor(kg.val_triples, dtype=torch.long)
    
    from intelligraphs.data_loaders import load_data_as_list

    (train_g, val_g, test_g,
    (e2i, i2e), (r2i, i2r), *_ ) = load_data_as_list(config['dataset'])  # e.g., "syn-paths"

    train_triples = [torch.tensor(g, dtype=torch.long) for g in train_g]
    val_triples = [torch.tensor(g, dtype=torch.long) for g in val_g]

    config['n_entities'] = len(e2i)
    config['n_relations'] = len(r2i)

    
    
    train_dataset = KGDataset(train_triples, config['max_edges'])
    val_dataset = KGDataset(val_triples, config['max_edges'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )
    
    # config['n_entities'] = kg.n_entities
    # config['n_relations'] = kg.n_relations
    
    model = KGVAE(config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    if config.get('lr_scheduler', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        train_loss, train_recon_loss, train_kl_loss = train_epoch(
            model, train_loader, optimizer, config, device
        )
        
        val_loss, val_recon_loss, val_kl_loss = validate(
            model, val_loader, config, device
        )
        
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/reconstruction_loss': train_recon_loss,
            'train/kl_loss': train_kl_loss,
            'val/loss': val_loss,
            'val/reconstruction_loss': val_recon_loss,
            'val/kl_loss': val_kl_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f})")
        
        if config.get('lr_scheduler', False):
            scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(
                checkpoint,
                os.path.join(args.checkpoint_dir, f'best_model.pt')
            )
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        if (epoch + 1) % config.get('save_every', 10) == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(
                checkpoint,
                os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )
    
    wandb.finish()
    print("Training completed!")


if __name__ == "__main__":
    main()
import torch
import torch.optim as optim
import wandb
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np

from intelligraphs import DataLoader
from intelligraphs.data_loaders import DatasetDownloader

from kgvae.model.models import KGVAE
from kgvae.model.utils import (
    compute_kl_divergence,
    compute_reconstruction_loss,
    create_padding_mask,
)
from kgvae.model.verification import get_verifier, sample_and_verify


def process_batch(batch_triples, max_edges, device):
    """
    Process a batch of triples from IntelliGraphs DataLoader.
    """
    # batch_triples is already a tensor from IntelliGraphs
    batch_size = batch_triples.size(0)
    
    # Pad/truncate to max_edges
    current_edges = batch_triples.size(1)
    
    if current_edges > max_edges:
        # Truncate
        batch_triples = batch_triples[:, :max_edges, :]
    elif current_edges < max_edges:
        # Pad with zeros
        padding = torch.zeros(batch_size, max_edges - current_edges, 3, 
                             dtype=batch_triples.dtype, device=batch_triples.device)
        batch_triples = torch.cat([batch_triples, padding], dim=1)
    
    # Create mask for padding
    mask = create_padding_mask(batch_triples)
    
    return batch_triples, mask


def train_epoch(model, dataloader, optimizer, config, device):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    for batch_idx, batch_triples in enumerate(tqdm(dataloader, desc="Training")):
        # Process batch from IntelliGraphs DataLoader
        triples, mask = process_batch(batch_triples, config['max_edges'], device)
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
        num_batches += 1
        
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def validate(model, dataloader, config, device):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_triples in tqdm(dataloader, desc="Validation"):
            # Process batch from IntelliGraphs DataLoader
            triples, mask = process_batch(batch_triples, config['max_edges'], device)
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
            num_batches += 1
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--wandb-project', type=str, default='submission', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default='a-vozikis-vrije-universiteit-amsterdam', help='Weights & Biases entity')
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
    
    # Initialize dataset handler and check/download datasets
    dataset_name = config['dataset']  # e.g. 'syn-paths', 'syn-tipr', etc.
    dataset_handler = DatasetDownloader()
    
    # Check if the datasets already exist and download if needed
    if not dataset_handler.check_datasets_exist():
        print("Downloading datasets...")
        dataset_handler.download_and_verify_all()
    else:
        print("Verifying existing datasets...")
        dataset_handler.verify_datasets()
    
    # Load data using IntelliGraphs DataLoader
    data_loader = DataLoader(dataset_name=dataset_name)
    
    # Load PyTorch DataLoaders with padding
    train_loader, val_loader, test_loader = data_loader.load_torch(
        batch_size=config['batch_size'],
        padding=True,  # Enable padding for consistent batch sizes
        shuffle_train=True,
        shuffle_valid=False,
        shuffle_test=False
    )
    
    # Get entity and relation mappings
    entity_map = data_loader.entity_to_id
    relation_map = data_loader.relation_to_id
    
    # Create inverse mappings
    i2e = {idx: entity for entity, idx in entity_map.items()}
    i2r = {idx: relation for relation, idx in relation_map.items()}
    
    # Update config with dataset statistics
    config['n_entities'] = len(entity_map)
    config['n_relations'] = len(relation_map)
    
    print(f"Dataset: {dataset_name}")
    print(f"Entities: {config['n_entities']}, Relations: {config['n_relations']}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize verifier for the dataset
    verifier = get_verifier(dataset_name)
    if verifier is None:
        print(f"Warning: No verifier available for dataset {dataset_name}")
    
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
        
        # Log basic metrics
        log_dict = {
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/reconstruction_loss': train_recon_loss,
            'train/kl_loss': train_kl_loss,
            'val/loss': val_loss,
            'val/reconstruction_loss': val_recon_loss,
            'val/kl_loss': val_kl_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # Periodically verify generated graphs
        if verifier and (epoch + 1) % config.get('verify_every', 10) == 0:
            verification_results = sample_and_verify(
                model, config, verifier, i2e, i2r, device, 
                num_samples=config.get('verify_samples', 100)
            )
            log_dict.update({
                'verification/validity_rate': verification_results['validity_rate'],
                'verification/valid_count': verification_results['valid_count'],
                'verification/total_count': verification_results['total_count']
            })
            print(f"Graph Verification: {verification_results['valid_count']}/{verification_results['total_count']} "
                  f"valid ({verification_results['validity_rate']:.2%})")
        
        wandb.log(log_dict)
        
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
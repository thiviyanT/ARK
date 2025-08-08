import torch
import torch.optim as optim
import wandb
import yaml
import argparse
import os
import warnings
from tqdm import tqdm
import numpy as np

from intelligraphs import DataLoader
from intelligraphs.data_loaders import DatasetDownloader

from kgvae.model.models import KGVAE
from kgvae.model.rescal_vae_model import RESCALVAE
from kgvae.model.utils import (
    compute_kl_divergence,
    compute_reconstruction_loss,
    create_padding_mask,
)
from kgvae.model.verification import get_verifier, sample_and_verify


def process_batch(batch_triples, max_edges, max_nodes, device):
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
    
    # Extract unique nodes from triples for RESCALVAE
    nodes = torch.zeros(batch_size, max_nodes, dtype=torch.long, device=device)
    for b in range(batch_size):
        unique_nodes = torch.unique(batch_triples[b, :, [0, 2]].flatten())
        # Filter out padding (0) and place in nodes tensor
        unique_nodes = unique_nodes[unique_nodes != 0]
        num_nodes = min(len(unique_nodes), max_nodes)
        if num_nodes > 0:
            nodes[b, :num_nodes] = unique_nodes[:num_nodes]
    
    return batch_triples, nodes, mask


def train_epoch(model, dataloader, optimizer, config, device):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_entity_loss = 0
    num_batches = 0
    
    model_type = config.get('model_type', 'kgvae')
    
    for batch_idx, batch_triples in enumerate(tqdm(dataloader, desc="Training")):
        # Process batch from IntelliGraphs DataLoader
        triples, nodes, mask = process_batch(batch_triples, config['max_edges'], config['max_nodes'], device)
        triples = triples.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'rescal_vae':
            # RESCALVAE forward pass and loss computation
            outputs = model(triples, nodes, mask)
            loss_dict = model.compute_loss(outputs, triples, nodes, mask)
            loss = loss_dict['loss']
            recon_loss = loss_dict['edge_loss']
            kl_loss = loss_dict['kl_loss']
            entity_loss = loss_dict.get('entity_loss', 0)
            total_entity_loss += entity_loss.item() if torch.is_tensor(entity_loss) else entity_loss
        else:
            # Original KGVAE forward pass
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
    avg_entity_loss = total_entity_loss / num_batches if num_batches > 0 else 0
    
    if model_type == 'rescal_vae':
        return avg_loss, avg_recon_loss, avg_kl_loss, avg_entity_loss
    return avg_loss, avg_recon_loss, avg_kl_loss


def validate(model, dataloader, config, device):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_entity_loss = 0
    num_batches = 0
    
    model_type = config.get('model_type', 'kgvae')
    
    with torch.no_grad():
        for batch_triples in tqdm(dataloader, desc="Validation"):
            # Process batch from IntelliGraphs DataLoader
            triples, nodes, mask = process_batch(batch_triples, config['max_edges'], config['max_nodes'], device)
            triples = triples.to(device)
            mask = mask.to(device)
            
            if model_type == 'rescal_vae':
                # RESCALVAE forward pass and loss computation
                outputs = model(triples, nodes, mask)
                loss_dict = model.compute_loss(outputs, triples, nodes, mask)
                loss = loss_dict['loss']
                recon_loss = loss_dict['edge_loss']
                kl_loss = loss_dict['kl_loss']
                entity_loss = loss_dict.get('entity_loss', 0)
                total_entity_loss += entity_loss.item() if torch.is_tensor(entity_loss) else entity_loss
            else:
                # Original KGVAE forward pass
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
    avg_entity_loss = total_entity_loss / num_batches if num_batches > 0 else 0
    
    if model_type == 'rescal_vae':
        return avg_loss, avg_recon_loss, avg_kl_loss, avg_entity_loss
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
    
    # Warning for test set evaluation
    if config.get('use_test_for_final_eval', False):
        warnings.warn(
            "Test set evaluation ENABLED! Only use for final evaluation, NOT for hyperparameter tuning!",
            UserWarning,
            stacklevel=2
        )
    
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
    
    # Initialize model based on config
    model_type = config.get('model_type', 'kgvae')
    if model_type == 'rescal_vae':
        model = RESCALVAE(config).to(device)
    else:
        model = KGVAE(config).to(device)
    print(f"Using model: {model_type}")
    
    # Support multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    if config.get('lr_scheduler', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        train_results = train_epoch(model, train_loader, optimizer, config, device)
        val_results = validate(model, val_loader, config, device)
        
        # Handle different return values based on model type
        if config.get('model_type', 'kgvae') == 'rescal_vae':
            train_loss, train_recon_loss, train_kl_loss, train_entity_loss = train_results
            val_loss, val_recon_loss, val_kl_loss, val_entity_loss = val_results
        else:
            train_loss, train_recon_loss, train_kl_loss = train_results
            val_loss, val_recon_loss, val_kl_loss = val_results
            train_entity_loss = 0
            val_entity_loss = 0
        
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
        
        # Add entity loss for RESCALVAE
        if config.get('model_type', 'kgvae') == 'rescal_vae':
            log_dict['train/entity_loss'] = train_entity_loss
            log_dict['val/entity_loss'] = val_entity_loss
        
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
            # Handle DataParallel when saving
            model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_state,
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
            # Handle DataParallel when saving
            model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(
                checkpoint,
                os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )
    
    # Final evaluation
    use_test = config.get('use_test_for_final_eval', False)
    eval_set_name = "test" if use_test else "validation"
    eval_loader = test_loader if use_test else val_loader
    
    print(f"\n{'='*50}\nFinal evaluation on {eval_set_name} set...")
    if use_test:
        warnings.warn("Using TEST SET for final evaluation", UserWarning)
    
    # Run final evaluation using existing validate function
    final_results = validate(model, eval_loader, config, device)
    
    # Unpack results based on model type
    if config.get('model_type', 'kgvae') == 'rescal_vae':
        final_loss, final_recon_loss, final_kl_loss, final_entity_loss = final_results
        log_dict = {
            f'final_{eval_set_name}/loss': final_loss,
            f'final_{eval_set_name}/reconstruction_loss': final_recon_loss,
            f'final_{eval_set_name}/kl_loss': final_kl_loss,
            f'final_{eval_set_name}/entity_loss': final_entity_loss
        }
        print(f"\nFinal {eval_set_name}: Loss={final_loss:.4f}, Recon={final_recon_loss:.4f}, "
              f"KL={final_kl_loss:.4f}, Entity={final_entity_loss:.4f}")
    else:
        final_loss, final_recon_loss, final_kl_loss = final_results
        log_dict = {
            f'final_{eval_set_name}/loss': final_loss,
            f'final_{eval_set_name}/reconstruction_loss': final_recon_loss,
            f'final_{eval_set_name}/kl_loss': final_kl_loss
        }
        print(f"\nFinal {eval_set_name}: Loss={final_loss:.4f}, Recon={final_recon_loss:.4f}, KL={final_kl_loss:.4f}")
    
    # Final verification
    if verifier:
        final_verification = sample_and_verify(
            model, config, verifier, i2e, i2r, device,
            num_samples=config.get('verify_samples', 100)
        )
        print(f"Final generation validity: {final_verification['validity_rate']:.2%} "
              f"({final_verification['valid_count']}/{final_verification['total_count']})")
        
        log_dict.update({
            f'final_{eval_set_name}/validity_rate': final_verification['validity_rate'],
            f'final_{eval_set_name}/valid_count': final_verification['valid_count'],
            f'final_{eval_set_name}/total_count': final_verification['total_count']
        })
    
    wandb.log(log_dict)
    print("="*50)
    
    wandb.finish()
    print("\nTraining and evaluation completed!")


if __name__ == "__main__":
    main()
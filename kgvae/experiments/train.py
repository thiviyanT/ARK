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

from kgvae.model.rescal_vae_model import RESCALVAE
from kgvae.model.utils import create_padding_mask
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
    
    # Extract unique nodes from triples
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
    
    for batch_idx, batch_triples in enumerate(tqdm(dataloader, desc="Training")):
        # Process batch from IntelliGraphs DataLoader
        triples, nodes, mask = process_batch(batch_triples, config['max_edges'], config['max_nodes'], device)
        triples = triples.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        # RESCALVAE forward pass and loss computation
        outputs = model(triples, nodes, mask)
        loss_dict = model.compute_loss(outputs, triples, nodes, mask)
        loss = loss_dict['loss']
        recon_loss = loss_dict['edge_loss']
        kl_loss = loss_dict['kl_loss']
        entity_loss = loss_dict.get('entity_loss', 0)
        total_entity_loss += entity_loss.item() if torch.is_tensor(entity_loss) else entity_loss
        
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
    
    return avg_loss, avg_recon_loss, avg_kl_loss, avg_entity_loss


def validate(model, dataloader, config, device, compute_compression=False):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_entity_loss = 0
    total_compression_bits = 0
    total_kl_bits = 0
    total_edge_bits = 0
    total_entity_bits = 0
    total_graphs = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_triples in tqdm(dataloader, desc="Validation"):
            # Process batch from IntelliGraphs DataLoader
            triples, nodes, mask = process_batch(batch_triples, config['max_edges'], config['max_nodes'], device)
            triples = triples.to(device)
            mask = mask.to(device)
            
            # RESCALVAE forward pass and loss computation
            outputs = model(triples, nodes, mask)
            loss_dict = model.compute_loss(outputs, triples, nodes, mask)
            loss = loss_dict['loss']
            recon_loss = loss_dict['edge_loss']
            kl_loss = loss_dict['kl_loss']
            entity_loss = loss_dict.get('entity_loss', 0)
            total_entity_loss += entity_loss.item() if torch.is_tensor(entity_loss) else entity_loss
            
            # Compute compression bits if requested
            if compute_compression:
                scoring_mode = config.get('compression_scoring_mode', 'sparse')
                compression_dict = model.compute_compression_bits(outputs, triples, nodes, 
                                                                   scoring_mode=scoring_mode)
                total_compression_bits += compression_dict['total_bits']
                total_kl_bits += compression_dict['kl_bits']
                total_edge_bits += compression_dict['edge_bits']
                total_entity_bits += compression_dict['entity_bits']
                total_graphs += compression_dict['batch_size']
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0
    avg_entity_loss = total_entity_loss / num_batches if num_batches > 0 else 0
    
    if compute_compression:
        avg_compression_bits = total_compression_bits / total_graphs if total_graphs > 0 else 0
        avg_kl_bits = total_kl_bits / total_graphs if total_graphs > 0 else 0
        avg_edge_bits = total_edge_bits / total_graphs if total_graphs > 0 else 0
        avg_entity_bits = total_entity_bits / total_graphs if total_graphs > 0 else 0
        return (avg_loss, avg_recon_loss, avg_kl_loss, avg_entity_loss, 
                avg_compression_bits, avg_kl_bits, avg_edge_bits, avg_entity_bits)
    return avg_loss, avg_recon_loss, avg_kl_loss, avg_entity_loss


def final_validation(model, test_loader, val_loader, config, device, verifier, i2e, i2r):
    """
    Perform final validation on either test or validation set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to run on
        verifier: Graph verifier for the dataset
        i2e: Index to entity mapping
        i2r: Index to relation mapping
    
    Returns:
        Dictionary of final metrics to log to wandb
    """
    # Determine which set to use for final evaluation
    use_test = config.get('use_test_for_final_eval', False)
    eval_set_name = "test" if use_test else "validation"
    eval_loader = test_loader if use_test else val_loader
    
    print(f"\n{'='*50}\nFinal evaluation on {eval_set_name} set...")
    if use_test:
        warnings.warn("Using TEST SET for final evaluation", UserWarning)
    
    # Run final evaluation with compression bits
    compute_final_compression = True
    final_results = validate(model, eval_loader, config, device, compute_compression=compute_final_compression)
    
    # Unpack results from RESCAL-VAE
    if compute_final_compression:
        (final_loss, final_recon_loss, final_kl_loss, final_entity_loss,
         final_compression_bits, final_kl_bits, final_edge_bits, final_entity_bits) = final_results
    else:
        final_loss, final_recon_loss, final_kl_loss, final_entity_loss = final_results
        final_compression_bits = final_kl_bits = final_edge_bits = final_entity_bits = None
    
    log_dict = {
        f'final_{eval_set_name}/loss': final_loss,
        f'final_{eval_set_name}/reconstruction_loss': final_recon_loss,
        f'final_{eval_set_name}/kl_loss': final_kl_loss,
        f'final_{eval_set_name}/entity_loss': final_entity_loss
    }
    
    # Add compression bits if computed
    if final_compression_bits is not None:
        log_dict.update({
            f'final_{eval_set_name}/compression_bits': final_compression_bits,
            f'final_{eval_set_name}/compression_kl_bits': final_kl_bits,
            f'final_{eval_set_name}/compression_edge_bits': final_edge_bits,
            f'final_{eval_set_name}/compression_entity_bits': final_entity_bits
        })
        
    print(f"\nFinal {eval_set_name}: Loss={final_loss:.4f}, Recon={final_recon_loss:.4f}, "
          f"KL={final_kl_loss:.4f}, Entity={final_entity_loss:.4f}")
    
    if final_compression_bits is not None:
        print(f"Final compression: {final_compression_bits:.2f} bits/graph "
              f"(KL: {final_kl_bits:.2f}, Edge: {final_edge_bits:.2f}, Entity: {final_entity_bits:.2f})")
    
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
    
    print("="*50)
    return log_dict


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
        name=config.get('experiment_name', 'rescal_vae_experiment')
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
    
    # Initialize RESCAL-VAE model
    scoring_mode = config.get('compression_scoring_mode', 'sparse')
    print(f"\n{'='*60}")
    print(f"RESCAL-VAE Configuration:")
    print(f"  Training/Evaluation scoring mode: {scoring_mode.upper()}")
    print(f"  Compression logging every: {config.get('compression_log_every', 5)} epochs")
    if scoring_mode == 'dense':
        print(f"  WARNING: Dense scoring is memory intensive!")
        print(f"  Total edges to score per graph: {config['max_nodes']}×{config['max_nodes']}×{config['n_relations']} = {config['max_nodes']*config['max_nodes']*config['n_relations']}")
    else:
        print(f"  Efficient sparse scoring: Only {config['max_edges']} edges per graph")
    print(f"{'='*60}\n")
    model = RESCALVAE(config).to(device)
    print(f"Using model: RESCAL-VAE")
    
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
        
        # Unpack results from RESCAL-VAE
        train_loss, train_recon_loss, train_kl_loss, train_entity_loss = train_results
        val_loss, val_recon_loss, val_kl_loss, val_entity_loss = val_results
        
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
        
        # Add entity loss
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
    
    # Perform final validation
    final_metrics = final_validation(model, test_loader, val_loader, config, device, verifier, i2e, i2r)
    wandb.log(final_metrics)
    
    wandb.finish()
    print("\nTraining and evaluation completed!")


if __name__ == "__main__":
    main()

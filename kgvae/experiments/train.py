import math
import torch
import wandb
import yaml
import torch.optim as optim
import argparse
import os
import warnings
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


from intelligraphs import DataLoader
from intelligraphs.data_loaders import DatasetDownloader
from intelligraphs.data_loaders import load_data_as_list

from torch.utils.data import Dataset
from torch.utils.data import DataLoader as PDataLoader


from kgvae.model.models import SAIL, ARK
from kgvae.model.utils import GraphSeqDataset
from kgvae.model.verification import get_verifier
from kgvae.model.utils import  ints_to_labels,  seq_to_triples
from kgvae.model.verification import run_semantic_evaluation




def train_epoch(model, dataloader, optimizer, config, device, b=1.0): 
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_entity_loss = 0
    num_batches = 0
    
    model_type = config.get('model_type', 'ARK')

    
    for batch_idx, batch_triples in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()
        triples, seq = batch_triples
        seq = seq.to(device)
        logits = model(seq[:, :-1])  
        vocab = logits.size(-1)
        ce = F.cross_entropy(
            logits.reshape(-1, vocab),
            seq[:, 1:].reshape(-1),
            ignore_index=config["special_tokens"]["PAD"]
        )
        loss = ce
        recon_loss = ce
        kl_loss = torch.tensor(0.0, device=device)
                
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



def validate(model, dataloader, config, device, compute_compression=False, b=1.0,special_tokens=None):
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
    
    model_type = config.get('model_type', 'ARK')

    
    with torch.no_grad():
        for batch_triples in tqdm(dataloader, desc="Validation"):
            triples, seq = batch_triples
            seq = seq.to(device)
            logits = model(seq[:, :-1])  # predict next tokens
            vocab = logits.size(-1)
            ce = F.cross_entropy(
                logits.reshape(-1, vocab),
                seq[:, 1:].reshape(-1),
                ignore_index=config["special_tokens"]["PAD"]
            )
            loss = ce
            recon_loss = ce
            kl_loss = torch.tensor(0.0, device=device)    
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0
    avg_entity_loss = total_entity_loss / num_batches if num_batches > 0 else 0

    stats = model.posterior_bits(
        dataloader.dataset,
        device,
        pad_id=special_tokens["PAD"],
        sample_frac=config['sample_frac'],
        desc="Posterior compression"
    )
    print("\n[Final Posterior Compression on Validation/Test Set]")
    print(f" Final Avg total bits: {stats['avg_total_bits']:.2f}")
    print(f" Final Avg AR bits:    {stats['avg_ar_bits']:.2f}")
    avg_compression_bits = stats['avg_total_bits']
    avg_kl_bits = stats['avg_kl_bits']        
    avg_entity_bits = stats['avg_ar_bits']   
    avg_edge_bits = stats['avg_ar_bits']
    return (avg_loss, avg_recon_loss, avg_kl_loss, avg_entity_loss,
            avg_compression_bits, avg_kl_bits, avg_edge_bits, avg_entity_bits)


def final_validation(model, test_loader, val_loader, config, device, verifier, i2e, i2r, b = 1.0, special_tokens=None, seq_len=None, ENT_BASE=None, REL_BASE=None, train_g=None):    
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
    model_type = config.get('model_type', 'ARK')
    compute_final_compression = model_type in ('ARK', 't-ARK', 'SAIL', 't-SAIL')

    final_results = validate(model, eval_loader, config, device, compute_compression=compute_final_compression, b=b, special_tokens=special_tokens)
    
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
        target_N = config.get('num_generated_latent_graphs', 1000)
        chunk_size = 50
        all_batches = []

        while sum(x.size(0) for x in all_batches) < target_N:
            with torch.inference_mode():
                seq_batch = model.generate(
                    seq_len, special_tokens,
                    device=device,
                    batch_size=chunk_size,
                    beam=1,
                    sample=True,
                    temperature=config.get('temperature', 1.0),
                    top_p=config.get('top_p', 0.9),
                    top_k=config.get('top_k', 0)
                ).cpu()
            all_batches.append(seq_batch)

        seq_batch = torch.cat(all_batches, dim=0)[:target_N]

        graphs = [seq_to_triples(row.cpu(), special_tokens, ENT_BASE, REL_BASE) for row in seq_batch]
        labels = ints_to_labels(graphs, i2e, i2r)

        print("\nExample graphs (ARK):")
        for k in range(min(5, len(labels))):
            print(f"[{k}] {labels[k]}")              

        sem_eval = run_semantic_evaluation(labels, train_g, i2e, i2r, verifier, title="ARK samples")
        res = sem_eval.organized_results["results"]

        # Also include in the final metrics dict returned by final_validation
        log_dict.update({
            f'final_{eval_set_name}/validity_rate':      res.get("semantics", 0.0) / 100.0,
            f'final_{eval_set_name}/novelty_rate':       res.get("novel", 0.0) / 100.0,
            f'final_{eval_set_name}/valid_novelty_rate': res.get("novel_semantics", 0.0) / 100.0,
        })

        print(f"Final {eval_set_name} — validity: {res.get('semantics',0.0):.2f}% | "
            f"novelty: {res.get('novel',0.0):.2f}% | "
            f"valid&novel: {res.get('novel_semantics',0.0):.2f}%")

            
    print("="*50)
    return log_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--wandb-project', type=str, default='submission', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Weights & Biases entity')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    def apply_overrides(cfg, overrides):
        for k, v in dict(overrides).items():
            cfg[k] = v
        return cfg
        
    model_type = config.get('model_type', 'ARK')
    
    entity = args.wandb_entity or os.getenv("WANDB_ENTITY")

    init_kwargs = dict(
        project=args.wandb_project,
        config=config,
        name=config.get('experiment_name', 'ARK_experiment'),
        anonymous="allow",  
    )
    if entity:
        init_kwargs["entity"] = entity

    wandb.init(**init_kwargs)

    
    config = apply_overrides(config, wandb.config)
    config['learning_rate'] = float(config.get('learning_rate', 1e-3))
    run_dir = os.path.join(args.checkpoint_dir, wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)
    args.checkpoint_dir = run_dir

    with open(os.path.join(run_dir, "effective_config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    best_comp_bits = 1e12
    wandb.log({'objective': best_comp_bits})

    
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
    
    (train_g, val_g, test_g,(e2i, i2e),(r2i, i2r),(min_edges, max_edges), _) = load_data_as_list(dataset_name)

    num_entities  = len(e2i)
    num_relations = len(r2i)
    use_padding  = config.get("use_padding", dataset_name.startswith("wd-"))

    if use_padding:
        PAD_EID = num_entities
        PAD_RID = num_relations
        num_entities  += 1
        num_relations += 1
    else:
        PAD_EID = None
        PAD_RID = None

    #necessary tokens for the autoregressive model (begining and end of sequence)
    special_tokens    = {"PAD": 0, "BOS": 1, "EOS": 2}
    ENT_BASE   = 3
    REL_BASE   = ENT_BASE + num_entities  
    VOCAB_SIZE = REL_BASE + num_relations
    seq_len    = 1 + max_edges * 3 + 1
    
    #custom dataset class because of different ordering/shuffling etc 
    train_loader = PDataLoader(
    GraphSeqDataset(
        graphs=train_g,
        i2e=i2e,
        i2r=i2r,
        triple_order=config["triple_order"],
        permute= config.get("permute_triples", False),
        use_padding=use_padding,
        pad_eid=PAD_EID,
        pad_rid=PAD_RID,
        max_triples=max_edges,
        special_tokens=special_tokens,
        ent_base=ENT_BASE,
        rel_base=REL_BASE,
        seq_len=seq_len,
    ),
    batch_size=config['batch_size'],
    shuffle=config['shuffle_train'],
    drop_last=True,
)

    val_loader = PDataLoader(
        GraphSeqDataset(
            graphs=val_g,
            i2e=i2e,
            i2r=i2r,
            triple_order=config['triple_order'],
            permute=config.get("permute_triples", False),
            use_padding=use_padding,
            pad_eid=PAD_EID,
            pad_rid=PAD_RID,
            max_triples=max_edges,
            special_tokens=special_tokens,
            ent_base=ENT_BASE,
            rel_base=REL_BASE,
            seq_len=seq_len,
        ),
        batch_size=config['batch_size']
    )

    test_loader = PDataLoader(
        GraphSeqDataset(
            graphs=test_g,
            i2e=i2e,
            i2r=i2r,
            triple_order=config['triple_order'],
            permute=config.get("permute_triples", False),
            use_padding=use_padding,
            pad_eid=PAD_EID,
            pad_rid=PAD_RID,
            max_triples=max_edges,
            special_tokens=special_tokens,
            ent_base=ENT_BASE,
            rel_base=REL_BASE,
            seq_len=seq_len,
        ),
        batch_size=config['batch_size']
    )
    
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
    model_type = config.get('model_type', 'ARK')
    
   
    if model_type == 'ARK' or model_type == 't-ARK':
        config.update({
            "n_entities": num_entities,
            "n_relations": num_relations,
            "pad_eid": PAD_EID,
            "pad_rid": PAD_RID,
            "seq_len": seq_len,
            "vocab_size": VOCAB_SIZE,
            "special_tokens": special_tokens,
            "ENT_BASE": ENT_BASE,
            "REL_BASE": REL_BASE
        })
        model = ARK(config).to(device)
        
    else:
        raise NotImplementedError(
            f"Model type '{model_type}' is not implemented. Use one of: 'ARK','t-ARK'."
        )

    
    print(f"Using model: {model_type}")
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    scheduler = None
    if config.get('lr_scheduler', False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('eta_min', 1e-6)
        )
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        b = 1

        train_results = train_epoch(model, train_loader, optimizer, config, device,b)
        
        comp_every = int(config.get('compression_log_every', 5))
        do_comp   = ((epoch + 1) % comp_every == 0)
            
        val_results = validate(model, val_loader, config, device, compute_compression= do_comp,b=b, special_tokens=special_tokens)
        
        # Handle different return values based on model type
        train_loss, train_recon_loss, train_kl_loss, train_entity_loss = train_results
        val_loss,   val_recon_loss,   val_kl_loss,   val_entity_loss   = val_results[:4]

        
        if do_comp and len(val_results) == 8:
            _, _, _, _, val_comp_bits, val_kl_bits, val_edge_bits, val_entity_bits = val_results
            wandb.log({
            'val/compression_bits': val_comp_bits,
            'val/compression_kl_bits': val_kl_bits,
            'val/compression_edge_bits': val_edge_bits,
            'val/compression_entity_bits': val_entity_bits,
        })
            
            vcb = float(val_comp_bits)
            if math.isfinite(vcb) and vcb < best_comp_bits:
                best_comp_bits = vcb
            wandb.log({'objective': best_comp_bits})
        else:
            wandb.log({'objective': best_comp_bits})


        
        # Log basic metrics
        log_dict = {
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/reconstruction_loss': train_recon_loss,
            'val/loss': val_loss,
            'val/reconstruction_loss': val_recon_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }

        if model_type in ('SAIL', 't-SAIL'):
            log_dict['train/kl_loss'] = train_kl_loss
            log_dict['val/kl_loss']   = val_kl_loss

        
        # Periodically verify generated graphs
        if verifier and (epoch + 1) % config.get('verify_every', 10) == 0:
            target_N = config.get('num_generated_latent_graphs', 1000)
            chunk_size = 50
            all_batches = []

            while sum(x.size(0) for x in all_batches) < target_N:
                with torch.inference_mode():
                    seq_batch = model.generate(
                        seq_len, special_tokens,
                        device=device,
                        batch_size=chunk_size,
                        beam=1,
                        sample=True,
                        temperature=config.get('temperature', 1.0),
                        top_p=config.get('top_p', 0.9),
                        top_k=config.get('top_k', 0)
                    ).cpu()
                all_batches.append(seq_batch)

            seq_batch = torch.cat(all_batches, dim=0)[:target_N]

            graphs = [seq_to_triples(row.cpu(), special_tokens, ENT_BASE, REL_BASE) for row in seq_batch]
            labels = ints_to_labels(graphs, i2e, i2r)

            print("\nExample graphs (decoder-only):")
            for k in range(min(5, len(labels))):
                print(f"[{k}] {labels[k]}")

            sem_eval = run_semantic_evaluation(labels, train_g, i2e, i2r, verifier, title="decoder-only samples")
            res = sem_eval.organized_results["results"]

            wandb.log({
                "verification/validity_rate":       res.get("semantics", 0.0) / 100.0,
                "verification/novelty_rate":        res.get("novel", 0.0) / 100.0,
                "verification/valid_novelty_rate":  res.get("novel_semantics", 0.0) / 100.0,
            })

            print(f"Verification — validity: {res.get('semantics',0.0):.2f}% | "
                f"novelty: {res.get('novel',0.0):.2f}% | "
                f"valid&novel: {res.get('novel_semantics',0.0):.2f}%")

        
        wandb.log(log_dict)
        
        print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f})")
        print(f"Val   Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f})")

        if scheduler is not None:
            if config.get('lr_scheduler', False):
                scheduler.step() 

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_state = model.state_dict()            
            vocabs = {
        'e2i': e2i, 'i2e': i2e,
        'r2i': r2i, 'i2r': i2r,
    }
            dataset_meta = {
        'dataset': dataset_name,
        'n_entities': len(i2e),
        'n_relations': len(i2r),
    }
            checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': (scheduler.state_dict() if scheduler is not None else None),
        'val_loss': val_loss,
        'config': config,
        'vocabs': vocabs,
        'dataset_meta': dataset_meta,
    }

            torch.save(
                checkpoint,
                os.path.join(args.checkpoint_dir, f'{dataset_name}_{model_type}_best_model.pt'), _use_new_zipfile_serialization=False

            )
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        if (epoch + 1) % config.get('save_every', 10) == 0:
            model_state = model.state_dict()
            vocabs = {
        'e2i': e2i, 'i2e': i2e,
        'r2i': r2i, 'i2r': i2r,
    }
            dataset_meta = {
        'dataset': dataset_name,
        'n_entities': len(i2e),
        'n_relations': len(i2r),
    }
            checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': (scheduler.state_dict() if scheduler is not None else None), 
        'val_loss': val_loss,
        'config': config,
        'vocabs': vocabs,
        'dataset_meta': dataset_meta,
    }
            torch.save(
                checkpoint,
                os.path.join(args.checkpoint_dir, f'{dataset_name}_{model_type}_checkpoint_epoch_{epoch+1}.pt'),     _use_new_zipfile_serialization=False
            )
    # Perform final validation
    final_metrics = final_validation(model, test_loader, val_loader, config, device, verifier, i2e, i2r, b=1.0, special_tokens=special_tokens, seq_len=seq_len, ENT_BASE=ENT_BASE, REL_BASE=REL_BASE, train_g=train_g)
    wandb.log(final_metrics)
    
    wandb.finish()
    print("\nTraining and evaluation completed!")


if __name__ == "__main__":
    main()


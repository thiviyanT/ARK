import argparse
import torch
import torch.nn as nn
import wandb
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
try:
    import scienceplots
    plt.style.use(['science', 'ieee'])
except ImportError:
    print("SciencePlots not available, using default style")
from sklearn.manifold import TSNE
import networkx as nx
from typing import List, Dict, Tuple, Set
from kgvae.model.rescal_vae_model import RESCALVAE
from kgvae.model.models import AutoRegModel
from kgvae.model.utils import seq_to_triples, ints_to_labels
from intelligraphs.data_loaders import load_data_as_list


def jaccard(a: set, b: set) -> float:
    """
    Calculate Jaccard similarity between two sets.
    
    Args:
        a: First set to compare
        b: Second set to compare
    
    Returns:
        float: Jaccard similarity score between 0 and 1
    """
    if not a and not b: 
        return 1.0
    if not a or not b: 
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union


def decode_to_triple_set(
    model_unwrapped,
    z: torch.Tensor,
    seq_len: int,
    special_tokens: dict,
    entity_base_idx: int,
    relation_base_idx: int,
    beam: int = 1
) -> set:
    """
    Decode a latent vector to a set of Knowledge Graph triples.
    
    Takes a latent representation and decodes it into a set of (head, relation, tail)
    triples representing edges in a knowledge graph. The triples are returned as
    integer IDs for efficient comparison.
    
    Args:
        model_unwrapped: The unwrapped model (not DataParallel wrapped)
        z: Latent vector to decode (shape: [latent_dim])
        seq_len: Maximum sequence length for decoding
        special_tokens: Dictionary of special tokens used in decoding
        entity_base_idx: Base index for entity IDs in vocabulary
        relation_base_idx: Base index for relation IDs in vocabulary
        beam: Beam size for beam search decoding (1 = greedy)
    
    Returns:
        set: Set of tuples, each containing (head_id, relation_id, tail_id) as integers
    """
    decoded_graph = model_unwrapped.decode_latent(
        z.unsqueeze(0), seq_len, special_tokens, seq_to_triples, 
        entity_base_idx, relation_base_idx, beam=beam
    )[0]
    return set(tuple(map(int, t)) for t in decoded_graph)



def load_model(checkpoint_dir, dataset, model_type, epoch=None, device=None):
    """
    Load a trained VAE model from checkpoint.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        dataset: Name of the dataset (e.g., 'syn-paths')
        model_type: Type of model ('autoreg' or 'rescal_vae')
        epoch: Specific epoch to load, or None for best model
        device: Device to load model on ('cuda' or 'cpu'), auto-detected if None
    
    Returns:
        tuple: (model, config, checkpoint_path)
            - model: Loaded model in eval mode
            - config: Model configuration dictionary
            - checkpoint_path: Path to loaded checkpoint file
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if epoch is None:
        ckpt_path = os.path.join(checkpoint_dir, f"{dataset}_{model_type}_best_model.pt")
    else:
        ckpt_path = os.path.join(checkpoint_dir, f"{dataset}_{model_type}_checkpoint_epoch_{epoch}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if model_type == "autoreg":
        model = AutoRegModel(config).to(device)
    elif model_type == "rescal_vae":
        model = RESCALVAE(config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.load_state_dict(state)
    model.eval()
    return model, config, ckpt_path


@torch.no_grad()
def random_steps_latent_autoreg(model, i2e, i2r, n_directions=20, epsilon=1.2, device=None):
    """
    Explore local latent space neighborhood by perturbing in random directions.
    
    Samples a random latent point z₀ and perturbs it in n_directions random unit
    directions scaled by epsilon. Decodes each perturbed point and compares the
    resulting graphs to understand local latent space structure.
    
    Args:
        model: Trained autoregressive VAE model
        i2e: Index-to-entity mapping dictionary
        i2r: Index-to-relation mapping dictionary
        n_directions: Number of random directions to explore
        epsilon: Step size for perturbations
        device: Device for computation, auto-detected if None
    """
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    config = model_unwrapped.config
    seq_len        = config["seq_len"]
    special_tokens = config["special_tokens"]
    entity_base_idx       = config["ENT_BASE"]
    relation_base_idx       = config["REL_BASE"]
    latent_dim     = config["d_latent"]

    if device is None:
        device = next(model_unwrapped.parameters()).device
    z0 = torch.randn(latent_dim, device=device)
    directions = torch.randn(n_directions, latent_dim, device=device)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    perturbed_zs = z0.unsqueeze(0) + epsilon * directions 
    ref_graphs = model_unwrapped.decode_latent(
        z0.unsqueeze(0), seq_len, special_tokens, seq_to_triples, entity_base_idx, relation_base_idx, beam=1
    )
    ref_triples = ints_to_labels(ref_graphs, i2e, i2r)[0]
    decoded_graphs = model_unwrapped.decode_latent(
        perturbed_zs, seq_len, special_tokens, seq_to_triples, entity_base_idx, relation_base_idx, beam=1
    )
    decoded_triples = ints_to_labels(decoded_graphs, i2e, i2r)
    print("\n=== Local Latent Neighborhood Exploration ===")
    print("\n--- Reference Graph (z₀) ---")
    for h, r, t in ref_triples:
        print(f"({h}, {r}, {t})")

    for i, graph in enumerate(decoded_triples):
        print(f"\n--- Perturbed z #{i+1} ---")
        for h, r, t in graph:
            print(f"({h}, {r}, {t})")
        overlap = set(ref_triples) & set(graph)
        denom = max(1, len(ref_triples))
        print(f"# Overlapping triples with z₀: {len(overlap)} / {denom}")
          
@torch.no_grad()
def smoothness_line_check_autoreg(model, i2e, i2r, steps: int = 10, epsilon: float = 0.1, device: str = None, beam: int = 1):
    """
    Walk along a line in latent space to measure smoothness.
    
    Starting from a random point z₀, takes steps along a random unit direction,
    decoding at each point. Measures both local smoothness (similarity to previous
    step) and global drift (similarity to starting point).
    
    Args:
        model: Trained autoregressive VAE model
        i2e: Index-to-entity mapping dictionary
        i2r: Index-to-relation mapping dictionary
        steps: Number of steps to take along the line
        epsilon: Step size in latent space
        device: Device for computation, auto-detected if None
        beam: Beam size for decoding (1 = greedy)
    """
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    config = model_unwrapped.config
    seq_len        = config["seq_len"]
    special_tokens = config["special_tokens"]
    entity_base_idx       = config["ENT_BASE"]
    relation_base_idx       = config["REL_BASE"]
    latent_dim       = config["d_latent"]

    if device is None:
        device = next(model_unwrapped.parameters()).device

    # Anchor and direction
    z0  = torch.randn(latent_dim, device=device)
    direction = torch.randn(latent_dim, device=device)
    direction = direction / direction.norm().clamp_min(1e-12)

    # Decode anchor
    anchor_graph_int = model_unwrapped.decode_latent(
        z0.unsqueeze(0), seq_len, special_tokens, seq_to_triples, entity_base_idx, relation_base_idx, beam=beam
    )
    anchor_graph = ints_to_labels(anchor_graph_int, i2e, i2r)[0]

    print("\n=== Latent Smoothness Line Walk ===")
    print(f"Steps: {steps} | step size ε = {epsilon}")
    print("\n--- Anchor (z₀) ---")
    for h, r, t in anchor_graph:
        print(f"({h}, {r}, {t})")

    prev_graph = anchor_graph
    prev_z = z0.clone()

    total_local = 0.0
    total_global = 0.0
    denom_anchor = max(1, len(anchor_graph))

    for s in range(1, steps + 1):
        z = z0 + (s * epsilon) * direction
        dec_int = model_unwrapped.decode_latent(
            z.unsqueeze(0), seq_len, special_tokens, seq_to_triples, entity_base_idx, relation_base_idx, beam=beam
        )
        graph = ints_to_labels(dec_int, i2e, i2r)[0]

        # Overlaps
        denom_prev = max(1, len(prev_graph))
        local_overlap  = len(set(prev_graph) & set(graph)) / denom_prev
        global_overlap = len(set(anchor_graph) & set(graph)) / denom_anchor

        total_local  += local_overlap
        total_global += global_overlap

        print(f"\n--- Step {s}: z = z₀ + {s}·ε·direction ---")
        for h, r, t in graph:
            print(f"({h}, {r}, {t})")
        print(f"Local smoothness (vs step {s-1}): {local_overlap:.2f}")
        print(f"Global overlap (vs anchor)     : {global_overlap:.2f}")

        prev_graph = graph
        prev_z = z

    print("\n=== Summary ===")
    print(f"Avg local smoothness over {steps} steps: {total_local/steps:.2f}")
    print(f"Avg global overlap over {steps} steps : {total_global/steps:.2f}")


@torch.no_grad()
def latent_smoothness_score_autoreg(model, steps:int=10, epsilon:float=0.1, n_anchors:int=3, n_dirs:int=3, beam:int=1, device:str=None):
    """
    Compute quantitative smoothness scores using Jaccard similarity.
    
    Performs multiple random walks from different anchor points in latent space,
    computing Jaccard similarity between consecutive steps (local smoothness)
    and between each step and the anchor (global consistency).
    
    Args:
        model: Trained autoregressive VAE model
        steps: Number of steps per walk
        epsilon: Step size in latent space
        n_anchors: Number of random starting points to test
        n_dirs: Number of random directions per anchor
        beam: Beam size for decoding (1 = greedy)
        device: Device for computation, auto-detected if None
    
    Returns:
        tuple: (avg_local_jaccard, avg_global_jaccard)
            - avg_local_jaccard: Average Jaccard between consecutive steps
            - avg_global_jaccard: Average Jaccard between steps and anchors
    """
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    config = model_unwrapped.config
    seq_len        = config["seq_len"]
    special_tokens = config["special_tokens"]
    entity_base_idx       = config["ENT_BASE"]
    relation_base_idx       = config["REL_BASE"]
    latent_dim       = config["d_latent"]
    if device is None:
        device = next(model_unwrapped.parameters()).device

    total_local = 0.0
    total_global = 0.0
    count_local = 0
    count_global = 0

    for _ in range(n_anchors):
        z0  = torch.randn(latent_dim, device=device)
        anchor = decode_to_triple_set(model_unwrapped, z0, seq_len, special_tokens, entity_base_idx, relation_base_idx, beam)
        for _ in range(n_dirs):
            direction = torch.randn(latent_dim, device=device)
            direction = direction / direction.norm().clamp_min(1e-12)

            prev = anchor
            # march: z_s = z0 + s*epsilon*d
            for s in range(1, steps+1):
                z = z0 + (s * epsilon) * direction
                cur = decode_to_triple_set(model_unwrapped, z, seq_len, special_tokens, entity_base_idx, relation_base_idx, beam)
                total_local  += jaccard(cur, prev)
                total_global += jaccard(cur, anchor)
                count_local  += 1
                count_global += 1
                prev = cur

    avg_local  = total_local / max(1, count_local)
    avg_global = total_global / max(1, count_global)
    print(f"\n[SMOOTHNESS SCORE] anchors={n_anchors}, dirs={n_dirs}, steps={steps}, ε={epsilon}")
    print(f"Avg local Jaccard : {avg_local:.3f}")
    print(f"Avg global Jaccard: {avg_global:.3f}")
    return avg_local, avg_global


@torch.no_grad()
def latent_flip_rate_autoreg(model, steps:int=30, epsilon:float=0.05, n_anchors:int=5, n_dirs:int=4, beam:int=1, device:str=None):
    """
    Measure discreteness of latent space by tracking graph changes.
    
    Walks through latent space with small steps and tracks how often the decoded
    graph changes. High flip rates indicate discrete/non-smooth latent space,
    while low flip rates suggest smooth interpolation. Basin length measures
    the average number of consecutive steps producing identical graphs.
    
    Args:
        model: Trained autoregressive VAE model
        steps: Number of steps per walk
        epsilon: Step size in latent space
        n_anchors: Number of random starting points to test
        n_dirs: Number of random directions per anchor
        beam: Beam size for decoding (1 = greedy)
        device: Device for computation, auto-detected if None
    
    Returns:
        tuple: (flip_rate, avg_basin)
            - flip_rate: Fraction of steps that change the decoded graph (0-1)
            - avg_basin: Average number of consecutive steps with same graph
    """
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    config = model_unwrapped.config
    seq_len        = config["seq_len"]
    special_tokens = config["special_tokens"]
    entity_base_idx       = config["ENT_BASE"]
    relation_base_idx       = config["REL_BASE"]
    latent_dim       = config["d_latent"]
    if device is None:
        device = next(model_unwrapped.parameters()).device

    total_flips = 0
    total_steps = 0
    all_basin_lengths = []

    for _ in range(n_anchors):
        z0  = torch.randn(latent_dim, device=device)
        for _ in range(n_dirs):
            direction = torch.randn(latent_dim, device=device)
            direction = direction / direction.norm().clamp_min(1e-12)

            prev_set = decode_to_triple_set(model_unwrapped, z0, seq_len, special_tokens, entity_base_idx, relation_base_idx, beam)
            basin_len = 1
            last_was_flip = False
            for s in range(1, steps+1):
                z = z0 + (s * epsilon) * direction
                cur_set = decode_to_triple_set(model_unwrapped, z, seq_len, special_tokens, entity_base_idx, relation_base_idx, beam)
                flipped = int(cur_set != prev_set)
                total_flips += flipped
                total_steps += 1
                if flipped:
                    all_basin_lengths.append(basin_len)
                    basin_len = 1
                    last_was_flip = True
                else:
                    basin_len += 1
                    last_was_flip = False
                prev_set = cur_set
            # Only append the final basin if the last step wasn't a flip
            # (if it was a flip, we already recorded that basin)
            if not last_was_flip and basin_len > 0:
                all_basin_lengths.append(basin_len)

    flip_rate = total_flips / max(1, total_steps)           # 0..1
    avg_basin = (sum(all_basin_lengths) / max(1, len(all_basin_lengths)))
    print(f"\n[FLIP RATE] anchors={n_anchors}, dirs={n_dirs}, steps={steps}, ε={epsilon}")
    print(f"Flip rate      : {flip_rate:.3f} (fraction of step transitions that change graph)")
    print(f"Avg basin len  : {avg_basin:.2f} steps")
    return flip_rate, avg_basin


@torch.no_grad()
def qualitative_latent_analysis_wd_movies(model, output_dir: str = "figures", n_samples: int = 500, use_all_test: bool = False, device: str = None):
    """
    Perform qualitative analysis of latent space for wd-movies dataset.
    
    Generates three visualizations:
    1. t-SNE projection colored by movie genre
    2. Linear interpolation path visualization
    3. Decoded graphs at interpolation points
    
    Args:
        model: Trained autoregressive VAE model
        output_dir: Directory to save generated figures
        n_samples: Number of test graphs to encode for t-SNE (ignored if use_all_test=True)
        use_all_test: If True, use all graphs in test dataset
        device: Device for computation, auto-detected if None
    """
    import scienceplots
    plt.style.use(['science', 'ieee'])
    
    os.makedirs(output_dir, exist_ok=True)

    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    config = model_unwrapped.config
    seq_len = config["seq_len"]
    special_tokens = config["special_tokens"]
    entity_base_idx = config["ENT_BASE"]
    relation_base_idx = config["REL_BASE"]
    
    if device is None:
        device = next(model_unwrapped.parameters()).device

    # load wd-movies dataset
    train_list, valid_list, test_list, (e2i, i2e), (r2i, i2r), _, _ = load_data_as_list("wd-movies")

    def extract_genres_from_graph(graph_triples: List[Tuple]) -> List[str]:
        """Extract all genres from movie graph triples using 'has_genre' relationship."""
        genres = []
        for h, r, t in graph_triples:
            # Look specifically for 'has_genre' relationship
            if 'has_genre' in r.lower() or 'genre' == r.lower():
                genres.append(t)  # t is the genre entity
        return genres if genres else ['unknown']

    def get_primary_genre(genres: List[str]) -> str:
        """Get the primary (first) genre for visualization."""
        return genres[0] if genres else 'unknown'

    print("\n=== Preparing data for analysis ===")
    
    if use_all_test:
        test_sample = test_list
        print(f"Using all {len(test_sample)} test graphs")
    else:
        test_sample = test_list[:min(n_samples, len(test_list))]
        print(f"Using {len(test_sample)} sampled test graphs")

    print("\n=== Extracting genres from dataset ===")
    all_genres_set = set()
    for graph in test_sample:
        graph_labels = ints_to_labels([graph], i2e, i2r)[0]
        genres = extract_genres_from_graph(graph_labels)
        all_genres_set.update(genres)
    
    print(f"Found {len(all_genres_set)} unique genres in dataset")
    print(f"Genre examples: {list(all_genres_set)[:10]}")
    
    # create color mapping for genres (use a colormap for many genres)
    import matplotlib.cm as cm
    unique_genres = sorted(list(all_genres_set))
    n_genres = len(unique_genres)

    # use distinct colors for small number of genres, otherwise use continuous colormap for many genres
    if n_genres <= 20:
        cmap = cm.get_cmap('tab20')
        genre_colors = {genre: cmap(i/20) for i, genre in enumerate(unique_genres)}
    else:
        cmap = cm.get_cmap('hsv')
        genre_colors = {genre: cmap(i/n_genres) for i, genre in enumerate(unique_genres)}

    print("\n=== Encoding graphs to latent space ===")
    
    latent_vectors = []
    all_genres_list = []
    primary_genres = []  # primary genre for each graph (for coloring)
    
    for graph_idx, graph in enumerate(test_sample):
        if graph_idx % 50 == 0:
            print(f"Encoding graph {graph_idx}/{len(test_sample)}")
        
        # Convert graph to tensor format for encoding
        # The AutoRegEncoder expects triples in shape (batch_size, num_triples, 3)
        # where each triple is [subject_id, relation_id, object_id]
        
        try:
            # convert graph (list of triples) to tensor
            # graph is already in integer format from the dataset
            if len(graph) == 0:
                continue
                
            # pad graph to fixed size if needed
            max_triples = config.get('max_edges', 100)
            
            # convert to tensor and pad if necessary
            graph_tensor = torch.zeros((1, max_triples, 3), dtype=torch.long, device=device)
            num_triples = min(len(graph), max_triples)
            
            for i in range(num_triples):
                if len(graph[i]) == 3:  # ensure triple has 3 elements
                    graph_tensor[0, i, 0] = graph[i][0]  # subject
                    graph_tensor[0, i, 1] = graph[i][1]  # relation
                    graph_tensor[0, i, 2] = graph[i][2]  # object
            
            # use padding IDs if available
            pad_rid = config.get('pad_rid', 0)
            if num_triples < max_triples:
                graph_tensor[0, num_triples:, 1] = pad_rid  # Mark padded triples
            
            # encode using the model's encoder
            with torch.no_grad():
                z, mu, logv = model_unwrapped.enc(graph_tensor)
                latent_vectors.append(mu[0].cpu().numpy())  # Take first element of batch
                    
        except Exception as e:
            print(f"Skipping graph {graph_idx} due to error: {e}")
            continue
        
        # extract genres
        graph_labels = ints_to_labels([graph], i2e, i2r)[0]
        genres = extract_genres_from_graph(graph_labels)
        all_genres_list.append(genres)
        primary_genres.append(get_primary_genre(genres))
    
    if len(latent_vectors) == 0:
        print("Warning: No graphs could be encoded. Please implement encode_graph method or graph tensor preparation.")
        return
    
    latent_vectors = np.vstack(latent_vectors)
    print(f"Encoded {len(latent_vectors)} graphs successfully")

    # perform t-SNE
    print("\n=== Running t-SNE projection ===")
    tsne = TSNE(n_components=2, perplexity=min(30, len(latent_vectors)-1), 
                random_state=42, n_iter=1000)
    latent_2d = tsne.fit_transform(latent_vectors)

    # plot t-SNE with actual genre colors
    print("\n=== Generating t-SNE visualization ===")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    # plot points colored by primary genre
    for genre in unique_genres[:20]:  # Limit legend to 20 genres for readability
        mask = [g == genre for g in primary_genres]
        if any(mask):
            points = latent_2d[mask]
            ax1.scatter(points[:, 0], points[:, 1], 
                       c=[genre_colors[genre]], 
                       label=genre, 
                       alpha=0.6, 
                       s=30)
    
    # handle remaining genres if more than 20
    if n_genres > 20:
        other_mask = [g not in unique_genres[:20] for g in primary_genres]
        if any(other_mask):
            points = latent_2d[other_mask]
            ax1.scatter(points[:, 0], points[:, 1], 
                       c='gray', 
                       label='Other genres', 
                       alpha=0.4, 
                       s=20)
    
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.set_title(f'Latent Space Structure (wd-movies)', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2 if n_genres > 10 else 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_tsne_movies.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n=== Generating interpolation path visualization ===")
    
    # find two movies with different genres
    genre_pairs = []
    for i, genres_i in enumerate(all_genres_list):
        for j, genres_j in enumerate(all_genres_list[i+1:], i+1):
            # Check if genres are different
            if set(genres_i).isdisjoint(set(genres_j)):
                genre_pairs.append((i, j, genres_i[0], genres_j[0]))
                if len(genre_pairs) >= 5:  # Get a few pairs to choose from
                    break
        if len(genre_pairs) >= 5:
            break
    
    if len(genre_pairs) > 0:
        # select first pair with different genres
        idx1, idx2, genre1, genre2 = genre_pairs[0]
        print(f"Interpolating between {genre1} and {genre2}")
        
        z1 = torch.tensor(latent_vectors[idx1], device=device, dtype=torch.float32)
        z2 = torch.tensor(latent_vectors[idx2], device=device, dtype=torch.float32)
        
        # generate interpolation path
        n_interp_points = 20
        alphas = np.linspace(0, 1, n_interp_points)
        interp_points = []
        
        for alpha in alphas:
            z_alpha = (1 - alpha) * z1 + alpha * z2
            interp_points.append(z_alpha.cpu().numpy())
        
        interp_points = np.vstack(interp_points)
        
        # project interpolation path to 2D using existing t-SNE model
        # note: This is an approximation; proper way would be to retrain t-SNE
        interp_2d = tsne.fit_transform(np.vstack([latent_vectors, interp_points]))
        
        fig2, ax2 = plt.subplots(figsize=(10, 8))

        ax2.scatter(interp_2d[:len(latent_vectors), 0], 
                   interp_2d[:len(latent_vectors), 1], 
                   c='lightgray', alpha=0.3, s=10)
        
        # plot interpolation path
        path_2d = interp_2d[len(latent_vectors):]
        ax2.plot(path_2d[:, 0], path_2d[:, 1], 'b-', linewidth=2, alpha=0.7, label='Interpolation path')
        ax2.scatter(path_2d[0, 0], path_2d[0, 1], c='red', s=150, marker='s', 
                   label=f'Start: {genre1}', zorder=5, edgecolor='black')
        ax2.scatter(path_2d[-1, 0], path_2d[-1, 1], c='blue', s=150, marker='s', 
                   label=f'End: {genre2}', zorder=5, edgecolor='black')
        
        # mark intermediate points
        for i in [5, 10, 15]:
            ax2.scatter(path_2d[i, 0], path_2d[i, 1], c='orange', s=80, marker='o', zorder=4)
        
        ax2.set_xlabel('t-SNE Component 1', fontsize=12)
        ax2.set_ylabel('t-SNE Component 2', fontsize=12)
        ax2.set_title('Linear Interpolation in Latent Space', fontsize=14)
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latent_interpolation.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

        print("\n=== Generating interpolation sequence ===")
        
        fig3 = plt.figure(figsize=(18, 4))
        gs = GridSpec(1, 5, figure=fig3, wspace=0.3)
        
        alpha_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for idx, alpha in enumerate(alpha_points):
            ax = fig3.add_subplot(gs[0, idx])

            z_alpha = (1 - alpha) * z1 + alpha * z2
            
            # decode to graph
            decoded_graph = decode_to_triple_set(
                model_unwrapped, z_alpha, seq_len, special_tokens,
                entity_base_idx, relation_base_idx, beam=1
            )
            
            # convert to labels
            decoded_labels = ints_to_labels([list(decoded_graph)], i2e, i2r)[0]
            
            # extract genres from decoded graph
            decoded_genres = extract_genres_from_graph(decoded_labels)
            
            # create a small graph visualization
            G = nx.DiGraph()
            for h, r, t in decoded_labels[:6]:  # Show first 6 triples
                # Truncate labels for display
                h_short = h[:15] + '...' if len(h) > 15 else h
                t_short = t[:15] + '...' if len(t) > 15 else t
                r_short = r[:10]
                G.add_edge(h_short, t_short, label=r_short)

            pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                  node_size=600, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                  arrows=True, arrowsize=10, ax=ax, width=1.5)

            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
            
            ax.set_title(f'α = {alpha}', fontsize=11, fontweight='bold')
            ax.axis('off')

            genres_text = ', '.join(decoded_genres[:3])  # Show up to 3 genres
            if len(decoded_genres) > 3:
                genres_text += '...'
            ax.text(0.5, -0.15, f'Genres: {genres_text}', 
                   transform=ax.transAxes,
                   ha='center', fontsize=8, style='italic')
        
        fig3.suptitle('Decoded Graphs Along Interpolation Path', fontsize=14, y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'interpolation_sequence.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ Qualitative analysis complete. Figures saved to {output_dir}/")
    print("  - latent_tsne_movies.pdf")
    print("  - latent_interpolation.pdf") 
    print("  - interpolation_sequence.pdf")



def main():
    """
    Main entry point for latent space interpolation experiments.
    
    Loads a trained VAE model and runs various interpolation experiments to
    analyze the structure and smoothness of the learned latent space.
    Experiments include random perturbations, line walks, smoothness scoring,
    and flip rate analysis across multiple epsilon values.
    
    Command-line Arguments:
        --config: Path to model configuration YAML file (required)
        --checkpoint-dir: Directory containing model checkpoints
        --wandb-project: Weights & Biases project name for logging
        --wandb-entity: W&B entity/username
        --directions: Number of random directions for perturbation experiment
        --epsilon: Base step size for interpolation (overridden in main loop)
        --epoch: Specific epoch to load, or None for best model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--wandb-project', type=str, default='submission', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default='a-vozikis-vrije-universiteit-amsterdam', help='W&B entity')
    parser.add_argument('--directions', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=None, help='If set, load that epoch; else load best')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset    = config['dataset']
    model_type = config.get('model_type', 'autoreg')
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, config, ckpt_path = load_model(
        checkpoint_dir=args.checkpoint_dir,
        dataset=dataset,
        model_type=model_type,
        epoch=args.epoch,
        device=device,
    )
    wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    config=config,
    name=f"latent_interp_{config['dataset']}_{config.get('model_type','autoreg')}"
)

    kind = f"epoch {args.epoch}" if args.epoch is not None else "best"
    print(f"✅ Loaded {model_type} for {dataset} ({kind}) from {ckpt_path} on {device}")

    if model_type == "autoreg":
        _, _, _, (e2i, i2e), (r2i, i2r), _, _ = load_data_as_list(dataset)
        for e in [0.02, 0.05, 0.07, 0.1, 0.12, 0.15, 0.17, 0.2]:
            print("----------------------------------------------------------------------")
            print ("epsilon value is:", e)
            print("----------------------------------------------------------------------")
            
            random_steps_latent_autoreg(
                model,
                i2e=i2e,
                i2r=i2r,
                n_directions=args.directions,
                epsilon=e,
                device=device
            )
            smoothness_line_check_autoreg(
                model,
                i2e=i2e,
                i2r=i2r,
                steps=10,             
                epsilon=e, 
                device=device,
                beam=1
            )
            latent_smoothness_score_autoreg(
                model,
                steps=10,
                epsilon=e,  # uses your existing flag
                n_anchors=3,
                n_dirs=3,
                beam=1,
                device=device,
            )
            latent_flip_rate_autoreg(
            model,
            steps=30,
            epsilon=e,  # reuse your flag
            n_anchors=5,
            n_dirs=4,
            beam=1,
            device=device,
            )

    wandb.finish()

if __name__ == "__main__":
    main()

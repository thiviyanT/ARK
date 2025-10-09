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
    plt.style.use(['science', 'ieee', 'no-latex'])

except ImportError:
    print("SciencePlots not available, using default style")
from sklearn.manifold import TSNE
import networkx as nx
from typing import List, Dict, Tuple, Set
from kgvae.model.models import SAIL
from kgvae.model.utils import seq_to_triples, ints_to_labels
from intelligraphs.data_loaders import get_file_paths, parse_files_to_subgraphs



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
    beam: int = 3
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


def load_graphs_with_checkpoint_vocab(dataset: str, e2i: Dict[str, int], r2i: Dict[str, int]):
    """Load dataset triples and re-map them to checkpoint vocab ids."""

    def _map_graphs(raw_graphs):
        mapped = []
        for graph in raw_graphs:
            triples = []
            for s, p, o in graph:
                if s in e2i and p in r2i and o in e2i:
                    triples.append((e2i[s], r2i[p], e2i[o]))
            mapped.append(triples)
        return mapped

    train_file, val_file, test_file = get_file_paths(dataset)
    train_raw, val_raw, test_raw = parse_files_to_subgraphs(train_file, val_file, test_file, split_tab=True)

    return (
        _map_graphs(train_raw),
        _map_graphs(val_raw),
        _map_graphs(test_raw),
    )


def load_model(checkpoint_dir, dataset, model_type, epoch=None, device=None):
    """
    Load a trained VAE model from checkpoint.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        dataset: Name of the dataset (e.g., 'syn-paths')
        model_type: Type of model ('SAIL' or 't-SAIL')
        epoch: Specific epoch to load, or None for best model
        device: Device to load model on ('cuda' or 'cpu'), auto-detected if None
    
    Returns:
        tuple: (model, config, checkpoint_path)
            - model: Loaded model in eval mode
            - config: Model configuration dictionary
            - checkpoint_path: Path to loaded checkpoint file
            - vocabs: Dictionary of vocabularies (e.g., {'e2i', 'i2e', 'r2i', 'i2r'})
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if epoch is None:
        ckpt_path = os.path.join(checkpoint_dir, f"{dataset}_{model_type}_best_model.pt")
    else:
        ckpt_path = os.path.join(checkpoint_dir, f"{dataset}_{model_type}_checkpoint_epoch_{epoch}.pt")

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    val = config.get("ablation_encoder")
    if not val or str(val).lower() == "none":
        config["ablation_encoder"] = "Transformer"

    val = config.get("ablation_decoder")
    if not val or str(val).lower() == "none":
        config["ablation_decoder"] = "Transformer"

    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    if model_type == "SAIL" or model_type == "t-SAIL":
        model = SAIL(config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state)
    model.eval()

    vocabs = ckpt.get("vocabs", None)
    dataset_meta = ckpt.get("dataset_meta", None)

    return model, config, ckpt_path, vocabs, dataset_meta



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
        z0.unsqueeze(0), seq_len, special_tokens, seq_to_triples, entity_base_idx, relation_base_idx, beam=3
    )
    ref_triples = ints_to_labels(ref_graphs, i2e, i2r)[0]
    decoded_graphs = model_unwrapped.decode_latent(
        perturbed_zs, seq_len, special_tokens, seq_to_triples, entity_base_idx, relation_base_idx, beam=3
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
def smoothness_line_check_autoreg(model, i2e, i2r, steps: int = 10, epsilon: float = 0.1, device: str = None, beam: int = 3):
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
def latent_smoothness_score_autoreg(model, steps:int=10, epsilon:float=0.1, n_anchors:int=3, n_dirs:int=3, beam:int=3, device:str=None):
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
def latent_flip_rate_autoreg(model, steps:int=30, epsilon:float=0.05, n_anchors:int=5, n_dirs:int=4, beam:int=3, device:str=None):
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
            if not last_was_flip and basin_len > 0:
                all_basin_lengths.append(basin_len)

    flip_rate = total_flips / max(1, total_steps)           
    avg_basin = (sum(all_basin_lengths) / max(1, len(all_basin_lengths)))
    print(f"\n[FLIP RATE] anchors={n_anchors}, dirs={n_dirs}, steps={steps}, ε={epsilon}")
    print(f"Flip rate      : {flip_rate:.3f} (fraction of step transitions that change graph)")
    print(f"Avg basin len  : {avg_basin:.2f} steps")
    return flip_rate, avg_basin


@torch.no_grad()
def qualitative_latent_analysis_wd_movies(
    model,
    vocabs: Dict[str, Dict],
    output_dir: str = "figures",
    n_samples: int = 5000,
    use_all_test: bool = False,
    device: str = None,
    target_genres: List[str] = None,
):
    """
    Qualitative analysis of latent space (wd-movies) with t-SNE restricted to 10 famous, diverse genres.
    """
    import matplotlib.cm as cm
    import matplotlib
    try:
        import scienceplots
        plt.style.use(['science', 'ieee', 'no-latex'])
    except Exception:
        pass

    # ---- 10 target genres (canonical names) ----
    if target_genres is None:
        target_genres = [
            'Action film',
            'Comedy film',
            'Drama film',
            'Horror film',
            'Romance film',
            'Musical film',
            'Science fiction film',
            'Western film',
            'Bollywood',
            'Documentary film',
        ]
    TARGET_SET = set(target_genres)

    _lower_to_canon = {g.lower(): g for g in TARGET_SET}

    def extract_genres_from_graph(graph_triples: List[Tuple[str, str, str]]) -> List[str]:
        """Extract canonical genres intersecting TARGET_SET without any manual normalization."""
        raw = []
        for h, r, t in graph_triples:
            if 'has_genre' in (r or '').lower() or (r or '').lower() == 'genre':
                raw.append((t or '').strip())
        canon = []
        for g in raw:
            g_can = _lower_to_canon.get(g.lower())
            if g_can is not None:
                canon.append(g_can)
        return list(dict.fromkeys(canon))


    def get_primary_genre(genres: List[str]) -> str:
        return genres[0] if genres else None

    os.makedirs(output_dir, exist_ok=True)

    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    config = model_unwrapped.config
    seq_len = config["seq_len"]
    special_tokens = config["special_tokens"]
    entity_base_idx = config["ENT_BASE"]
    relation_base_idx = config["REL_BASE"]

    if device is None:
        device = next(model_unwrapped.parameters()).device

    if vocabs is None:
        raise ValueError("qualitative_latent_analysis_wd_movies requires checkpoint vocabularies.")

    required_keys = ("e2i", "i2e", "r2i", "i2r")
    missing = [k for k in required_keys if k not in vocabs or vocabs[k] is None]
    if missing:
        raise KeyError(f"Checkpoint vocabs missing keys: {missing}")

    e2i = vocabs["e2i"]
    i2e = vocabs["i2e"]
    r2i = vocabs["r2i"]
    i2r = vocabs["i2r"]

    _, _, test_list = load_graphs_with_checkpoint_vocab("wd-movies", e2i, r2i)

    test_sample = test_list if use_all_test else test_list[:min(n_samples, len(test_list))]

    latent_vectors = []
    primary_genres = []
    kept_idx = []

    for graph_idx, graph in enumerate(test_sample):
        # labels for genre extraction
        graph_labels = ints_to_labels([graph], i2e, i2r)[0]
        genres = extract_genres_from_graph(graph_labels)
        primary = get_primary_genre(genres)

        if primary is None:
            continue  

        try:
            max_triples = config.get('max_edges', 100)
            graph_tensor = torch.zeros((1, max_triples, 3), dtype=torch.long, device=device)
            num_triples = min(len(graph), max_triples)
            for i in range(num_triples):
                if len(graph[i]) == 3:
                    graph_tensor[0, i, 0] = graph[i][0]
                    graph_tensor[0, i, 1] = graph[i][1]
                    graph_tensor[0, i, 2] = graph[i][2]
            pad_rid = config.get('pad_rid', 0)
            if num_triples < max_triples:
                graph_tensor[0, num_triples:, 1] = pad_rid

            z, mu, logv = model_unwrapped.enc(graph_tensor)
            latent_vectors.append(mu[0].cpu().numpy())
            primary_genres.append(primary)
            kept_idx.append(graph_idx)
        except Exception as e:
            continue

    if len(latent_vectors) == 0:
        print("Warning: No graphs matched the 10 target genres or encoding failed.")
        return
    latent_vectors = np.vstack(latent_vectors)

    tsne = TSNE(
        n_components=2,
        perplexity=max(5, min(30, len(latent_vectors) - 1)),
        random_state=42,
        max_iter=1000
    )
    latent_2d = tsne.fit_transform(latent_vectors)


    cmap = cm.get_cmap('tab10')
    ordered_targets = list(target_genres)  
    genre_colors = {g: cmap(i / 10) for i, g in enumerate(ordered_targets)}


    fig, ax = plt.subplots(figsize=(10, 10))
    for g in ordered_targets:
        mask = np.array([pg == g for pg in primary_genres])
        if mask.any():
            pts = latent_2d[mask]
            ax.scatter(pts[:, 0], pts[:, 1], c=[genre_colors[g]], s=30, alpha=0.7, label=g)

    ax.set_xlabel('t-SNE Component 1', fontsize=32)
    ax.set_ylabel('t-SNE Component 2', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(loc='upper right', fontsize=16, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_tsne_movies_top10.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ t-SNE (10 genres) saved to {os.path.join(output_dir, 'latent_tsne_movies_top10.pdf')}")


@torch.no_grad()
def qualitative_latent_analysis_wd_movies_with_vocab(
    model,
    vocabs: Dict[str, Dict],
    output_dir: str = "figures",
    n_samples: int = 500,
    use_all_test: bool = False,
    device: str = None,
):
    """Replicate the original wd-movies qualitative analysis while relying on checkpoint vocabs.

    Generates the three legacy PDFs (latent_tsne_movies.pdf, latent_interpolation.pdf,
    interpolation_sequence.pdf) and then invokes ``qualitative_latent_analysis_wd_movies``
    to add the canonical top-10 genre visualization (latent_tsne_movies_top10.pdf).
    """

    if vocabs is None:
        raise ValueError("qualitative_latent_analysis_wd_movies_with_vocab requires checkpoint vocabs.")

    required_keys = ("e2i", "i2e", "r2i", "i2r")
    missing = [k for k in required_keys if k not in vocabs or vocabs[k] is None]
    if missing:
        raise KeyError(f"Checkpoint vocabs missing keys: {missing}")

    e2i = vocabs["e2i"]
    i2e = vocabs["i2e"]
    r2i = vocabs["r2i"]
    i2r = vocabs["i2r"]

    os.makedirs(output_dir, exist_ok=True)

    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    config = model_unwrapped.config
    seq_len = config["seq_len"]
    special_tokens = config["special_tokens"]
    entity_base_idx = config["ENT_BASE"]
    relation_base_idx = config["REL_BASE"]

    if device is None:
        device = next(model_unwrapped.parameters()).device

    _, _, test_list = load_graphs_with_checkpoint_vocab("wd-movies", e2i, r2i)

    if use_all_test:
        test_sample = test_list
    else:
        test_sample = test_list[:min(n_samples, len(test_list))]

    if len(test_sample) == 0:
        print("Warning: wd-movies test split is empty after vocabulary filtering.")
        return

    target_genres = [
        'Action film',
        'Comedy film',
        'Drama film',
        'Horror film',
        'Romance film',
        'Musical film',
        'Science fiction film',
        'Western film',
        'Bollywood',
        'Documentary film',
    ]

    target_lookup = {genre.lower(): genre for genre in target_genres}

    distinct_colors = [
        '#FF0000',
        '#FFD700',
        '#0000FF',
        '#000000',
        '#FF69B4',
        '#FF8C00',
        '#00FF00',
        '#8B4513',
        '#9370DB',
        '#00CED1',
    ]

    genre_colors = {genre: color for genre, color in zip(target_genres, distinct_colors)}
    genre_colors['other'] = '#C0C0C0'

    def extract_genres_from_graph(graph_triples: List[Tuple[str, str, str]]) -> List[str]:
        genres = []
        for _, r, t in graph_triples:
            relation = (r or "").lower()
            if 'has_genre' in relation or relation == 'genre':
                genre = (t or '').strip()
                if genre and genre not in genres:
                    genres.append(genre)
        return genres

    def get_primary_genre(genres: List[str]) -> str:
        for genre in genres:
            canonical = target_lookup.get(genre.lower())
            if canonical:
                return canonical
        return 'other'

    latent_vectors = []
    all_genres_list = []
    primary_genres = []
    kept_indices = []  


    for graph_idx, graph in enumerate(test_sample):
        if not graph:
            continue

        max_triples = config.get('max_edges', 100)
        graph_tensor = torch.zeros((1, max_triples, 3), dtype=torch.long, device=device)
        num_triples = min(len(graph), max_triples)

        for i in range(num_triples):
            triple = graph[i]
            if len(triple) == 3:
                graph_tensor[0, i, 0] = triple[0]
                graph_tensor[0, i, 1] = triple[1]
                graph_tensor[0, i, 2] = triple[2]

        pad_rid = config.get('pad_rid', 0)
        if num_triples < max_triples:
            graph_tensor[0, num_triples:, 1] = pad_rid

        try:
            _, mu, _ = model_unwrapped.enc(graph_tensor)
        except Exception as exc:
            print(f"Skipping graph {graph_idx} during encoding: {exc}")
            continue

        latent_vectors.append(mu[0].cpu().numpy())
        kept_indices.append(graph_idx)  


        graph_labels = ints_to_labels([graph], i2e, i2r)[0]
        raw_genres = extract_genres_from_graph(graph_labels)
        all_genres_list.append(raw_genres)
        primary_genres.append(get_primary_genre(raw_genres))

    if len(latent_vectors) == 0:
        print("Warning: No wd-movies graphs could be encoded with the checkpoint vocabulary.")
        return

    latent_vectors = np.vstack(latent_vectors)

    tsne = TSNE(
        n_components=2,
        perplexity=max(5, min(30, len(latent_vectors) - 1)),
        random_state=42,
        max_iter=1000
    )

    latent_2d = tsne.fit_transform(latent_vectors)


    fig1, ax1 = plt.subplots(figsize=(10, 10))
    for genre in target_genres:
        mask = [pg == genre for pg in primary_genres]
        if any(mask):
            points = latent_2d[mask]
            ax1.scatter(
                points[:, 0],
                points[:, 1],
                c=genre_colors[genre],
                label=genre.title(),
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5,
            )

    other_mask = [pg == 'other' for pg in primary_genres]
    if any(other_mask):
        points = latent_2d[other_mask]
        ax1.scatter(
            points[:, 0],
            points[:, 1],
            c=genre_colors['other'],
            label='Other genres',
            alpha=0.3,
            s=20,
        )

    ax1.set_xlabel('t-SNE Component 1', fontsize=32)
    ax1.set_ylabel('t-SNE Component 2', fontsize=32)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    legend = ax1.legend(
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=16,
        framealpha=0.95,
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    ax1.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_tsne_movies.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    genre_pairs = []
    for i, genres_i in enumerate(all_genres_list):
        for j, genres_j in enumerate(all_genres_list[i + 1:], i + 1):
            target_i = [g for g in genres_i if g in target_genres]
            target_j = [g for g in genres_j if g in target_genres]
            if target_i and target_j and set(target_i).isdisjoint(set(target_j)):
                genre_pairs.append((i, j, target_i[0], target_j[0]))
                if len(genre_pairs) >= 5:
                    break
        if len(genre_pairs) >= 5:
            break

    if genre_pairs:
        idx1, idx2, genre1, genre2 = genre_pairs[0]
        print(f"Interpolating between {genre1.title()} and {genre2.title()}")

        z1 = torch.tensor(latent_vectors[idx1], device=device, dtype=torch.float32)
        z2 = torch.tensor(latent_vectors[idx2], device=device, dtype=torch.float32)

        n_interp_points = 20
        alphas = np.linspace(0, 1, n_interp_points)
        interp_points = np.vstack([
            ((1 - alpha) * z1 + alpha * z2).cpu().numpy() for alpha in alphas
        ])

        interp_2d = tsne.fit_transform(np.vstack([latent_vectors, interp_points]))
        path_2d = interp_2d[len(latent_vectors):]

        fig2, ax2 = plt.subplots(figsize=(10, 10))
        ax2.scatter(
            interp_2d[:len(latent_vectors), 0],
            interp_2d[:len(latent_vectors), 1],
            c='lightgray',
            alpha=0.3,
            s=10,
        )
        ax2.plot(path_2d[:, 0], path_2d[:, 1], 'b-', linewidth=2, alpha=0.7, label='Interpolation path')
        ax2.scatter(path_2d[0, 0], path_2d[0, 1], c='red', s=150, marker='s', label=f'Start: {genre1.title()}', zorder=5, edgecolor='black')
        ax2.scatter(path_2d[-1, 0], path_2d[-1, 1], c='blue', s=150, marker='s', label=f'End: {genre2.title()}', zorder=5, edgecolor='black')
        for marker_idx in [5, 10, 15]:
            if marker_idx < len(path_2d):
                ax2.scatter(path_2d[marker_idx, 0], path_2d[marker_idx, 1], c='orange', s=80, marker='o', zorder=4)
        ax2.set_xlabel('t-SNE Component 1', fontsize=32)
        ax2.set_ylabel('t-SNE Component 2', fontsize=32)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_aspect('equal', adjustable='box')
        ax2.legend(fontsize=16, loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.95)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latent_interpolation.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

        fig3 = plt.figure(figsize=(18, 4))
        gs = GridSpec(1, 5, figure=fig3, wspace=0.3)
        alpha_points = [0.0, 0.25, 0.5, 0.75, 1.0]


        beam_width = config.get('beam_width', 3)
        for idx, alpha in enumerate(alpha_points):
            ax = fig3.add_subplot(gs[0, idx])

            z_alpha = (1 - alpha) * z1 + alpha * z2
            decoded_graph = decode_to_triple_set(
                model_unwrapped, z_alpha, seq_len, special_tokens,
                entity_base_idx, relation_base_idx, beam=beam_width
            )
            decoded_labels = ints_to_labels([list(decoded_graph)], i2e, i2r)[0]
            decoded_genres = extract_genres_from_graph(decoded_labels)

            G = nx.DiGraph()
            for h, r, t in decoded_labels[:6]:
                h_short = h[:15] + '...' if len(h) > 15 else h
                t_short = t[:15] + '...' if len(t) > 15 else t
                r_short = r[:10]
                G.add_edge(h_short, t_short, label=r_short)

            pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=10, ax=ax, width=1.5)
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
            ax.axis('off')

            genres_text = ', '.join(decoded_genres[:3]) if decoded_genres else 'unknown'
            if len(decoded_genres) > 3:
                genres_text += '...'
            ax.text(0.5, -0.15, f'Genres: {genres_text}', transform=ax.transAxes, ha='center', fontsize=8, style='italic')


        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'interpolation_sequence.pdf'), dpi=300, bbox_inches='tight')
        plt.close()


    else:
        print("Skipping interpolation path visualization: no suitable genre pair found.")

    qualitative_latent_analysis_wd_movies(
        model=model,
        vocabs=vocabs,
        output_dir=output_dir,
        n_samples=n_samples,
        use_all_test=use_all_test,
        device=device,
    )

    print(f"\n✅ Qualitative analysis complete. Figures saved to {output_dir}/")
    print("  - latent_tsne_movies.pdf")
    print("  - latent_interpolation.pdf")
    print("  - interpolation_sequence.pdf")
    print("  - latent_tsne_movies_top10.pdf")


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
    model_type = config.get('model_type', 'SAIL')
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    beam = config.get("beam_width", 3)


    model, config, ckpt_path, vocabs, _ = load_model(
        checkpoint_dir=args.checkpoint_dir,
        dataset=dataset,
        model_type=model_type,
        epoch=args.epoch,
        device=device,
    )

    if vocabs is None:
        raise KeyError("Checkpoint missing 'vocabs'; retrain and save with vocabulary mappings.")

    required_vocab_keys = ("i2e", "i2r")
    missing = [k for k in required_vocab_keys if vocabs.get(k) is None]
    if missing:
        raise KeyError(f"Checkpoint vocabulary missing keys: {missing}")

    i2e = vocabs["i2e"]
    i2r = vocabs["i2r"]

    wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    config=config,
    name=f"latent_interp_{config['dataset']}_{config.get('model_type','SAIL')}"
)

    kind = f"epoch {args.epoch}" if args.epoch is not None else "best"
    print(f"✅ Loaded {model_type} for {dataset} ({kind}) from {ckpt_path} on {device}")
    if dataset == "wd-movies":
        qualitative_latent_analysis_wd_movies_with_vocab(
            model,
            vocabs=vocabs,
            output_dir="figures",
            n_samples=10000,
            use_all_test=True
        )

    if model_type == "SAIL" or model_type == "t-SAIL":
        assert i2e is not None and i2r is not None, "Checkpoint missing vocabs; retrain.py must save them."
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
                beam=beam
            )
            latent_smoothness_score_autoreg(
                model,
                steps=10,
                epsilon=e,
                n_anchors=3,
                n_dirs=3,
                beam=beam,
                device=device,
            )
            latent_flip_rate_autoreg(
            model,
            steps=30,
            epsilon=e,
            n_anchors=5,
            n_dirs=4,
            beam=beam,
            device=device,
            )

    wandb.finish()

if __name__ == "__main__":
    main()

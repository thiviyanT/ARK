"""
Latent space interpolation experiments for Knowledge Graph VAE models.

We load trained VAE models and perform interpolation between learned latent
representations to analyze whether the model has learned smooth transitions
between different graph structures in the latent space.
"""

import argparse
import torch
import torch.nn as nn
import wandb
import yaml
import os
from kgvae.model.rescal_vae_model import RESCALVAE
from kgvae.model.models import AutoRegModel
from kgvae.model.utils import seq_to_triples, ints_to_labels
from intelligraphs.data_loaders import load_data_as_list



def load_model(checkpoint_dir, dataset, model_type, epoch=None, device=None):
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
    m = model.module if isinstance(model, nn.DataParallel) else model
    config = m.config
    seq_len        = config["seq_len"]
    special_tokens = config["special_tokens"]
    ENT_BASE       = config["ENT_BASE"]
    REL_BASE       = config["REL_BASE"]
    latent_dim     = config["d_latent"]

    if device is None:
        device = next(m.parameters()).device
    z0 = torch.randn(latent_dim, device=device)
    directions = torch.randn(n_directions, latent_dim, device=device)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    perturbed_zs = z0.unsqueeze(0) + epsilon * directions 
    ref_graphs = m.decode_latent(
        z0.unsqueeze(0), seq_len, special_tokens, seq_to_triples, ENT_BASE, REL_BASE, beam=1
    )
    ref_triples = ints_to_labels(ref_graphs, i2e, i2r)[0]
    decoded_graphs = m.decode_latent(
        perturbed_zs, seq_len, special_tokens, seq_to_triples, ENT_BASE, REL_BASE, beam=1
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
def smoothness_line_check_autoreg(
    model,
    i2e,
    i2r,
    steps: int = 10,
    epsilon: float = 0.1,
    device: str = None,
    beam: int = 1,
):
    """
    Walk in latent space from z0 along a single random unit direction with small step size `epsilon`.
    At each step, decode and measure:
      - overlap with previous step's triples  (local smoothness)
      - overlap with the anchor z0 triples    (global drift)
    """
    m = model.module if isinstance(model, nn.DataParallel) else model
    cfg = m.config
    seq_len        = cfg["seq_len"]
    special_tokens = cfg["special_tokens"]
    ENT_BASE       = cfg["ENT_BASE"]
    REL_BASE       = cfg["REL_BASE"]
    d_latent       = cfg["d_latent"]

    if device is None:
        device = next(m.parameters()).device

    # Anchor and direction
    z0  = torch.randn(d_latent, device=device)
    dir = torch.randn(d_latent, device=device)
    dir = dir / dir.norm().clamp_min(1e-12)

    # Decode anchor
    anchor_graph_int = m.decode_latent(
        z0.unsqueeze(0), seq_len, special_tokens, seq_to_triples, ENT_BASE, REL_BASE, beam=beam
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
        z = z0 + (s * epsilon) * dir
        dec_int = m.decode_latent(
            z.unsqueeze(0), seq_len, special_tokens, seq_to_triples, ENT_BASE, REL_BASE, beam=beam
        )
        graph = ints_to_labels(dec_int, i2e, i2r)[0]

        # Overlaps
        denom_prev = max(1, len(prev_graph))
        local_overlap  = len(set(prev_graph) & set(graph)) / denom_prev
        global_overlap = len(set(anchor_graph) & set(graph)) / denom_anchor

        total_local  += local_overlap
        total_global += global_overlap

        print(f"\n--- Step {s}: z = z₀ + {s}·ε·dir ---")
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
def latent_smoothness_score_autoreg(
    model,
    steps:int=10,
    epsilon:float=0.1,
    n_anchors:int=3,
    n_dirs:int=3,
    beam:int=1,
    device:str=None,
):
    """
    Returns (avg_local_jaccard, avg_global_jaccard).
    Local = J(acc(step s), acc(step s-1))
    Global = J(acc(step s), acc(anchor))
    where acc(x) is the set of decoded triples at x.
    """
    m = model.module if isinstance(model, nn.DataParallel) else model
    cfg = m.config
    seq_len        = cfg["seq_len"]
    special_tokens = cfg["special_tokens"]
    ENT_BASE       = cfg["ENT_BASE"]
    REL_BASE       = cfg["REL_BASE"]
    d_latent       = cfg["d_latent"]
    if device is None:
        device = next(m.parameters()).device

    def decode_to_set(z):
        # decode_latent returns triples as integer ids (h,r,t); we can compare in int space
        g_int = m.decode_latent(
            z.unsqueeze(0), seq_len, special_tokens, seq_to_triples, ENT_BASE, REL_BASE, beam=beam
        )[0]
        return set(tuple(map(int, t)) for t in g_int)  # {(h,r,t), ...}

    def jaccard(a:set, b:set):
        if not a and not b: 
            return 1.0
        if not a or not b: 
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / max(1, union)

    total_local = 0.0
    total_global = 0.0
    count_local = 0
    count_global = 0

    for _ in range(n_anchors):
        z0  = torch.randn(d_latent, device=device)
        anchor = decode_to_set(z0)
        for _ in range(n_dirs):
            d = torch.randn(d_latent, device=device)
            d = d / d.norm().clamp_min(1e-12)

            prev = anchor
            # march: z_s = z0 + s*epsilon*d
            for s in range(1, steps+1):
                z = z0 + (s * epsilon) * d
                cur = decode_to_set(z)
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
def latent_flip_rate_autoreg(
    model,
    steps:int=30,
    epsilon:float=0.05,
    n_anchors:int=5,
    n_dirs:int=4,
    beam:int=1,
    device:str=None,
):
    """
    Measures how often the decoded graph changes as you take ε-steps in latent space.
    Returns:
      flip_rate: fraction of step transitions that change the decoded set of triples
      avg_basin: average contiguous run length with identical decoded graph
    """
    m = model.module if isinstance(model, nn.DataParallel) else model
    cfg = m.config
    seq_len        = cfg["seq_len"]
    special_tokens = cfg["special_tokens"]
    ENT_BASE       = cfg["ENT_BASE"]
    REL_BASE       = cfg["REL_BASE"]
    d_latent       = cfg["d_latent"]
    if device is None:
        device = next(m.parameters()).device

    def decode_set(z):
        g = m.decode_latent(
            z.unsqueeze(0), seq_len, special_tokens, seq_to_triples, ENT_BASE, REL_BASE, beam=beam
        )[0]
        return set(tuple(map(int, t)) for t in g)

    total_flips = 0
    total_steps = 0
    all_basin_lengths = []

    for _ in range(n_anchors):
        z0  = torch.randn(d_latent, device=device)
        for _ in range(n_dirs):
            d = torch.randn(d_latent, device=device)
            d = d / d.norm().clamp_min(1e-12)

            prev_set = decode_set(z0)
            basin_len = 1
            last_was_flip = False
            for s in range(1, steps+1):
                z = z0 + (s * epsilon) * d
                cur_set = decode_set(z)
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



def main():

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

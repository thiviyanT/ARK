#!/usr/bin/env python3
import argparse
import torch
import yaml
import wandb
from kgvae.model.models import AutoRegModel
from kgvae.model.utils import seq_to_triples, ints_to_labels
from kgvae.model.verification import get_verifier, run_semantic_evaluation
from intelligraphs.data_loaders import load_data_as_list


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    vocabs = ckpt.get("vocabs", None)
    if vocabs is None:
        raise RuntimeError("Checkpoint missing 'vocabs' (expected keys: e2i, i2e, r2i, i2r).")
    return config, state, vocabs

def build_model(config, state, device):
    ''' Build model from config and load state dict. '''
    
    needed = ["vocab_size","seq_len","special_tokens","ENT_BASE","REL_BASE",
              "n_entities","n_relations","d_latent"]
    missing = [k for k in needed if k not in config]
    if missing:
        raise RuntimeError(f"Checkpoint config missing keys: {missing}")
    model = AutoRegModel(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def get_logits_fn(model):
    def fn(z, prefix_ids):
        out = model.dec(z, prefix_ids)   
        return out[:, -1, :]             
    return fn

def force_token(logits, forced_id):
    """Replace distribution by a delta on forced_id."""
    out = torch.full_like(logits, float('-inf'))
    out[..., forced_id] = 0.0
    return out


def ids_for_labels(relation_label, object_label, r2i, e2i, REL_BASE, ENT_BASE):
    if relation_label not in r2i:
        raise SystemExit(f"Unknown relation label: {relation_label}")
    if object_label  not in e2i:
        raise SystemExit(f"Unknown entity label: {object_label}")
    rid = r2i[relation_label] + REL_BASE
    oid = e2i[object_label]  + ENT_BASE
    return rid, oid

@torch.no_grad()
def conditional_decode(
    model,
    num_samples,
    seq_len,
    special_tokens,
    d_latent,
    force_rid,       
    force_oid,     
    device="cuda",
):
    """
    conditional greedy generation:
      - sample z ~ N(0, I)
      - decode from BOS
      - at t==2 force r1 == force_rid
      - at t==3 force o1 == force_oid
      - otherwise greedy argmax until EOS or seq_len
    """
    BOS = special_tokens["BOS"]
    EOS = special_tokens["EOS"]

    z = torch.randn(num_samples, d_latent, device=device)

    logits_fn = get_logits_fn(model)
    seq = torch.full((num_samples, 1), BOS, dtype=torch.long, device=device)

    while seq.size(1) < seq_len:
        t = seq.size(1) 
        logits_next = logits_fn(z, seq) 

        if t == 2:
            logits_next = force_token(logits_next, force_rid)  
        elif t == 3:
            logits_next = force_token(logits_next, force_oid)  

        next_ids = torch.argmax(logits_next, dim=-1, keepdim=True)
        seq = torch.cat([seq, next_ids], dim=1)

        if (seq[:, -1] == EOS).all():
            break

    return seq


@torch.no_grad()
def unconditional_decode(
    model,
    num_samples,
    seq_len,
    special_tokens,
    d_latent,
    device="cuda",
    beam=1,  
):
    """
    Unconditional greedy generation:
      1) sample z ~ N(0, I)
      2) decode from BOS until EOS or seq_len
    Returns:
      LongTensor of token ids with shape (num_samples, T<=seq_len)
    """
    BOS = special_tokens["BOS"]
    EOS = special_tokens["EOS"]

    z = torch.randn(num_samples, d_latent, device=device)
    logits_fn = get_logits_fn(model)  
    seq = torch.full((num_samples, 1), BOS, dtype=torch.long, device=device)

    while seq.size(1) < seq_len:
        logits_next = logits_fn(z, seq)                
        next_ids = torch.argmax(logits_next, dim=-1)  
        seq = torch.cat([seq, next_ids.unsqueeze(-1)], dim=1)
        if (seq[:, -1] == EOS).all():
            break

    return seq


def main():
    parser = argparse.ArgumentParser("Constrained first (relation, object) decoding")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--relation", required=True, help="e.g., has_actor")
    parser.add_argument("--object",   required=True, help="e.g., Al Pacino")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument('--wandb-project', type=str, default='submission', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default='a-vozikis-vrije-universiteit-amsterdam', help='W&B entity')
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", required=True, help="IntelliGraphs dataset name, e.g., wd-movies")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config, state, vocabs = load_model(args.checkpoint, device)
    model = build_model(config, state, device)

    i2e, i2r = vocabs["i2e"], vocabs["i2r"]
    e2i, r2i = vocabs["e2i"], vocabs["r2i"]
    special_tokens = config["special_tokens"]
    ENT_BASE = config["ENT_BASE"]
    REL_BASE = config["REL_BASE"]
    seq_len = config["seq_len"]
    d_latent = config["d_latent"]
    verifier = get_verifier(args.dataset)
    train_g, _, _, i2e_data, i2r_data, e2i_data, r2i_data = load_data_as_list(args.dataset)
    wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    config=config,
    name=f"latent_interp_{config['dataset']}_{config.get('model_type','autoreg')}"
)


    # ---------- Constrained generation ----------
    force_rid, force_oid = ids_for_labels(args.relation, args.object, r2i, e2i, REL_BASE, ENT_BASE)

    z = torch.randn(args.num_samples, d_latent, device=device)
    
    print("----------------------------------------------")
    force_rid, force_oid = ids_for_labels(args.relation, args.object, r2i, e2i, REL_BASE, ENT_BASE)

    seq_constrained = conditional_decode(
        model=model,
        num_samples=args.num_samples,
        seq_len=seq_len,
        special_tokens=special_tokens,
        d_latent=d_latent,
        force_rid=force_rid,
        force_oid=force_oid,
        device=device,
    )

    print("\n=== Constrained samples ===")
    triples_per_graph = []
    for s in seq_constrained.tolist():
        g = seq_to_triples(s, special_tokens, ENT_BASE, REL_BASE)
        labeled = ints_to_labels([g], i2e, i2r)[0]
        triples_per_graph.append(labeled)

    for i, triples in enumerate(triples_per_graph, 1):
        print(f"\n[{i}] Triples (labels):")
        for t in triples:
            print("  ", t)
        print("-" * 60)
        
    print("\n=== Conditioned semantic evaluation ===")
    run_semantic_evaluation(
        triples_per_graph,  
        train_g,             
        i2e, i2r,            
        verifier,           
        title="graphs conditioned on (relation, object)"
    )

    # ---------- Unconditional generation (added here) ----------
    seq_uncond = unconditional_decode(
        model=model,
        num_samples=args.num_samples,
        seq_len=seq_len,
        special_tokens=special_tokens,
        d_latent=d_latent,
        device=device,
        beam=4
    )

    print("\n=== Unconditional samples ===")
    triples_per_graph_uncond = []
    for s in seq_uncond.tolist():
        g = seq_to_triples(s, special_tokens, ENT_BASE, REL_BASE)
        labeled = ints_to_labels([g], i2e, i2r)[0]
        triples_per_graph_uncond.append(labeled)

    for i, triples in enumerate(triples_per_graph_uncond, 1):
        print(f"\n[{i}] Triples (labels):")
        for t in triples:
            print("  ", t)
        print("-" * 60)
        
    print("\n=== Unconditioned semantic evaluation ===")
    run_semantic_evaluation(
        triples_per_graph_uncond,
        train_g,
        i2e, i2r,
        verifier,
        title="graphs from random latent (unconditional)"
    )

    wandb.finish()

if __name__ == "__main__":
    main()
"""
Graph reconstruction experiments for Knowledge Graph VAE models.

We will evaluate the model's ability to accurately reconstruct input graphs
from their latent representations, measuring fidelity and information preservation.
"""


import argparse
from email import parser
import os
import yaml
import torch
import wandb
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader as PDataLoader

from intelligraphs.data_loaders import load_data_as_list
from kgvae.model.models import AutoRegModel
from kgvae.model.utils import GraphSeqDataset, seq_to_triples 

from kgvae.model.utils import ints_to_labels
from kgvae.model.verification import get_verifier, run_semantic_evaluation



def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def graph_tensor_row_to_set(row_triples, ent_base, rel_base, pad_eid, pad_rid):
    """
    row_triples: (max_triples, 3) integer tensor in TOKEN SPACE.
    Filters out padded triples using tokenized pad IDs:
      subject/object pad token = ent_base + pad_eid   (if pad_eid is not None)
      relation pad token       = rel_base + pad_rid   (if pad_rid is not None)
    Returns a set of (h,r,t) tuples (ints).
    """
    pad_subj_tok = (ent_base + pad_eid) if pad_eid is not None else None
    pad_obj_tok  = pad_subj_tok
    pad_rel_tok  = (rel_base + pad_rid) if pad_rid is not None else None

    triples = []
    for (h, r, t) in row_triples.tolist():
        # Skip padding by relation token
        if pad_rel_tok is not None and r == pad_rel_tok:
            continue
        # Skip padding by entity tokens (subject/object)
        if pad_subj_tok is not None and (h == pad_subj_tok or t == pad_obj_tok):
            continue
        # Also ignore all-zero rows (defensive)
        if h == 0 and r == 0 and t == 0:
            continue
        triples.append((int(h), int(r), int(t)))
    return set(triples)



def decode_batch_to_sets(model_unwrapped, z, seq_len, special_tokens, ent_base, rel_base, beam=1) -> list[set]:
    """
    Decode to VOCAB space via decode_latent, then convert to TOKEN space
    so we can fairly compare against loader outputs (which are in TOKEN space).
    """
    decoded_vocab = model_unwrapped.decode_latent(
        z, seq_len, special_tokens, seq_to_triples, ent_base, rel_base, beam=beam
    ) 

    out = []
    for g in decoded_vocab:
        tok_set = set((int(h) + ent_base, int(r) + rel_base, int(t) + ent_base) for (h, r, t) in g)
        out.append(tok_set)
    return out



def build_autoreg_dataset_loaders(dataset: str, cfg: dict, i2e: dict, i2r: dict, batch_size: int):
    """
    Build GraphSeqDataset/DataLoaders using the SAME vocab & AR layout
    stored in the checkpoint config and vocabs.
    """
    (train_g, val_g, test_g, _, _, (min_edges, max_edges), _) = load_data_as_list(dataset)

    special_tokens = cfg["special_tokens"]
    ENT_BASE       = cfg["ENT_BASE"]
    REL_BASE       = cfg["REL_BASE"]
    seq_len        = cfg["seq_len"]

    PAD_EID = cfg.get("pad_eid", None)
    PAD_RID = cfg.get("pad_rid", None)

    def make_loader(graphs, shuffle=False):
        ds = GraphSeqDataset(
            graphs=graphs,
            i2e=i2e,
            i2r=i2r,
            triple_order=cfg.get("triple_order", "keep"),
            permute=False,
            use_padding=(PAD_EID is not None or PAD_RID is not None),
            pad_eid=PAD_EID,
            pad_rid=PAD_RID,
            max_triples=max_edges,
            special_tokens=special_tokens,
            ent_base=ENT_BASE,
            rel_base=REL_BASE,
            seq_len=seq_len,
        )
        return PDataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_g, shuffle=False)
    val_loader   = make_loader(val_g,   shuffle=False)
    test_loader  = make_loader(test_g,  shuffle=False)

    ds_cfg = {
        "PAD_EID": PAD_EID,  
        "PAD_RID": PAD_RID,
        "special_tokens": special_tokens,
        "ENT_BASE": ENT_BASE,
        "REL_BASE": REL_BASE,
        "seq_len": seq_len,
        "max_edges": max_edges,
        "i2e": i2e,
        "i2r": i2r,
    }
    return train_loader, val_loader, test_loader, ds_cfg



def load_model(checkpoint_dir, dataset, model_type, epoch=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = (
        os.path.join(checkpoint_dir, f"{dataset}_{model_type}_best_model.pt")
        if epoch is None else
        os.path.join(checkpoint_dir, f"{dataset}_{model_type}_checkpoint_epoch_{epoch}.pt")
    )
    print(f"\n>>> Loading checkpoint: {ckpt_path}")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model = AutoRegModel(config).to(device)
    model.load_state_dict(state)
    model.eval()

    vocabs = ckpt.get("vocabs", None)
    
    print("\n=== Checkpoint contents ===")
    print("Keys in checkpoint:", list(ckpt.keys()))
    print("Model state_dict keys:", list(state.keys())[:10], "...")

    if vocabs is None:
        raise KeyError("Checkpoint missing 'vocabs' (expected keys: 'i2e','i2r',...). Re-save training with vocabs.")
    return model, config, ckpt_path, vocabs




@torch.no_grad()
def reconstruct(model, loader, ds_cfg, max_batches=None, beam=1, device=None):
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    device = device or next(model_unwrapped.parameters()).device

    PAD_EID = ds_cfg["PAD_EID"]
    PAD_RID = ds_cfg["PAD_RID"]
    seq_len = ds_cfg["seq_len"]
    special_tokens = ds_cfg["special_tokens"]
    ENT_BASE = ds_cfg["ENT_BASE"]
    REL_BASE = ds_cfg["REL_BASE"]

    jac_scores, exact_hits = [], []
    n_graphs = 0

    for b_idx, batch in enumerate(loader):
        triples = batch[0] if isinstance(batch, (list, tuple)) else batch
        triples = triples.to(device)

        enc_out = model_unwrapped.enc(triples)
        if isinstance(enc_out, (list, tuple)):
            if len(enc_out) == 3:
                z, mu, logv = enc_out
            elif len(enc_out) == 2:
                mu, logv = enc_out
                z = mu
            else:
                z = enc_out[0]
        else:
            z = enc_out
        if 'mu' in locals():
            z = mu

        dec_sets = decode_batch_to_sets(
            model_unwrapped, z, seq_len, special_tokens, ENT_BASE, REL_BASE, beam=beam
        )

        B = triples.size(0)
        for i in range(B):
            orig_set = graph_tensor_row_to_set(triples[i], ENT_BASE, REL_BASE, PAD_EID, PAD_RID)
            rec_set  = dec_sets[i]
            j = jaccard(orig_set, rec_set)
            jac_scores.append(j)
            exact_hits.append(int(orig_set == rec_set))
            n_graphs += 1
            if n_graphs <= 5:
                print(f"\n--- Example {n_graphs} ---", flush=True)
                print("Input triples :", orig_set, flush=True)
                print("Reconstructed :", rec_set, flush=True)

        if (max_batches is not None) and (b_idx + 1 >= max_batches):
            break

    return {
        "avg_jaccard": float(np.mean(jac_scores)) if jac_scores else 0.0,
        "exact_match_rate": float(np.mean(exact_hits)) if exact_hits else 0.0,
        "num_graphs": n_graphs,
    }

@torch.no_grad()
def sample_from_latent(
    model,
    model_cfg,
    ds_cfg,
    i2e,
    i2r,
    device,
    dataset,         
    n_samples=5,    
    beam=1
):
    """
    Sample graphs with z ~ N(0, I), decode, map to labels, print them,
    and run semantic evaluation (validity + novelty vs train set).
    """
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    z_dim = getattr(model_unwrapped, "z_dim", None) \
            or model_cfg.get("d_latent", model_cfg.get("latent_dim"))
    if z_dim is None:
        raise ValueError("Cannot infer latent dim. Set config['d_latent'] or expose model.z_dim.")

    seq_len        = ds_cfg["seq_len"]
    special_tokens = ds_cfg["special_tokens"]
    ENT_BASE       = ds_cfg["ENT_BASE"]
    REL_BASE       = ds_cfg["REL_BASE"]

    # 1) sample z ~ N(0, I)
    z = torch.randn(n_samples, z_dim, device=device)

    # 2) decode to vocab-space integer triples
    decoded_vocab = model_unwrapped.decode_latent(
        z, seq_len, special_tokens, seq_to_triples, ENT_BASE, REL_BASE, beam=beam
    )  # List[List[(h_id, r_id, t_id)]]

    # 3) map to labels and print (predicted graphs)
    label_graphs = ints_to_labels(decoded_vocab, i2e, i2r)
    print(f"\n[Sampled {n_samples} graphs from latent N(0, I)]:")
    for i, g in enumerate(label_graphs, 1):
        print(f"\n--- Graph {i} ---")
        for (h, r, t) in g:
            print(f"({h}, {r}, {t})")

    # 4) semantic evaluation (validity + novelty vs train) — MATCHES train.py
    verifier = get_verifier(dataset)
    if verifier is None:
        print(f"[warn] No verifier for dataset '{dataset}'. Skipping semantic eval.")
        return label_graphs

    (train_g, _, _, _, _, _, _) = load_data_as_list(dataset)

    run_semantic_evaluation(
        label_graphs,     
        train_g,        
        i2e, i2r,
        verifier,
        title=f"graphs from random latent ({n_samples})"
    )

    return label_graphs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument('--wandb-project', type=str, default='anonymized_project')
    parser.add_argument('--wandb-entity', type=str, default='anonymous')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    assert config.get("model_type", "autoreg") == "autoreg"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_cfg, ckpt_path, vocabs = load_model(
        args.checkpoint_dir, config["dataset"], "autoreg", epoch=args.epoch, device=device
    )
    
    wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    config=config,
    name=f"latent_interp_{config['dataset']}_{config.get('model_type','autoreg')}"
)
    i2e, i2r = vocabs["i2e"], vocabs["i2r"]

    train_loader, val_loader, test_loader, ds_cfg = build_autoreg_dataset_loaders(
        config["dataset"], model_cfg, i2e, i2r, batch_size=args.batch_size
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    stats = reconstruct(model, loader, ds_cfg,
                        max_batches=args.max_batches, beam=args.beam, device=device)
    latent_label_graphs = sample_from_latent(
    model, model_cfg, ds_cfg, i2e, i2r, device, dataset=config["dataset"], n_samples=100, beam=1
)


    print("\n=== Reconstruction Results ===", flush=True)
    print(f"Graphs evaluated : {stats['num_graphs']}", flush=True)
    print(f"Avg Jaccard      : {stats['avg_jaccard']*100:.2f}%", flush=True)
    print(f"Exact match rate : {stats['exact_match_rate']*100:.2f}%\n", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
"""
Graph reconstruction experiments for Knowledge Graph VAE models.

We will evaluate the model's ability to accurately reconstruct input graphs
from their latent representations, measuring fidelity and information preservation.
"""


import argparse
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader as PDataLoader

from intelligraphs.data_loaders import load_data_as_list
from kgvae.model.models import AutoRegModel
from kgvae.model.utils import GraphSeqDataset, seq_to_triples 


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def graph_tensor_row_to_set(row_triples, pad_rid):
    """
    row_triples: (max_triples, 3) integer tensor from GraphSeqDataset batch.
    Returns a set of (h,r,t) tuples (ints).
    """
    triples = []
    for (h, r, t) in row_triples.tolist():
        if pad_rid is not None and r == pad_rid:
            continue
        if h == 0 and r == 0 and t == 0:
            continue
        triples.append((int(h), int(r), int(t)))
    return set(triples)


def decode_batch_to_sets(model_unwrapped, z, seq_len, special_tokens, ent_base, rel_base, beam=1) -> list[set]:
    """
    Calls beam_generate through model.decode_latent() and converts
    each decoded graph (list of triples) to a set of (h,r,t) tuples.
    """
    decoded = model_unwrapped.decode_latent(
        z, seq_len, special_tokens, seq_to_triples, ent_base, rel_base, beam=beam
    )
    out = []
    for g in decoded:
        out.append(set((int(h), int(r), int(t)) for (h, r, t) in g))
    return out


def build_autoreg_dataset_loaders(dataset: str, config: dict, batch_size: int):
    """
    Recreates the same GraphSeqDataset/DataLoader setup we use in train.py,
    so reconstruction is measured on the same canonicalization & padding.
    """
    (train_g, val_g, test_g, (e2i, i2e), (r2i, i2r), (min_edges, max_edges), _) = load_data_as_list(dataset)

    num_entities  = len(e2i)
    num_relations = len(r2i)
    use_padding   = config.get("use_padding", dataset.startswith("wd-"))

    if use_padding:
        PAD_EID = num_entities
        PAD_RID = num_relations
        num_entities  += 1
        num_relations += 1
    else:
        PAD_EID = None
        PAD_RID = None

    special_tokens = {"PAD": 0, "BOS": 1, "EOS": 2}
    ENT_BASE = 3
    REL_BASE = ENT_BASE + num_entities
    VOCAB_SIZE = REL_BASE + num_relations
    seq_len = 1 + max_edges * 3 + 1

    def make_loader(graphs, shuffle=False):
        ds = GraphSeqDataset(
            graphs=graphs,
            i2e=i2e,
            i2r=i2r,
            triple_order="keep",
            permute=False,
            use_padding=use_padding,
            pad_eid=PAD_EID,
            pad_rid=PAD_RID,
            max_triples=max_edges,
            special_tokens=special_tokens,
            ent_base=ENT_BASE,
            rel_base=REL_BASE,
            seq_len=seq_len,
        )
        return PDataLoader(ds, batch_size=batch_size, shuffle=shuffle), PAD_RID, special_tokens, ENT_BASE, REL_BASE, seq_len

    train_loader, *_ = make_loader(train_g, shuffle=False)
    val_loader,   PAD_RID, special_tokens, ENT_BASE, REL_BASE, seq_len = make_loader(val_g, shuffle=False)
    test_loader,  *_ = make_loader(test_g, shuffle=False)

    ds_cfg = {
        "PAD_RID": PAD_RID,
        "special_tokens": special_tokens,
        "ENT_BASE": ENT_BASE,
        "REL_BASE": REL_BASE,
        "seq_len": seq_len,
        "max_edges": max_edges,
        "num_entities": num_entities,
        "num_relations": num_relations,
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
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model = AutoRegModel(config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, config, ckpt_path


@torch.no_grad()
def reconstruct(model, loader, ds_cfg, max_batches=None, beam=1, device=None):
    """
    For each batch: z = enc(triples) -> decode(z) -> compare with original triples.
    Reports Jaccard per graph and exact match, plus dataset averages.
    """
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    device = device or next(model_unwrapped.parameters()).device

    PAD_RID = ds_cfg["PAD_RID"]
    seq_len = ds_cfg["seq_len"]
    special_tokens = ds_cfg["special_tokens"]
    ENT_BASE = ds_cfg["ENT_BASE"]
    REL_BASE = ds_cfg["REL_BASE"]

    jac_scores = []
    exact_hits = []
    n_graphs = 0

    for b_idx, (triples, _seq) in enumerate(loader):
        triples = triples.to(device)      
        z, mu, logv = model_unwrapped.enc(triples)
        z = mu   

        dec_sets = decode_batch_to_sets(
            model_unwrapped, z, seq_len, special_tokens, ENT_BASE, REL_BASE, beam=beam
        )

        B = triples.size(0)
        for i in range(B):
            orig_set = graph_tensor_row_to_set(triples[i], PAD_RID)
            rec_set  = dec_sets[i]

            j = jaccard(orig_set, rec_set)
            jac_scores.append(j)
            exact_hits.append(int(orig_set == rec_set))
            n_graphs += 1
            if n_graphs <= 10:
                print(f"\n--- Example {n_graphs} ---")
                print("Input triples :", orig_set)
                print("Reconstructed :", rec_set)

        if (max_batches is not None) and (b_idx + 1 >= max_batches):
            break

    avg_jaccard = float(np.mean(jac_scores)) if jac_scores else 0.0
    exact_rate  = float(np.mean(exact_hits)) if exact_hits else 0.0

    return {
        "avg_jaccard": avg_jaccard,
        "exact_match_rate": exact_rate,
        "num_graphs": n_graphs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to YAML used to train")
    ap.add_argument("--checkpoint-dir", default="checkpoints", type=str)
    ap.add_argument("--epoch", type=int, default=None, help="Load a specific epoch instead of best")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--beam", type=int, default=1, help="Beam width (1 = greedy)")
    ap.add_argument("--max-batches", type=int, default=None, help="Limit for quick tests")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    assert cfg_yaml.get("model_type", "autoreg") == "autoreg", "This simple reconstruction script currently supports model_type=='autoreg'."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_cfg, ckpt_path = load_model(
        args.checkpoint_dir, cfg_yaml["dataset"], "autoreg", epoch=args.epoch, device=device
    )
    print(f"Loaded autoreg for {cfg_yaml['dataset']} from {ckpt_path} on {device}")

    train_loader, val_loader, test_loader, ds_cfg = build_autoreg_dataset_loaders(
        cfg_yaml["dataset"], cfg_yaml, batch_size=args.batch_size
    )

    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    stats = reconstruct(
        model, loader, ds_cfg, max_batches=args.max_batches, beam=args.beam, device=device
    )

    print("\n=== Reconstruction Results ===")
    print(f"Graphs evaluated : {stats['num_graphs']}")
    print(f"Avg Jaccard      : {stats['avg_jaccard']*100:.2f}%")
    print(f"Exact match rate : {stats['exact_match_rate']*100:.2f}%\n")


if __name__ == "__main__":
    main()

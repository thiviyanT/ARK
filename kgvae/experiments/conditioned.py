import argparse
import copy
from pathlib import Path

import torch
import yaml

from kgvae.model.models import SAIL, ARK
from kgvae.model.utils import seq_to_triples, ints_to_labels


CONDITION_RELATION = "has_director"
CONDITION_OBJECT = "Tim Burton"


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    vocabs = ckpt.get("vocabs")
    if vocabs is None:
        raise KeyError(f"Checkpoint {path} is missing 'vocabs'.")
    return config, state, vocabs


def normalize_config(config, model_type_override=None):
    cfg = copy.deepcopy(config)
    raw_type = model_type_override or cfg.get("model_type", "ARK")
    raw_type = str(raw_type)
    lower = raw_type.lower()

    if lower in {"sail", "autoreg", "autoregressive"}:
        resolved = "SAIL"
    elif lower in {"t-sail", "tsail"}:
        resolved = "t-SAIL"
    elif lower in {"ark"}:
        resolved = "ARK"
    elif lower in {"t-ark", "tark"}:
        resolved = "t-ARK"
    elif lower == "dec_only":
        decoder = str(cfg.get("ablation_decoder", "Transformer")).lower()
        resolved = "ARK" if decoder == "gru" else "t-ARK"
    else:
        raise ValueError(f"Unsupported model_type '{raw_type}'.")

    cfg["model_type"] = resolved
    return cfg, resolved


def resolve_model_variant(config, raw_type=None):
    """Return canonical model string without mutating the original config."""
    _, resolved = normalize_config(config, raw_type)
    return resolved


def build_model(config, state, device, model_type_override=None):
    cfg, resolved = normalize_config(config, model_type_override)
    if resolved in {"SAIL", "t-SAIL"}:
        model = SAIL(cfg).to(device)
        model_kind = "autoreg"
    elif resolved in {"ARK", "t-ARK"}:
        model = ARK(cfg).to(device)
        model_kind = "decoder_only"
    else:
        raise ValueError(f"Unhandled resolved model type '{resolved}'.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, model_kind


def force_token(logits, token_id):
    forced = torch.full_like(logits, float('-inf'))
    forced[..., token_id] = 0.0
    return forced


def sample_from_logits(logits, temperature=1.0, top_p=0.0, top_k=0):
    if temperature and temperature != 1.0:
        logits = logits / float(temperature)

    probs = torch.softmax(logits, dim=-1)

    vocab_size = probs.size(-1)
    if top_k and top_k > 0 and top_k < vocab_size:
        top_values, top_indices = probs.topk(top_k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(-1, top_indices, 1.0)
        probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    if top_p and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled_idx = torch.multinomial(sorted_probs, 1)
        next_tokens = torch.gather(sorted_idx, -1, sampled_idx)
    else:
        next_tokens = torch.multinomial(probs, 1)

    return next_tokens


@torch.no_grad()
def conditional_generate(model, model_kind, cfg, forced_relation_id, forced_object_id,
                         num_samples, device):
    special_tokens = cfg["special_tokens"]
    seq_len = cfg["seq_len"]
    bos = special_tokens["BOS"]
    eos = special_tokens["EOS"]

    seq = torch.full((num_samples, 1), bos, dtype=torch.long, device=device)

    if model_kind == "autoreg":
        latent_dim = cfg["d_latent"]
        z = torch.randn(num_samples, latent_dim, device=device)
        def next_logits(prefix):
            return model.dec(z, prefix)[:, -1, :]
    else:
        def next_logits(prefix):
            return model(prefix)[:, -1, :]

    decoder_sample = False
    temperature = cfg.get("temperature", 1.0)
    top_p = cfg.get("top_p", 0.0)
    top_k = cfg.get("top_k", 0)
    if model_kind == "decoder_only":
        decoder_sample = bool(
            cfg.get("sample", True)
            or (top_p and top_p > 0.0)
            or (top_k and top_k > 0)
            or (temperature and temperature != 1.0)
        )

    while seq.size(1) < seq_len:
        logits = next_logits(seq)
        step = seq.size(1)
        if step == 2:
            logits = force_token(logits, forced_relation_id)
        elif step == 3:
            logits = force_token(logits, forced_object_id)
        if decoder_sample:
            next_token = sample_from_logits(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)
        if (seq[:, -1] == eos).all():
            break

    return seq.cpu()


def ids_for_condition(vocabs, cfg, relation_label, object_label):
    e2i = vocabs.get("e2i")
    r2i = vocabs.get("r2i")
    if e2i is None or r2i is None:
        raise KeyError("Checkpoint vocabs require 'e2i' and 'r2i'.")
    try:
        rid = r2i[relation_label] + cfg["REL_BASE"]
    except KeyError as err:
        raise KeyError(f"Relation '{relation_label}' not found in checkpoint vocab.") from err
    try:
        oid = e2i[object_label] + cfg["ENT_BASE"]
    except KeyError as err:
        raise KeyError(f"Entity '{object_label}' not found in checkpoint vocab.") from err
    return rid, oid


def to_labeled_triples(seqs, cfg, vocabs):
    special_tokens = cfg["special_tokens"]
    ent_base = cfg["ENT_BASE"]
    rel_base = cfg["REL_BASE"]
    graphs = [seq_to_triples(seq, special_tokens, ent_base, rel_base) for seq in seqs]
    i2e = vocabs.get("i2e")
    i2r = vocabs.get("i2r")
    if i2e is None or i2r is None:
        raise KeyError("Checkpoint vocabs require 'i2e' and 'i2r' for decoding.")
    return ints_to_labels(graphs, i2e, i2r)


def discover_checkpoints(explicit, checkpoint_dir):
    if explicit:
        return [Path(p) for p in explicit]
    directory = Path(checkpoint_dir)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pt"))


def main():
    parser = argparse.ArgumentParser("Conditioned decoding for WD Movies")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file.')
    parser.add_argument('--checkpoints', nargs='+', default=None, help='One or more checkpoint files to load.')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Fallback directory to scan for checkpoints.')
    parser.add_argument('--num-samples', type=int, default=4, help='Number of graphs to generate per checkpoint.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--relation', type=str, default=CONDITION_RELATION, help='Relation label to force in the first triple.')
    parser.add_argument('--tail', type=str, default=CONDITION_OBJECT, help='Tail entity label to force in the first triple.')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name used to filter checkpoints (overrides config).')
    parser.add_argument('--model-type', type=str, default=None, choices=['SAIL', 't-SAIL', 'ARK', 't-ARK'], help='Override model type if checkpoint config is ambiguous.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    with open(args.config, 'r') as f:
        cfg_file = yaml.safe_load(f)
    expected_dataset = cfg_file.get('dataset')
    config_model_type = cfg_file.get('model_type')

    dataset_filter = args.dataset or expected_dataset
    model_type_override = args.model_type or config_model_type

    checkpoints = discover_checkpoints(args.checkpoints, args.checkpoint_dir)
    if not checkpoints:
        raise SystemExit("No checkpoints found. Provide --checkpoints or populate the checkpoint directory.")

    for ckpt_path in checkpoints:
        config, state, vocabs = load_checkpoint(ckpt_path, device)
        dataset = config.get("dataset")
        if dataset_filter and dataset != dataset_filter:
            print(f"Skipping {ckpt_path} (dataset={dataset}).")
            continue

        try:
            resolved_checkpoint_type = resolve_model_variant(config)
        except ValueError as err:
            print(f"Skipping {ckpt_path}: {err}")
            continue

        override_choice = model_type_override
        if override_choice is not None:
            try:
                resolved_override = resolve_model_variant(config, override_choice)
            except ValueError as err:
                print(f"Warning: override '{override_choice}' invalid for {ckpt_path} ({err}); using checkpoint model type instead.")
                override_choice = None
            else:
                if resolved_override != resolved_checkpoint_type:
                    print(f"Warning: override '{override_choice}' mapped to {resolved_override} but checkpoint is {resolved_checkpoint_type}; using checkpoint model type.")
                    override_choice = None

        try:
            model, cfg, model_kind = build_model(config, state, device, override_choice)
        except ValueError as err:
            print(f"Skipping {ckpt_path}: {err}")
            continue

        required_special = {"PAD", "BOS", "EOS"}
        special_tokens = cfg.get("special_tokens", {})
        if not required_special.issubset(special_tokens):
            missing = required_special.difference(special_tokens)
            print(f"Skipping {ckpt_path}: missing special tokens {sorted(missing)}")
            continue

        try:
            forced_relation_id, forced_object_id = ids_for_condition(vocabs, cfg, args.relation, args.tail)
        except KeyError as err:
            print(f"Skipping {ckpt_path}: {err}")
            continue

        seqs = conditional_generate(
            model=model,
            model_kind=model_kind,
            cfg=cfg,
            forced_relation_id=forced_relation_id,
            forced_object_id=forced_object_id,
            num_samples=args.num_samples,
            device=device,
        )

        if isinstance(model, torch.nn.Module):
            model.to("cpu")

        labeled = to_labeled_triples(seqs, cfg, vocabs)
        print("\n===", ckpt_path, "===")
        for idx, triples in enumerate(labeled, start=1):
            print(f"[{idx}]")
            if not triples:
                print("  (empty graph)")
                continue
            for triple in triples:
                print("  ", triple)
        print("---")


if __name__ == '__main__':
    main()

import torch
import random
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as PDataLoader
from intelligraphs import DataLoader
from torch.utils.data import Subset
from intelligraphs.evaluators import post_process_data, SemanticEvaluator



def compute_kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def compute_reconstruction_loss(logits, targets, mask=None):
    subject_logits, relation_logits, object_logits = logits
    subject_targets, relation_targets, object_targets = targets[:, :, 0], targets[:, :, 1], targets[:, :, 2]
    
    subject_loss = F.cross_entropy(subject_logits.reshape(-1, subject_logits.size(-1)), subject_targets.reshape(-1), reduction='none')
    relation_loss = F.cross_entropy(relation_logits.reshape(-1, relation_logits.size(-1)), relation_targets.reshape(-1), reduction='none')
    object_loss = F.cross_entropy(object_logits.reshape(-1, object_logits.size(-1)), object_targets.reshape(-1), reduction='none')
    
    # total_loss = subject_loss + relation_loss + object_loss
    total_loss = torch.cat([subject_loss, relation_loss, object_loss], dim=0) 
    
    if mask is not None:
        mask = mask.reshape(-1)
        total_loss = total_loss * mask
        return total_loss.sum() / mask.sum()
    else:
        return total_loss.mean()
    
    

def pad_triples(triples, max_edges, pad_value=0):
    batch_size, n_triples, _ = triples.shape
    
    if n_triples >= max_edges:
        return triples[:, :max_edges, :]
    
    padding = torch.full((batch_size, max_edges - n_triples, 3), pad_value, dtype=triples.dtype, device=triples.device)
    padded_triples = torch.cat([triples, padding], dim=1)
    
    return padded_triples


# def create_padding_mask(triples, pad_value=0):
#     mask = (triples[:, :, 0] != pad_value).float()
#     return mask
def create_padding_mask(triples, pad_value=0):
    B, N, _ = triples.shape
    flat = triples.view(B, -1)  
    mask = (flat != pad_value).float()
    return mask 

def compute_entity_sorting_loss(entity_logits, sorted_entities, mask=None):
    loss = F.cross_entropy(entity_logits.reshape(-1, entity_logits.size(-1)), sorted_entities.reshape(-1), reduction='none')
    
    if mask is not None:
        mask = mask.reshape(-1)
        loss = loss * mask
        return loss.sum() / mask.sum()
    else:
        return loss.mean()
    



#aturogressive model specific functions    
def kl_mean(mu, logv):
    return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())

def canonical_graph_string(graph): 
    return str(sorted(graph))

#calculate the compression bits per sequence
def bits_per_sequence(model, seq, z, pad_id=0):
    LN2 = math.log(2)
    seq = seq.unsqueeze(0).to(z.device)  
    total = 0.0
    for t in range(1, seq.size(1)):     
        target = seq[0, t].item()
        if target == pad_id:
            break
        logits = model.dec(z, seq[:, :t])[:, -1]
        log_probs = F.log_softmax(logits, dim=-1)
        total += -log_probs[0, target].item() / LN2
    return total

#convert the sequence to triples
def seq_to_triples(seq, SPECIAL, ENT_BASE, REL_BASE):
    triples, i = [], 1
    if torch.is_tensor(seq):
        seq = seq.tolist()
    while i + 2 < len(seq) and seq[i] != SPECIAL["EOS"]:
        h, r, t = seq[i:i+3]
        triples.append((h - ENT_BASE, r - REL_BASE, t - ENT_BASE))
        i += 3
    return triples

#calculate the posterior compression bits
@torch.no_grad()
def posterior_bits(
    model,
    dataset,
    device,
    pad_id=0,
    sample_frac=0.01,
    return_latents=False,
    desc="posterior bits"
):
    LN2 = math.log(2)
    n = max(1, int(sample_frac * len(dataset)))
    subset = Subset(dataset, range(n))
    loader = PDataLoader(subset, batch_size=1, shuffle=False)

    records = []
    for triples, seq in tqdm(loader, desc=desc):
        triples = triples.to(device)
        seq = seq[0].to(device)

        z, mu, logv = model.enc(triples)
        ar_bits = bits_per_sequence(model, seq, z, pad_id)

        kl_nats = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp(), dim=1)
        kl_bits = (kl_nats / LN2).item()

        rec = {
            "ar_bits": ar_bits,
            "kl_bits": kl_bits,
            "total_bits": ar_bits + kl_bits,
        }
        if return_latents:
            rec.update({
                "z": z.squeeze(0).cpu().numpy(),
                "mu": mu.squeeze(0).cpu().numpy(),
                "logv": logv.squeeze(0).cpu().numpy(),
            })
        records.append(rec)

    total = np.array([r["total_bits"] for r in records])
    ar    = np.array([r["ar_bits"]    for r in records])
    kl    = np.array([r["kl_bits"]    for r in records])
    stats = {
        "avg_total_bits": float(total.mean()),
        "avg_ar_bits":    float(ar.mean()),
        "avg_kl_bits":    float(kl.mean()),
        "min_total_bits": float(total.min()),
        "max_total_bits": float(total.max()),
        "records": records,
    }
    return stats

#convert the ints to labels
def ints_to_labels(graphs, i2e, i2r):
    result = []
    skipped = 0
    for g in graphs:
        clean_graph = []
        for h, r, t in g:
            if h in i2e and r in i2r and t in i2e:
                clean_graph.append((i2e[h], i2r[r], i2e[t]))
            else:
                skipped += 1
        result.append(clean_graph)
    if skipped > 0:
        print(f"[!] Skipped {skipped} invalid triples")
    return result

def canonicalize(triples, i2e = None, i2r = None, mode="alpha_name"):
    if mode == "keep":
        return triples
    return sorted(triples, key=lambda x: (i2e[x[0]], i2r[x[1]], i2e[x[2]]))

#convert the triples to sequence with extra BOS and EOS tokens and PAD
def triples_to_seq(triples, SPECIAL, ENT_BASE, REL_BASE, seq_len):
    seq = [SPECIAL["BOS"]]
    for h, r, t in triples:
        seq += [ENT_BASE + h, REL_BASE + r, ENT_BASE + t]
    seq.append(SPECIAL["EOS"])
    seq += [SPECIAL["PAD"]] * (seq_len - len(seq))
    return torch.tensor(seq, dtype=torch.long)

#generate the beam search sequences
@torch.no_grad()
def beam_generate(model, seq_len, special, seq_to_triples, z, ent_base, rel_base, beam=4):
    device = z.device
    B = z.size(0)
    BOS = torch.full((B, 1), special["BOS"], dtype=torch.long, device=device)
    seqs = [(BOS, torch.zeros(B, device=device))]
    for _ in range(seq_len - 1):
        cand = []
        for s, lp in seqs:
            logits = model.dec(z, s)[:, -1]
            logp = F.log_softmax(logits, dim=-1)
            top_lp, ids = logp.topk(beam, dim=-1)
            for k in range(beam):
                cand.append((torch.cat([s, ids[:, k, None]], 1), lp + top_lp[:, k]))
        seqs = sorted(cand, key=lambda x: x[1].mean().item(), reverse=True)[:beam]
        if all((s[:, -1] == special["EOS"]).all() for s, _ in seqs):
            break
    best = seqs[0][0].cpu()
    return [seq_to_triples(row, special, ent_base, rel_base) for row in best]

@torch.no_grad()
def decode_latent(model, z, seq_len, special, seq_to_triples, ent_base, rel_base, beam=4):
    z = z.to(next(model.parameters()).device, dtype=torch.float32)
    return beam_generate(model, seq_len, special, seq_to_triples, z, ent_base, rel_base, beam=beam)

#function that counts the unique graphs generated 
@torch.no_grad()
def count_unique_graphs(model, latent_dim, decode_latent_fn, num_samples=1000, beam=1):
    model.eval()
    z_samples = torch.randn((num_samples, latent_dim), device=next(model.parameters()).device)
    decoded_graphs = decode_latent_fn(model, z_samples, beam=beam)
    graph_strings = [canonical_graph_string(g) for g in decoded_graphs]
    unique_graphs = set(graph_strings)
    print(f"\n[Graph Diversity from {num_samples} Random Latents]")
    print(f"  Unique graphs generated: {len(unique_graphs)}")
    print(f"  Diversity ratio: {len(unique_graphs) / num_samples:.3f}")
    return unique_graphs


@torch.no_grad()
def generate_test_graphs(model, test_loader, seq_len, special, seq_to_triples,
                         ent_base, rel_base, beam_width=4, num_generated_test_graphs=1000, device="cuda"):
    generated_graphs = []
    for triples, _ in test_loader:
        z, *_ = model.enc(triples.to(device))
        generated_graphs.extend(
            beam_generate(model, seq_len, special, seq_to_triples, z, ent_base, rel_base, beam=beam_width)
        )
        if len(generated_graphs) >= num_generated_test_graphs:
            generated_graphs = generated_graphs[:num_generated_test_graphs]
            break
    return generated_graphs

#custom dataset class with ordering and padding
class GraphSeqDataset(Dataset):
    def __init__(self, graphs, i2e, i2r, triple_order="alpha_name", permute=False,
                 use_padding=False, pad_eid=None, pad_rid=None, max_triples=None,
                 special=None, ent_base=None, rel_base=None, seq_len=None):
        self.use_padding = use_padding
        self.pad_eid = pad_eid
        self.pad_rid = pad_rid
        self.max_triples = max_triples
        self.special = special
        self.ent_base = ent_base
        self.rel_base = rel_base
        self.seq_len = seq_len

        self.graphs = [canonicalize(g, i2e, i2r, triple_order) for g in graphs]
        self.permute = permute

    def __len__(self): 
        return len(self.graphs)

    def __getitem__(self, idx):
        triples = self.graphs[idx]
        if not self.use_padding and self.permute:
            triples = random.sample(triples, k=len(triples))

        if self.use_padding:
            pad = (self.pad_eid, self.pad_rid, self.pad_eid)
            triples_tensor = torch.tensor(
                triples + [pad] * (self.max_triples - len(triples)),
                dtype=torch.long
            )
        else:
            triples_tensor = torch.tensor(triples, dtype=torch.long)

        seq_tensor = triples_to_seq(triples, self.special, self.ent_base, self.rel_base, self.seq_len)
        return triples_tensor, seq_tensor






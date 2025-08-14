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

def canonical_graph_string(graph): 
    return str(sorted(graph))

#convert the sequence to triples
def seq_to_triples(seq, special_tokens, ENT_BASE, REL_BASE):
    triples, i = [], 1
    if torch.is_tensor(seq):
        seq = seq.tolist()
    while i + 2 < len(seq) and seq[i] != special_tokens["EOS"]:
        h, r, t = seq[i:i+3]
        triples.append((h - ENT_BASE, r - REL_BASE, t - ENT_BASE))
        i += 3
    return triples

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
def triples_to_seq(triples, special_tokens, ENT_BASE, REL_BASE, seq_len):
    seq = [special_tokens["BOS"]]
    for h, r, t in triples:
        seq += [ENT_BASE + h, REL_BASE + r, ENT_BASE + t]
    seq.append(special_tokens["EOS"])
    seq += [special_tokens["PAD"]] * (seq_len - len(seq))
    return torch.tensor(seq, dtype=torch.long)


#custom dataset class with ordering and padding
class GraphSeqDataset(Dataset):
    def __init__(self, graphs, i2e, i2r, triple_order="alpha_name", permute=False,
                 use_padding=False, pad_eid=None, pad_rid=None, max_triples=None,
                 special_tokens=None, ent_base=None, rel_base=None, seq_len=None):
        self.use_padding = use_padding
        self.pad_eid = pad_eid
        self.pad_rid = pad_rid
        self.max_triples = max_triples
        self.special_tokens = special_tokens
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

        seq_tensor = triples_to_seq(triples, self.special_tokens, self.ent_base, self.rel_base, self.seq_len)
        return triples_tensor, seq_tensor






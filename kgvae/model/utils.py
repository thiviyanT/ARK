import torch
import torch.nn.functional as F


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
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import numpy as np

from .layers import EncodingTransformer, DecodingTransformer
from torch.utils.data import DataLoader as PDataLoader
from torch.utils.data import Subset
from kgvae.model.utils import canonical_graph_string


class EdgeEmbeddings(nn.Module):
    def __init__(self, n_entities, n_relations, d_model):
        super().__init__()
        self.entity_embeddings = nn.Embedding(n_entities, d_model)
        self.relation_embeddings = nn.Embedding(n_relations, d_model)
        self.type_embeddings = nn.Embedding(3, d_model)
        
    def forward(self, triples):
        batch_size, n_triples, _ = triples.shape
        
        subjects = self.entity_embeddings(triples[:, :, 0])
        relations = self.relation_embeddings(triples[:, :, 1])
        objects = self.entity_embeddings(triples[:, :, 2])
        
        subject_type = self.type_embeddings(torch.zeros(batch_size, n_triples, dtype=torch.long, device=triples.device))
        relation_type = self.type_embeddings(torch.ones(batch_size, n_triples, dtype=torch.long, device=triples.device))
        object_type = self.type_embeddings(torch.full((batch_size, n_triples), 2, dtype=torch.long, device=triples.device))
        
        edge_embeddings = torch.stack([
            subjects + subject_type,
            relations + relation_type,
            objects + object_type
        ], dim=2)
        
        return edge_embeddings.view(batch_size, n_triples * 3, -1)


class Encoder(nn.Module):
    def __init__(self, n_entities, n_relations, d_model, n_layers, n_heads, d_ff, d_latent, dropout=0.1):
        super().__init__()
        self.edge_embeddings = EdgeEmbeddings(n_entities, n_relations, d_model)
        self.transformer = EncodingTransformer(n_layers, d_model, n_heads, d_ff, dropout)
        self.mu_layer = nn.Linear(d_model, d_latent)
        self.logvar_layer = nn.Linear(d_model, d_latent)
        
    def forward(self, triples, mask=None):
        edge_embeds = self.edge_embeddings(triples)
        transformer_output = self.transformer(edge_embeds, mask)
        
        pooled_output = transformer_output.mean(dim=1)
        
        mu = self.mu_layer(pooled_output)
        logvar = self.logvar_layer(pooled_output)
        
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, n_entities, n_relations, d_model, n_layers, n_heads, d_ff, d_latent, max_nodes, max_edges, dropout=0.1):
        super().__init__()
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.d_model = d_model
        
        self.latent_projection = nn.Linear(d_latent, d_model)
        self.transformer = DecodingTransformer(n_layers, d_model, n_heads, d_ff, dropout)
        
        self.entity_predictor = nn.Linear(d_model, n_entities)
        self.relation_predictor = nn.Linear(d_model, n_relations)
        
        self.positional_encoding = nn.Parameter(torch.randn(1, max_edges * 3, d_model))
        
    def forward(self, z, mask=None):
        batch_size = z.size(0)
        
        z_projected = self.latent_projection(z).unsqueeze(1)
        z_expanded = z_projected.expand(batch_size, self.max_edges * 3, self.d_model)
        
        decoder_input = z_expanded + self.positional_encoding
        transformer_output = self.transformer(decoder_input, mask)
        
        transformer_output = transformer_output.view(batch_size, self.max_edges, 3, self.d_model)
        
        subject_logits = self.entity_predictor(transformer_output[:, :, 0, :])
        relation_logits = self.relation_predictor(transformer_output[:, :, 1, :])
        object_logits = self.entity_predictor(transformer_output[:, :, 2, :])
        
        return subject_logits, relation_logits, object_logits


class MLP_Encoder(nn.Module):
    def __init__(self, n_entities, n_relations, d_model, d_hidden, d_latent, dropout=0.1):
        super().__init__()
        self.edge_embeddings = EdgeEmbeddings(n_entities, n_relations, d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.mu_layer = nn.Linear(d_hidden, d_latent)
        self.logvar_layer = nn.Linear(d_hidden, d_latent)
        
    def forward(self, triples, mask=None):
        edge_embeds = self.edge_embeddings(triples)
        pooled = edge_embeds.mean(dim=1)
        
        hidden = self.mlp(pooled)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        
        return mu, logvar


class MLP_Decoder(nn.Module):
    def __init__(self, n_entities, n_relations, d_hidden, d_latent, max_edges, dropout=0.1):
        super().__init__()
        self.max_edges = max_edges
        
        self.mlp = nn.Sequential(
            nn.Linear(d_latent + 3, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.entity_predictor = nn.Linear(d_hidden, n_entities)
        self.relation_predictor = nn.Linear(d_hidden, n_relations)
        
    def forward(self, z, mask=None):
        batch_size = z.size(0)
        outputs = []
        
        for i in range(self.max_edges):
            position_encoding = torch.zeros(batch_size, 3, device=z.device)
            position_encoding[:, i % 3] = 1
            
            decoder_input = torch.cat([z, position_encoding], dim=1)
            hidden = self.mlp(decoder_input)
            
            if i % 3 == 0:
                output = self.entity_predictor(hidden)
            elif i % 3 == 1:
                output = self.relation_predictor(hidden)
            else:
                output = self.entity_predictor(hidden)
                
            outputs.append(output)
            
        subject_logits = torch.stack(outputs[0::3], dim=1)
        relation_logits = torch.stack(outputs[1::3], dim=1)
        object_logits = torch.stack(outputs[2::3], dim=1)
        
        return subject_logits, relation_logits, object_logits


class ScoringFunction(nn.Module):
    def __init__(self, n_entities, n_relations, d_model):
        super().__init__()
        self.entity_embeddings = nn.Embedding(n_entities, d_model)
        # self.relation_embeddings = nn.Embedding(n_relations, d_model, d_model)
        self.relation_embeddings = nn.Embedding(n_relations, d_model)
        
    def forward(self, subjects, relations, objects):
        s_embed = self.entity_embeddings(subjects)
        r_embed = self.relation_embeddings(relations)
        o_embed = self.entity_embeddings(objects)
        
        scores = torch.sum(s_embed * torch.matmul(r_embed, o_embed.transpose(-2, -1)), dim=-1)
        return scores


class KGVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config['encoder_type'] == 'transformer':
            self.encoder = Encoder(
                config['n_entities'],
                config['n_relations'],
                config['d_model'],
                config['n_layers'],
                config['n_heads'],
                config['d_ff'],
                config['d_latent'],
                config['dropout']
            )
        else:
            self.encoder = MLP_Encoder(
                config['n_entities'],
                config['n_relations'],
                config['d_model'],
                config['d_hidden'],
                config['d_latent'],
                config['dropout']
            )
            
        if config['decoder_type'] == 'transformer':
            self.decoder = Decoder(
                config['n_entities'],
                config['n_relations'],
                config['d_model'],
                config['n_layers'],
                config['n_heads'],
                config['d_ff'],
                config['d_latent'],
                config['max_nodes'],
                config['max_edges'],
                config['dropout']
            )
        else:
            self.decoder = MLP_Decoder(
                config['n_entities'],
                config['n_relations'],
                config['d_hidden'],
                config['d_latent'],
                config['max_edges'],
                config['dropout']
            )
            
        self.scoring_function = ScoringFunction(
            config['n_entities'],
            config['n_relations'],
            config['d_model']
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, triples, mask=None):
        mu, logvar = self.encoder(triples, mask)
        z = self.reparameterize(mu, logvar)
        
        subject_logits, relation_logits, object_logits = self.decoder(z, mask)
        
        return {
            'subject_logits': subject_logits,
            'relation_logits': relation_logits,
            'object_logits': object_logits,
            'mu': mu,
            'logvar': logvar
        }
        
    def sample(self, batch_size, device):
        z = torch.randn(batch_size, self.config['d_latent'], device=device)
        subject_logits, relation_logits, object_logits = self.decoder(z)
        
        subjects = torch.argmax(subject_logits, dim=-1)
        relations = torch.argmax(relation_logits, dim=-1)
        objects = torch.argmax(object_logits, dim=-1)

        return torch.stack([subjects, relations, objects], dim=-1)
    
    
    
class AutoRegDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, seq_len, vocab_size, latent_dim):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len,   d_model)
        self.z_proj  = nn.Linear(latent_dim,   d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.txf = nn.TransformerDecoder(layer, num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, z, tgt):
        B, L = tgt.shape
        tok = self.tok_emb(tgt)
        pos = self.pos_emb(torch.arange(L, device=z.device)).unsqueeze(0)
        mem = self.z_proj(z).unsqueeze(1).repeat(1, L, 1)
        mask = torch.triu(torch.ones(L, L, device=z.device, dtype=torch.bool), 1)
        return self.out(self.txf(tok + pos, mem, tgt_mask=mask))
    
    
class AutoRegDecoderGRU(nn.Module):
    def __init__(self, d_model, num_layers, seq_len, vocab_size, latent_dim, dropout=0.1, tie_weights=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.z_proj  = nn.Linear(latent_dim, d_model)   
        self.gru     = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out     = nn.Linear(d_model, vocab_size)

        if tie_weights:
            if self.out.weight.shape == self.tok_emb.weight.shape:
                self.out.weight = self.tok_emb.weight
            else:
                pass

    def forward(self, z, tgt):              
        B, L = tgt.shape
        x  = self.tok_emb(tgt)              
        h0 = torch.tanh(self.z_proj(z))     
        h0 = h0.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)  
        y, _ = self.gru(x, h0)              
        return self.out(y) 

class AutoRegEncoder(nn.Module):
    def __init__(self, num_entities, num_relations, d_model, nhead, latent_dim,
                 pad_eid=None, pad_rid=None, n_layers=2):
        super().__init__()
        self.pad_rid = pad_rid
        self.e_emb = nn.Embedding(num_entities,  d_model, padding_idx=pad_eid)
        self.r_emb = nn.Embedding(num_relations, d_model, padding_idx=pad_rid)
        layer = nn.TransformerEncoderLayer(d_model*3, nhead, batch_first=True)
        self.txf  = nn.TransformerEncoder(layer, n_layers)
        self.mu   = nn.Linear(d_model*3, latent_dim)
        self.logv = nn.Linear(d_model*3, latent_dim)

    def forward(self, triples):  
        B, T, _ = triples.shape
        h = self.e_emb(triples[:, :, 0])
        r = self.r_emb(triples[:, :, 1])
        t = self.e_emb(triples[:, :, 2])
        x = torch.cat([h, r, t], -1)

        if self.pad_rid is not None:
            mask = triples[:, :, 1] != self.pad_rid              
            x = self.txf(x, src_key_padding_mask=~mask)
            denom = mask.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
            x = (x * mask.unsqueeze(-1)).sum(1) / denom.squeeze(-1)
        else:
            x = self.txf(x).mean(1)

        mu, logv = self.mu(x), self.logv(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logv)
        return z, mu, logv
    
    

class AutoRegEncoderMLP(nn.Module):
    def __init__(self, num_entities, num_relations, d_model, latent_dim,
                 pad_eid=None, pad_rid=None, hidden=None, dropout=0.0):
        super().__init__()
        self.pad_rid = pad_rid
        self.e_emb = nn.Embedding(num_entities,  d_model, padding_idx=pad_eid)
        self.r_emb = nn.Embedding(num_relations, d_model, padding_idx=pad_rid)

        d_in = d_model * 3
        hidden = hidden or max(d_in, d_model * 2)

        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mu   = nn.Linear(hidden, latent_dim)
        self.logv = nn.Linear(hidden, latent_dim)

    def forward(self, triples): 
        h = self.e_emb(triples[:, :, 0])
        r = self.r_emb(triples[:, :, 1])
        t = self.e_emb(triples[:, :, 2])
        x = torch.cat([h, r, t], dim=-1)       

        if self.pad_rid is not None:
            mask = (triples[:, :, 1] != self.pad_rid)       
            x = x * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            g = x.sum(dim=1) / denom                        
        else:
            g = x.mean(dim=1)

        g = self.mlp(g)                                      
        mu   = self.mu(g)
        logv = self.logv(g).clamp(-10, 10) 
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logv)
        return z, mu, logv

class AutoRegModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config['ablation_encoder'] == 'MLP':
            print("Using MLP encoder")
            self.enc = AutoRegEncoderMLP(
                num_entities=config["n_entities"],
                num_relations=config["n_relations"],
                d_model=config["d_model"],
                latent_dim=config["d_latent"],
                pad_eid=config.get("pad_eid"),
                pad_rid=config.get("pad_rid"),
            )
        else:
            self.enc = AutoRegEncoder(
                num_entities=config["n_entities"],
                num_relations=config["n_relations"],
                d_model=config["d_model"],
                nhead=config["n_heads"],
                latent_dim=config["d_latent"],
                pad_eid=config.get("pad_eid", None),
                pad_rid=config.get("pad_rid", None),
                n_layers=config.get("n_layers", 2)
            )
        if config['ablation_decoder'] == 'GRU':
            print("Using GRU Decoder")
            self.dec = AutoRegDecoderGRU(
                d_model=self.config["d_model"],
                num_layers=self.config["n_layers"],
                seq_len=self.config["seq_len"],
                vocab_size=self.config["vocab_size"],
                latent_dim=self.config["d_latent"],
                dropout=self.config.get("dec_dropout", 0.1),
                tie_weights=self.config.get("tie_weights", True),
            )
        else:
            self.dec = AutoRegDecoder(
                d_model=config["d_model"],
                nhead=config["n_heads"],
                num_layers=config["n_layers"],
                seq_len=config["seq_len"],
                vocab_size=config["vocab_size"],
                latent_dim=config["d_latent"]
            )
    def kl_mean(self,mu, logv):
        return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
    
    #calculate the compression bits per sequence
    def bits_per_sequence(self, seq, z, pad_id=0):
        LN2 = math.log(2)
        seq = seq.unsqueeze(0).to(z.device)  
        total = 0.0
        for t in range(1, seq.size(1)):     
            target = seq[0, t].item()
            if target == pad_id:
                break
            logits = self.dec(z, seq[:, :t])[:, -1]
            log_probs = F.log_softmax(logits, dim=-1)
            total += -log_probs[0, target].item() / LN2
        return total
    
    #calculate the posterior compression bits. We calculate the KL divergence and the autoregressive loss and divide by LN2 for each sequence
    #KL term: how many bits to encode the latent representation
    #AR term (autoregressive bits): how many bits to reconstruct the data given the latent
    @torch.no_grad()
    def posterior_bits(
        self,
        dataset,
        device,
        pad_id=0,
        sample_frac=0.1,
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

            z, mu, logv = self.enc(triples)
            ar_bits = self.bits_per_sequence(seq, z, pad_id)

            kl_nats = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp(), dim=1)
            kl_bits = (kl_nats / LN2).item()

            records.append({
            "ar_bits": ar_bits,
            "kl_bits": kl_bits,
            "total_bits": ar_bits + kl_bits,
        })

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
    
    @torch.no_grad()
    def decode_latent(self, z, seq_len, special_tokens, seq_to_triples, ent_base, rel_base, beam=4):
        self.eval()
        z = z.to(next(self.parameters()).device, dtype=torch.float32)
        return self.beam_generate (seq_len, special_tokens, seq_to_triples, z, ent_base, rel_base, beam=beam)

    #function that counts the unique graphs generated 
    @torch.no_grad()
    def count_unique_graphs(self, latent_dim, decode_latent_fn, num_samples=1000, beam=1):
        self.eval()
        z_samples = torch.randn((num_samples, latent_dim), device=next(self.parameters()).device)
        decoded_graphs = decode_latent_fn(z_samples, beam=beam)
        graph_strings = [canonical_graph_string(g) for g in decoded_graphs]
        unique_graphs = set(graph_strings)
        print(f"\n[Graph Diversity from {num_samples} Random Latents]")
        print(f"  Unique graphs generated: {len(unique_graphs)}")
        print(f"  Diversity ratio: {len(unique_graphs) / num_samples:.3f}")
        return unique_graphs
    
    #generate the beam search sequences
    @torch.no_grad()
    def beam_generate(self, seq_len, special_tokens, seq_to_triples, z, ent_base, rel_base, beam=4):
        device = z.device
        B = z.size(0)
        BOS = torch.full((B, 1), special_tokens["BOS"], dtype=torch.long, device=device)
        seqs = [(BOS, torch.zeros(B, device=device))]
        for _ in range(seq_len - 1):
            cand = []
            for s, lp in seqs:
                logits = self.dec(z, s)[:, -1]
                logp = F.log_softmax(logits, dim=-1)
                top_lp, ids = logp.topk(beam, dim=-1)
                for k in range(beam):
                    cand.append((torch.cat([s, ids[:, k, None]], 1), lp + top_lp[:, k]))
            seqs = sorted(cand, key=lambda x: x[1].mean().item(), reverse=True)[:beam]
            if all((s[:, -1] == special_tokens["EOS"]).all() for s, _ in seqs):
                break
        best = seqs[0][0].cpu()
        return [seq_to_triples(row, special_tokens, ent_base, rel_base) for row in best]
    
    
    @torch.no_grad()
    def generate_test_graphs(self, test_loader, seq_len, special_tokens, seq_to_triples,
                            ent_base, rel_base, beam_width=4, num_generated_test_graphs=1000, device="cuda"):
        generated_graphs = []
        for triples, _ in test_loader:
            z, *_ = self.enc(triples.to(device))
            generated_graphs.extend(
                self.beam_generate(seq_len, special_tokens, seq_to_triples, z, ent_base, rel_base, beam=beam_width)
            )
            if len(generated_graphs) >= num_generated_test_graphs:
                generated_graphs = generated_graphs[:num_generated_test_graphs]
                break
        return generated_graphs

    def forward(self, triples, seq_in):
        z, mu, logv = self.enc(triples)
        logits = self.dec(z, seq_in)
        return logits, mu, logv    

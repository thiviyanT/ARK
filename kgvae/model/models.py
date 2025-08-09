import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import EncodingTransformer, DecodingTransformer


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

    def forward(self, triples):  # (B, T, 3)
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
    
    
class AutoRegModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.enc = AutoRegEncoder(
            num_entities=config["n_entities"],
            num_relations=config["n_relations"],
            d_model=config["d_model"],
            nhead=config["n_heads"],
            latent_dim=config["d_latent"],
            pad_eid=config.get("pad_eid", None), #this needs to be changed or set in the config
            pad_rid=config.get("pad_rid", None), #this needs to be changed or set in the config
            n_layers=config.get("n_layers", 2)
        )

        self.dec = AutoRegDecoder(
            d_model=config["d_model"],
            nhead=config["n_heads"],
            num_layers=config["n_layers"],
            seq_len=config["seq_len"],
            vocab_size=config["vocab_size"],
            latent_dim=config["d_latent"]
        )
    
    # def forward(self, triples, seq_in):
    #     z, mu, logvar = self.encoder(triples)
    #     logits = self.decoder(z, seq_in)
    #     return {
    #         "logits": logits,
    #         "mu": mu,
    #         "logvar": logvar
    #     }
    def forward(self, triples, seq_in):
        z, mu, logv = self.enc(triples)
        logits = self.dec(z, seq_in)
        return logits, mu, logv    

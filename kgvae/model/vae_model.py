import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEmbeddings(nn.Module):
    """
    Generates edge embeddings from triples (subject, predicate, object).
    Projects concatenated [s, p, o] embeddings to a fixed dimension.
    """
    def __init__(self, num_relations, emb_size):
        super().__init__()
        self.relation_embeddings = nn.Embedding(num_relations, emb_size)
        # Project concatenated s,p,o embeddings (3*emb_size) to emb_size
        self.down_projection = nn.Linear(emb_size * 3, emb_size)
        
    def forward(self, triples, node_embeddings):
        """
        Generate embeddings to encode edge connectivity.
        
        Args:
            triples: [batch_size, num_edges, 3] containing (s, p, o) indices
            node_embeddings: Node embedding layer
        
        Returns:
            Edge embeddings of shape [batch_size, num_edges, emb_size]
        """
        assert triples.size(-1) == 3
        
        # Split triples into s, p, o
        s_index = triples[:, :, 0]
        p_index = triples[:, :, 1]
        o_index = triples[:, :, 2]
        
        # Get embeddings
        s = node_embeddings(s_index)  # [batch, num_edges, emb_size]
        p = self.relation_embeddings(p_index)  # [batch, num_edges, emb_size]
        o = node_embeddings(o_index)  # [batch, num_edges, emb_size]
        
        # Concatenate and project
        x = torch.cat([s, p, o], dim=-1)  # [batch, num_edges, 3*emb_size]
        return self.down_projection(x)  # [batch, num_edges, emb_size]


class ScoringFunction(nn.Module):
    """
    RESCAL-style scoring function for knowledge graph completion.
    Computes scores using bilinear tensor factorization.
    """
    def __init__(self, emb_size, num_relations):
        super().__init__()
        # Relation-specific matrices for RESCAL scoring
        self.relation_matrices = nn.Parameter(
            torch.randn(num_relations, emb_size, emb_size)
        )
        nn.init.xavier_normal_(self.relation_matrices)
        
    def forward(self, node_emb):
        """
        Compute RESCAL scores for all possible triples.
        
        Args:
            node_emb: Node embeddings [batch_size, num_nodes, emb_size]
        
        Returns:
            Scores of shape [batch_size, num_relations, num_nodes, num_nodes]
        """
        # RESCAL: score(s,r,o) = s^T * M_r * o
        # Using einsum for efficient computation
        return torch.einsum('bnd,rde,bke->brnk', node_emb, self.relation_matrices, node_emb)


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder that processes node and edge embeddings.
    Applies self-attention and produces latent representations.
    """
    def __init__(self, max_nodes, max_edges, emb_size, latent_dim, 
                 num_heads=8, num_layers=4, d_ff=2048, dropout=0.1):
        super().__init__()
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        
        # Transformer blocks
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # VAE latent projections
        self.mu_projection = nn.Linear(emb_size, latent_dim)
        self.logvar_projection = nn.Linear(emb_size, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_embeddings, edge_embeddings):
        """
        Encode graph structure into latent distribution parameters.
        
        Args:
            node_embeddings: [batch_size, max_nodes, emb_size]
            edge_embeddings: [batch_size, max_edges, emb_size]
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution  
            node_emb: Updated node embeddings after attention
        """
        # Concatenate node and edge embeddings
        sequence = torch.cat([node_embeddings, edge_embeddings], dim=1)
        sequence = self.dropout(sequence)
        
        # Apply transformer blocks
        for layer in self.transformer_layers:
            sequence = layer(sequence)
        
        # Extract node embeddings
        node_emb = sequence[:, :self.max_nodes, :]
        
        # Pool to get graph representation
        graph_emb = node_emb.mean(dim=1)  # Global average pooling
        
        # Compute VAE parameters
        mu = self.mu_projection(graph_emb)
        logvar = self.logvar_projection(graph_emb)
        
        return mu, logvar, node_emb
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder that generates node embeddings from latent code.
    """
    def __init__(self, max_nodes, emb_size, latent_dim, num_entities,
                 num_heads=8, num_layers=4, d_ff=2048, dropout=0.1):
        super().__init__()
        self.max_nodes = max_nodes
        
        # Project latent to embedding space
        self.latent_projection = nn.Linear(latent_dim, emb_size)
        
        # Learnable positional embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, max_nodes, emb_size))
        
        # Transformer blocks
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Entity prediction head
        self.entity_classifier = nn.Linear(emb_size, num_entities)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latent_representation):
        """
        Decode latent representation into node embeddings and entity predictions.
        
        Args:
            latent_representation: [batch_size, latent_dim]
        
        Returns:
            node_emb: Decoded node embeddings [batch_size, max_nodes, emb_size]
            entity_pred: Entity predictions [batch_size, max_nodes, num_entities]
        """
        batch_size = latent_representation.size(0)
        
        # Project latent to embedding space
        latent_emb = self.latent_projection(latent_representation)  # [batch, emb_size]
        
        # Expand to sequence length and add positional embeddings
        sequence = latent_emb.unsqueeze(1).expand(batch_size, self.max_nodes, -1)
        sequence = sequence + self.position_embeddings
        sequence = self.dropout(sequence)
        
        # Apply transformer blocks
        for layer in self.transformer_layers:
            sequence = layer(sequence)
        
        # Entity predictions
        entity_pred = self.entity_classifier(sequence)
        
        return sequence, entity_pred


class TransformerBlock(nn.Module):
    """
    Standard transformer block with multi-head attention and feed-forward network.
    """
    def __init__(self, emb_size, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, emb_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class RESCALVAE(nn.Module):
    """
    RESCAL-VAE model combining:
    - Node and edge embeddings
    - Transformer encoder and decoder
    - RESCAL scoring function
    - Type embeddings for distinguishing nodes and edges
    """
    def __init__(self, config):
        super().__init__()
        
        # Extract config parameters
        num_entities = config['n_entities'] + 1  # Add padding entity
        num_relations = config['n_relations'] + 1  # Add padding relation
        emb_size = config['d_model']
        latent_dim = config['d_latent']
        max_nodes = config['max_nodes']
        max_edges = config['max_edges']
        num_heads = config.get('n_heads', 8)
        num_layers = config.get('n_layers', 4)
        d_ff = config.get('d_ff', 2048)
        dropout = config.get('dropout', 0.1)
        
        # Embeddings
        self.node_embeddings = nn.Embedding(num_entities, emb_size, padding_idx=0)
        self.edge_embedder = EdgeEmbeddings(num_relations, emb_size)
        
        # Type embeddings (0 for nodes, 1 for edges)
        self.type_embeddings = nn.Embedding(2, emb_size)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(
            max_nodes, max_edges, emb_size, latent_dim,
            num_heads, num_layers, d_ff, dropout
        )
        
        self.decoder = TransformerDecoder(
            max_nodes, emb_size, latent_dim, num_entities,
            num_heads, num_layers, d_ff, dropout
        )
        
        # Scoring function
        self.scoring_function = ScoringFunction(emb_size, num_relations)
        
        # Store config
        self.config = config
        
    def forward(self, triples, nodes, mask=None):
        """
        Forward pass through the RESCAL-VAE.
        
        Args:
            triples: Edge triples [batch_size, max_edges, 3]
            nodes: Node indices [batch_size, max_nodes]
            mask: Optional padding mask
        
        Returns:
            Dictionary containing:
                - decoder_scores: Reconstruction scores
                - entity_predictions: Entity predictions
                - mu: Latent mean
                - logvar: Latent log variance
        """
        batch_size = triples.size(0)
        device = triples.device
        
        # Get node embeddings and add type embeddings
        node_sequence = self.node_embeddings(nodes)
        node_type = torch.zeros(batch_size, self.config['max_nodes'], dtype=torch.long, device=device)
        node_sequence = node_sequence + self.type_embeddings(node_type)
        
        # Get edge embeddings and add type embeddings
        edge_sequence = self.edge_embedder(triples, self.node_embeddings)
        edge_type = torch.ones(batch_size, self.config['max_edges'], dtype=torch.long, device=device)
        edge_sequence = edge_sequence + self.type_embeddings(edge_type)
        
        # Encode
        mu, logvar, encoder_node_emb = self.encoder(node_sequence, edge_sequence)
        
        # Reparameterize
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode
        decoder_node_emb, entity_predictions = self.decoder(z)
        
        # Score
        decoder_scores = self.scoring_function(decoder_node_emb)
        
        return {
            'decoder_scores': decoder_scores,
            'entity_predictions': entity_predictions,
            'mu': mu,
            'logvar': logvar,
            'encoder_scores': self.scoring_function(encoder_node_emb)  # Optional
        }
    
    def sample(self, batch_size, device):
        """
        Sample new graphs from the prior distribution.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            Generated triples [batch_size, max_edges, 3]
        """
        # Sample from standard normal
        z = torch.randn(batch_size, self.config['d_latent'], device=device)
        
        # Decode
        node_emb, entity_pred = self.decoder(z)
        
        # Get scores
        scores = self.scoring_function(node_emb)
        
        # Convert scores to triples (simplified - just take argmax)
        # In practice, you might want more sophisticated sampling
        generated_triples = []
        for b in range(batch_size):
            triples = []
            for r in range(self.config['n_relations']):
                # Get top-k edges for this relation
                relation_scores = scores[b, r]
                flat_scores = relation_scores.flatten()
                _, top_indices = torch.topk(flat_scores, min(3, flat_scores.size(0)))
                
                for idx in top_indices:
                    s = idx // self.config['max_nodes']
                    o = idx % self.config['max_nodes']
                    triples.append([s.item(), r, o.item()])
                    
                    if len(triples) >= self.config['max_edges']:
                        break
                        
                if len(triples) >= self.config['max_edges']:
                    break
            
            # Pad if necessary
            while len(triples) < self.config['max_edges']:
                triples.append([0, 0, 0])
            
            generated_triples.append(triples[:self.config['max_edges']])
        
        return torch.tensor(generated_triples, device=device)

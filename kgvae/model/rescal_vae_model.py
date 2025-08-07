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
        if triples.size(-1) != 3:
            raise ValueError(f"Expected triples to have shape [..., 3], got [..., {triples.size(-1)}]")
        
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
    
    Memory optimization: Computing all possible edges creates huge tensors
    (batch × relations × nodes × nodes), causing CUDA OOM. Instead, we score
    only observed/candidate edges when edge_indices are provided.
    """
    def __init__(self, emb_size, num_relations):
        super().__init__()
        # Relation-specific matrices for RESCAL scoring
        self.relation_matrices = nn.Parameter(
            torch.randn(num_relations, emb_size, emb_size)
        )
        nn.init.normal_(self.relation_matrices, std=0.01)  # Smaller init for bilinear
        
    def forward(self, node_emb, edge_indices=None, entity_embeddings=None):
        """
        Compute RESCAL scores for triples.
        
        Args:
            node_emb: Node embeddings [batch_size, num_nodes, emb_size] or entity embedding layer
            edge_indices: Optional tensor of edges to score [batch_size, num_edges, 3] (s, r, o)
                         If None, computes all possible combinations (memory intensive!)
            entity_embeddings: Optional entity embedding layer to use for lookups
        
        Returns:
            If edge_indices provided: Scores [batch_size, num_edges]
            Otherwise: Scores [batch_size, num_relations, num_nodes, num_nodes]
        """
        if edge_indices is not None:
            # Memory-efficient sparse scoring: only compute scores for specified edges
            batch_size = edge_indices.size(0)
            num_edges = edge_indices.size(1)
            device = edge_indices.device
            
            # Extract indices for specified edges
            s_idx = edge_indices[:, :, 0]  # [batch_size, num_edges]
            r_idx = edge_indices[:, :, 1]  # [batch_size, num_edges]
            o_idx = edge_indices[:, :, 2]  # [batch_size, num_edges]
            
            # Get embeddings - use entity_embeddings if provided, otherwise use node_emb
            if entity_embeddings is not None:
                # Direct lookup from entity embedding table
                s_emb = entity_embeddings(s_idx)  # [batch_size, num_edges, emb_size]
                o_emb = entity_embeddings(o_idx)  # [batch_size, num_edges, emb_size]
            else:
                # Use provided node embeddings (assumed to be positional)
                # This only works if indices < node_emb.size(1)
                max_idx = node_emb.size(1)
                # Clamp indices to valid range and gather
                s_idx_clamped = torch.clamp(s_idx, 0, max_idx - 1)
                o_idx_clamped = torch.clamp(o_idx, 0, max_idx - 1)
                s_emb = torch.gather(node_emb, 1, s_idx_clamped.unsqueeze(-1).expand(-1, -1, node_emb.size(-1)))
                o_emb = torch.gather(node_emb, 1, o_idx_clamped.unsqueeze(-1).expand(-1, -1, node_emb.size(-1)))
            
            # Vectorized scoring for all edges
            scores = torch.zeros(batch_size, num_edges, device=device)
            for b in range(batch_size):
                for e in range(num_edges):
                    r = r_idx[b, e]
                    if r < self.relation_matrices.size(0):  # Check valid relation
                        scores[b, e] = torch.sum(
                            s_emb[b, e] * torch.matmul(self.relation_matrices[r], o_emb[b, e])
                        )
            
            return scores
        else:
            # Full scoring: computes all possible combinations
            # WARNING: Creates tensor of size [batch × relations × nodes × nodes]  #TODO Put warning here
            # Can easily exceed GPU memory (e.g., 32×50×100×100 = 6.4GB)
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
        
    def forward(self, node_embeddings, edge_embeddings, node_mask=None, edge_mask=None):
        """
        Encode graph structure into latent distribution parameters.
        
        Args:
            node_embeddings: [batch_size, max_nodes, emb_size]
            edge_embeddings: [batch_size, max_edges, emb_size]
            node_mask: [batch_size, max_nodes] - True for padding
            edge_mask: [batch_size, max_edges] - True for padding
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution  
            node_emb: Updated node embeddings after attention
        """
        # Concatenate node and edge embeddings
        sequence = torch.cat([node_embeddings, edge_embeddings], dim=1)
        sequence = self.dropout(sequence)
        
        # Create combined mask if provided
        if node_mask is not None or edge_mask is not None:
            batch_size = sequence.size(0)
            if node_mask is None:
                node_mask = torch.zeros(batch_size, self.max_nodes, device=sequence.device, dtype=torch.bool)
            if edge_mask is None:
                edge_mask = torch.zeros(batch_size, self.max_edges, device=sequence.device, dtype=torch.bool)
            combined_mask = torch.cat([node_mask, edge_mask], dim=1)
        else:
            combined_mask = None
        
        # Apply transformer blocks with mask
        for layer in self.transformer_layers:
            sequence = layer(sequence, mask=combined_mask)
        
        # Extract node embeddings
        node_emb = sequence[:, :self.max_nodes, :]
        
        # Masked pooling for graph representation
        if node_mask is not None:
            # Mask out padding before pooling
            node_emb_masked = node_emb.masked_fill(node_mask.unsqueeze(-1), 0)
            valid_nodes = (~node_mask).sum(dim=1, keepdim=True).float()
            graph_emb = node_emb_masked.sum(dim=1) / valid_nodes.clamp(min=1)
        else:
            graph_emb = node_emb.mean(dim=1)
        
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
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_nodes, emb_size))
        nn.init.normal_(self.position_embeddings, std=0.02)
        
        # Transformer blocks
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Entity prediction head
        self.entity_classifier = nn.Linear(emb_size, num_entities)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latent_representation, mask=None):
        """
        Decode latent representation into node embeddings and entity predictions.
        
        Args:
            latent_representation: [batch_size, latent_dim]
            mask: Optional [batch_size, max_nodes] - True for padding
        
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
        
        # Apply transformer blocks with mask
        for layer in self.transformer_layers:
            sequence = layer(sequence, mask=mask)
        
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
        
        # Create masks for padding
        node_mask = (nodes == 0)  # Padding idx is 0
        edge_mask = (triples.sum(dim=-1) == 0)  # All zeros = padding
        
        # Get node embeddings and add type embeddings
        node_sequence = self.node_embeddings(nodes)
        node_type = torch.zeros(batch_size, self.config['max_nodes'], dtype=torch.long, device=device)
        node_sequence = node_sequence + self.type_embeddings(node_type)
        
        # Get edge embeddings and add type embeddings
        edge_sequence = self.edge_embedder(triples, self.node_embeddings)
        edge_type = torch.ones(batch_size, self.config['max_edges'], dtype=torch.long, device=device)
        edge_sequence = edge_sequence + self.type_embeddings(edge_type)
        
        # Encode with masks
        mu, logvar, encoder_node_emb = self.encoder(node_sequence, edge_sequence, node_mask, edge_mask)
        
        # Reparameterize
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode with node mask
        decoder_node_emb, entity_predictions = self.decoder(z, node_mask)
        
        # Score - pass edge indices and entity embeddings for memory-efficient sparse scoring
        # Use entity embeddings directly since triples contain entity IDs, not positions
        decoder_scores = self.scoring_function(
            decoder_node_emb, 
            edge_indices=triples, 
            entity_embeddings=self.node_embeddings
        )
        
        # For encoder scores, also use sparse scoring with entity embeddings
        encoder_scores = self.scoring_function(
            encoder_node_emb, 
            edge_indices=triples,
            entity_embeddings=self.node_embeddings
        )
        
        return {
            'decoder_scores': decoder_scores,
            'entity_predictions': entity_predictions,
            'mu': mu,
            'logvar': logvar,
            'encoder_scores': encoder_scores
        }
    
    def sample(self, batch_size, device, sampling_method=None, temperature=1.0):
        """
        Sample new graphs from the prior distribution.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate samples on
            sampling_method: 'probabilistic' (default) or 'deterministic'
                - probabilistic: Sample edges from multinomial distribution based on scores
                - deterministic: Select top-k edges based on scores
            temperature: Temperature for probabilistic sampling (higher = more random)
        
        Returns:
            Generated triples [batch_size, max_edges, 3]
        """
        # Use config sampling method if not specified
        if sampling_method is None:
            sampling_method = self.config.get('sampling_method', 'probabilistic')
        temperature = self.config.get('sampling_temperature', temperature)
        
        # Sample from standard normal
        z = torch.randn(batch_size, self.config['d_latent'], device=device)
        
        # Decode
        node_emb, entity_pred = self.decoder(z)
        
        # Get parameters
        max_nodes = self.config['max_nodes']
        max_edges = self.config['max_edges']
        num_relations = self.config['n_relations']
        num_entities = self.config['n_entities']
        
        # Generate all possible candidate edges for comprehensive sampling
        # Create grid of all possible (s, r, o) combinations
        all_candidates = []
        for b in range(batch_size):
            batch_candidates = []
            # Generate more diverse candidates
            for r in range(1, min(num_relations, 10)):  # Limit relations for efficiency
                # Sample entity pairs for this relation
                num_samples = min(max_nodes * 2, 50)  # Sample more candidates per relation
                for _ in range(num_samples):
                    s = torch.randint(0, min(max_nodes, num_entities), (1,), device=device).item()
                    o = torch.randint(0, min(max_nodes, num_entities), (1,), device=device).item()
                    batch_candidates.append([s, r, o])
            all_candidates.append(batch_candidates)
        
        candidate_triples = torch.tensor(all_candidates, device=device)
        
        # Score candidates using entity embeddings for proper scoring
        scores = self.scoring_function(
            node_emb, 
            edge_indices=candidate_triples,
            entity_embeddings=self.node_embeddings
        )
        
        # Select edges based on sampling method
        generated_triples = []
        for b in range(batch_size):
            batch_scores = scores[b]
            num_candidates = batch_scores.size(0)
            
            if sampling_method == 'probabilistic':
                # Probabilistic sampling from multinomial distribution
                # Apply temperature scaling and convert to probabilities
                scaled_scores = batch_scores / temperature
                probs = torch.softmax(scaled_scores, dim=0)
                
                # Sample edges without replacement
                num_samples = min(max_edges, num_candidates)
                if num_samples > 0 and probs.sum() > 0:
                    # Handle edge case where all probs might be 0
                    sampled_indices = torch.multinomial(probs, num_samples, replacement=False)
                    triples = candidate_triples[b, sampled_indices].tolist()
                else:
                    triples = []
                    
            else:  # deterministic
                # Deterministic: select top-k edges
                k = min(max_edges, num_candidates)
                if k > 0:
                    _, top_indices = torch.topk(batch_scores, k)
                    triples = candidate_triples[b, top_indices].tolist()
                else:
                    triples = []
            
            # Pad if necessary
            while len(triples) < max_edges:
                triples.append([0, 0, 0])
            
            generated_triples.append(triples[:max_edges])
        
        return torch.tensor(generated_triples, device=device)
    
    def compute_loss(self, outputs, target_triples, target_nodes, mask=None):
        """
        Compute VAE loss including reconstruction and KL divergence.
        
        Args:
            outputs: Dictionary from forward pass
            target_triples: Ground truth triples [batch_size, max_edges, 3]
            target_nodes: Ground truth nodes [batch_size, max_nodes]
            mask: Optional padding mask
        
        Returns:
            Dictionary with loss components
        """
        device = target_triples.device
        batch_size = target_triples.size(0)
        
        # 1. KL Divergence Loss
        kl_loss = -0.5 * torch.sum(
            1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
        ) / batch_size
        
        # 2. Entity Prediction Loss
        entity_loss = F.cross_entropy(
            outputs['entity_predictions'].reshape(-1, outputs['entity_predictions'].size(-1)),
            target_nodes.reshape(-1),
            ignore_index=0,  # Ignore padding
            reduction='mean'
        )
        
        # 3. Graph Reconstruction Loss (RESCAL scores)
        # Since we're using sparse scoring, decoder_scores are [batch_size, num_edges]
        num_edges = target_triples.size(1)
        
        # Create target labels and mask
        target_labels = (target_triples[:, :, 1] != 0).float()  # 1 for real edges, 0 for padding
        edge_mask = (target_triples.sum(dim=-1) == 0)  # True for padding
        
        # Compute loss per edge
        edge_losses = F.binary_cross_entropy_with_logits(
            outputs['decoder_scores'],
            target_labels,
            reduction='none'
        )
        
        # Mask out padding positions
        edge_losses = edge_losses.masked_fill(edge_mask, 0)
        valid_edges = (~edge_mask).sum()
        edge_loss = edge_losses.sum() / valid_edges.clamp(min=1)
        
        # Total loss with configurable weights
        beta = self.config.get('beta', 1e-4)  # KL weight
        gamma = self.config.get('gamma', 1.0)  # Entity loss weight
        zeta = self.config.get('zeta', 1.0)   # Edge loss weight
        
        total_loss = zeta * edge_loss + gamma * entity_loss + beta * kl_loss
        
        return {
            'loss': total_loss,
            'edge_loss': edge_loss,
            'entity_loss': entity_loss,
            'kl_loss': kl_loss
        }
    
    def discretize_output(self, outputs, input_triples, threshold=0.5):
        """
        Convert continuous outputs to discrete graph structure.
        
        Args:
            outputs: Dictionary from forward pass
            input_triples: Original triples used for scoring [batch_size, max_edges, 3]
            threshold: Threshold for edge existence
        
        Returns:
            Dictionary with discretized outputs
        """
        # Discretize entity predictions
        entity_predictions = torch.argmax(outputs['entity_predictions'], dim=-1)
        
        # Discretize edge predictions using threshold
        # decoder_scores: [batch_size, num_edges] from sparse scoring
        edge_scores = torch.sigmoid(outputs['decoder_scores'])
        edge_predictions = (edge_scores > threshold).float()
        
        # Filter input triples based on predictions
        batch_size = edge_scores.size(0)
        discretized_triples = []
        
        for b in range(batch_size):
            triples = []
            for e in range(input_triples.size(1)):
                # Include edge if score exceeds threshold
                if edge_predictions[b, e] > 0:
                    s, r, o = input_triples[b, e]
                    if r != 0:  # Not padding
                        triples.append([s.item(), r.item(), o.item()])
            
            # Pad if necessary
            while len(triples) < self.config['max_edges']:
                triples.append([0, 0, 0])
            
            discretized_triples.append(triples[:self.config['max_edges']])
        
        return {
            'entity_predictions': entity_predictions,
            'edge_predictions': edge_predictions,
            'triples': torch.tensor(discretized_triples, device=edge_scores.device)
        }

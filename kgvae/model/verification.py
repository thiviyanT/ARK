import torch
from intelligraphs.verifier.synthetic import SynPathsVerifier, SynTIPRVerifier, SynTypesVerifier
from intelligraphs.verifier.wikidata import WDMoviesVerifier, WDArticlesVerifier


def get_verifier(dataset_name):
    """Returns the appropriate verifier for the given dataset."""
    verifiers = {
        "syn-paths": SynPathsVerifier(),
        "syn-tipr": SynTIPRVerifier(),
        "syn-types": SynTypesVerifier(),
        "wd-movies": WDMoviesVerifier(),
        "wd-articles": WDArticlesVerifier()
    }
    return verifiers.get(dataset_name)


def verify_generated_graphs(generated_triples, verifier, i2e, i2r):
    """
    Verify generated graphs using IntelliGraphs verifiers.
    
    Args:
        generated_triples: Tensor of generated triples [batch_size, num_triples, 3]
        verifier: IntelliGraphs verifier for the dataset
        i2e: Index to entity mapping
        i2r: Index to relation mapping
    
    Returns:
        Dictionary with verification statistics
    """
    # Transfer to CPU for verification (memory efficient)
    if generated_triples.is_cuda:
        generated_triples = generated_triples.cpu()
    
    batch_size = generated_triples.size(0)
    valid_count = 0
    invalid_reasons = []
    
    for i in range(batch_size):
        graph = generated_triples[i]
        
        # Remove padding (zeros)
        non_zero_mask = (graph != 0).any(dim=1)
        graph = graph[non_zero_mask]
        
        if len(graph) == 0:
            invalid_reasons.append("empty_graph")
            continue
        
        # Convert indices back to labels if needed
        labeled_graph = []
        for triple in graph:
            s, r, o = triple.tolist()
            # Skip if any component is padding
            if s >= len(i2e) or o >= len(i2e) or r >= len(i2r):
                continue
            labeled_graph.append([i2e[s], i2r[r], i2e[o]])
        
        if len(labeled_graph) == 0:
            invalid_reasons.append("all_padding")
            continue
        
        try:
            # Evaluate the graph - returns list of violations
            violations = verifier.evaluate_graph(labeled_graph)
            if len(violations) == 0:  # No violations means valid graph
                valid_count += 1
            else:
                # Extract violation reasons
                for violation_msg, _ in violations:
                    invalid_reasons.append(violation_msg)
        except Exception as e:
            invalid_reasons.append(f"verification_error: {str(e)}")
    
    validity_rate = valid_count / batch_size if batch_size > 0 else 0.0
    
    return {
        "valid_count": valid_count,
        "total_count": batch_size,
        "validity_rate": validity_rate,
        "invalid_reasons": invalid_reasons
    }


def sample_and_verify(model, config, verifier, i2e, i2r, device, num_samples=100):
    """
    Sample graphs from the model and verify them.
    
    Args:
        model: The KGVAE model
        config: Configuration dictionary
        verifier: IntelliGraphs verifier for the dataset
        i2e: Index to entity mapping
        i2r: Index to relation mapping
        device: PyTorch device
        num_samples: Number of samples to generate
    
    Returns:
        Dictionary with sampling and verification results
    """
    model.eval()
    
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, config['d_latent'], device=device)
        
        # Generate graphs using the model's sample method
        generated_triples = model.sample(num_samples, device)
        
        # Transfer to CPU for verification
        if generated_triples.is_cuda:
            generated_triples = generated_triples.cpu()
        
        # Verify the generated graphs on CPU
        verification_results = verify_generated_graphs(
            generated_triples, verifier, i2e, i2r
        )
    
    return verification_results

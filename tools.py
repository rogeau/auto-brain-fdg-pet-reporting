import torch.nn.functional as F
import torch
import math

def top_p_sampling(logits, p=0.9, temperature=1.0):
    logits = logits / temperature  # Optional: control randomness
    probs = F.softmax(logits, dim=-1)

    # Sort the probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above p
    cutoff = cumulative_probs > p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = 0

    # Mask out the tokens that are not in the nucleus
    sorted_probs[cutoff] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()  # Re-normalize

    # Sample from the filtered distribution
    next_token = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(-1, next_token).item()

def get_sampling_ratio(current_epoch, total_epochs, start_epoch=10, min_ratio=0.0, max_ratio=1.0, decay_rate=5.0):
    if current_epoch < start_epoch:
        return max_ratio  # Full teacher forcing before decay starts
    else:
        # Normalize progress *after* start_epoch to [0,1]
        progress = (current_epoch - start_epoch) / (total_epochs - start_epoch)
        ratio = min_ratio + (max_ratio - min_ratio) * math.exp(-decay_rate * progress)
        # Clamp for safety
        return max(min(ratio, max_ratio), min_ratio)
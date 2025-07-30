import torch.nn.functional as F
import torch
import math

def make_causal_mask(L, device):
    return torch.triu(torch.full((L, L), float("-inf"), device=device), 1)

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
    

def beam_search(model, image, tokenizer, BOS, EOS, seq_len, beam_width=5, device='cuda'):
    beams = [(torch.tensor([BOS], device=device), 0.0)]  # (tokens, score)

    for _ in range(seq_len):
        new_beams = []
        for tokens, score in beams:
            if tokens[-1] == EOS:
                new_beams.append((tokens, score))
                continue

            input_tensor = tokens.unsqueeze(0)  # [1, T]
            mask = make_causal_mask(input_tensor.size(1), device)
            logits = model(image.unsqueeze(0), input_tensor, mask)  # [1, T, vocab_size]
            logits = logits[:, -1, :]  # Last token logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)  # [vocab_size]

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

            for log_prob, idx in zip(topk_log_probs, topk_indices):
                new_seq = torch.cat([tokens, idx.unsqueeze(0)])
                new_score = score + log_prob.item()
                new_beams.append((new_seq, new_score))

        # Keep top-k beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        # Early stopping: if all beams end in EOS
        if all(tokens[-1] == EOS for tokens, _ in beams):
            break

    best_beam = beams[0][0]  # Tokens with highest score
    return tokenizer.decode(best_beam.tolist(), skip_special_tokens=True)

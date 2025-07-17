from tokenizers import Tokenizer
import json

def sequence_length(codebook_path, reports_json):
    token_sequences = []
    tokenizer = Tokenizer.from_file(codebook_path)

    with open(reports_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        text = entry.get("PET_results", "").strip()
        if not text:
            continue

        tokened_seq = tokenizer.encode(text)
        token_sequences.append(tokened_seq.ids)  # Store list of token IDs

    # Compute sequence lengths
    lengths = [len(seq) for seq in token_sequences]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        max_len = max(lengths)
        print(f"Average tokenized length: {avg_len:.2f}")
        print(f"Maximum tokenized length: {max_len}")
    else:
        print("No valid sequences found.")

    return token_sequences
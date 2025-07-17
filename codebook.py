from tokenizers import Tokenizer, trainers, models, pre_tokenizers, normalizers
from tokenizers.pre_tokenizers import Whitespace
import json

input_file = "dataset/reports.json"
output_file = "dataset/corpus.txt"
codebook = "dataset/codebook.json"

def make_corpus(reports_path, corpus_path):
    with open(reports_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(corpus_path, "w", encoding="utf-8") as f_out:
        for item in data:
            report = item.get("PET_results", "").strip()
            if report:
                f_out.write(report.replace("\n", " ") + "\n")

def make_codebook(corpus_path, codebook_path):
    tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(
        vocab_size=10000,
        min_frequency=1,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
    )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(codebook_path)

def main():
    make_corpus(input_file, output_file)
    make_codebook(output_file, codebook)

if __name__ == "__main__":
    main()
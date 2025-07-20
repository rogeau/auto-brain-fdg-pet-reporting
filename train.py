from dataset import RawDataset, TransformedDataset, filenames, transform
from tokenizers import Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from ViT import ReportingModel
from tqdm import tqdm
import torch.nn.functional as F
from tools import top_p_sampling

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
codebook_path = "dataset/codebook.json"
img_size = 64
patch_size = 8
in_channels = 1
embed_dim = 1152
depth_enc = 12
depth_dec = 12
num_heads = 12
vocab_size = 2300
batch_size = 6
seq_len = 150
num_epochs = 20


# Model
model = ReportingModel(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    embed_dim=embed_dim,
    depth_enc=depth_enc,
    depth_dec=depth_dec,
    num_heads=num_heads,
    vocab_size=vocab_size,
    seq_len=seq_len
).to(device)

# state_dict = torch.load("model_epoch_6.pth")
# model.load_state_dict(state_dict)

# Tokenizer
tokenizer = Tokenizer.from_file(codebook_path)

BOS = tokenizer.token_to_id("<s>")
EOS = tokenizer.token_to_id("</s>")
PAD = tokenizer.token_to_id("<pad>")

# Dataset and Splits
ds = RawDataset(
    data_root="dataset/",
    json_path="dataset/reports.json",
    filenames=filenames,
    target_shape=img_size
)

# torch.manual_seed(40)

# val_len = int(0.1 * len(ds))
# train_len = len(ds) - val_len
# train_set, val_set = torch.utils.data.random_split(ds, [train_len, val_len])

torch.manual_seed(35)
val_len = int(0.1 * len(ds))
train_len = len(ds) - val_len
train_set_base, val_set_base = torch.utils.data.random_split(ds, [train_len, val_len])

train_set = TransformedDataset(train_set_base, transform=transform)
val_set = TransformedDataset(val_set_base, transform=None)


# Causal Mask
def make_causal_mask(L, device):
    return torch.triu(torch.full((L, L), float("-inf"), device=device), 1)

# Collate Function
def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs)  # [B, 1, D, H, W]

    token_lists = []
    for c in caps:
        ids = tokenizer.encode(c).ids
        ids = [tokenizer.token_to_id("<s>")] + ids + [tokenizer.token_to_id("</s>")]
        token_lists.append(torch.tensor(ids, dtype=torch.long))

    tgt = torch.full((len(token_lists), seq_len), PAD, dtype=torch.long)
    for i, ids in enumerate(token_lists):
        tgt[i, :ids.size(0)] = ids

    tgt_mask = make_causal_mask(seq_len, tgt.device)
    return imgs, tgt, tgt_mask

# Dataloader
loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
loader_val = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

val_imgs, val_tgt, val_mask = next(iter(loader_val))
val_imgs, val_tgt, val_mask = (
    val_imgs.to(device),
    val_tgt.to(device),
    val_mask.to(device)
)

# Optimizer and Scaler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler("cuda")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    loader = tqdm(loader_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for step, (imgs, tgt, tgt_mask) in enumerate(loader):
        imgs = imgs.to(device)
        tgt = tgt.to(device)
        tgt_mask = tgt_mask.to(device)

        in_tokens = tgt[:, :-1]
        gold = tgt[:, 1:]
        gold_mask = gold != PAD

        optimizer.zero_grad()

        with autocast("cuda"):
            logits = model(imgs, in_tokens, tgt_mask[:-1, :-1])  # [B, T-1, V]
            loss = nn.functional.cross_entropy(
                logits.transpose(1, 2), gold, ignore_index=PAD
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loader.set_postfix(train_loss=loss.item())

        if step % 10 == 0:
            model.eval()
            with torch.no_grad(), autocast("cuda"):
                val_imgs, val_tgt, val_mask = next(iter(loader_val))
                val_imgs = val_imgs.to(device)
                val_tgt = val_tgt.to(device)
                val_mask = val_mask.to(device)

                v_logits = model(val_imgs, val_tgt[:, :-1], val_mask[:-1, :-1])
                v_loss = nn.functional.cross_entropy(
                    v_logits.transpose(1, 2),
                    val_tgt[:, 1:],
                    ignore_index=PAD
                )

                # ---- ground‑truth caption ----
                gt_caption = tokenizer.decode(
                    [tid for tid in val_tgt[0, 1:].tolist()
                     if tid not in {PAD, EOS}],
                    skip_special_tokens=True
                )

                # deterministic prediction
                # pred_ids = v_logits.argmax(dim=-1)[0].tolist()
                
                # probablistic prediction
                # probs = F.softmax(v_logits, dim=-1)
                # pred_ids = torch.multinomial(probs[0], 1).squeeze(1).tolist()

                # trimmed = []
                # for tid in pred_ids:
                #     if tid in {PAD, EOS}:
                #         break
                #     trimmed.append(tid)


                # generated = [BOS]
                # for _ in range(seq_len):
                #     input_tensor = torch.tensor(generated, device=device).unsqueeze(0)  # [1, T]
                #     mask = make_causal_mask(input_tensor.size(1), device)
                #     logits = model(val_imgs[0].unsqueeze(0), input_tensor, mask)
                #     next_token = logits.argmax(dim=-1)[:, -1].item()
                #     if next_token == EOS:
                #         break
                #     generated.append(next_token)


                # pred_caption = tokenizer.decode(generated, skip_special_tokens=True)

                generated = [BOS]
                for _ in range(seq_len):
                    input_tensor = torch.tensor(generated, device=device).unsqueeze(0)  # [1, T]
                    mask = make_causal_mask(input_tensor.size(1), device)
                    logits = model(val_imgs[0].unsqueeze(0), input_tensor, mask)
                    logits = logits[:, -1, :]  # Take logits for last token

                    next_token = top_p_sampling(logits.squeeze(0), p=0.9, temperature=1.0)

                    if next_token == EOS:
                        break
                    generated.append(next_token)

                pred_caption = tokenizer.decode(generated, skip_special_tokens=True)


                print(f"\nEpoch {epoch+1} | Step {step:4d}")
                print(f"Train loss: {loss.item():.4f} | Val loss: {v_loss.item():.4f}")
                print("GT :", gt_caption)
                print("Pred:", pred_caption)
            model.train()

    print(f"[Epoch {epoch+1}] Average train loss: {total_loss / len(loader_train):.4f}")
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
from dataset import RawDataset, TransformedDataset, transform
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from MAE import PretrainingModel, visualize_mask_reconstruction
from tqdm import tqdm
from configs import *
import matplotlib.pyplot as plt
import os

# Model
model = PretrainingModel(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    encoder_embed_dim=encoder_embed_dim,
    decoder_embed_dim=mask_decoder_embed_dim,
    depth=depth_enc,
    num_heads=num_heads
).to(device)


# Dataset and Splits
torch.manual_seed(35)

ds = RawDataset(
    data_root="adni_nifd/",
    target_shape=img_size,
    json_path=None,
    filenames=None,
    pretraining=True
)

transformed_ds = TransformedDataset(ds, transform=transform, pretraining=True)
loader = DataLoader(transformed_ds, batch_size=pretrain_batch_size, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_lr, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_pretraining)
scaler = GradScaler("cuda")

resume_path = 'pretraining_checkpoints/checkpoint_epoch190.pth'
if os.path.exists(resume_path):
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0


for epoch in range(start_epoch, num_epochs_pretraining):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs_pretraining}", leave=False)

    for step, imgs in enumerate(pbar):
        imgs = imgs.to(device)
        optimizer.zero_grad()

        with autocast("cuda"):
            loss, reconstructed, ids_shuffle = model(imgs, mask_ratio)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(train_loss=loss.item())

        if step % 20 == 0:
            model.eval()
            val_imgs = next(iter(loader)).to(device)
            with torch.no_grad(), autocast("cuda"):
                os.makedirs("imgs_pretraining/", exist_ok=True)
                visualize_mask_reconstruction(model, val_imgs, path=f"imgs_pretraining/epoch{epoch}_step{step}.png", mask_ratio=mask_ratio)
            model.train()

    scheduler.step()
    with open("pretraining.log", "a") as f:
        f.write(f"[Epoch {epoch+1}] Average train loss: {total_loss / len(loader):.4f}\n")
        f.flush()
    
    if (epoch + 1) % 10 == 0:
        checkpoint_dir = 'pretraining_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch + 1}.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, checkpoint_path)
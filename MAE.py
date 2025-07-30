import torch
import torch.nn as nn
from ViT import ViTEncoder3D, TransformerBlock
import matplotlib.pyplot as plt

# class MaskedDecoder(nn.Module):
#     def __init__(self, embed_dim, num_patches, patch_dim):
#         super().__init__()
#         self.reconstruct = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.GELU(),
#             nn.LayerNorm(embed_dim)
#         )
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         self.decoder_pred = nn.Linear(embed_dim, patch_dim)

#     def forward(self, encoded_visible, ids_restore):
#         B, N_vis, D = encoded_visible.shape
#         N_total = ids_restore.shape[1]

#         # Restore masked positions
#         full_tokens = torch.zeros(B, N_total, D, device=encoded_visible.device)
#         full_tokens.scatter_(1, ids_restore[:, :N_vis].unsqueeze(-1).expand(-1, -1, D), encoded_visible)

#         # Add positional encoding
#         full_tokens = full_tokens + self.pos_embed
#         decoded = self.reconstruct(full_tokens)
#         pixel_preds = self.decoder_pred(decoded)
#         return pixel_preds


class MaskedDecoder(nn.Module):
    def __init__(self, num_patches, patch_dim, mask_dec_embed_dim=512, decoder_depth=4, decoder_heads=4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, mask_dec_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, mask_dec_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.decoder_blocks = nn.Sequential(
                            *[TransformerBlock(mask_dec_embed_dim, num_heads=decoder_heads, mlp_ratio=4.0, dropout=0)
                            for _ in range(decoder_depth)]
                        )
        self.norm = nn.LayerNorm(mask_dec_embed_dim)
        self.decoder_pred = nn.Linear(mask_dec_embed_dim, patch_dim)

    def forward(self, encoded_visible, ids_restore):
        B, N_vis, D = encoded_visible.shape
        N_total = ids_restore.shape[1]

        # Prepare mask tokens
        mask_tokens = self.mask_token.expand(B, N_total - N_vis, D).to(encoded_visible.dtype)

        # Reconstruct full sequence (visible + masked tokens)
        all_tokens = torch.cat([encoded_visible, mask_tokens], dim=1)  # [B, N, D]
        full_tokens = torch.gather(all_tokens, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))  # [B, N_total, D]
        full_tokens = full_tokens + self.pos_embed

        full_tokens = self.decoder_blocks(full_tokens)

        full_tokens = self.norm(full_tokens)
        pixel_preds = self.decoder_pred(full_tokens)

        return pixel_preds

def mae_loss(reconstructed_pixels, original_pixels, ids_keep):
    B, N, D = reconstructed_pixels.shape
    mask = torch.ones_like(reconstructed_pixels, dtype=torch.bool)
    mask.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, D), False)
    return ((reconstructed_pixels - original_pixels) ** 2)[mask].mean()

class PretrainingModel(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1,
                 encoder_embed_dim=1152, decoder_embed_dim=512, depth=6, num_heads=8):
        super().__init__()
        self.encoder = ViTEncoder3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_embed_dim,
            depth=depth,
            num_heads=num_heads
        )
        self.num_patches = (img_size // patch_size) ** 3
        patch_dim = (patch_size ** 3) * in_channels
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.decoder = MaskedDecoder(self.num_patches, patch_dim, decoder_embed_dim)

    def forward(self, x, mask_ratio=0.6):
        x_patched_pixels = self.encoder.patch_embed.patchify(x)  # [B, N, D]
        x_patched = self.encoder.patch_embed.proj(x_patched_pixels)
        B, N, embed_dim = x_patched.shape

        # Random masking
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_visible = torch.gather(x_patched, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, embed_dim))
        pos_embed = self.encoder.pos_embed.expand(B, -1, -1)
        pos_embed_visible = torch.gather(pos_embed, 1, ids_keep.unsqueeze(-1).expand(-1, -1, pos_embed.shape[2]))  # [B, N_vis, D]
        x_visible = x_visible + pos_embed_visible
        
        encoded = self.encoder.blocks(x_visible)
        encoded = self.encoder.norm(encoded)
        encoded = self.encoder_to_decoder(encoded)
        reconstructed = self.decoder(encoded, ids_restore)
        loss = mae_loss(reconstructed, x_patched_pixels, ids_keep)
        return loss, reconstructed, ids_shuffle
    

def visualize_mask_reconstruction(model, x, path, mask_ratio=0.6):
    """
    x: input volume tensor, shape [B, C, H, W, D]
    model: instance of PretrainingModel
    """

    model.eval()
    with torch.no_grad():
        # Forward pass returns loss, reconstructed patch embeddings, ids_shuffle
        loss, reconstructed, ids_shuffle = model(x, mask_ratio=mask_ratio)

    B, C, H, W, D = x.shape
    patch_size = model.encoder.patch_size
    num_patches_per_dim = H // patch_size  # assuming cubic volume and equal dims

    # Step 1: Get patchified version of original input (flattened patches)
    x_patched = model.encoder.patch_embed.patchify(x)  # [B, N, D]

    N = x_patched.shape[1]

    # Step 2: Determine visible patch indices
    len_keep = int(N * (1 - mask_ratio))
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones((B, N), dtype=torch.bool, device=x.device)
    mask.scatter_(1, ids_keep, False)  # True for masked patches

    # Step 3: Create masked version of input volume by zeroing masked patches
    # First unpatchify original patches:
    original_vol = model.encoder.patch_embed.unpatchify(x_patched)  # [B, C, H, W, D]

    # Create masked patched tensor: zero out masked patches
    masked_patched = x_patched.clone()
    masked_patched[mask] = 0

    masked_vol = model.encoder.patch_embed.unpatchify(masked_patched)

    # Step 4: Reconstruct full volume from decoder output (embeddings -> patches -> volume)
    reconstructed_vol = model.encoder.patch_embed.unpatchify(reconstructed)

    combined_patched = x_patched.clone()
    combined_patched[mask] = reconstructed[mask].to(x_patched.dtype)
    combined_vol = model.encoder.patch_embed.unpatchify(combined_patched)

    # Step 5: Extract middle slice along the last dimension (D)
    mid_slice = D // 2

    # Plot original, masked, and reconstructed slices for batch index 0
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    axs[0].imshow(original_vol[0, 0, :, :, mid_slice].cpu(), cmap='gray')
    axs[0].set_title("Original Middle Slice")
    axs[0].axis('off')

    axs[1].imshow(masked_vol[0, 0, :, :, mid_slice].cpu(), cmap='gray')
    axs[1].set_title("Masked Middle Slice")
    axs[1].axis('off')

    axs[2].imshow(reconstructed_vol[0, 0, :, :, mid_slice].cpu(), cmap='gray')
    axs[2].set_title("Reconstructed Middle Slice")
    axs[2].axis('off')

    axs[3].imshow(combined_vol[0, 0, :, :, mid_slice].cpu(), cmap='gray')
    axs[3].set_title("Original + Reconstructed Patches")
    axs[3].axis('off')

    plt.savefig(path)
    plt.close(fig)
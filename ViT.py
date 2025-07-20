import torch
import torch.nn as nn
import time

class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels=1, patch_size=8, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # Shape: [B, embed_dim, D/P, H/P, W/P]
        x = x.flatten(2)  # Flatten spatial dims
        x = x.transpose(1, 2)  # Shape: [B, N_patches, embed_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder3D(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=1,
                 embed_dim=128, depth=6, num_heads=4, fused=False):
        super().__init__()
        num_patches = (img_size // patch_size) ** 3
        self.patch_embed = PatchEmbed3D(in_channels, patch_size, embed_dim)
        self.pos_embed = PositionalEncoding(num_patches, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.depth = depth
        self.fused = fused
        self.fusion_weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        if self.fused:
            features = []
            checkpoints = [int(self.depth / 3),
                           int(2 * self.depth / 3),
                           self.depth]

            for i, block in enumerate(self.blocks):
                x = block(x)
                if (i + 1) in checkpoints:
                    features.append(self.norm(x))

            if len(features) != 3:
                raise ValueError("Did not collect exactly 3 features for fusion.")

            w = torch.softmax(self.fusion_weights, dim=0)
            fused = sum(wi * fi for wi, fi in zip(w, features))
            return fused

        else:
            x = self.blocks(x)
            return self.norm(x)



class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, tgt, memory, attn_mask=None):
        # tgt: [B, 1, embed_dim] — current predicted token
        # memory: [B, N, embed_dim] — full encoder output (e.g., patch tokens from ViT)

        # 1. Self-attention on target (typically just 1 token at inference)
        tgt2 = self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt), attn_mask=attn_mask)[0]
        tgt = tgt + tgt2 * 0.3

        # 2. Cross-attention over encoder memory
        tgt2 = self.cross_attn(self.norm2(tgt), self.norm2(memory), self.norm2(memory))[0]
        tgt = tgt + tgt2 * 2

        # 3. Feedforward
        tgt = tgt + self.mlp(self.norm3(tgt))

        return tgt

class Decoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, num_heads=4, depth=2, seq_len=150, mlp_ratio=4.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim)) 
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)  # output logits for next token

    def forward(self, decoder_input_ids, encoder_output, caption_mask=None):
        x = self.token_embedding(decoder_input_ids)
        x = x + self.pos_embedding[:, :x.size(1)] 
        for layer in self.layers:
            x = layer(x, encoder_output, caption_mask)
        x = self.norm(x)
        logits = self.head(x)  # [B, 1, vocab_size]
        return logits

class ReportingModel(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_channels=1,
                 embed_dim=128, depth_enc=6, depth_dec=6, num_heads=4, vocab_size=2300, seq_len=150):
        super().__init__()
        self.encoder = ViTEncoder3D(img_size, patch_size, in_channels, embed_dim, depth_enc, num_heads)
        self.decoder = Decoder(embed_dim, vocab_size, num_heads, depth_dec, seq_len)

    def forward(self, images, captions, caption_mask=None):
        encoder_out = self.encoder(images)  # [B, N_patches, embed_dim]
        output_logits = self.decoder(captions, encoder_out, caption_mask)
        return output_logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration
    img_size = 64
    patch_size = 8
    in_channels = 1
    embed_dim = 256
    depth_enc = 6
    depth_dec = 6
    num_heads = 8
    vocab_size = 2300
    batch_size = 6
    seq_len = 50

    x = torch.randn(batch_size, in_channels, img_size, img_size, img_size).to(device)
    captions = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len)).to(device)

    model = ReportingModel(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth_enc=depth_enc,
        depth_dec=depth_dec,
        num_heads=num_heads,
        vocab_size=vocab_size
    ).to(device)

    # Forward pass with timing
    torch.cuda.empty_cache()
    start = time.time()
    out = model(x, captions)
    end = time.time()

    print(f"Output shape: {out.shape}  # [B, T, vocab_size]")
    print(f"Forward pass time: {end - start:.3f}s")

    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"GPU memory allocated: {mem_mb:.2f} MB")

if __name__ == "__main__":
    main()
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

codebook_path = "dataset/codebook.json"
checkpoint_pretraining = "pretraining_checkpoints/checkpoint_epoch190.pth"
img_size = 64
patch_size = 8
in_channels = 1
encoder_embed_dim = 1152
depth_enc = 12
depth_dec = 12
num_heads = 12
vocab_size = 2300
batch_size = 6
seq_len = 150
num_epochs = 200
fused=True
freeze_epochs = 3


# pretraining
num_epochs_pretraining = 200
pretrain_batch_size = 32
mask_decoder_embed_dim = 256
mask_ratio = 0.75
pretrain_lr=4e-4
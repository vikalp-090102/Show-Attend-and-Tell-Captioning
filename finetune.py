import os
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from model import EncoderCNN, DecoderRNN
from dataset import ImageCaptionDataset
from utils import collate_fn

# -------------------------
# Argument Parsing
# -------------------------
parser = argparse.ArgumentParser(description="Fine-tune image captioning model on Indiana dataset")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint .pth file")
parser.add_argument("--data_path", type=str, default="./data", help="Path to Indiana dataset")
parser.add_argument("--network", choices=["vgg19", "resnet152", "densenet161"], default="vgg19", help="Network to use in the encoder")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=60, help="Number of fine-tuning epochs")
args = parser.parse_args()

# -------------------------
# Load checkpoint
# -------------------------
checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
word_dict = checkpoint['word_dict']

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Models
# -------------------------
encoder = EncoderCNN(arch=args.network)
decoder = DecoderRNN(
    attention_dim=512,
    embed_dim=256,
    decoder_dim=512,
    vocab_size=len(word_dict)
)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder = encoder.to(device)
decoder = decoder.to(device)

# -------------------------
# Data transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# -------------------------
# Dataset and Dataloader
# -------------------------
dataset = ImageCaptionDataset(
    transform=transform,
    data_path=args.data_path,
    split_type='train',
    word2idx=word_dict
)

train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

# -------------------------
# Loss and Optimizer
# -------------------------
criterion = nn.CrossEntropyLoss(ignore_index=word_dict['<pad>'])
params = list(decoder.parameters()) + list(encoder.parameters())
optimizer = optim.Adam(params, lr=1e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# -------------------------
# Fine-tuning Loop
# -------------------------
start_epoch = checkpoint['epoch'] + 1
end_epoch = start_epoch + args.num_epochs

for epoch in range(start_epoch, end_epoch):
    encoder.train()
    decoder.train()
    total_loss = 0.0

    for imgs, caps in train_loader:
        imgs = imgs.to(device)
        caps = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(cap) for cap in caps],
            batch_first=True,
            padding_value=word_dict['<pad>']
        ).to(device)

        optimizer.zero_grad()

        encoder_out = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoder_out, caps)

        targets = caps_sorted[:, 1:]
        scores = scores[:, :max(decode_lengths), :].contiguous().view(-1, scores.size(-1))
        targets = targets[:, :max(decode_lengths)].contiguous().view(-1)

        loss = criterion(scores, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch}/{end_epoch-1}], Loss: {total_loss:.4f}")

    # Save fine-tuned checkpoint
    save_path = f"model_checkpoint_{args.network}_epoch{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'word_dict': word_dict
    }, save_path)

print("Finetuning completed and checkpoints saved.")

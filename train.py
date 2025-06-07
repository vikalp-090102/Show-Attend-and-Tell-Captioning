import argparse
import json
import os
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import ImageCaptionDataset
from decoder import Decoder
from encoder import Encoder
from utils import AverageMeter, accuracy, calculate_caption_lengths

# -----------------------------------------------------------------------------
# Collate functions to handle variable-length caption sequences.
# -----------------------------------------------------------------------------
def train_collate_fn(batch):
    """
    Expects a list of tuples (img_tensor, cap_ids) where cap_ids is a list of integers.
    Returns:
      - images: Tensor of shape (B, C, H, W)
      - padded_caps: Tensor of shape (B, max_cap_len)
      - lengths: List of actual caption lengths
    """
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, caps = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in caps]
    max_len = lengths[0]
    padded = torch.zeros(len(caps), max_len, dtype=torch.long)
    for i, cap in enumerate(caps):
        padded[i, :lengths[i]] = torch.tensor(cap, dtype=torch.long)
    return images, padded, lengths

def val_collate_fn(batch):
    """
    Expects a list of tuples (img_tensor, cap_ids, all_ref_caps)
    Returns:
      - images, padded_caps, lengths, and all_ref_caps (as a nested list)
    """
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, caps, all_caps = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in caps]
    max_len = lengths[0]
    padded = torch.zeros(len(caps), max_len, dtype=torch.long)
    for i, cap in enumerate(caps):
        padded[i, :lengths[i]] = torch.tensor(cap, dtype=torch.long)
    return images, padded, lengths, list(all_caps)

# -----------------------------------------------------------------------------
# Main Training/Validation Function
# -----------------------------------------------------------------------------
def main(args):
    writer = SummaryWriter()

    # --- Vocabulary Handling ---
    # Instead of saving to the (read-only) input folder, we use the working directory.
    word_dict_path = "word_dict.json"
    if os.path.exists(word_dict_path):
        word_dict = json.load(open(word_dict_path, "r"))
    else:
        print("word_dict.json not found. Generating vocabulary from training dataset...")
        # Create a temporary training dataset instance without supplying word2idx.
        train_temp_ds = ImageCaptionDataset(transform=data_transforms,
                                            data_path=args.data,
                                            split_type="train")
        counter = Counter()
        for cap in train_temp_ds.captions:
            # Captions here are raw strings (since no vocabulary is provided yet)
            # counter.update(cap.lower().split())
            if isinstance(cap, str):
                counter.update(cap.lower().split())
            elif isinstance(cap, list):
                counter.update(cap)

        tokens = list(counter.keys())
        # Reserve special tokens; adjust these as needed.
        vocab = ["<pad>", "<start>", "<end>", "<unk>"] + sorted(tokens)
        word_dict = {w: i for i, w in enumerate(vocab)}
        with open(word_dict_path, "w") as f:
            json.dump(word_dict, f)
        print(f"Generated vocabulary with {len(word_dict)} tokens and saved to {word_dict_path}")

    vocab_size = len(word_dict)

    # --- Build the Encoder and Decoder ---
    encoder = Encoder(args.network).cuda()
    decoder = Decoder(vocab_size, encoder.dim, args.tf).cuda()

    if args.model:
        decoder.load_state_dict(torch.load(args.model))

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)
    criterion = nn.CrossEntropyLoss().cuda()

    # --- Create the Datasets and DataLoaders ---
    # The dataset uses the provided CSVs (indiana_projections.csv and indiana_reports.csv)
    # and images from the folder: images/images_normalized/
    train_ds = ImageCaptionDataset(transform=data_transforms,
                                   data_path=args.data,
                                   split_type="train",
                                   word2idx=word_dict)
    val_ds = ImageCaptionDataset(transform=data_transforms,
                                 data_path=args.data,
                                 split_type="val",
                                 word2idx=word_dict)

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=train_collate_fn)
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=val_collate_fn)

    print("Starting training with:", args)
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        train(epoch, encoder, decoder, optimizer, criterion,
              train_loader, word_dict, args.alpha_c, args.log_interval, writer)
        validate(epoch, encoder, decoder, criterion,
                 val_loader, word_dict, args.alpha_c, args.log_interval, writer)
        os.makedirs("model", exist_ok=True)
        model_file = f"model/model_{args.network}_{epoch}.pth"
        torch.save(decoder.state_dict(), model_file)
        print("Saved model to", model_file)

    writer.close()

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train(epoch, encoder, decoder, optimizer, criterion, data_loader, word_dict, alpha_c, log_interval, writer):
    # Encoder remains in eval mode, assuming it is pre-trained.
    encoder.eval()
    decoder.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for bi, (imgs, caps, lengths) in enumerate(data_loader):
        imgs = imgs.cuda()
        caps = caps.cuda()

        with torch.no_grad():
            img_feats = encoder(imgs)

        optimizer.zero_grad()
        preds, alphas = decoder(img_feats, caps)
        targets = caps[:, 1:]
        pack_lens = [l - 1 for l in lengths]
        preds_p = pack_padded_sequence(preds, pack_lens, batch_first=True)[0]
        targ_p = pack_padded_sequence(targets, pack_lens, batch_first=True)[0]
        att_reg = alpha_c * ((1 - alphas.sum(1))**2).mean()
        loss = criterion(preds_p, targ_p) + att_reg
        loss.backward()
        optimizer.step()

        total_len = calculate_caption_lengths(word_dict, caps)
        a1 = accuracy(preds_p, targ_p, 1)
        a5 = accuracy(preds_p, targ_p, 5)
        losses.update(loss.item(), total_len)
        top1.update(a1, total_len)
        top5.update(a5, total_len)

        if bi % log_interval == 0:
            print(f"Train Epoch {epoch} [{bi}/{len(data_loader)}]  Loss {losses.val:.4f} ({losses.avg:.4f})  " +
                  f"Top1 {top1.val:.3f} ({top1.avg:.3f})  Top5 {top5.val:.3f} ({top5.avg:.3f})")
    writer.add_scalar("train_loss", losses.avg, epoch)
    writer.add_scalar("train_top1", top1.avg, epoch)
    writer.add_scalar("train_top5", top5.avg, epoch)

# -----------------------------------------------------------------------------
# Validation Loop (with BLEU score calculation)
# -----------------------------------------------------------------------------
def validate(epoch, encoder, decoder, criterion, data_loader, word_dict, alpha_c, log_interval, writer):
    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    references = []
    hypotheses = []

    with torch.no_grad():
        for bi, (imgs, caps, lengths, all_caps) in enumerate(data_loader):
            imgs = imgs.cuda()
            caps = caps.cuda()

            img_feats = encoder(imgs)
            preds, alphas = decoder(img_feats, caps)
            targets = caps[:, 1:]
            pack_lens = [l - 1 for l in lengths]
            preds_p = pack_padded_sequence(preds, pack_lens, batch_first=True)[0]
            targ_p = pack_padded_sequence(targets, pack_lens, batch_first=True)[0]
            att_reg = alpha_c * ((1 - alphas.sum(1))**2).mean()
            loss = criterion(preds_p, targ_p) + att_reg

            total_len = calculate_caption_lengths(word_dict, caps)
            a1 = accuracy(preds_p, targ_p, 1)
            a5 = accuracy(preds_p, targ_p, 5)
            losses.update(loss.item(), total_len)
            top1.update(a1, total_len)
            top5.update(a5, total_len)

            # Build reference captions list (each image may have multiple reference captions)
            for ref_set in all_caps:
                refs = [
                    [w for w in ref if w not in (word_dict["<start>"], word_dict["<pad>"])]
                    for ref in ref_set
                ]
                references.append(refs)

            # Build hypothesis captions from the model predictions (using argmax)
            pred_idxs = torch.argmax(preds, dim=2).cpu().tolist()
            for seq in pred_idxs:
                hyp = [w for w in seq if w not in (word_dict["<start>"], word_dict["<pad>"])]
                hypotheses.append(hyp)

            if bi % log_interval == 0:
                print(f"Val Epoch {epoch} [{bi}/{len(data_loader)}]  Loss {losses.val:.4f} ({losses.avg:.4f})  " +
                      f"Top1 {top1.val:.3f} ({top1.avg:.3f})  Top5 {top5.val:.3f} ({top5.avg:.3f})")
    writer.add_scalar("val_loss", losses.avg, epoch)
    writer.add_scalar("val_top1", top1.avg, epoch)
    writer.add_scalar("val_top5", top5.avg, epoch)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses)
    writer.add_scalar("val_bleu1", bleu1, epoch)
    writer.add_scalar("val_bleu2", bleu2, epoch)
    writer.add_scalar("val_bleu3", bleu3, epoch)
    writer.add_scalar("val_bleu4", bleu4, epoch)
    print(f" * BLEU@1 {bleu1:.4f}   BLEU@2 {bleu2:.4f}   BLEU@3 {bleu3:.4f}   BLEU@4 {bleu4:.4f}")

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show, Attend and Tell")
    parser.add_argument("--batch-size", type=int, default=64, help="mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--step-size", type=int, default=5, help="step size for learning rate reduction")
    parser.add_argument("--alpha-c", type=float, default=1.0, help="regularization constant")
    parser.add_argument("--log-interval", type=int, default=100, help="logging interval")
    # For the Indiana dataset, --data should point to:
    #   /kaggle/input/chest-xrays-indiana-university
    parser.add_argument("--data", type=str,
                        default="/kaggle/input/chest-xrays-indiana-university",
                        help="path to data directory")
    parser.add_argument("--network", choices=["vgg19", "resnet152", "densenet161"],
                        default="vgg19", help="Network to use in the encoder")
    parser.add_argument("--model", type=str, help="path to decoder checkpoint")
    parser.add_argument("--tf", action="store_true", default=False, help="use teacher forcing")
    args = parser.parse_args()

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    main(args)

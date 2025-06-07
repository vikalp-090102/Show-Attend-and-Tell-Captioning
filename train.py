import argparse, json, os, re
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
# Collate functions to pad variable‐length captions in a batch
# -----------------------------------------------------------------------------
def train_collate_fn(batch):
    """
    batch: list of (img_tensor, cap_ids : torch.LongTensor)
    returns: images (B,C,H,W), padded_caps (B, max_len), lengths (list[int])
    """
    # sort by descending length
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, caps = zip(*batch)

    images = torch.stack(images, 0)
    lengths = [len(c) for c in caps]
    max_len = lengths[0]

    padded = torch.zeros(len(caps), max_len, dtype=torch.long)
    for i, cap in enumerate(caps):
        end = lengths[i]
        padded[i, :end] = cap[:end]

    return images, padded, lengths

def val_collate_fn(batch):
    """
    batch: list of (img_tensor, cap_ids : LongTensor, all_ref_caps : List[List[int]])
    returns: images, padded_caps, lengths, all_caps (nested list)
    """
    # sort by descending length of primary cap
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, caps, all_caps = zip(*batch)

    images = torch.stack(images, 0)
    lengths = [len(c) for c in caps]
    max_len = lengths[0]

    padded = torch.zeros(len(caps), max_len, dtype=torch.long)
    for i, cap in enumerate(caps):
        end = lengths[i]
        padded[i, :end] = cap[:end]

    return images, padded, lengths, list(all_caps)

# -----------------------------------------------------------------------------
# Main training / validation
# -----------------------------------------------------------------------------
def main(args):
    writer = SummaryWriter()

    # 1) Load your pre‐built word_dict.json
    word_dict = json.load(open(os.path.join(args.data, 'word_dict.json'), 'r'))
    vocab_size = len(word_dict)

    # 2) Build encoder / decoder
    encoder = Encoder(args.network).cuda()
    decoder = Decoder(vocab_size, encoder.dim, args.tf).cuda()

    if args.model:
        decoder.load_state_dict(torch.load(args.model))

    # 3) Optimizer / scheduler / loss
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)
    criterion = nn.CrossEntropyLoss().cuda()

    # 4) DataLoaders — pass the word_dict so the dataset can encode text → IDs
    train_ds = ImageCaptionDataset(
        transform=data_transforms,
        data_path=args.data,
        split_type='train',
        word2idx=word_dict
    )
    val_ds = ImageCaptionDataset(
        transform=data_transforms,
        data_path=args.data,
        split_type='val',
        word2idx=word_dict
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=train_collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=val_collate_fn
    )

    print('Starting training with:\n', args)
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        train(
            epoch, encoder, decoder, optimizer,
            criterion, train_loader,
            word_dict, args.alpha_c, args.log_interval, writer
        )
        validate(
            epoch, encoder, decoder, criterion,
            val_loader, word_dict, args.alpha_c,
            args.log_interval, writer
        )

        # save checkpoint
        os.makedirs('model', exist_ok=True)
        ckpt = f'model/model_{args.network}_{epoch}.pth'
        torch.save(decoder.state_dict(), ckpt)
        print('Saved model to', ckpt)

    writer.close()

# -----------------------------------------------------------------------------
# Training loop (unchanged except for unpacking lengths)
# -----------------------------------------------------------------------------
def train(epoch, encoder, decoder,
          optimizer, criterion, data_loader,
          word_dict, alpha_c, log_interval, writer):
    encoder.eval()
    decoder.train()

    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

    for bi, (imgs, caps, lengths) in enumerate(data_loader):
        imgs = imgs.cuda()
        caps = caps.cuda()
        # forward through encoder
        with torch.no_grad():
            img_feats = encoder(imgs)

        # zero grads
        optimizer.zero_grad()
        # decode (teacher forcing)
        preds, alphas = decoder(img_feats, caps)

        # chop off <start> for loss & packing
        targets = caps[:, 1:]
        pack_lens = [l - 1 for l in lengths]

        preds_p = pack_padded_sequence(preds, pack_lens, batch_first=True)[0]
        targ_p  = pack_padded_sequence(targets, pack_lens, batch_first=True)[0]

        # attention reg
        att_reg = alpha_c * ((1 - alphas.sum(1))**2).mean()

        loss = criterion(preds_p, targ_p) + att_reg
        loss.backward()
        optimizer.step()

        # metrics
        total_len = calculate_caption_lengths(word_dict, caps)
        a1 = accuracy(preds_p, targ_p, 1)
        a5 = accuracy(preds_p, targ_p, 5)
        losses.update(loss.item(), total_len)
        top1.update(a1, total_len)
        top5.update(a5, total_len)

        if bi % log_interval == 0:
            print(f'Train Epoch {epoch} [{bi}/{len(data_loader)}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Top5 {top5.val:.3f} ({top5.avg:.3f})')

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1',  top1.avg,  epoch)
    writer.add_scalar('train_top5',  top5.avg,  epoch)

# -----------------------------------------------------------------------------
# Validation loop (unpacked lengths + all_captions)
# -----------------------------------------------------------------------------
def validate(epoch, encoder, decoder,
             criterion, data_loader,
             word_dict, alpha_c, log_interval, writer):
    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

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
            targ_p  = pack_padded_sequence(targets, pack_lens, batch_first=True)[0]
            att_reg = alpha_c * ((1 - alphas.sum(1))**2).mean()

            loss = criterion(preds_p, targ_p) + att_reg

            total_len = calculate_caption_lengths(word_dict, caps)
            a1 = accuracy(preds_p, targ_p, 1)
            a5 = accuracy(preds_p, targ_p, 5)
            losses.update(loss.item(), total_len)
            top1.update(a1, total_len)
            top5.update(a5, total_len)

            # accumulate refs & hyps for BLEU
            for ref_set in all_caps:
                # each ref_set is List[List[int]] for that image
                refs = [
                    [w for w in ref if w not in (word_dict['<start>'], word_dict['<pad>'])]
                    for ref in ref_set
                ]
                references.append(refs)

            # take argmax predictions
            pred_idxs = torch.argmax(preds, dim=2).cpu().tolist()
            for seq in pred_idxs:
                hy = [w for w in seq if w not in (word_dict['<start>'], word_dict['<pad>'])]
                hypotheses.append(hy)

            if bi % log_interval == 0:
                print(f'Val Epoch {epoch} [{bi}/{len(data_loader)}]\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Top5 {top5.val:.3f} ({top5.avg:.3f})')

    writer.add_scalar('val_loss', losses.avg, epoch)
    writer.add_scalar('val_top1', top1.avg,   epoch)
    writer.add_scalar('val_top5', top5.avg,   epoch)

    # compute BLEU
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses)
    writer.add_scalar('val_bleu1', bleu1, epoch)
    writer.add_scalar('val_bleu2', bleu2, epoch)
    writer.add_scalar('val_bleu3', bleu3, epoch)
    writer.add_scalar('val_bleu4', bleu4, epoch)
    print(f' * BLEU@1 {bleu1:.4f}   BLEU@2 {bleu2:.4f}   '
          f'BLEU@3 {bleu3:.4f}   BLEU@4 {bleu4:.4f}')

# -----------------------------------------------------------------------------
# entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--batch-size',   type=int,   default=64)
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--step-size',    type=int,   default=5)
    parser.add_argument('--alpha-c',      type=float, default=1.0)
    parser.add_argument('--log-interval', type=int,   default=100)
    parser.add_argument('--data',         type=str,
                        default='/kaggle/input/chest-xrays-indiana-university')
    parser.add_argument('--network',      choices=['vgg19','resnet152','densenet161'],
                        default='vgg19')
    parser.add_argument('--model',        type=str,   help='path to decoder checkpoint')
    parser.add_argument('--tf',           action='store_true', default=False,
                        help='use teacher forcing')
    args = parser.parse_args()

    # image transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225]),
    ])

    main(args)

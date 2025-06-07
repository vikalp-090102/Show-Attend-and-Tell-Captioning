import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from PIL import Image

def pil_loader(path):
    """Load an image from disk and convert to RGB."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, split_type='train', word2idx=None):
        super(ImageCaptionDataset, self).__init__()
        self.split_type     = split_type
        self.transform      = transform
        self.word_count     = Counter()
        self.caption_img_idx = {}
        self.word2idx       = word2idx  # Provided vocabulary mapping (or None)

        # 1) Load & merge your CSVs
        proj_fp = os.path.join(data_path, 'indiana_projections.csv')
        rep_fp  = os.path.join(data_path, 'indiana_reports.csv')
        projections = pd.read_csv(proj_fp)
        reports     = pd.read_csv(rep_fp)
        merged      = pd.merge(projections, reports, on='uid')

        # 2) Extract (filename, caption) pairs, flattening out any numbered impressions
        self.img_paths = []
        self.captions  = []
        for _, row in merged.iterrows():
            fn  = row['filename']
            imp = row['impression']
            if not isinstance(imp, str) or pd.isna(imp):
                continue

            # Strip leading "1. " or "2. " if present
            m = re.match(r'^\d+\.', imp)
            cap = imp.split('. ', 1)[1] if m else imp

            full_img_path = os.path.join(data_path, 'images', 'images_normalized', fn)
            self.img_paths.append(full_img_path)
            
            # Update word counts for vocabulary building
            for w in cap.lower().split():
                self.word_count[w] += 1

            # If a vocabulary is provided, encode caption to token IDs; otherwise, store raw text.
            if self.word2idx is not None:
                encoded = self._encode_caption(cap)
                self.captions.append(encoded)
            else:
                self.captions.append(cap)

        # 3) Build a lookup: image_path -> indices of all associated captions
        for idx, p in enumerate(self.img_paths):
            self.caption_img_idx.setdefault(p, []).append(idx)

    def _encode_caption(self, caption):
        """
        Encode a caption string into a list of integer token IDs.
        This method assumes that a word2idx mapping is available.
        It converts the caption to lowercase, splits it into tokens,
        prepends a '<start>' token and appends an '<end>' token.
        Unknown words are mapped to the token ID for '<unk>'.
        """
        tokens = caption.lower().split()
        tokens = ['<start>'] + tokens + ['<end>']
        # Use the provided vocabulary; if a token isn't found, fallback to '<unk>'
        return [self.word2idx.get(token, self.word2idx.get('<unk>', 0)) for token in tokens]

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        # Ensure the image is a torch.Tensor (convert if needed)
        img_tensor = img if isinstance(img, torch.Tensor) else torch.FloatTensor(img)

        if self.split_type == 'train':
            # For training, return a single image/caption pair.
            return img_tensor, self.captions[index]

        # For validation/testing, also retrieve all captions associated with the image.
        all_idxs     = self.caption_img_idx[img_path]
        all_captions = [self.captions[i] for i in all_idxs]
        return img_tensor, self.captions[index], all_captions

    def __len__(self):
        return len(self.img_paths)

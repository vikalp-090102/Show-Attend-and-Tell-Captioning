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
    def __init__(self, transform, data_path, split_type='train', word2idx=None, min_word_freq=5):
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
        raw_captions = []

        for _, row in merged.iterrows():
            fn  = row['filename']
            imp = row['impression']
            if not isinstance(imp, str) or pd.isna(imp):
                continue

            m = re.match(r'^\d+\.', imp)
            cap = imp.split('. ', 1)[1] if m else imp

            full_img_path = os.path.join(data_path, 'images', 'images_normalized', fn)
            self.img_paths.append(full_img_path)
            raw_captions.append(cap)

            for w in cap.lower().split():
                self.word_count[w] += 1

        # 3) Auto-vocab if not provided
        if self.word2idx is None:
            self.word2idx = self._build_vocab(min_word_freq)
        
        # 4) Encode captions
        for cap in raw_captions:
            self.captions.append(self._encode_caption(cap))

        # 5) Build caption lookup
        for idx, p in enumerate(self.img_paths):
            self.caption_img_idx.setdefault(p, []).append(idx)

    def _build_vocab(self, min_word_freq):
        """
        Build vocabulary dictionary from word frequency count.
        """
        word2idx = {
            '<pad>': 0,
            '<start>': 1,
            '<end>': 2,
            '<unk>': 3
        }
        idx = 4
        for word, count in self.word_count.items():
            if count >= min_word_freq:
                word2idx[word] = idx
                idx += 1
        print(f"Built vocabulary of size: {len(word2idx)}")
        return word2idx

    def _encode_caption(self, caption):
        tokens = caption.lower().split()
        tokens = ['<start>'] + tokens + ['<end>']
        return [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        img_tensor = img if isinstance(img, torch.Tensor) else torch.FloatTensor(img)

        if self.split_type == 'train':
            return img_tensor, self.captions[index]
        
        all_idxs     = self.caption_img_idx[img_path]
        all_captions = [self.captions[i] for i in all_idxs]
        return img_tensor, self.captions[index], all_captions

    def __len__(self):
        return len(self.img_paths)

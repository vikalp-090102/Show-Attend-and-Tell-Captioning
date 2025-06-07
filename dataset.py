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
    def __init__(self, transform, data_path, split_type='train'):
        super(ImageCaptionDataset, self).__init__()
        self.split_type     = split_type
        self.transform      = transform
        self.word_count     = Counter()
        self.caption_img_idx = {}

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

            # strip leading "1. " or "2. " if present
            m = re.match(r'^\d+\.', imp)
            cap = imp.split('. ', 1)[1] if m else imp

            full_img_path = os.path.join(data_path, 'images/images_normalized', fn)
            self.img_paths.append(full_img_path)
            self.captions.append(cap)

            # update word counts for later vocab building
            for w in cap.lower().split():
                self.word_count[w] += 1

        # 3) Build a lookup: image_path -> all caption-indices
        for idx, p in enumerate(self.img_paths):
            self.caption_img_idx.setdefault(p, []).append(idx)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img      = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # match your original API: always wrap image as FloatTensor
        img_tensor = img if isinstance(img, torch.Tensor) else torch.FloatTensor(img)

        if self.split_type == 'train':
            # for training: return single (img, caption)
            return img_tensor, self.captions[index]

        # for val/test: also gather ALL captions for this image
        all_idxs     = self.caption_img_idx[img_path]
        all_captions = [self.captions[i] for i in all_idxs]
        return img_tensor, self.captions[index], all_captions

    def __len__(self):
        return len(self.img_paths)

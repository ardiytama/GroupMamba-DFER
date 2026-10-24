import csv
import functools
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def sample_indices(num_frames, T=16, train=False):
    # I_t = floor(L/T * t + delta), delta ~ U[0, floor(L/T) - 1]; eval takes the center frame of each segment
    seg = num_frames / T
    if train:
        delta = random.randint(0, max(int(seg) - 1, 0))
        idx = [int(seg * t + delta) for t in range(T)]
    else:
        idx = [int(seg * t + seg / 2) for t in range(T)]
    return [min(i, num_frames - 1) for i in idx]


@functools.lru_cache(maxsize=None)
def find_clips(root):
    # root/<id>/*.jpg, or root/<part>/<id>/*.jpg (full DFEW clips)
    clips = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        children = os.listdir(path)
        if any(c.lower().endswith(('.jpg', '.png')) for c in children):
            clips[name] = path
        else:
            for c in children:
                if os.path.isdir(os.path.join(path, c)):
                    clips[c] = os.path.join(path, c)
    return clips


def read_split(csv_path):
    # video_name,label (labels 1..7) -> (zero-padded id, 0..6)
    with open(csv_path, encoding='utf-8') as f:
        rows = list(csv.reader(f))[1:]
    return [(r[0].zfill(5), int(r[1]) - 1) for r in rows]


class DFEWClips(Dataset):
    def __init__(self, root, split_csv, train=False, T=16, size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.items = read_split(split_csv)
        self.clips = find_clips(root)
        missing = [v for v, _ in self.items if v not in self.clips]
        assert not missing, f"{len(missing)} clips from {split_csv} not found under {root}, e.g. {missing[:3]}"
        self.train, self.T, self.size = train, T, size
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.items)

    def _frames(self, vid):
        d = self.clips[vid]
        names = sorted((n for n in os.listdir(d) if n.lower().endswith(('.jpg', '.png'))),
                       key=lambda n: int(os.path.splitext(n)[0]))
        return [os.path.join(d, n) for n in names]

    def __getitem__(self, i):
        vid, label = self.items[i]
        paths = self._frames(vid)
        frames = []
        for j in sample_indices(len(paths), self.T, self.train):
            img = Image.open(paths[j]).convert('RGB')
            if img.size != (self.size, self.size):
                img = img.resize((self.size, self.size))
            frames.append(torch.from_numpy(np.array(img)))
        x = torch.stack(frames).permute(3, 0, 1, 2).float() / 255.0  # (3, T, H, W)
        if self.train and random.random() < 0.5:
            x = x.flip(-1)  # same flip for every frame
        return (x - self.mean) / self.std, label


def channel_stats(dataset, n=500, cache=None):
    # per-channel mean/std over the training split (Z-score normalization)
    if cache and os.path.isfile(cache):
        with open(cache, encoding='utf-8') as f:
            s = json.load(f)
        return s['mean'], s['std']
    saved = dataset.mean, dataset.std, dataset.train
    dataset.mean, dataset.std, dataset.train = torch.zeros(3, 1, 1, 1), torch.ones(3, 1, 1, 1), False
    ids = random.Random(0).sample(range(len(dataset)), min(n, len(dataset)))
    s = torch.zeros(3)
    sq = torch.zeros(3)
    count = 0
    for i in ids:
        x = dataset[i][0]
        s += x.sum(dim=(1, 2, 3))
        sq += (x ** 2).sum(dim=(1, 2, 3))
        count += x[0].numel()
    dataset.mean, dataset.std, dataset.train = saved
    mean = s / count
    std = (sq / count - mean ** 2).sqrt()
    out = (mean.tolist(), std.tolist())
    if cache:
        with open(cache, 'w', encoding='utf-8') as f:
            json.dump({'mean': out[0], 'std': out[1]}, f)
    return out

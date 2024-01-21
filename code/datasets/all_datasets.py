import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset



class ISIC2017Dataset(Dataset):
    def __init__(self, datapath, gtpath, mode='train', args=None):
        self.imgs = sorted([datapath + f for f in os.listdir(datapath) if not f.startswith('.')])
        self.gtpath = gtpath
        self.gts = sorted([gtpath + f for f in os.listdir(gtpath) if not f.startswith('.')])
        assert len(self.imgs) == len(self.gts), f"{len(self.imgs)}, {len(self.gts)}"

        self.mode = mode
        assert self.mode in ["train", "test"]

        self.args = args
        

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        img = np.asarray(img)
        img = torch.Tensor(img.copy())
        assert img.shape == (256, 256, 3)
        img = img.permute(2, 0, 1)

        label = Image.open(self.gts[index])
        label = np.asarray(label) / 255
        assert label.shape == (256, 256)
        label = torch.Tensor(label)
        label = label.unsqueeze(0)

        return {'image': img, 'mask': label, 'index': index}


class BreastTumorDataset(Dataset):
    def __init__(self, root, gtpath, mode='train', args=None):
        with open(os.path.join(root, f"{self.mode}.txt"), 'r') as f:
            f_names = [line.rstrip() for line in f.readlines()]
        self.f_names = f_names
        self.imgs = [os.path.join(root, self.mode, "imgs", f) for f in f_names]
        self.gts = [os.path.join(root, self.mode, gtpath, f) for f in f_names]
        self.gtpath = gtpath
        assert len(self.imgs) == len(self.gts), f"{len(self.imgs)}, {len(self.gts)}"

        self.mode = mode
        assert self.mode in ["train", "test"]

        self.args = args

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('L')
        img = np.asarray(img)
        img = torch.Tensor(img.copy())
        img = img.unsqueeze(0)

        label = Image.open(self.gts[index]).convert('L')
        label = np.asarray(label) / 255
        label = torch.Tensor(label)
        label = label.unsqueeze(0)

        return {'image': img, 'mask': label, 'index': index}

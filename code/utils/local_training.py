import logging
import numpy as np

import copy
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.losses import SoftDiceBCEWithLogitsLoss



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        sample = self.dataset[self.idxs[item]]
        return sample


class LocalUpdate(object):
    def __init__(self, args, id, dataset, idxs):
        self.args = args
        self.id = id
        self.idxs = idxs
        self.local_dataset = DatasetSplit(copy.deepcopy(dataset), idxs)
        logging.info(
            f'client{id} data size: {len(self.local_dataset)}, index len: {len(self.idxs)}')
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.epoch = 0
        self.iter_num = 0


    def train(self, net, writer):
        print(f"id: {self.id}, num: {len(self.local_dataset)}, lr: {self.lr}")

        net.train()
        # set the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, betas=(0.9, 0.99), amsgrad=False)
        # train and update
        epoch_loss = []
        # set the loss function
        loss_func = nn.BCEWithLogitsLoss() if not self.args.diceloss else SoftDiceBCEWithLogitsLoss(w=self.args.DiceLossWeight)
    
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for i, sample in enumerate(self.ldr_train):
                images, labels, indexs = sample["image"].cuda(), sample["mask"].cuda(), sample["index"]
                
                outputs = net(images)
                loss = loss_func(outputs, labels)
                loss_total = loss

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(f'client{self.id}/loss_train', loss.item(), self.iter_num)

                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        net.cpu()
        optimizer = None
        return net.state_dict(), np.array(epoch_loss).mean(0)


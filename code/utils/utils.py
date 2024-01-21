import copy
import random
import os
import sys
import shutil
import numpy as np
import logging
from tensorboardX import SummaryWriter

from scipy.ndimage.morphology import distance_transform_edt as distrans

import torch
import torch.nn as nn




def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_output_files(args):
    outputs_dir = 'output/' + str(args.dataset)
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    exp_dir = os.path.join(outputs_dir, args.exp+'_s'+str(args.strength)+'_e'+str(args.local_ep))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    code_dir = os.path.join(exp_dir, 'code')
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir)
    shutil.copytree('./code', code_dir, ignore=shutil.ignore_patterns('.git', '__pycache__'))

    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(tensorboard_dir)
    return writer, models_dir



def region(labels):
    labels = labels.cpu().numpy().astype("float")
    sdm = np.zeros_like(labels).astype("float")
    region_mask = np.zeros_like(labels).astype("float")
    # print(labels.dtype, sdm.dtype, region_mask.dtype)
    for i in range(labels.shape[0]):
        if labels[i].sum() < 1:
            pass
        else:
            pos_dis = distrans(labels[i])
            neg_dis = distrans(1-labels[i])
            sdm[i] = -pos_dis + neg_dis
            min_dis = sdm[i].min()
            max_dis = sdm[i].max()
            assert min_dis < 0
            assert max_dis > 0
            dis = -min_dis if -min_dis<max_dis else max_dis
            assert dis>0, f"{dis} is not bigger than 0"
            region_mask[i][(labels[i]==0) & (sdm[i]<=dis)] = 2
            region_mask[i][(labels[i]==1) & (sdm[i]>=-dis)] = 1

    return region_mask


def cal_loss_two_directions(loader, net, args, criterion=None):
    net.eval()
    with torch.no_grad():
        for i, sample in enumerate(loader):
            images, labels = sample["image"].cuda(), sample["mask"].cuda()
            outputs = net(images)
            if criterion is None:
                raise
            else:
                loss = criterion(outputs, labels)  # (b, c, h, w)
                loss_feature = torch.zeros((labels.shape[0], 2*args.n_classes))
                for c in range(outputs.shape[1]):
                    region_mask = torch.from_numpy(region(labels[:, c].unsqueeze(1))).cuda()
                    assert (region_mask == 0).any()
                    loss_n = loss[:, c].unsqueeze(1)
                    assert region_mask.shape == loss_n.shape
                    loss_n_in = (loss_n * (region_mask==1).float()).view(loss_n.shape[0], -1).sum(1) / (region_mask==1).float().view(loss_n.shape[0], -1).sum(1)
                    loss_n_out = (loss_n * (region_mask==2).float()).view(loss_n.shape[0], -1).sum(1) / (region_mask==2).float().view(loss_n.shape[0], -1).sum(1)
                    assert loss_n_in.shape[0] == loss_n_out.shape[0] == images.shape[0]
                    loss_feature[:, c*2] = loss_n_in
                    loss_feature[:, c*2+1] = loss_n_out

            if i == 0:
                loss_whole_n = loss_feature.cpu().numpy()
            else:
                loss_whole_n = np.concatenate((loss_whole_n, loss_feature.cpu().numpy()), axis=0)
    return loss_whole_n



def get_outputs(loader, net):
    net.eval()
    with torch.no_grad():
        for i, sample in enumerate(loader):
            images, indexs, labels = sample["image"].cuda(), sample["index"], sample["mask"]
            outputs = net(images)
            assert outputs.shape[1:] == (1, 256, 256)
            probs = torch.sigmoid(outputs)

            if i == 0:
                prob_whole = probs.cpu().numpy()
                label_whole = labels.numpy()
                index_whole = np.array(indexs)
            else:
                prob_whole = np.concatenate((prob_whole, probs.cpu().numpy()), axis=0)
                label_whole = np.concatenate((label_whole, labels.numpy()), axis=0)
                index_whole = np.concatenate([index_whole, np.array(indexs)], axis=0)
    return prob_whole, label_whole, index_whole



def get_output(loader, net, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, sample in enumerate(loader):
            images, labels, indexs = sample["image"].cuda(), sample["mask"].cuda(), sample["index"]
            outputs = net(images)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(torch.sigmoid(outputs).cpu())
                loss_whole = np.array(loss.cpu())
                label_whole = np.array(labels.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, torch.sigmoid(outputs).cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
                label_whole = np.concatenate((label_whole, labels.cpu()), axis=0)

    assert len(output_whole.shape) == len(loss_whole.shape) == 4
    output_whole = output_whole[:, 0, :, :]
    loss_whole = loss_whole[:, 0, :, :]
    label_whole = label_whole[:, 0, :, :]
    return output_whole, loss_whole, label_whole
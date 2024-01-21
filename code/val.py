import numpy as np

import torch
from torch.utils.data import DataLoader

from medpy import metric


def cal_metric_3D(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)
            

def cal_metric_2D(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)



def test_all_slices(net, dataset, num_classes, writer, visualization, args):
    net.eval()
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    cnt = len(eval_loader.dataset)

    if args.dataset == "ISIC2017":
        assert cnt == 600
        assert num_classes == 1
    elif args.dataset == "BreastTS":
        assert cnt == 276
        assert num_classes == 1
    else: 
        raise NotImplementedError
    
    dval = np.zeros((num_classes, cnt))
    hdval = np.zeros((num_classes, cnt))
    for iter, batch in enumerate(eval_loader):
        image, target = batch['image'].cuda(), batch['mask'].cuda()
        index = batch['index']
        with torch.no_grad():
            output = net(image)
        prob = torch.sigmoid(output)
        pred = (prob > 0.5).float()

        if args.dataset == "ISIC2017":
            assert pred.shape[0] == target.shape[0] == 1
            assert pred.shape[1] == target.shape[1] == 1
        elif args.dataset == "BreastTS":
            assert pred.shape[0] == target.shape[0] == 1
            assert pred.shape[1] == target.shape[1] == 1
        elif args.dataset == "BraTS":
            assert pred.shape[0] == target.shape[0] == 1
            assert pred.shape[1] == target.shape[1] == 1
        else:
            raise NotImplementedError
        
        for j in range(num_classes):
            metric = cal_metric_2D(pred[0, j].detach().cpu().numpy(), target[0, j].cpu().numpy())
            dval[j][iter] = metric[0]
            hdval[j][iter] = metric[1]

    return dval, hdval

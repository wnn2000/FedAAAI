import copy
import torch
import numpy as np


def FedAvg(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():        
        w_avg[k] = w_avg[k] * dict_len[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg


def FedA3I_Agg(w, weight, dict_len, args):
    weight1 = weight  # quality-based weights
    weight2 = np.array(dict_len) / np.sum(dict_len)  # quantity-based weights
    assert np.allclose(np.sum(weight1), 1), f'{np.sum(weight1)} does not sum to 1.0'
    assert np.allclose(np.sum(weight2), 1), f'{np.sum(weight2)} does not sum to 1.0'
    w_avg = copy.deepcopy(w[0])
    alpha = np.linspace(0., 1., 10)
    alpha = args.weight * np.power(alpha, args.power) + args.bias
    # print(alpha)
    for k in w_avg.keys():
        if "inc" in k:
            temp = 0 
        elif "down1" in k:
            temp = 1
        elif "down2" in k:
            temp = 2
        elif "down3" in k:
            temp = 3
        elif "down4" in k:
            temp = 4
        elif "up1" in k:
            temp = 5
        elif "up2" in k:
            temp = 6
        elif "up3" in k:
            temp = 7
        elif "up4" in k:
            temp = 8
        elif "outc" in k:
            temp = 9
        else:
            raise
        weight3 = alpha[temp]*weight1 + (1-alpha[temp])*weight2  # layer-wise combinition
        assert len(weight3) == len(dict_len) == len(w)
        assert np.allclose(np.sum(weight3), 1), f'{np.sum(weight3)} does not sum to 1.0'
        w_avg[k] = w_avg[k] * weight3[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * weight3[i]

    return w_avg

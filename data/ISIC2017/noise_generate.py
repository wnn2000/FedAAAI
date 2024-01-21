import random
import argparse
from typing import List, Tuple
from pathlib import Path
from pprint import pprint
from argparse import Namespace
import os
import cv2
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy import ndimage



def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)
    dict_users, all_idxs = {}, np.arange(n_train) # initial user and index for whole dataset
    np.random.shuffle(all_idxs)
    split = np.array_split(all_idxs, num_users, axis=0)
    for i in range(num_users):
        dict_users[i] = list(split[i])
    # examine
    all = 0
    for i in range(num_users):
        all += len(set(dict_users[i]))
        if i < num_users-1:
            assert set(dict_users[i]).isdisjoint(set(dict_users[i+1]))
        else:
            assert set(dict_users[i]).isdisjoint(set(dict_users[0]))
    assert all == n_train
    
    return dict_users


def pixel_move(offset, pixel, x_near, y_near, label): # This function will make the pixel on the boundary move to a new position
    x, y = pixel

    tag = 1 if len(np.unique(x_near)) < len(np.unique(y_near)) else 0
    k = np.polyfit(x_near, y_near, deg=1)[0] if tag == 0 else 1/(1e-9+np.polyfit(y_near, x_near, deg=1)[0])
    if k != 0.:
        k_ = -1 / k           # k_ denotes the normal direction
        if np.abs(k_) >= 20:
            k_ = np.inf
    else:
        k_ = np.inf
    del k

    dis_abs = np.abs(offset)
    if k_ != np.inf:
        delta_x = (dis_abs**2 / (1 + k_**2))**0.5
        assert delta_x >= 0
        delta_y = k_ * delta_x
        x_i = 1 / (1+k_**2)**0.5
        y_i = k_ / (1+k_**2)**0.5
    else:
        delta_x = 0
        delta_y = dis_abs
        assert delta_y >= 0
        x_i = 0
        y_i = 1

    
    for temp in range(1, 6):    # inward or outward?
        x_ii = int(np.round(x_i*temp))
        y_ii = int(np.round(y_i*temp))
        x_p = np.clip(x+x_ii, 0, label.shape[0]-1)
        y_p = np.clip(y+y_ii, 0, label.shape[0]-1)
        x_m = np.clip(x-x_ii, 0, label.shape[0]-1)
        y_m = np.clip(y-y_ii, 0, label.shape[0]-1)
        if int(label[x_p, y_p]) ^ int(label[x_m, y_m]):
            break
        else:
            if temp == 5:
                return (np.nan, np.nan, np.nan)  # fail

    indicator = label[x_p, y_p]
    if (indicator == 1 and offset <= 0) or (indicator == 0 and offset > 0):
        pass
    else:
        delta_x, delta_y = -delta_x, -delta_y

    new_x = int(np.round(x + delta_x))
    new_y = int(np.round(y + delta_y))
    new_x = np.clip(new_x, 0, label.shape[0]-1)
    new_y = np.clip(new_y, 0, label.shape[1]-1)

    if (label[new_x, new_y] == 1 and offset > 1) or (label[new_x, new_y] == 0 and offset <= -1):
        return (np.nan, np.nan, np.nan)

    return (new_x, new_y, k_)


def add_noise(orig_mask, mean, std):
    orig_arr: np.ndarray = np.array(orig_mask, dtype=np.uint8)
    res_arr: np.ndarray = np.zeros_like(orig_arr)
    assert orig_arr.dtype == res_arr.dtype

    # mean: float = args.mean
    # std: float = args.std
    segment: int = args.segment
    deg: int = args.deg
    num: int = args.num_pixel  

    contours, hierarchy = cv2.findContours(
        orig_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(contours) == 1  # only one object

    # for a random starting point
    X = contours[0][:, 0, 1]
    Y = contours[0][:, 0, 0]
    X_2 = np.concatenate([X, X], axis=0)
    Y_2 = np.concatenate([Y, Y], axis=0)
    begin = np.random.randint(0, len(X))
    X = X_2[begin:begin+len(X)]
    Y = Y_2[begin:begin+len(Y)]

    idxs = np.arange(len(X))
    margin = len(idxs) // segment
    if margin == 0:
        return np.zeros_like(orig_arr), 0
    else:
        select_idxs = idxs[::margin][:segment]

    while 1: # Obtain the offset for each pixel on the boundary
        offset = np.zeros_like(idxs).astype("float")
        offset[select_idxs] = np.random.normal(mean, std, size=select_idxs.shape)
        w = np.ones(len(select_idxs)+1)
        w[0] = 10.
        w[-1] = 10.
        poly = np.polyfit(np.concatenate([idxs[select_idxs], [len(idxs)]], axis=0),
                        np.concatenate([offset[select_idxs], [offset[0]]], axis=0), deg=deg, w=w)  # Ensure continuity between beginning and end
        offset = np.polyval(poly, idxs)
        offset = np.clip(offset, -30, 30)
        if np.abs(offset[0]-offset[-1]) < 3:
            break

    X_new = np.array([]).astype("int")
    Y_new = np.array([]).astype("int")
    X_c = np.concatenate([X, X, X], axis=0)
    Y_c = np.concatenate([Y, Y, Y], axis=0)

    if (offset==0).all():
        return orig_arr
    else:
        for i in range(X.shape[0]):
            pixel = X[i], Y[i]
            j = i + len(X)
            x_near = X_c[j-num: j+num+1]
            y_near = Y_c[j-num: j+num+1]
            x_new, y_new, k = pixel_move(offset[i], pixel, x_near, y_near, copy.deepcopy(orig_arr))

            if np.isnan(x_new):
                continue

            X_new = np.append(X_new, x_new)
            Y_new = np.append(Y_new, y_new)
            res_arr[x_new, y_new] = 1

    contours_new = np.concatenate([Y_new.reshape(-1, 1, 1), X_new.reshape(-1, 1, 1)], axis=2)
    contours_new = (contours_new, )
    try:
        res_arr = cv2.drawContours(copy.deepcopy(res_arr), contours_new, 0, 1, cv2.FILLED)
    except:
        if res_arr.sum() == 0:
            return res_arr
        else:
            raise

    if args.open and offset.min()<=-10: # post-processing
        temp = copy.deepcopy(res_arr)
        k = np.ones((3, 3), np.uint8)
        res_arr = cv2.erode(res_arr, k, iterations=1)
        labeled_array, num = ndimage.label(res_arr)
        if num == 0:
            return temp
        max_label = 0
        max_num = 0
        for i in range(1, num+1):
            if np.sum(labeled_array == i) > max_num:
                max_num = np.sum(labeled_array == i)
                max_label = i
        res_arr = (labeled_array == max_label).astype("uint8")
        res_arr = cv2.dilate(res_arr, k, iterations=1)

    return res_arr


def weaken_img(pn, mean, std) -> Tuple[int, int]:
    p, n = pn    # p: path     n: file name
    img = Image.open(p)
    selected_class: list = args.selected_class  # list of objects

    new_img = np.zeros_like(np.array(img))
    for c in selected_class:
        ni = (np.array(img) == selected_class)
        assert set(np.unique(ni)).issubset({False, True})
        labeled_array, num_features = ndimage.label(ni)

        if num_features == 1:
            res_img = add_noise(ni, mean, std)
        else:
            area = np.array([0])
            for id in range(1, num_features+1):
                area = np.append(area, (labeled_array==id).sum())
            idx = np.argsort(-area)
            assert len(idx) == num_features+1
            assert idx[-1] == 0
            assert area[idx[-1]] == 0

            for i in range(len(idx)-1):
                if i==0:
                    res_img = add_noise((labeled_array==idx[0]), mean, std)
                else:
                    if area[idx[i]] < 15:
                        continue # very small areas are ignored
                    res_img_temp = add_noise((labeled_array==idx[i]), mean, std)
                    res_img = res_img | res_img_temp
            
        new_img[res_img == 1] = c    # the initial label
    
    new_img = Image.fromarray(new_img.astype(np.uint8), mode='L')
    res_arr: np.ndarray = np.array(new_img)
    
    save_path = Path(args.base_folder, args.save_subfolder, n)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    new_img.save(save_path)



def main(args: Namespace) -> None:
    inputs: List[Path] = list(Path(args.base_folder, args.GT_subfolder).glob(args.regex))
    inputs.sort()
    names: List[str] = [p.name for p in inputs]
    print(f"Found {len(names)} images to weaken")
    pprint(inputs[:5])
    pprint(names[:5])

    # obtain data partition of different clients
    if os.path.exists("dict_user.npy"):
        dict_user = np.load("dict_user.npy", allow_pickle=True).item()
    else:
        dict_user = iid_sampling(len(names), args.n_clients, seed=0)
        np.save('dict_user.npy', dict_user)

    random.seed(args.seed)
    np.random.seed(args.seed)

    # different noise types hold different parameters for the proposed noise model
    if args.n_type == 1:
        mean_clients = np.random.uniform(0, args.mean, args.n_clients)
        idxs = np.random.choice(a=args.n_clients, size=int(0.2*args.n_clients), replace=False)
        mean_clients[idxs] *= -1
        std_clients = np.random.uniform(0.5*args.std, args.std, args.n_clients)
        print(mean_clients, std_clients)

    elif args.n_type == 2:
        mean_clients = np.random.uniform(0, args.mean, args.n_clients)
        idxs = np.random.choice(a=args.n_clients, size=int(0.8*args.n_clients), replace=False)
        mean_clients[idxs] *= -1
        std_clients = np.random.uniform(0.5*args.std, args.std, args.n_clients)
        print(mean_clients, std_clients)

    else:
        raise


    for i in tqdm(range(args.n_clients)):
        mean = mean_clients[i]
        std = std_clients[i]
        for j in tqdm(range(len(dict_user[i]))):
            pn = (inputs[dict_user[i][j]], names[dict_user[i][j]])
            weaken_img(pn, mean, std)



def get_args() -> Namespace:
    parser = argparse.ArgumentParser(description='Dataset params')
    parser.add_argument("--base_folder", type=str, default='./train')
    parser.add_argument("--n_clients", type=int, default=50)
    parser.add_argument("--mean", type=float, default=20.)
    parser.add_argument("--std", type=float, default=10.)
    parser.add_argument("--n_type", type=int, default=1)

    parser.add_argument("--segment", type=int, default=30)
    parser.add_argument("--deg", type=int, default=10)
    parser.add_argument("--num_pixel", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selected_class", type=list, default=[255])
    parser.add_argument("--GT_subfolder", default='gt', type=str)
    parser.add_argument("--regex", type=str, default="*.png")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args: Namespace = get_args()
    args.save_subfolder = args.GT_subfolder + '_n' + str(args.n_type) + '_m' + str(int(args.mean)) + '_s' + str(int(args.std))
    args.open = False
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)

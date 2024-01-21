import numpy as np

from utils.sampling import iid_sampling
from .all_datasets import ISIC2017Dataset, BreastTumorDataset


def get_dataset(args):
    if args.dataset == "ISIC2017":
        args.n_clients = 50
        args.client_num = args.n_clients
        args.n_classes = 1
        args.input_channel = 3
        args.twoD = True

        if args.strength == 0: # clean
            train_dataset = ISIC2017Dataset(
                datapath="data/ISIC2017/train/imgs/", gtpath="data/ISIC2017/train/gt/", mode="train", args=args)
            
        # Other dataset with different annotation noise. You can add new datasets here
        elif args.strength == 21:
            train_dataset = ISIC2017Dataset(
                datapath="data/ISIC2017/train/imgs/", gtpath="data/ISIC2017/train/gt_n1_m20_s10/", mode="train", args=args)
            
        else:
            raise NotImplementedError

        test_dataset = ISIC2017Dataset(
            datapath="data/ISIC2017/test/imgs/", gtpath=f"data/ISIC2017/test/gt/", mode="test", args=args)

        dict_users = np.load(
            "data/ISIC2017/dict_user.npy", allow_pickle=True).item()
        assert len(dict_users.keys()) == args.n_clients


    elif args.dataset == "BreastTS":
        args.n_clients = 50
        args.client_num = args.n_clients
        args.n_classes = 1
        args.input_channel = 1
        args.twoD = True

        if args.strength == 0: # clean
            train_dataset = BreastTumorDataset(
                root="data/BreastTS", gtpath="gt", mode="train", args=args)
            
        # Other dataset with different annotation noise. You can add new datasets here
        else:
            raise NotImplementedError

        test_dataset = BreastTumorDataset(root="data/BreastTS", gtpath="gt", mode="test", args=args)

        dict_users = np.load(
            "data/BreastTS/dict_users.npy", allow_pickle=True).item()
        assert len(dict_users.keys()) == args.n_clients


    else:
        raise NotImplementedError

    return train_dataset, test_dataset, dict_users
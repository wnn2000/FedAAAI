import os
import copy
import logging
import numpy as np

from sklearn.mixture import GaussianMixture

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.options import args_parser
from utils.local_training import LocalUpdate
from utils.FedAvg import FedAvg, FedA3I_Agg
from utils.utils import set_seed, set_output_files, cal_loss_two_directions

from datasets.dataset import get_dataset
from models.build_model import build_model
from val import test_all_cases, test_all_slices

np.set_printoptions(threshold=np.inf)




if __name__ == '__main__':
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------ output files ------------------------------
    writer, models_dir = set_output_files(args)

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed)

    # ------------------------------ global dataset ------------------------------
    dataset_train, dataset_test, dict_users = get_dataset(args)
    logging.info(f"train, total: {len(dataset_train)}")
    logging.info(f"test, total {len(dataset_test)}")

    # --------------------- build models and local set ---------------------------
    netglob = build_model(args)
    net_ini = copy.deepcopy(netglob)
    user_id = list(range(args.n_clients))
    trainer_locals = []
    for id in user_id:
        trainer_locals.append(LocalUpdate(
            args, id, copy.deepcopy(dataset_train), dict_users[id]))
    dict_len = [len(dict_users[idx]) for idx in user_id]

    # ------------------------------ begin training ------------------------------
    set_seed(args.seed)
    s1 = args.s1
    Metrics = np.zeros((args.run, 2*args.n_classes))


    for run in range(args.run):
        logging.info('\n')
        logging.info(
            f"--------------------- beging training, run: {run} --------------------- \n")
        lr = args.base_lr

        for rnd in range(args.rounds):
            writer.add_scalar(f'train_{run}/lr', lr, rnd)
            w_locals, loss_locals = [], []

            logging.info(
                f"===========> Round: {rnd}, lr: {lr}")
            for idx in user_id:  # training over the subset
                local = trainer_locals[idx]
                local.lr = lr
                local.run = run
                w_local, loss_local = local.train(
                    net=copy.deepcopy(netglob).to(args.device), writer=writer)
                # store every updated model
                w_locals.append(copy.deepcopy(w_local))
                loss_locals.append(copy.deepcopy(loss_local))
            
            assert len(dict_len) == args.n_clients == len(w_locals) == idx+1


            
            if rnd < s1:  # warming up stage
                w_glob_fl = FedAvg(w_locals, dict_len)
                netglob.load_state_dict(copy.deepcopy(w_glob_fl))

                ############ ↓↓↓↓↓↓ Key of Our Method FedA3I ↓↓↓↓↓↓ ############
                if rnd == s1-1:
                    """ collect loss value """
                    loss_client = np.zeros((args.n_clients, 2*args.n_classes))
                    criterion = nn.BCEWithLogitsLoss(reduction='none')
                    for id in user_id:
                        dataset_local = copy.deepcopy(trainer_locals[id].local_dataset)
                        loader_local = DataLoader(dataset_local, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
                        loss_n = cal_loss_two_directions(loader_local, copy.deepcopy(netglob).to(args.device), args, criterion=criterion)
                        # loss_client[id] = loss_n.mean(0)
                        loss_client[id] = np.nanmean(loss_n, axis=0)
                    logging.info(loss_client)

                    """ noise classification """
                    gmm = GaussianMixture(n_components=2).fit(loss_client)
                    gmm_pred = gmm.predict(loss_client)
                    odd0 = np.sum(gmm.means_[0][1::2])
                    even0 = np.sum(gmm.means_[0][::2])
                    odd1 = np.sum(gmm.means_[1][1::2])
                    even1 = np.sum(gmm.means_[1][::2])
                    y_expand = 0 if (even0-odd0) > (even1-odd1) else 1
                    gmm_ex = np.where(gmm_pred == y_expand)[0]
                    gmm_sh = np.where(gmm_pred == (1-y_expand))[0]
                    logging.info(f"=====> selected expand clients: {gmm_ex}")
                    logging.info(f"=====> selected shrink clients: {gmm_sh}")
                    loss_ex = loss_client[gmm_ex][:, 0] - loss_client[gmm_ex][:, 1]
                    loss_sh = loss_client[gmm_sh][:, 1] - loss_client[gmm_sh][:, 0]
                    weight = np.zeros(args.n_clients)
                    
                    weight[gmm_ex] = 1 - (loss_ex - loss_ex.min()) / \
                        (loss_ex.max() - loss_ex.min())
                    weight[gmm_ex] = args.InterW * \
                        weight[gmm_ex] / weight[gmm_ex].sum()
                    weight[gmm_sh] = 1 - (loss_sh - loss_sh.min()) / \
                        (loss_sh.max() - loss_sh.min())
                    weight[gmm_sh] = (1-args.InterW) * \
                        weight[gmm_sh] / weight[gmm_sh].sum()
                    
                    logging.info(f"=====> Quality-based weights: {weight}")

                ############ ↑↑↑↑↑↑ Key of Our Method FedA3I ↑↑↑↑↑↑ ############

    
            else:  #further training stage
                w_glob_fl = FedA3I_Agg(w_locals, weight, dict_len, args)
            
                netglob.load_state_dict(copy.deepcopy(w_glob_fl))


            if args.twoD:
                dval, hdval = test_all_slices(copy.deepcopy(netglob).to(args.device),
                                              copy.deepcopy(dataset_test),
                                              num_classes=args.n_classes,
                                              writer=writer,
                                              visualization=False,
                                              args=args
                                              )
                if args.dataset in ["ISIC2017", "BreastTS"]:
                    classwise_mean_dc = np.mean(dval, axis=1)
                    classwise_mean_hd = np.mean(hdval, axis=1)
                    assert len(classwise_mean_dc) == len(
                        classwise_mean_hd) == 1
                    dc = np.mean(classwise_mean_dc)
                    hd = np.mean(classwise_mean_hd)
                    logging.info(
                        "******** Evaluation ******** >>>> round: %d, Dice: %.4f, HD: %.4f \n" % (rnd, dc, hd))
                    writer.add_scalar(f'val_{run}/Dice', dc, rnd)
                    writer.add_scalar(f'val_{run}/HD', hd, rnd)
                    if rnd == args.rounds-1:
                        Metrics[run, :] = np.array([dc, hd])
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            if args.save:
                if (rnd+1)%10 == 0 and rnd<50:
                    torch.save(netglob.state_dict(), models_dir +
                            f'/model_run_{run}_rnd_{rnd}.pth')
                if rnd==args.s1-1:
                    torch.save(netglob.state_dict(), models_dir +
                            f'/model_run_{run}_rnd_{rnd}.pth')
        # save the last model
        torch.save(netglob.state_dict(), models_dir +
                   f'/model_run_{run}.pth')

        # a new run
        netglob = build_model(args)
        weight = None

    logging.info("\n################ mean: ################")
    logging.info(Metrics.mean(0))
    logging.info("\n################ std: ################")
    logging.info(Metrics.std(0))
    logging.info("\n################ all: ################")
    logging.info(Metrics)

    torch.cuda.empty_cache()

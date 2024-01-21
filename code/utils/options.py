import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')

    # basic setting
    parser.add_argument('--exp', type=str,
                        default='Fed', help='experiment name')
    parser.add_argument('--run', type=int,
                        default=5, help='run times')
    parser.add_argument('--dataset', type=str,
                        default='ISIC2017', help='dataset name')
    parser.add_argument('--model', type=str,
                        default='Unet', help='model name')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=0.005,
                        help='base learning rate')
    parser.add_argument('--save', action="store_true",
                        help='save every model?')
    parser.add_argument('--diceloss', action="store_true",
                        help='using dice loss?')
    parser.add_argument('--DiceLossWeight', type=float, default=0.5)

    # for FL
    parser.add_argument('--local_ep', type=int, default=5, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=100, help='rounds')

    parser.add_argument('--power', type=float, default=1.0)
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--bias', type=float, default=0.0)

    # for Noise
    parser.add_argument('--strength', type=int,  default=0)
    parser.add_argument('--s1', type=int, default=10)
    parser.add_argument('--InterW', type=float,  default=0.5)


    args = parser.parse_args()
    return args
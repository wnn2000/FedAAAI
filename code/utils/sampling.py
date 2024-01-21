import numpy as np


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
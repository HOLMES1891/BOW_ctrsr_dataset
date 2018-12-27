import numpy as np


def read_mult(f_in, D=8000):
    fp = open(f_in)
    lines = fp.readlines()
    X = np.zeros((len(lines), D))
    for i, line in enumerate(lines):
        strs = line.strip().split(' ')[1:]
        for strr in strs:
            segs = strr.split(':')
            X[i, int(segs[0])] = float(segs[1])
    arr_max = np.amax(X, axis=1)
    X = (X.T/arr_max).T
    return X.T


def get_mult():
    X = read_mult('ctrsr_datasets/citeulike-a/mult.dat', 8000).astype(np.float32)
    return X


def read_user(f_in='ctrsr_datasets/citeulike-a/cf-train-1-users.dat', num_u=5551, num_v=16980):
    fp = open(f_in)
    R = np.zeros((num_u, num_v))
    for i, line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        for seg in segs:
            R[i, int(seg)] = 1
    return R


if __name__ == "__main__":
    # 5551 users and 16980 items
    # bag of words feature vector only on item nodes
    feature_matrix = get_mult()    # (16980, 8000)
    R = read_user()     # (5551, 16980) user-item matrix




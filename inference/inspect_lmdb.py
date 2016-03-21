import numpy as np
import lmdb
import caffe

env = lmdb.open('mnist_fc1_train_lmdb', readonly=True)
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        print(key, value)

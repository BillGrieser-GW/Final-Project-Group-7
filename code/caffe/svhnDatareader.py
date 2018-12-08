import scipy.io as sio
import numpy as np
import caffe
import lmdb
import random
from caffe.proto import caffe_pb2

def main():
    #load matlab data
    train = sio.loadmat("Data/train_32x32.mat")
    test = sio.loadmat("Data/test_32x32.mat")

    X_train = train['X']
    y_train = train['y']
    X_test = test['X']
    y_test = test['y']

    #transpose the data

    X_train = X_train.transpose((3,2,0,1))
    X_test = X_test.transpose((3,2,0,1))

    N = X_train.shape[0]
    map_size = X_train.nbytes * 10

    env = lmdb.open("train_lmdb", map_size=map_size)
    txn = env.begin(write=True)
    r = list(range(N))
    random.shuffle(r)

    count = 0
    for i in r:
        datum = caffe_pb2.Datum()
        label = int(y_train[i][0])
        if label == 10:
            label = 0
        datum = caffe.io.array_to_datum(X_train[i], label)
        str_id = '{:08}'.format(count)
        txn.put(str_id, datum.SerializeToString())
        count = count + 1

    txn.commit()
    env.close()

    map_size_test = X_test.nbytes * 10

    env = lmdb.open("test_lmdb", map_size=map_size_test)
    txn = env.begin(write=True)

    count = 0
    for i in range(X_test.shape[0]):
        datum = caffe_pb2.Datum()
        label = int(y_test[i][0])
        if label == 10:
            label = 0
        datum = caffe.io.array_to_datum(X_test[i], label)
        str_id = '{:08}'.format(count)
        txn.put(str_id, datum.SerializeToString())
        count = count + 1

    txn.commit()
    env.close()

    print "Finished creating train and test lmdb"
    return 0


main()
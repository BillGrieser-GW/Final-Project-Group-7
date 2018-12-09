"""
Author: Darshan Kasat
reference: https://blog.csdn.net/yj3254/article/details/52370767
"""



import caffe
import numpy as np
import lmdb
from caffe.proto import caffe_pb2
import time

train_env = lmdb.open("train_lmdb")
train_txn = train_env.begin()
train_cursor = train_txn.cursor()
datum = caffe_pb2.Datum()

N=0
mean_array = np.zeros((1, 3, 32, 32))
for key, value in train_cursor:
    datum.ParseFromString(value)
    data = caffe.io.datum_to_array(datum)
    image = data.transpose(1,2,0)
    mean_array[0, 0] += image[:, :, 0]
    mean_array[0, 1] += image[:, :, 1]
    mean_array[0, 2] += image[:, :, 2]
    N = N+1

mean_array[0] /= N
blob = caffe.io.array_to_blobproto(mean_array)
with open('mean.binaryproto', 'wb') as f:
    f.write(blob.SerializeToString())

train_env.close()
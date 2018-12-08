import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import caffe
import lmdb
import random
from caffe.proto import caffe_pb2

print "Loading matlab data: "
mat_data = sio.loadmat("train_32x32.mat")
data = mat_data['X']
label = mat_data['y']
print(label)
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title(label[i][0])
    plt.imshow(data[..., i])
    plt.axis('off')
plt.show()

data = data.transpose((3, 2, 0, 1))
print(data.shape)

N = label.shape[0]
map_size = data.nbytes * 10
env = lmdb.open("train_lmdb", map_size=map_size, writemap=True)
txn = env.begin(write=True)
r = list(range(N))
random.shuffle(r)


count = 0
for i in r:
    datum = caffe_pb2.Datum()
    label_number = int(label[i][0])
    if label_number == 10:
        label_number == 0
    datum = caffe.io.array_to_datum(data[i], label_number)
    str_id = '{:08}'.format(count)
    txn.put(str_id, datum.SerializeToString())
    count = count + 1

    if count%1000 == 0:
        print("Already handled with {} pictures".format(count))
    txn.commit()
env.close()

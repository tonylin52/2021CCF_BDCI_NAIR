import numpy as np
import os

data_root = "/aiot_nfs/jzz_data/paddle_jzz/data/5folder_Resample_origin/"
for i in range(1, 6):
    print(data_root+str(i)+"/vallabel.npy")
    label = np.load(data_root+str(i)+"/vallabel.npy")
    print(len(label))

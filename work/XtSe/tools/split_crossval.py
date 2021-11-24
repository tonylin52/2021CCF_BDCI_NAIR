import numpy as np
import sys
import os

import paddle
sys.path.append("/home/aistudio/work/XtSe")
from paddlevideo.utils import get_config
import argparse
from sklearn.model_selection import KFold, StratifiedKFold

kps_origin, labels_origin = np.load("/home/aistudio/data/Res152/train_data.npy"), \
    np.load("/home/aistudio/data/Res152/train_label.npy")
kps_mask_plus_plus, labels_mask_plus_plus = None, None

def splitAll():
    kf = StratifiedKFold(n_splits=5,shuffle=True)
    num_folder = 1
    for train_index , test_index in kf.split(kps_origin, labels_origin):
        print(test_index)
        data_path = "/home/aistudio/data/5folder_new_"
        for suffix in ["origin"]:
            data_path = data_path + suffix + "/" + str(num_folder)
            print(data_path)
            if not os.path.exists(data_path):
                print(data_path+" not exists, create one!")
                os.makedirs(data_path)
            
            train_data = kps_origin[train_index] if suffix == "origin" else kps_mask_plus_plus[train_index]
            train_label = labels_origin[train_index] if suffix == "origin" else labels_mask_plus_plus[train_index]
            
            test_data = kps_origin[test_index] if suffix == "origin" else kps_mask_plus_plus[test_index]
            test_label = labels_origin[test_index] if suffix == "origin" else labels_mask_plus_plus[test_index]
            
            assert train_data.shape[0] == train_label.shape[0]
            assert test_data.shape[0] == test_label.shape[0]
            
            np.save(data_path+"/traindata.npy", train_data)
            np.save(data_path+"/trainlabel.npy", train_label)
            np.save(data_path+"/valdata.npy", test_data)
            np.save(data_path+"/vallabel.npy", test_label)
        
        num_folder += 1
    print("Already Have!!!")
        
splitAll()

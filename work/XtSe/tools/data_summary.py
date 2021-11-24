import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
sys.path.append("/aiot_nfs/chao/PaddleVideo_backbonewc")
from paddlevideo.utils import get_config
import argparse
from sklearn.model_selection import KFold, StratifiedKFold
from tools.utils import generate_a_heatmap

data_origin = np.load("/aiot_nfs/jzz_data/paddle_jzz/data/5folder_Resample_origin/1/traindata.npy")
labels_origin = np.load("/aiot_nfs/jzz_data/paddle_jzz/data/5folder_Resample_origin/1/trainlabel.npy")


def show_heatmap(source, kfolder=1):
    w, h = 200, 200
    
    data = np.load(os.path.join(source, str(kfolder), "traindata.npy"))
    # Sample
    for i in range(data.shape[0]):
        # Frame
        for j in range(data.shape[2]):
            # num_kps
            heatmaps = []
            for k in range(data.shape[3]):
                x, y = w//2*(1+data[i,0,j,k,0]), h//2*(1+data[i,1,j,k,0])
                heatmaps.append(generate_a_heatmap(w, h, (x, y), sigma=1))
            one_frame = np.stack(heatmaps, axis=-1)
            print(one_frame.shape)
            os._exit(0)
                
                

def distribution(labels, tag="origin"):
    x = [0 for _ in range(max(labels)+1)]
    for i in range(len(labels)):
        x[labels[i]] += 1

    
    plt.bar(range(len(x)), x, align='center', color='steelblue',alpha=0.8)
    plt.title('类别分布')
    plt.savefig("./tools/dist_"+tag+".png")
    return np.argsort(x)[:10]

def rewind(source, target, Flip=False, kfolder=1):
    supply_data, supply_label = [], []
    data_origin = np.load(os.path.join(source, str(kfolder), "traindata.npy"))
    label_origin = np.load(os.path.join(source, str(kfolder), "trainlabel.npy"))
    small_samples = distribution(label_origin)
    for i in tqdm(range(data_origin.shape[0])):
        if label_origin[i] not in small_samples:    continue
        # total_frame = 0
        # for j in range(data_origin.shape[2]-1, -1, -1):
        #     total_frame = j
        #     invalid = data_origin[i,:, j, :] == np.zeros((3, 25, 1))
        #     if (np.sum(invalid) != 75): break
        if Flip:
            supply_data.append(np.vstack([np.expand_dims(-1*data_origin[i,0,], 0), data_origin[i,1:3,]]))
            supply_label.append(label_origin[i])
        # data_origin[i,:,:total_frame,] = data_origin[i,:,total_frame:0:-1,]
        # supply_data.append(data_origin[i])
        # supply_label.append(label_origin[i])
    print(np.array(supply_data).shape)
        
    assert len(supply_data) == len(supply_label)
    
    if not os.path.exists(os.path.join(target, str(kfolder))):
        os.makedirs(os.path.join(target, str(kfolder)))
    
    np.save(os.path.join(target, str(kfolder), "traindata.npy"), np.vstack([data_origin, np.array(supply_data)]))
    np.save(os.path.join(target, str(kfolder), "trainlabel.npy"), np.hstack([label_origin, np.array(supply_label)]))
    distribution(np.load(os.path.join(target, str(kfolder), "trainlabel.npy")), "supply_filter_flip")


def Flip(source, target, kfolder=1):
    data_origin = np.load(os.path.join(source, str(kfolder), "traindata.npy"))
    label_origin = np.load(os.path.join(source, str(kfolder), "trainlabel.npy"))
    small_samples = distribution(label_origin)
    for i in tqdm(range(data_origin.shape[0])):
        if label_origin[i] not in small_samples:    continue
        data_origin[i,:2,] = -data_origin[i,:2,]
    
rewind("/aiot_nfs/jzz_data/paddle_jzz/data/5folder_Resample_copy", "/aiot_nfs/jzz_data/paddle_jzz/data/5folder_Resample_filter_flip", True)
# show_heatmap("/aiot_nfs/jzz_data/paddle_jzz/data/5folder_Resample_copy")
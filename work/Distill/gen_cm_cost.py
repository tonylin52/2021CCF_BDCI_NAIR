import sys,os
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse


def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo train script")
    # 模型预测的5折数据验证集标签地址，数字部分使用%d代替，例如： ../valid/nwc5-%d.npy
    parser.add_argument('-valpred',
                        type=str,
                        default='../../data/Distill/5fold_val_pred/nwc5-%d.npy',
                        help='5fold validate prediction')
    # 输出目录
    parser.add_argument('--output',
                        type=str,
                        default='../../data/Distill/new_train1/',
                        help='output dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # val_pred_dir = sys.argv[1]  # 模型预测的5折数据验证集标签地址，数字部分使用%d代替，例如： ../valid/nwc5-%d.npy
    # out_dir = sys.argv[2]  # 输出目录
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    cm = np.zeros((30,30))
    for i in range(1,6):
        data = np.load(args.valpred % i)
        pred = np.argmax(data[:,1:],axis=-1)
        label = data[:, 0]
        cm += confusion_matrix(label, pred)
    cm_cost = cm - np.diag(np.diag(cm))
    wrong_rate = np.sum(cm_cost,axis=1)/np.sum(cm,axis=1)
    wrong_rate = wrong_rate/np.sum(wrong_rate)
    wrong_rate = wrong_rate *100
    cm_cost_last = cm_cost * wrong_rate
    np.save("%s/cm_cost.npy"%args.output,cm_cost_last)
    # print(cm_cost_last)

import sys,os
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo train script")
    # 5折数据验证集的地址，数字部分使用%d代替
    parser.add_argument('--valdata',
                        type=str,
                        default='../../data/Distill/5fold_val_data/valdata_%d.npy',
                        help='5fold validate data')
    # 模型预测的5折数据验证集标签地址，数字部分使用%d代替，例如： ../valid/nwc5-%d.npy
    parser.add_argument('--valpred',
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
    # data_dir = sys.argv[1] # 5折数据验证集的地址，数字部分使用%d代替
    # val_pred_dir = sys.argv[2] # 模型预测的5折数据验证集标签地址，数字部分使用%d代替，例如： ../valid/nwc5-%d.npy
    # out_dir = sys.argv[3] # 输出目录
    args = parse_args()

    os.makedirs(args.output,exist_ok=True)

    val_data = []
    for i in range(1,6):
        val_data_path = args.valdata % i
        val_data_ = np.load(val_data_path)
        val_data.append(val_data_)
    val_data_np = np.vstack(val_data)
    np.save("%s/train_data.npy"%args.output, val_data_np)


    val_label = []
    for i in range(1,6):
        # val_label_path = "../valid/nwc5-%d.npy" % i
        val_label_path = args.valpred % i
        val_label_pred = np.load(val_label_path)
        pred_index = np.argmax(val_label_pred[:, 1:], axis=-1)
        lamda = np.max(val_label_pred[:, 1:], axis=-1)
        lamda[np.where(lamda > 0.428)] = 0.428
        label = val_label_pred[:, 0]
        val_label_i = np.stack([label, pred_index, lamda], axis=-1)
        val_label.append(val_label_i)

    val_label_np = np.vstack(val_label)
    np.save("%s/train_label.npy"%args.output,val_label_np)

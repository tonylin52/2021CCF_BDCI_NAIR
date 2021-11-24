export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#export FLAGS_conv_workspace_size_limit=800 #MB
#export FLAGS_cudnn_exhaustive_search=1
#export FLAGS_cudnn_batchnorm_spatial_persistent=1

t_path="/home/aistudio/data/Res152/train_data.npy"
label_path="/home/aistudio/data/Res152/train_label.npy"

start_time=$(date +%s)


CUDA_VISIBLE_DEVICES=0 nohup python3.7 main.py -c ./config_old/PoseC3D_total.yaml --t_path $t_path --label_path $label_path  > ./logs/preBN_ls_5E-4_retrain.log  2>&1 &


end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "Time to train is $(($cost_time/60))min $(($cost_time%60))s"

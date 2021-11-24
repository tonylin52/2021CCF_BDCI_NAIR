export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#export FLAGS_conv_workspace_size_limit=800 #MB
#export FLAGS_cudnn_exhaustive_search=1
#export FLAGS_cudnn_batchnorm_spatial_persistent=1

test_path="/home/aistudio/data/test_B_data.npy"
weight_path="/home/aistudio/work/Res152/output/PoseC3D_olddata_res152_5E-4_preBN_ls_epoch_00191.pdparams"

start_time=$(date +%s)


CUDA_VISIBLE_DEVICES=0 nohup python3.7 main.py --test -c ./config_old/PoseC3D_total.yaml --t_path $test_path --output_name submission152_olddata_5E-4_200_preBN_ls_B.csv -w $weight_path  > ./logs/preBN_ls_5E-4_test.log  2>&1 &


end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "Time to train is $(($cost_time/60))min $(($cost_time%60))s"

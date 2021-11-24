CUDA_VISIBLE_DEVICES=0 nohup python main.py -c resnet152_5folder/PoseC3D-resnet152-1.yaml -w resnet152_5folder/nwc5-1-resnet152_epoch_00181.pdparams --val  2>&1 
CUDA_VISIBLE_DEVICES=0 nohup python main.py -c resnet152_5folder/PoseC3D-resnet152-2.yaml -w resnet152_5folder/nwc5-2-resnet152_epoch_00186.pdparams --val  2>&1 
CUDA_VISIBLE_DEVICES=0 nohup python main.py -c resnet152_5folder/PoseC3D-resnet152-3.yaml -w resnet152_5folder/nwc5-3-resnet152_epoch_00171.pdparams --val  2>&1 
CUDA_VISIBLE_DEVICES=0 nohup python main.py -c resnet152_5folder/PoseC3D-resnet152-4.yaml -w resnet152_5folder/nwc5-4-resnet152_epoch_00181.pdparams --val  2>&1 
CUDA_VISIBLE_DEVICES=0 nohup python main.py -c resnet152_5folder/PoseC3D-resnet152-5.yaml -w resnet152_5folder/nwc5-5-resnet152_epoch_00186.pdparams --val  2>&1 

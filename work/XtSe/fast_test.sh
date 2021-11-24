# test DSE50 with trained weight
CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose.yaml -m DSe -n submission_ResDSe50_1.csv -w output/TestB/ResDSe50_1_folder_1_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose.yaml -m DSe -n submission_ResDSe50_2.csv -w output/TestB/ResDSe50_2_folder_2_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose.yaml -m DSe -n submission_ResDSe50_3.csv -w output/TestB/ResDSe50_3_folder_3_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python main.py -c configs/pose.yaml -m DSe -n submission_ResDSe50_4.csv -w output/TestB/ResDSe50_4_folder_4_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python main.py -c configs/pose.yaml -m DSe -n submission_ResDSe50_5.csv -w output/TestB/ResDSe50_5_folder_5_best.pdparams --test 2>&1 &

# test Xt50 with trained weight
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose.yaml -n submission_ResXt50_1.csv -w output/TestB/ResXt50_1_folder_1_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose.yaml -n submission_ResXt50_2.csv -w output/TestB/ResXt50_2_folder_2_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose.yaml -n submission_ResXt50_3.csv -w output/TestB/ResXt50_3_folder_3_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python main.py -c configs/pose.yaml -n submission_ResXt50_4.csv -w output/TestB/ResXt50_4_folder_4_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python main.py -c configs/pose.yaml -n submission_ResXt50_5.csv -w output/TestB/ResXt50_5_folder_5_best.pdparams --test 2>&1 &

# test Xt128 with trained weight
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose128.yaml -n submission_ResXt128_1.csv -w output/TestB/ResXt128_1_folder_1_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose128.yaml -n submission_ResXt128_2.csv -w output/TestB/ResXt128_2_folder_2_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose128.yaml -n submission_ResXt128_3.csv -w output/TestB/ResXt128_3_folder_3_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python main.py -c configs/pose128.yaml -n submission_ResXt128_4.csv -w output/TestB/ResXt128_4_folder_4_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python main.py -c configs/pose128.yaml -n submission_ResXt128_5.csv -w output/TestB/ResXt128_5_folder_5_best.pdparams --test 2>&1 &

# test Xt101 with trained weight
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose101.yaml -n submission_ResXt101_1.csv -w output/TestB/ResXt101_1_folder_1_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose101.yaml -n submission_ResXt101_2.csv -w output/TestB/ResXt101_2_folder_2_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose101.yaml -n submission_ResXt101_3.csv -w output/TestB/ResXt101_3_folder_3_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python main.py -c configs/pose101.yaml -n submission_ResXt101_4.csv -w output/TestB/ResXt101_4_folder_4_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python main.py -c configs/pose101.yaml -n submission_ResXt101_5.csv -w output/TestB/ResXt101_5_folder_5_best.pdparams --test 2>&1 &

# test Xt56 with trained weight
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose5.yaml -n submission_Xt56_1.csv -w output/TestB/ResXt56_1_folder_1_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose5.yaml -n submission_Xt56_2.csv -w output/TestB/ResXt56_2_folder_2_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/pose5.yaml -n submission_Xt56_3.csv -w output/TestB/ResXt56_3_folder_3_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python main.py -c configs/pose5.yaml -n submission_Xt56_4.csv -w output/TestB/ResXt56_4_folder_4_best.pdparams --test 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python main.py -c configs/pose5.yaml -n submission_Xt56_5.csv -w output/TestB/ResXt56_5_folder_5_best.pdparams --test 2>&1 &
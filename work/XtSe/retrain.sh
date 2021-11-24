# retrain XtSe50
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose.yaml -w output/ResXt50_1/ResXt50_1_folder_1_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 1 -n ResXt50_1 --validate >> logs/ResXt50_1.out 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose.yaml -w output/ResXt50_2/ResXt50_2_folder_2_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 2 -n ResXt50_2 --validate >> logs/ResXt50_2.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose.yaml -w output/ResXt50_3/ResXt50_3_folder_3_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 3 -n ResXt50_3 --validate >> logs/ResXt50_3.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose.yaml -w output/ResXt50_4/ResXt50_4_folder_4_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 4 -n ResXt50_4 --validate >> logs/ResXt50_4.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -c configs/pose.yaml -w output/ResXt50_5/ResXt50_5_folder_5_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 5 -n ResXt50_5 --validate >> logs/ResXt50_5.out 2>&1 &

# retrain XtSe56
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose5.yaml -w output/ResXt56_1/ResXt56_1_folder_1_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 1 -n ResXt56_1 --validate >> logs/ResXt56_1.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose5.yaml -w output/ResXt56_2/ResXt56_2_folder_2_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 2 -n ResXt56_2 --validate >> logs/ResXt56_2.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -c configs/pose5.yaml -w output/ResXt56_3/ResXt56_3_folder_3_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 3 -n ResXt56_3 --validate >> logs/ResXt56_3.out 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -u main.py -c configs/pose5.yaml -w output/ResXt56_4/ResXt56_4_folder_4_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 4 -n ResXt56_4 --validate >> logs/ResXt56_4.out 2>&1 &
#CUDA_VISIBLE_DEVICES=4 nohup python -u main.py -c configs/pose5.yaml -w output/ResXt56_5/ResXt56_5_folder_5_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 5 -n ResXt56_5 --validate >> logs/ResXt56_5.out 2>&1 &

# retrain XtSe128
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose128.yaml -w output/ResXt128_1/ResXt128_1_folder_1_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 1 -n ResXt128_1 --validate >> logs/ResXt128_1.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose128.yaml -w output/ResXt128_2/ResXt128_2_folder_2_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 2 -n ResXt128_2 --validate >> logs/ResXt128_2.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -c configs/pose128.yaml -w output/ResXt128_3/ResXt128_3_folder_3_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 3 -n ResXt128_3 --validate >> logs/ResXt128_3.out 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -u main.py -c configs/pose128.yaml -w output/ResXt128_4/ResXt128_4_folder_4_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 4 -n ResXt128_4 --validate >> logs/ResXt128_4.out 2>&1 &
#CUDA_VISIBLE_DEVICES=4 nohup python -u main.py -c configs/pose128.yaml -w output/ResXt128_5/ResXt128_5_folder_5_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 5 -n ResXt128_5 --validate >> logs/ResXt128_5.out 2>&1 &

# retrain XtSe101
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose101.yaml -w output/ResXt101_1/ResXt101_1_folder_1_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 1 -n ResXt101_1 --validate >> logs/ResXt101_1.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose101.yaml -w output/ResXt101_2/ResXt101_2_folder_2_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 2 -n ResXt101_2 --validate >> logs/ResXt101_2.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -c configs/pose101.yaml -w output/ResXt101_3/ResXt101_3_folder_3_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 3 -n ResXt101_3 --validate >> logs/ResXt101_3.out 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -u main.py -c configs/pose101.yaml -w output/ResXt101_4/ResXt101_4_folder_4_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 4 -n ResXt101_4 --validate >> logs/ResXt101_4.out 2>&1 &
#CUDA_VISIBLE_DEVICES=4 nohup python -u main.py -c configs/pose101.yaml -w output/ResXt101_5/ResXt101_5_folder_5_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 5 -n ResXt101_5 --validate >> logs/ResXt101_5.out 2>&1 &


# retrain DSe50
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose.yaml -w output/ResDSe50_1/ResDSe50_1_folder_1_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 1 -m DSe -n ResDSe50_1 --validate >> logs/ResDSe50_1.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose.yaml -w output/ResDSe50_2/ResDSe50_2_folder_2_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 2 -m DSe -n ResDSe50_2 --validate >> logs/ResDSe50_2.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -c configs/pose.yaml -w output/ResDSe50_3/ResDSe50_3_folder_3_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 3 -m DSe -n ResDSe50_3 --validate >> logs/ResDSe50_3.out 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -u main.py -c configs/pose.yaml -w output/ResDSe50_4/ResDSe50_4_folder_4_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 4 -m DSe -n ResDSe50_4 --validate >> logs/ResDSe50_4.out 2>&1 &
#CUDA_VISIBLE_DEVICES=4 nohup python -u main.py -c configs/pose.yaml -w output/ResDSe50_5/ResDSe50_5_folder_5_best.pdparams -f /home/aistudio/data/XtSe -l 0.07 -e 120 -k 5 -m DSe -n ResDSe50_5 --validate >> logs/ResDSe50_5.out 2>&1 &


# train XtSe50
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose.yaml -f /home/aistudio/data/XtSe -k 1 -n ResXt50_1 --validate > logs/ResXt50_1.out 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose.yaml -f /home/aistudio/data/XtSe -k 2 -n ResXt50_2 --validate > logs/ResXt50_2.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose.yaml -f /home/aistudio/data/XtSe -k 3 -n ResXt50_3 --validate > logs/ResXt50_3.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose.yaml -f /home/aistudio/data/XtSe -k 4 -n ResXt50_4 --validate > logs/ResXt50_4.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -c configs/pose.yaml -f /home/aistudio/data/XtSe -k 5 -n ResXt50_5 --validate > logs/ResXt50_5.out 2>&1 &

# train XtSe56
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose5.yaml -f /home/aistudio/data/XtSe -k 1 -n ResXt56_1 --validate > logs/ResXt56_1.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose5.yaml -f /home/aistudio/data/XtSe -k 2 -n ResXt56_2 --validate > logs/ResXt56_2.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -c configs/pose5.yaml -f /home/aistudio/data/XtSe -k 3 -n ResXt56_3 --validate > logs/ResXt56_3.out 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -u main.py -c configs/pose5.yaml -f /home/aistudio/data/XtSe -k 4 -n ResXt56_4 --validate > logs/ResXt56_4.out 2>&1 &
#CUDA_VISIBLE_DEVICES=4 nohup python -u main.py -c configs/pose5.yaml -f /home/aistudio/data/XtSe -k 5 -n ResXt56_5 --validate > logs/ResXt56_5.out 2>&1 &

# train XtSe128
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose128.yaml -f /home/aistudio/data/XtSe -k 1 -n ResXt128_1 --validate > logs/ResXt128_1.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose128.yaml -f /home/aistudio/data/XtSe -k 2 -n ResXt128_2 --validate > logs/ResXt128_2.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -c configs/pose128.yaml -f /home/aistudio/data/XtSe -k 3 -n ResXt128_3 --validate > logs/ResXt128_3.out 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -u main.py -c configs/pose128.yaml -f /home/aistudio/data/XtSe -k 4 -n ResXt128_4 --validate > logs/ResXt128_4.out 2>&1 &
#CUDA_VISIBLE_DEVICES=4 nohup python -u main.py -c configs/pose128.yaml -f /home/aistudio/data/XtSe -k 5 -n ResXt128_5 --validate > logs/ResXt128_5.out 2>&1 &

# train XtSe101
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose101.yaml -f /home/aistudio/data/XtSe -k 1 -n ResXt101_1 --validate > logs/ResXt101_1.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose101.yaml -f /home/aistudio/data/XtSe -k 2 -n ResXt101_2 --validate > logs/ResXt101_2.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -c configs/pose101.yaml -f /home/aistudio/data/XtSe -k 3 -n ResXt101_3 --validate > logs/ResXt101_3.out 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -u main.py -c configs/pose101.yaml -f /home/aistudio/data/XtSe -k 4 -n ResXt101_4 --validate > logs/ResXt101_4.out 2>&1 &
#CUDA_VISIBLE_DEVICES=4 nohup python -u main.py -c configs/pose101.yaml -f /home/aistudio/data/XtSe -k 5 -n ResXt101_5 --validate > logs/ResXt101_5.out 2>&1 &

# train DSe50
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -c configs/pose.yaml -f /home/aistudio/data/XtSe -k 1 -m DSe -n ResDSe50_1 --validate > logs/ResDSe50_1.out 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u main.py -c configs/pose.yaml -f /home/aistudio/data/XtSe -k 2 -m DSe -n ResDSe50_2 --validate > logs/ResDSe50_2.out 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -c configs/pose.yaml -f /home/aistudio/data/XtSe -k 3 -m DSe -n ResDSe50_3 --validate > logs/ResDSe50_3.out 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -u main.py -c configs/pose.yaml -f /home/aistudio/data/XtSe -k 4 -m DSe -n ResDSe50_4 --validate > logs/ResDSe50_4.out 2>&1 &
#CUDA_VISIBLE_DEVICES=4 nohup python -u main.py -c configs/pose.yaml -f /home/aistudio/data/XtSe -k 5 -m DSe -n ResDSe50_5 --validate > logs/ResDSe50_5.out 2>&1 &

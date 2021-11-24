

test_B=../../data/test_B_data.npy
CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/self_distill_confusion.yaml -w model_for_test//self_distill_confusion_epoch_00120.pdparams --test -o METRIC.out_file=B_submission_self_distill_confusion_vote.csv -o DATASET.test.keypoint_file=$test_B   2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python main.py -c configs/self_distill_confusion.yaml -w model_for_test//self_distill2_epoch_00120.pdparams --test -o METRIC.out_file=B_submission_self_distill2_vote.csv -o DATASET.test.keypoint_file=$test_B   2>&1 &

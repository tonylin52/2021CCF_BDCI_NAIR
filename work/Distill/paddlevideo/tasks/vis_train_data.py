from paddlevideo.utils import get_config
from paddlevideo.loader.builder import build_dataset

config_file = "/aiot_nfs/zhaohe/PaddleCompete/PaddleVideo_jzz2/configs_new/nwc5-1-filtered.yaml"
cfg = get_config(config_file)
del cfg.PIPELINE.valid.transform.GeneratePoseTarget
del cfg.PIPELINE.valid.transform.FormatShape

valid_dataset = build_dataset((cfg.DATASET.valid, cfg.PIPELINE.valid))
for results in valid_dataset:
    print(results['keypoint'].shape)
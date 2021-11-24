# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import paddle
import os.path as osp
from paddlevideo.utils import get_logger
from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from ..modeling.builder import build_model
from paddlevideo.utils import load

logger = get_logger("paddlevideo")


def load_model(cfg,weights,parallel=True):
    cfg.MODEL.head.num_classes = cfg.DATASET.train.num_classes - len(cfg.DATASET.train.class_filter) + 1
    if cfg.MODEL.backbone.get('pretrained'):
        cfg.MODEL.backbone.pretrained = ''  # disable pretrain model init
    model = build_model(cfg.MODEL)
    if parallel:
        model = paddle.DataParallel(model)

    model.eval()

    state_dicts = load(weights)
    model.set_state_dict(state_dicts)
    return model

@paddle.no_grad()
def val_2model(cfg_large, weights_large,cfg_small,weights_small, parallel=False):
    """Test model entry

    Args:
        cfg (dict): configuration.
        weights (str): weights path to load.
        parallel (bool): Whether to do multi-cards testing. Default: True.

    """
    # 1. Construct model.
    model_large = load_model(cfg_large,weights_large,parallel)
    model_small = load_model(cfg_small, weights_small, parallel)
    # 2. Construct dataset and dataloader.
    small_class = cfg_large.DATASET.valid.class_filter
    remain_classes = [i for i in range(cfg_large.DATASET.train.num_classes) if i not in small_class]

    id2class = {i: remain_classes[i] for i in range(len(remain_classes))}
    id2class[len(remain_classes)] = -1

    cfg_large.DATASET.valid.class_filter = []
    dataset = build_dataset((cfg_large.DATASET.valid, cfg_large.PIPELINE.valid))
    batch_size = cfg_large.DATASET.get("test_batch_size", 8)
    places = paddle.set_device('gpu')
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg_large.DATASET.get('num_workers', 0)
    num_workers = cfg_large.DATASET.get('test_num_workers', num_workers)
    dataloader_setting = dict(batch_size=batch_size,
                              num_workers=num_workers,
                              places=places,
                              drop_last=False,
                              shuffle=False)

    data_loader = build_dataloader(dataset, **dataloader_setting)

    # add params to metrics
    cfg_large.METRIC.data_size = len(dataset)
    cfg_large.METRIC.batch_size = batch_size

    total_num = len(dataset)
    true1 = 0
    true5 = 0
    res = []
    for batch_id, data in enumerate(data_loader):
        label = data[1].numpy()[0]
        outputs, classcore = model_large(data, mode='valid')
        classcore = classcore.numpy()
        pred = np.argmax(classcore,axis=-1)[0]
        # print(outputs)
        pred_cls = id2class[pred]
        if pred_cls == -1:
            outputs, classcore = model_small(data, mode='valid')
            pred = np.argmax(classcore,axis=-1)[0]
            print(pred,label)
            pred_cls = small_class[pred]
        r = 0
        if pred_cls == label:
            r = 1
        true1 += r
        # true5 += outputs['top5'].item()
        res.append([label, pred_cls, r])

        print(label, true1, batch_id + 1, "top1:", true1 / (batch_id + 1), len(res))

    res = np.array(res, dtype=np.int32)
    np.save(osp.join(cfg_large.output_dir,"merge"+cfg_large.METRIC.val_npy), res)
    print("total_top1:", true1 / total_num, "total_top5:", true5 / total_num)




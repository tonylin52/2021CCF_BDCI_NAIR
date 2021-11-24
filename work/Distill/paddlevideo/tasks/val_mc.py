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
import os
import numpy as np
import paddle
from paddlevideo.utils import get_logger
from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from ..modeling.builder import build_model
from paddlevideo.utils import load

logger = get_logger("paddlevideo")


@paddle.no_grad()
def mcval_model(cfg, weights, parallel=True):
    """Test model entry

    Args:
        cfg (dict): configuration.
        weights (str): weights path to load.
        parallel (bool): Whether to do multi-cards testing. Default: True.

    """
    # 1. Construct model.
    if cfg.MODEL.backbone.get('pretrained'):
        cfg.MODEL.backbone.pretrained = ''  # disable pretrain model init
    model = build_model(cfg.MODEL)
    if parallel:
        model = paddle.DataParallel(model)
        
    model.eval()
    model.head.dropout.train()
    # model.backbone.dropout1.train()
    # model.backbone.dropout2.train()

    state_dicts = load(weights)
    model.set_state_dict(state_dicts)
    # 2. Construct dataset and dataloader.
    # cfg.DATASET.test.test_mode = True
    dataset = build_dataset((cfg.DATASET.valid, cfg.PIPELINE.valid))
    print("valid data size: ", len(dataset))
    batch_size = cfg.DATASET.get("test_batch_size", 8)
    places = paddle.set_device('gpu')
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    dataloader_setting = dict(batch_size=batch_size,
                              num_workers=num_workers,
                              places=places,
                              drop_last=False,
                              shuffle=False)

    data_loader = build_dataloader(dataset, **dataloader_setting)



    # add params to metrics
    cfg.METRIC.data_size = len(dataset)
    cfg.METRIC.batch_size = batch_size

    total_num = len(dataset)
    # true1 = 0
    # true5 = 0
    # res = []
    sd_total = []
    pred_total = []
    targets_total = []
    mean_total = []
    for batch_id, data in enumerate(data_loader):
        # label = data[1].numpy()[0]
        repeat_result = []
        for i in range(1):
            outputs,classcore = model(data, mode='valid')
            # classcore = classcore.numpy()
            probs = paddle.nn.Softmax(axis=-1)(classcore)
            repeat_result.append(classcore)
        repeat_result = paddle.stack(repeat_result)
        m = paddle.mean(repeat_result, axis=0)
        sd = paddle.var(repeat_result, axis=0)
        mprob = paddle.nn.Softmax(axis=-1)(m)
        predicts = paddle.argmax(mprob, 1)
        # sd_max = paddle.max(sd, 1)
        m_max= paddle.max(m, 1)
        # print(sd.shape)
        # print(predicts)
        sd_max = sd[0,predicts]

        sd_total.extend(sd_max.cpu().numpy().tolist())
        pred_total.extend(predicts.cpu().numpy().tolist())
        targets_total.extend(data[1].numpy().tolist())
        mean_total.extend(m_max.cpu().numpy().tolist())
        # print(label,true1,batch_id+1,"top1:",true1/(batch_id+1),true5,batch_id+1,"top1:",true5/(batch_id+1),len(res))

    # res = np.array(res,dtype=np.int32)
    # np.save(cfg.METRIC.val_npy,res)
    with open(os.path.join(cfg.output_dir,"mc_pred_mean_alldrop.csv"), "w") as f1:
        f1.write("grt,pred,max_sd,right,mean\n")
        for grt, pred, sd, m in zip(targets_total, pred_total, sd_total, mean_total):
            f1.write("{},{},{},{},{}\n".format(grt, pred, sd, pred == grt, m))

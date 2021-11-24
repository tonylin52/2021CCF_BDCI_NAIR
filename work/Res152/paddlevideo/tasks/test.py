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
import pandas as pd
import numpy as np
import paddle
from paddlevideo.utils import get_logger
from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from ..modeling.builder import build_model
from paddlevideo.utils import load

logger = get_logger("paddlevideo")

def write2csv(a, b,name):
    #字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'sample_index':a,'predict_category':b})

    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(name,index=False,sep=',')


@paddle.no_grad()
def test_model(cfg, weights, test_path, output, parallel=True):
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

    state_dicts = load(weights)
    model.set_state_dict(state_dicts)

    # 2. Construct dataset and dataloader.
    cfg.DATASET.test.test_mode = True
    cfg.DATASET.test.keypoint_file = test_path
    dataset = build_dataset((cfg.DATASET.test, cfg.PIPELINE.test))
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
    
    a = range(len(dataset))
    b = []
    conf = np.zeros([0,30],dtype=np.float64)
    # Metric = build_metric(cfg.METRIC)
    for batch_id, data in enumerate(data_loader):
        outputs,max_id = model(data, mode='test')
        
        # max_id = np.argmax(outputs.numpy())
        b.append(max_id)
        # co = outputs.numpy().reshape([-1,30])
        # conf = np.concatenate([conf,co],axis=0)
        print(batch_id,max_id)
    #     Metric.update(batch_id, data, outputs)
    # Metric.accumulate()
    write2csv(a, b,output)
    #np.save(cfg.METRIC.conf_npy,conf)

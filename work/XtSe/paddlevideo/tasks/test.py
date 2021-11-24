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
import os
import paddle
from tqdm import tqdm
from paddlevideo.utils import get_logger
from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from sklearn.metrics import confusion_matrix
from ..modeling.builder import build_model
from paddlevideo.utils import load

logger = get_logger("paddlevideo")


def write2csv(a, b,name):
    #字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'sample_index':a,'predict_category':b})

    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(name,index=False,sep=',')

@paddle.no_grad()
def test_model(cfg, weights, out_name, k = 0, parallel=True):
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
    # for p in model.state_dict():
    #     print(p,model.state_dict()[p])
    # 2. Construct dataset and dataloader.
    cfg.DATASET.test.test_mode = True
    dataset = build_dataset((cfg.DATASET.test, cfg.PIPELINE.test))
    batch_size = cfg.DATASET.get("test_batch_size", 8)
    places = paddle.set_device('gpu')
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 4)
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
    score = []
    Metric = build_metric(cfg.METRIC)
    for batch_id, data in tqdm(enumerate(data_loader)):
        outputs = model(data, mode='test')
        top5_idx = outputs.numpy().mean(axis=0).argsort()[::-1][:5]
        #score.append(outputs.mean(axis = 0))
        #max_id = np.argmax(outputs.numpy())
        max_id = np.argmax(np.bincount(paddle.argmax(outputs, axis=1).numpy()))
        b.append(max_id)
        # Metric.update(batch_id, data, outputs)
    # Metric.accumulate()
    #score_path = "./scores/"+out_name.split(".")[0]+"/"
    #if not os.path.exists(score_path):
    #    os.makedirs(score_path)
    #np.save("./scores/"+out_name.split(".")[0]+"/"+str(k)+".npy", np.array(score))
    write2csv(a, b,out_name.split(".")[0]+"_"+str(k)+".csv")

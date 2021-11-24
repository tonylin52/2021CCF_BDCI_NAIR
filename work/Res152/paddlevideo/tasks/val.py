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
from paddlevideo.utils import get_logger
from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from ..modeling.builder import build_model
from paddlevideo.utils import load

logger = get_logger("paddlevideo")


@paddle.no_grad()
def val_model(cfg, weights, parallel=True):
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
    dataset = build_dataset((cfg.DATASET.valid, cfg.PIPELINE.valid))
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
    true1 = 0
    true5 = 0
    res = []
    for batch_id, data in enumerate(data_loader):
        label = data[1].numpy()[0]
        outputs,classcore = model(data, mode='valid')
        classcore = classcore.numpy()
        true1 += outputs['top1'].item()
        true5 += outputs['top5'].item()
        # res.append([label,np.argmax(classcore),outputs['top1'].item()])
        list_ = []
        list_.append(float(label))
        for j in range(classcore.shape[1]):
            list_.append(classcore[0,j])

        res.append(list_)
        print(label,true1,batch_id+1,"top1:",true1/(batch_id+1),true5,batch_id+1,"top5:",true5/(batch_id+1),len(res))


    res = np.array(res,dtype=np.float32)
    np.save(cfg.METRIC.val_npy,res)
    print("total_top1:",true1/total_num,"total_top5:",true5/total_num)





# import numpy as np
# import paddle
# import os
# from paddlevideo.utils import get_logger
# from ..loader.builder import build_dataloader, build_dataset
# from ..metrics import build_metric
# from ..modeling.builder import build_model
# from paddlevideo.utils import load

# logger = get_logger("paddlevideo")

# def enable(model):
#     for m in model.modules():
#         if m.__class__.__name__.startswith('Dropout'):
#             print("&&&&&&&&",m.__class__.__name__)
#             # m.train()


# @paddle.no_grad()
# def val_model(cfg, weights, parallel=True):
#     """Test model entry

#     Args:
#         cfg (dict): configuration.
#         weights (str): weights path to load.
#         parallel (bool): Whether to do multi-cards testing. Default: True.

#     """
#     # 1. Construct model.
#     if cfg.MODEL.backbone.get('pretrained'):
#         cfg.MODEL.backbone.pretrained = ''  # disable pretrain model init
#     model = build_model(cfg.MODEL)
#     if parallel:
#         model = paddle.DataParallel(model)
        
#     # model.train()
#     model.eval()

#     state_dicts = load(weights)
#     model.set_state_dict(state_dicts)
#     # 2. Construct dataset and dataloader.
#     cfg.DATASET.test.test_mode = True
#     dataset = build_dataset((cfg.DATASET.valid, cfg.PIPELINE.valid))
#     batch_size = cfg.DATASET.get("test_batch_size", 8)
#     places = paddle.set_device('gpu')
#     # default num worker: 0, which means no subprocess will be created
#     num_workers = cfg.DATASET.get('num_workers', 0)
#     num_workers = cfg.DATASET.get('test_num_workers', num_workers)
#     dataloader_setting = dict(batch_size=batch_size,
#                               num_workers=num_workers,
#                               places=places,
#                               drop_last=False,
#                               shuffle=False)

#     data_loader = build_dataloader(dataset, **dataloader_setting)
#     # enable(model)
#     model.head.dropout.train()
#     print(model.head.dropout.training)

#     # add params to metrics
#     cfg.METRIC.data_size = len(dataset)
#     cfg.METRIC.batch_size = batch_size

#     total_num = len(dataset)
#     path_ = "/aiot_nfs/jzz_data/paddle_jzz/PaddleVideo_jzz2/cc/"
#     for batch_id, data in enumerate(data_loader):
#         label = data[1].numpy()[0]
#         res = np.zeros([0,30])
#         for i in range(20):
#             classcore = model(data, mode='valid')
#             classcore = classcore.numpy()
#             res = np.concatenate([res,classcore],axis=0)
        
#         print(batch_id,label,res.shape)
#         np.save(os.path.join(path_,str(batch_id)+"_"+str(label)+".npy"),res)
#     #     true1 += outputs['top1'].item()
#     #     true5 += outputs['top5'].item()
#     #     res.append([label,np.argmax(classcore),outputs['top1'].item()])
        
#     #     print(label,true1,batch_id+1,"top1:",true1/(batch_id+1),true5,batch_id+1,"top1:",true5/(batch_id+1),len(res))

#     # res = np.array(res,dtype=np.int32)
#     # np.save(cfg.METRIC.val_npy,res)
#     # print("total_top1:",true1/total_num,"total_top5:",true5/total_num)

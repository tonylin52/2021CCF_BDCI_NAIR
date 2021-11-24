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
import tqdm
from paddlevideo.utils import get_logger
from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from ..modeling.builder import build_model
from paddlevideo.utils import load

logger = get_logger("paddlevideo")


@paddle.no_grad()
def extract_model(cfg, weights, parallel=True):
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
    # cfg.DATASET.test.test_mode = True

    val_dataset = build_dataset((cfg.DATASET.valid, cfg.PIPELINE.test))
    batch_size = cfg.DATASET.get("test_batch_size", 8)
    places = paddle.set_device('gpu')
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    num_workers = cfg.DATASET.get('test_num_workers', num_workers)

    train_dataset = build_dataset((cfg.DATASET.train, cfg.PIPELINE.test))
    test_dataset = build_dataset((cfg.DATASET.test, cfg.PIPELINE.test))
    val_dataloader_setting = dict(batch_size=batch_size,
                              num_workers=num_workers,
                              places=places,
                              drop_last=False,
                              shuffle=False)

    train_loader = build_dataloader(train_dataset,  **val_dataloader_setting)
    val_data_loader = build_dataloader(val_dataset, **val_dataloader_setting)
    test_loader = build_dataloader(test_dataset,  **val_dataloader_setting)


    # train_features = []
    # for batch_id, data in tqdm.tqdm(enumerate(train_loader)):
    #     feature = model(data, mode='extract')
    #     # train_features.append(feature)
    #     np.save("extract_features/train_features_%d.npy"%batch_id, feature)
    #
    # # train_features = paddle.stack(train_features).numpy()
    # # np.save("train_features.npy", train_features)
    # print("save train feature to train_features.npy")
    #
    # # val_features = []
    # for batch_id, data in enumerate(val_data_loader):
    #     feature = model(data, mode='extract')
    #     np.save("extract_features/val_features_%d.npy" % batch_id, feature)
    #     # val_features.append(feature)
    #
    # # val_features = paddle.stack(val_features).numpy()
    #
    # # np.save("val_features.npy", val_features)
    # print("save val feature to val_features.npy")

    for batch_id, data in enumerate(test_loader):
        feature = model(data, mode='extract')
        np.save("extract_features/test_features_%d.npy" % batch_id, feature)
        # val_features.append(feature)

    # val_features = paddle.stack(val_features).numpy()

    # np.save("val_features.npy", val_features)
    print("save test feature to test_features.npy")

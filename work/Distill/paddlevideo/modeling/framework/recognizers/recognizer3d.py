# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ...registry import RECOGNIZERS
from .base import BaseRecognizer
from paddlevideo.utils import get_logger
import paddle

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class Recognizer3D(BaseRecognizer):
    """3D Recognizer model framework.
    """
    def forward_net(self, imgs,labels=None):
        """Define how the model is going to run, from input to output.
        """
        feature = self.backbone(imgs)
        cls_score = self.head(feature)
        return cls_score

    def train_step(self, data_batch):
        """Training step.
        """
        # imgs = data_batch[0:2]
        # labels = data_batch[2:]
        
        imgs = data_batch[0]
        labels = data_batch[1]
        # print(labels.shape)
        if labels.shape[-1] == 3:
            labels = [labels[:,0].astype('int64').unsqueeze(axis=-1),
                      labels[:,1].astype('int64').unsqueeze(axis=-1),labels[:,2]]
        else:
            labels = [paddle.reshape(labels, shape=(labels.shape[0], 1))]

        # call forward
        cls_score = self.forward_net(imgs,labels)
        loss_metrics = self.head.loss(cls_score, labels)
        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        # imgs = data_batch[0:2]
        # labels = data_batch[2:]
        imgs = data_batch[0]
        labels = data_batch[1]
        imgs = imgs.squeeze(0)
        # call forward
        cls_score = self.forward_net(imgs)
        cls_score = cls_score.mean(axis = 0)

        if labels.shape[-1] == 3:
            labels = [labels[:,0].astype('int64').unsqueeze(axis=-1),
                      labels[:,1].astype('int64').unsqueeze(axis=-1),labels[:,2]]
        else:
            labels = [paddle.reshape(labels, shape=(labels.shape[0], 1))]
  
        # labels = paddle.reshape(labels, shape=(labels.shape[0], 1,1))
        cls_score = paddle.reshape(cls_score, shape=(1,cls_score.shape[0]))
        
        loss_metrics = self.head.loss(cls_score, labels, valid_mode=True)
        # loss_metrics["cls_score"] = cls_score
        return loss_metrics,cls_score

    def test_step(self, data_batch):
        """Test step.
        """
        imgs = data_batch[0]
        imgs = imgs.squeeze(0)
        # call forward
        cls_score = self.forward_net(imgs)
        # cls_score = cls_score.mean(axis = 0)

        return cls_score

    def infer_step(self, data_batch):
        """Infer step.
        """
        imgs = data_batch[0:2]
        # call forward
        cls_score = self.forward_net(imgs)

        return cls_score

    def extract_feature(self,data_batch):
        imgs = data_batch[0]
        imgs = imgs.squeeze(0)
        # call forward
        feature = self.backbone(imgs)
        return feature
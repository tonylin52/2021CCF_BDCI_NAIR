# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..registry import HEADS
from .base import BaseHead
import paddle, os
import paddle.nn.functional as F

from ..weight_init import weight_init_

# get init parameters for conv layer
def get_conv_init(fan_out):
    return paddle.nn.initializer.KaimingNormal(fan_in=fan_out)

@HEADS.register()
class I3DHead(BaseHead):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """
    def __init__(self,
                 dim_in,
                 num_classes,
                 dropout_rate,
                 loss_cfg=dict(name='CrossEntropyLoss'),
                 multigrid_short=False,
                 **kwargs):
        """
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
        """
        super().__init__(num_classes, loss_cfg, **kwargs)
        self.multigrid_short = multigrid_short
        self.dim_in = dim_in
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        # self.centers = paddle.fluid.layers.create_parameter(shape=[30, 512], dtype='float32')
        # self.pool_size_ratio = pool_size_ratio

        # assert (len({len(self.pool_size), len(self.dim_in)
        #              }) == 1), "pathway dimensions are not consistent."
        # self.num_pathways = len(self.pool_size)

        self.dropout = paddle.nn.Dropout(p=self.dropout_rate)

        self.projection = paddle.nn.Linear(
            in_features=self.dim_in,
            out_features=self.num_classes,
        )
        
        fan = (self.dim_in) * (1 * 1 * 1)
        initializer_tmp = get_conv_init(fan)
        self.conv = paddle.nn.Conv3D(
            in_channels=self.dim_in,
            out_channels=num_classes,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)

    def init_weights(self):
        weight_init_(self.projection,
                     "Normal",
                     bias_value=0.0,
                     mean=0.0,
                     std=0.01)

    def forward(self, inputs):

        pathway = 0
        # x = F.adaptive_avg_pool3d(x=inputs[pathway],
        #                                 output_size=(1, 1, 1),
        #                                 data_format="NCDHW")

        x = F.adaptive_avg_pool3d(x=inputs,
                                  output_size=(1, 1, 1),
                                  data_format="NCDHW")
        

        # Perform dropout.
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        
        
        x = paddle.reshape_(x, shape=(x.shape[0], -1))
        # dist_mat = paddle.expand(self.centers, [x.shape[0], self.centers.shape[0], \
        #     self.centers.shape[1]])
        
        # x_ = paddle.expand_as(paddle.unsqueeze(x, axis=1), dist_mat)
        # dist_mat -= x_
        # dist_mat = paddle.sqrt(paddle.sum(paddle.pow(dist_mat, 2), axis=2))
        out = self.projection(x)
        
        # Performs fully convlutional inference.
        if not self.training:  # attr of base class
            out = F.softmax(out, axis=1)
            # x = paddle.mean(x, axis=[1, 2, 3])

        return out

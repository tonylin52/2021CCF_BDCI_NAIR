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

import paddle
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
        # self.pool_size_ratio = pool_size_ratio
        self.loss_cfg = loss_cfg

        
        

        # assert (len({len(self.pool_size), len(self.dim_in)
        #              }) == 1), "pathway dimensions are not consistent."
        # self.num_pathways = len(self.pool_size)

        self.dropout = paddle.nn.Dropout(p=self.dropout_rate)

        self.projection = paddle.nn.Linear(
            in_features=self.dim_in,
            out_features=int(self.num_classes),
        )
        
        #self.projection2 = paddle.nn.Linear(
        #    in_features=int(self.dim_in/4),
        #    out_features=self.num_classes,
        #)

        

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
        x = self.projection(x)
        
        
        #x = self.projection2(x)
        

        # Performs fully convlutional inference.
        if not self.training:  # attr of base class
            if self.loss_cfg["name"]=="LALoss" or self.loss_cfg["name"]=="LALDAMoss":
                N = x.shape[0]
                C = x.shape[1]
                weight_c = [0.04791238877481177, 0.04791238877481177, 0.04791238877481177, 0.007186858316221766, 0.04791238877481177,
                    0.04791238877481177, 0.018480492813141684, 0.02292950034223135, 0.04791238877481177, 0.04791238877481177,
                    0.022245037645448322, 0.04791238877481177, 0.007186858316221766, 0.04791238877481177, 0.007529089664613279,
                    0.019164955509924708, 0.03353867214236824, 0.014715947980835045, 0.04791238877481177, 0.02943189596167009,
                    0.01403148528405202, 0.039014373716632446, 0.04791238877481177, 0.04791238877481177, 0.014715947980835045,
                    0.01026694045174538, 0.04791238877481177, 0.04791238877481177, 0.02087611225188227, 0.04791238877481177]

                #  logit adjustment loss
                weight_c = paddle.to_tensor(weight_c)
                shift = paddle.log(weight_c) * 0.5
                shift = paddle.expand(shift,[N,C])
                x += shift
                
            x = F.softmax(x, axis=1)


        return x


@HEADS.register()
class MnistHead(BaseHead):
    def __init__(self,
                dim_in,
                num_classes,
                dropout_rate,
                loss_cfg=dict(name='CrossEntropyLoss'),
                multigrid_short=False,
                **kwargs):
        super().__init__(num_classes, loss_cfg, **kwargs)
        
        self.dim_in = dim_in
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.dropout = paddle.nn.Dropout(p=self.dropout_rate)

        self.projection = paddle.nn.Linear(
            in_features=self.dim_in,
            out_features=self.dim_in/2,
        )

        
        # fan = (self.dim_in) * (1 * 1 * 1)
        # initializer_tmp = get_conv_init(fan)
        # self.conv = paddle.nn.Conv3D(
        #     in_channels=self.dim_in,
        #     out_channels=self.dim_in,
        #     kernel_size=[1, 1, 1],
        #     stride=[1, 1, 1],
        #     padding=[0, 0, 0])
    def forward(self, inputs):
    
        
        # inputs = self.conv(inputs)
        x = F.adaptive_avg_pool3d(x=inputs,
                                  output_size=(1, 1, 1),
                                  data_format="NCDHW")
        

        # Perform dropout.
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        
        x = paddle.reshape_(x, shape=(x.shape[0], -1))
        x = self.projection(x)

        # inputs = self.conv(inputs)

        # Performs fully convlutional inference.
        if not self.training:  # attr of base class
            x = F.softmax(x, axis=4)
            x = paddle.mean(x, axis=[1, 2, 3])


        return x

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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal
from ..registry import BACKBONES
from paddlevideo.utils.multigrid import get_norm

# seed random seed
paddle.framework.seed(0)


# get init parameters for conv layer
def get_conv_init(fan_out):
    return KaimingNormal(fan_in=fan_out)


def get_bn_param_attr(bn_weight=1.0, coeff=0.0):
    param_attr = paddle.ParamAttr(
        initializer=paddle.nn.initializer.Constant(bn_weight),
        regularizer=paddle.regularizer.L2Decay(coeff))
    return param_attr


"""Video models."""


class BottleneckTransform(paddle.nn.Layer):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 temp_kernel_size,
                 stride,
                 temporal_stride,
                 dim_inner,
                 num_groups,
                 stride_1x1=False,
                 inplace_relu=True,
                 eps=1e-5,
                 dilation=1,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            dilation (int): size of dilation.
        """
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._stride_1x1 = stride_1x1
        self.norm_module = norm_module
        self.temporal_stride = temporal_stride
        self._construct(dim_in, dim_out, stride, temporal_stride, dim_inner, num_groups,
                        dilation)

    def _construct(self, dim_in, dim_out, stride, temporal_stride, dim_inner, num_groups,
                   dilation):
        str1x1, str3x3 = (stride, 1) if self._stride_1x1 else (1, stride)

        fan = (dim_inner) * (self.temp_kernel_size * 1 * 1)
        initializer_tmp = get_conv_init(fan)
        self.a = paddle.nn.Conv3D(
            in_channels=dim_in,
            out_channels=dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[temporal_stride, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self.a_bn = self.norm_module(num_features=dim_inner,
                                     epsilon=self._eps,
                                     weight_attr=get_bn_param_attr(),
                                     bias_attr=get_bn_param_attr(bn_weight=0.0))

        # 1x3x3, BN, ReLU.
        fan = (dim_inner) * (1 * 3 * 3)
        initializer_tmp = get_conv_init(fan)

        self.b = paddle.nn.Conv3D(
            in_channels=dim_inner,
            out_channels=dim_inner,
            kernel_size=[1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            dilation=[1, dilation, dilation],
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self.b_bn = self.norm_module(num_features=dim_inner,
                                     epsilon=self._eps,
                                     weight_attr=get_bn_param_attr(),
                                     bias_attr=get_bn_param_attr(bn_weight=0.0))

        # 1x1x1, BN.
        fan = (dim_out) * (1 * 1 * 1)
        initializer_tmp = get_conv_init(fan)

        self.c = paddle.nn.Conv3D(
            in_channels=dim_inner,
            out_channels=dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self.c_bn = self.norm_module(
            num_features=dim_out,
            epsilon=self._eps,
            weight_attr=get_bn_param_attr(bn_weight=0.0),
            bias_attr=get_bn_param_attr(bn_weight=0.0))
        
        # self.se = SEBlock(int(64/self.temp_kernel_size))

    def forward(self, x):
        # Branch2a.
        x = self.a(x)
        x = self.a_bn(x)
        x = F.relu(x)
        # Branch2b.
        x = self.b(x)
        x = self.b_bn(x)
        x = F.relu(x)
        # Branch2c
        x = self.c(x)
        x = self.c_bn(x)
        # print("cccccccccccccccc",x.shape[2],int(64/self.temporal_stride))
        # se = SEBlock(x.shape[2])
        # x = se(x)
        return x

class SEBlock(paddle.nn.Layer):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = paddle.nn.AdaptiveAvgPool3D(1)
        self.fc = paddle.nn.Sequential(
            paddle.nn.Linear(channel, channel * reduction),
            paddle.nn.ReLU(),
            paddle.nn.Linear(channel * reduction, channel),
            paddle.nn.Sigmoid()
        )
        
    def forward(self, x):
        x = paddle.transpose(x, [0,2,1,3,4])
        b, c, _, _, _ = x.shape
        y = paddle.reshape(self.avg_pool(x), (b,c))
        weights = self.fc(y)
        y = paddle.reshape(weights, (b, c, 1, 1, 1))
        return paddle.transpose(x*y, [0, 2, 1, 3, 4])




class ResBlock(paddle.nn.Layer):
    """
    Residual block.
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 temp_kernel_size,
                 stride,
                 temporal_stride,
                 dim_inner,
                 num_groups=1,
                 stride_1x1=False,
                 inplace_relu=True,
                 eps=1e-5,
                 dilation=1,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            dilation (int): size of dilation.
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self.norm_module = norm_module
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            temporal_stride,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        temporal_stride,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            fan = (dim_out) * (1 * 1 * 1)
            initializer_tmp = get_conv_init(fan)
            self.branch1 = paddle.nn.Conv3D(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=1,
                stride=[temporal_stride, stride, stride],
                padding=0,
                weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
                bias_attr=False,
                dilation=1)
            self.branch1_bn = self.norm_module(
                num_features=dim_out,
                epsilon=self._eps,
                weight_attr=get_bn_param_attr(),
                bias_attr=get_bn_param_attr(bn_weight=0.0))

        self.branch2 = BottleneckTransform(dim_in,
                                           dim_out,
                                           temp_kernel_size,
                                           stride,
                                           temporal_stride,
                                           dim_inner,
                                           num_groups,
                                           stride_1x1=stride_1x1,
                                           inplace_relu=inplace_relu,
                                           dilation=dilation,
                                           norm_module=self.norm_module)

    def forward(self, x):
        if hasattr(self, "branch1"):
            x1 = self.branch1(x)
            x1 = self.branch1_bn(x1)
            x2 = self.branch2(x)
            x = paddle.add(x=x1, y=x2)
            # print("branch1",x.shape)
        else:
            x2 = self.branch2(x)
            x = paddle.add(x=x, y=x2)
            # print("branch2",x.shape)
        x = F.relu(x)
        return x


class ResStage(paddle.nn.Layer):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        multi-pathway (SlowFast) cases.  More details can be found here:

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "Slowfast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 stride,
                 temporal_stride,
                 temp_kernel_sizes,
                 num_blocks,
                 dim_inner,
                 num_groups,
                 num_block_temp_kernel,
                 dilation,
                 stride_1x1=False,
                 inplace_relu=True,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            dilation (list): size of dilation for each pathway.
        """
        super(ResStage, self).__init__()
        assert all((num_block_temp_kernel[i] <= num_blocks[i]
                    for i in range(len(temp_kernel_sizes))))
        self.num_blocks = num_blocks
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[:num_block_temp_kernel[i]] +
            [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        assert (len({
            len(dim_in),
            len(dim_out),
            len(temp_kernel_sizes),
            len(stride),
            len(num_blocks),
            len(dim_inner),
            len(num_groups),
            len(num_block_temp_kernel),
        }) == 1)
        self.num_pathways = len(self.num_blocks)
        self.norm_module = norm_module
        self._construct(
            dim_in,
            dim_out,
            stride,
            temporal_stride,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        temporal_stride,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
    ):

        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                res_block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    stride[pathway] if i == 0 else 1,
                    temporal_stride[pathway] if i == 0 else 1,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=self.norm_module)
                self.add_sublayer("pathway{}_res{}".format(pathway, i),
                                  res_block)

    def forward(self, inputs):
        # output = []
        # for pathway in range(self.num_pathways):
        #     x = inputs[pathway]
        x = inputs
        pathway = 0
        for i in range(self.num_blocks[pathway]):
            m = getattr(self, "pathway{}_res{}".format(pathway, i))
            x = m(x)
        # output.append(x)

        # return output
        return x


class ResNetBasicStem(paddle.nn.Layer):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 kernel,
                 stride,
                 padding,
                 eps=1e-5,
                 norm_module=paddle.nn.BatchNorm3D):
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.norm_module = norm_module
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        fan = (dim_out) * (self.kernel[0] * self.kernel[1] * self.kernel[2])
        initializer_tmp = get_conv_init(fan)

        self._conv = paddle.nn.Conv3D(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self._bn = self.norm_module(num_features=dim_out,
                                    epsilon=self.eps,
                                    weight_attr=get_bn_param_attr(),
                                    bias_attr=get_bn_param_attr(bn_weight=0.0))

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x)
        x = F.relu(x)

        # x = F.max_pool3d(x=x,
        #                  kernel_size=[1, 3, 3],
        #                  stride=[1, 2, 2],
        #                  padding=[0, 1, 1],
        #                  data_format="NCDHW")
        return x


class VideoModelStem(paddle.nn.Layer):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for slow and fast pathways.
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 kernel,
                 stride,
                 padding,
                 eps=1e-5,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            eps (float): epsilon for batch norm.
        """
        super(VideoModelStem, self).__init__()

        # assert (len({
        #     len(dim_in),
        #     len(dim_out),
        #     len(kernel),
        #     len(stride),
        #     len(padding),
        # }) == 1), "Input pathway dimensions are not consistent."
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.norm_module = norm_module
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        for pathway in range(len(dim_in)):
            stem = ResNetBasicStem(dim_in[pathway], dim_out[pathway],
                                   self.kernel[pathway], self.stride[pathway],
                                   self.padding[pathway], self.eps,
                                   self.norm_module)
            self.add_sublayer("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        # x = [x]
        # assert (len(x) == self.num_pathways
        #         ), "Input tensor does not contain {} pathway".format(
        #             self.num_pathways)

        # for pathway in range(len(x)):
        pathway = 0
        m = getattr(self, "pathway{}_stem".format(pathway))
        x = m(x)

        return x



@BACKBONES.register()
class ResNetSlowOnly(paddle.nn.Layer):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """
    def __init__(
        self,
        bn_norm_type="batchnorm",
        bn_num_splits=1,
        num_pathways=1,
        depth=152,
        num_groups=1,
        input_channel_num=[25],
        width_per_group=64,
        pool_size_ratio=[[1, 1, 1]],
            dropout_rate=0.0
    ):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNetSlowOnly, self).__init__()
        self.norm_module = get_norm(bn_norm_type, bn_num_splits)
        self.num_pathways = num_pathways
        self.depth = depth
        self.num_groups = num_groups
        self.input_channel_num = input_channel_num
        self.width_per_group = width_per_group
        self.pool_size_ratio = pool_size_ratio
        self.dropout_rate = dropout_rate
        # self.fusion_conv_channel_ratio = fusion_conv_channel_ratio
        self._construct_network()

    def _construct_network(self):
        """
        Builds a SlowFast model.
        The first pathway is the Slow pathway
        and the second pathway is the Fast pathway.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        temp_kernel = [
            [[1]],  # conv1 temporal kernel for slow and fast pathway.
            [[1]],  # res3 temporal kernel for slow and fast pathway.
            [[3]],  # res4 temporal kernel for slow and fast pathway.
            [[3]],  # res5 temporal kernel for slow and fast pathway.
        ]  
        
        #self.rednetblock = RednetBlock()

        self.s1 = VideoModelStem(
            dim_in=self.input_channel_num,
            dim_out=[self.width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 1, 1]],
            padding=[
                [0, 3, 3]
            ],
            norm_module=self.norm_module)
       
        # ResNet backbone
        MODEL_STAGE_DEPTH = {18: (2, 2, 2),
                             34: (4, 6, 3),
                             50: (4, 6, 3),
                             101: (4, 23, 3),
                             152: (8, 36, 3)}
        (d3, d4, d5) = MODEL_STAGE_DEPTH[self.depth]

        num_block_temp_kernel = [[d3], [d4], [d5]]
        spatial_dilations = [[1], [1], [1]]
        spatial_strides = [[2], [2], [2]]
        temporal_stride = [[1], [1], [2]]

        # out_dim_ratio = self.beta // self.fusion_conv_channel_ratio  #4
        dim_inner = 32  #64


        self.s3 = ResStage(
            dim_in=[
                self.width_per_group
                # 25
            ],
            dim_out=[
                self.width_per_group * 4
            ],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=spatial_strides[0],
            temporal_stride = temporal_stride[0],
            num_blocks=[d3],
            num_groups=[self.num_groups],
            num_block_temp_kernel=num_block_temp_kernel[0],
            dilation=spatial_dilations[0],
            norm_module=self.norm_module,
        )

        self.s4 = ResStage(
            dim_in=[
                self.width_per_group * 4
                    # 25
            ],
            dim_out=[
                self.width_per_group * 8
            ],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=spatial_strides[1],
            temporal_stride = temporal_stride[1],
            num_blocks=[d4],
            num_groups=[self.num_groups],
            num_block_temp_kernel=num_block_temp_kernel[1],
            dilation=spatial_dilations[1],
            norm_module=self.norm_module,
        )



        self.s5 = ResStage(
            dim_in=[
                self.width_per_group * 8
            
            ],
            dim_out=[
                self.width_per_group * 16
            ],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=spatial_strides[2],
            temporal_stride = temporal_stride[2],
            num_blocks=[d5],
            num_groups=[self.num_groups],
            num_block_temp_kernel=num_block_temp_kernel[2],
            dilation=spatial_dilations[2],
            norm_module=self.norm_module,
        )
        if self.dropout_rate > 0.0:
            self.dropout1 = nn.Dropout3D(p=self.dropout_rate)
            self.dropout2 = nn.Dropout3D(p=self.dropout_rate)

    def init_weights(self):
        pass

    def forward(self, x):
        
        #x = self.rednetblock(x)
        # print(x.shape)# [16, 25, 64, 56, 28]
        x = self.s1(x)  #VideoModelStem
        # x = self.s1_fuse(x)  #FuseFastToSlow
        # x = self.s2(x)  #ResStage
        # x = self.s2_fuse(x)
        # print(x.shape) #[16, 32, 64, 56, 28]
        if self.dropout_rate > 0.0:
            x = self.dropout1(x)

        # for pathway in range(self.num_pathways):
        pathway = 0
        x = F.max_pool3d(x=x,kernel_size=[1,3,3],
                        stride=self.pool_size_ratio[pathway],
                        padding=[0, 1, 1],
                        data_format="NCDHW")
        # # x = [x]
        x = self.s3(x)
        if self.dropout_rate > 0.0:
            x = self.dropout2(x)
        # x = self.s3_fuse(x)
        x = self.s4(x)
        # x = self.s4_fuse(x)
        x = self.s5(x)
        return x




class involution(nn.Layer):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 5
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            ('conv', nn.Conv3D(
                in_channels=channels,
                out_channels=channels // reduction_ratio,
                kernel_size=(1, 1, 1),
                bias_attr=False
            )),
            ('bn', nn.BatchNorm3D(channels // reduction_ratio)),
            ('activate', nn.ReLU())
        )
        self.conv2 = nn.Sequential(
            ('conv', nn.Conv3D(
                in_channels=channels // reduction_ratio,
                out_channels=kernel_size**2 * self.groups,
                kernel_size=(1, 1, 1),
                stride=1))
        )
        if stride > 1:
            self.avgpool = nn.AvgPool2D(stride, stride)

    def forward(self, x):
        weight_3d = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, d, h, w = weight_3d.shape
        # print('weight_3d.shape')
        # print(weight_3d.shape)
        for i in range(d):
            if i == 0:
                weight = weight_3d[:,:,i,:,:].reshape((b, self.groups, self.kernel_size**2, h, w)).unsqueeze(2)
                out_start = nn.functional.unfold(x[:,:,i,:,:], self.kernel_size, strides=self.stride, paddings=(self.kernel_size-1)//2, dilations=1)
                out_start = out_start.reshape((b, self.groups, self.group_channels, self.kernel_size**2, h, w))
                out_start = (weight * out_start).sum(axis=3).reshape((b, self.channels, h, w))  
            else:
                weight = weight_3d[:,:,i,:,:].reshape((b, self.groups, self.kernel_size**2, h, w)).unsqueeze(2)
                out = nn.functional.unfold(x[:,:,i,:,:], self.kernel_size, strides=self.stride, paddings=(self.kernel_size-1)//2, dilations=1)
                out = out.reshape((b, self.groups, self.group_channels, self.kernel_size**2, h, w))
                out = (weight * out).sum(axis=3).reshape((b, self.channels, h, w))  
                out_start = paddle.concat(x=[out_start, out], axis=1)
                # print(out_start.shape)
        out = out_start.reshape((b, self.channels, d, h, w))
        return out

class RednetBlock(nn.Layer):
    def __init__(self,
                 inplanes=25,
                 planes=3,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(RednetBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3D
        # width = int(planes * (base_width / 64.)) * groups
        width = 25

        self.conv1 = nn.Conv3D(inplanes, width, (1, 1, 1), bias_attr=False)
        self.bn1 = norm_layer(width)

        self.conv2 = involution(width, 3, stride)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv3D(
            width, 25, (1, 1, 1), bias_attr=False)
        self.bn3 = norm_layer(25)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print('conv1')
        # print(out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # print('identity')
        # print(identity.shape)
        # print('out')
        # print(out.shape)
        out += identity
        out = self.relu(out)

        return out
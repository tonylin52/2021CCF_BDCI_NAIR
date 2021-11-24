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
import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal
from ..registry import BACKBONES
from paddlevideo.utils.multigrid import get_norm
from .resnet3d import ConvModule
from .resnet3d import ResNet3d
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
        self._construct(dim_in, dim_out, stride, dim_inner, num_groups,
                        dilation)

    def _construct(self, dim_in, dim_out, stride, dim_inner, num_groups,
                   dilation):
        str1x1, str3x3 = (stride, 1) if self._stride_1x1 else (1, stride)

        fan = (dim_inner) * (self.temp_kernel_size * 1 * 1)
        initializer_tmp = get_conv_init(fan)

        self.a = paddle.nn.Conv3D(
            in_channels=dim_in,
            out_channels=dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
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
        return x


class ResBlock(paddle.nn.Layer):
    """
    Residual block.
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 temp_kernel_size,
                 stride,
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
                stride=[1, stride, stride],
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
        else:
            x2 = self.branch2(x)
            x = paddle.add(x=x, y=x2)

        x = F.relu(x)
        return x


class ResNet3dPathway(ResNet3d):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        *args (arguments): Arguments same as :class:``ResNet3d``.
        lateral (bool): Determines whether to enable the lateral connection
            from another pathway. Default: False.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            ``alpha`` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to ``beta`` in the paper.
            Default: 8.
        fusion_kernel (int): The kernel size of lateral fusion.
            Default: 5.
        **kwargs (keyword arguments): Keywords arguments for ResNet3d.
    """

    def __init__(self,
                 *args,
                 lateral=False,
                 speed_ratio=8,
                 channel_ratio=8,
                 fusion_kernel=5,
                 **kwargs):
        self.lateral = lateral
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        super().__init__(*args, **kwargs)
        self.inplanes = self.base_channels
        if self.lateral:
            self.conv1_lateral = ConvModule(
                self.inplanes // self.channel_ratio,
                self.inplanes * 2 // self.channel_ratio,
                kernel_size=(fusion_kernel, 1, 1),
                stride=(self.speed_ratio, 1, 1),
                padding=((fusion_kernel - 1) // 2, 0, 0),
                bias=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                act_cfg=None)

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2**i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1:
                # no lateral connection needed in final stage
                lateral_name = f'layer{(i + 1)}_lateral'
                setattr(
                    self, lateral_name,
                    ConvModule(
                        self.inplanes // self.channel_ratio,
                        self.inplanes * 2 // self.channel_ratio,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=None,
                        act_cfg=None))
                self.lateral_connections.append(lateral_name)

    def make_res_layer(self,
                       block,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       style='pytorch',
                       mtype='XtSe',
                       inflate=1,
                       inflate_style='3x1x1',
                       non_local=0,
                       non_local_cfg=dict(),
                       conv_cfg=None,
                       norm_cfg=None,
                       act_cfg=None,
                       with_cp=False):
        """Build residual layer for Slowfast.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input
                feature in each block.
            planes (int): Number of channels for the output
                feature in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides
                in residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``,
                the stride-two layer is the 3x3 conv layer,
                otherwise the stride-two layer is the first 1x1 conv layer.
                Default: ``pytorch``.
            mtype (str): 'XtSe' or 'DSe'. if set to 'XtSe', make origin resnet3d to
                resXt3d with time dimension attention. if set to 'DSe', use origin 
                resnet3d with time dimension attention and channel dimension attention.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines
                the kernel sizes and padding strides for conv1 and
                conv2 in each block. Default: ``3x1x1``.
            non_local (int | Sequence[int]): Determine whether to apply
                non-local module in the corresponding block of each stages.
                Default: 0.
            non_local_cfg (dict): Config for non-local module.
                Default: ``dict()``.
            conv_cfg (dict | None): Config for conv layers. Default: None.
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save
                some memory while slowing down the training speed.
                Default: False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        non_local = non_local if not isinstance(
            non_local, int) else (non_local, ) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks
        if self.lateral:
            lateral_inplanes = inplanes * 2 // self.channel_ratio
        else:
            lateral_inplanes = 0
        if (spatial_stride != 1
                or (inplanes + lateral_inplanes) != planes * block.expansion):
            downsample = ConvModule(
                inplanes + lateral_inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
        else:
            downsample = None

        layers = []
        layers.append(
            block(
                inplanes + lateral_inplanes,
                planes,
                spatial_stride,
                temporal_stride,
                dilation,
                downsample,
                style=style,
                mtype=mtype,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp))
        inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    1,
                    1,
                    dilation,
                    style=style,
                    mtype=mtype,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp))

        return nn.Sequential(*layers)

    def inflate_weights(self, logger):
        """Inflate the resnet2d parameters to resnet3d pathway.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart. For pathway the ``lateral_connection`` part should
        not be inflated from 2d weights.

        Args:
            logger (logging.Logger): The logger used to print
                debugging information.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']

        inflated_param_names = []
        for name, module in self.named_modules():
            if 'lateral' in name:
                continue
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_conv_name}')
                else:
                    self._inflate_conv_params(module.conv, state_dict_r2d,
                                              original_conv_name,
                                              inflated_param_names)
                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_bn_name}')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d,
                                            original_bn_name,
                                            inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded'
                        f': {remaining_names}')

    def _inflate_conv_params(self, conv3d, state_dict_2d, module_name_2d,
                             inflated_param_names):
        """Inflate a conv module from 2d to 3d.

        The differences of conv modules betweene 2d and 3d in Pathway
        mainly lie in the inplanes due to lateral connections. To fit the
        shapes of the lateral connection counterpart, it will expand
        parameters by concatting conv2d parameters and extra zero paddings.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + '.weight'
        conv2d_weight = state_dict_2d[weight_2d_name]
        old_shape = conv2d_weight.shape
        new_shape = conv3d.weight.data.shape
        kernel_t = new_shape[2]

        if new_shape[1] != old_shape[1]:
            if new_shape[1] < old_shape[1]:
                warnings.warn(f'The parameter of {module_name_2d} is not'
                              'loaded due to incompatible shapes. ')
                return
            # Inplanes may be different due to lateral connections
            new_channels = new_shape[1] - old_shape[1]
            pad_shape = old_shape
            pad_shape = pad_shape[:1] + (new_channels, ) + pad_shape[2:]
            # Expand parameters by concat extra channels
            
            conv2d_weight = paddle.concat(
                (conv2d_weight,
                 paddle.zeros(pad_shape).type_as(conv2d_weight).to(
                     conv2d_weight.device)),
                dim=1)

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
            conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        `self.frozen_stages`."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i != len(self.res_layers) and self.lateral:
                # No fusion needed in the final stage
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.eval()
                for param in conv_lateral.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        # Override the init_weights of i3d
        # super().init_weights()
        for module_name in self.lateral_connections:
            layer = getattr(self, module_name)
            for m in layer.modules():
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    kaiming_init(m)




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
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=self.norm_module)
                self.add_sublayer("pathway{}_res{}".format(pathway, i),
                                  res_block)

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]

            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m(x)
            output.append(x)

        return output


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

        x = F.max_pool3d(x=x,
                         kernel_size=[1, 3, 3],
                         stride=[1, 2, 2],
                         padding=[0, 1, 1],
                         data_format="NCDHW")
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

        assert (len({
            len(dim_in),
            len(dim_out),
            len(kernel),
            len(stride),
            len(padding),
        }) == 1), "Input pathway dimensions are not consistent."
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
        assert (len(x) == self.num_pathways
                ), "Input tensor does not contain {} pathway".format(
                    self.num_pathways)

        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])

        return x


class FuseFastToSlow(paddle.nn.Layer):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """
    def __init__(self,
                 dim_in,
                 fusion_conv_channel_ratio,
                 fusion_kernel,
                 alpha,
                 eps=1e-5,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
        """
        super(FuseFastToSlow, self).__init__()
        fan = (dim_in * fusion_conv_channel_ratio) * (fusion_kernel * 1 * 1)
        initializer_tmp = get_conv_init(fan)

        self._conv_f2s = paddle.nn.Conv3D(
            in_channels=dim_in,
            out_channels=dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self._bn = norm_module(num_features=dim_in * fusion_conv_channel_ratio,
                               epsilon=eps,
                               weight_attr=get_bn_param_attr(),
                               bias_attr=get_bn_param_attr(bn_weight=0.0))

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self._conv_f2s(x_f)
        fuse = self._bn(fuse)
        fuse = F.relu(fuse)
        x_s_fuse = paddle.concat(x=[x_s, fuse], axis=1, name=None)

        return [x_s_fuse, x_f]


@BACKBONES.register()
class ResNetSlowFast(paddle.nn.Layer):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """
    def __init__(
        self,
        alpha,
        beta,
        bn_norm_type="batchnorm",
        bn_num_splits=1,
        num_pathways=1,
        depth=50,
        num_groups=1,
        input_channel_num=[3, 3],
        width_per_group=64,
        fusion_conv_channel_ratio=2,
        fusion_kernel_sz=7,  #5?
        pool_size_ratio=[[1, 1, 1], [1, 1, 1]],
    ):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNetSlowFast, self).__init__()

        self.alpha = alpha  #8
        self.beta = beta  #8
        self.norm_module = get_norm(bn_norm_type, bn_num_splits)
        self.num_pathways = num_pathways
        self.depth = depth
        self.num_groups = num_groups
        self.input_channel_num = input_channel_num
        self.width_per_group = width_per_group
        self.fusion_conv_channel_ratio = fusion_conv_channel_ratio
        self.fusion_kernel_sz = fusion_kernel_sz  # NOTE: modify to 7 in 8*8, 5 in old implement
        self.pool_size_ratio = pool_size_ratio
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
            [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
            [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
            [[3], [3]],
        ]  # res5 temporal kernel for slow and fast pathway.

        self.s1 = VideoModelStem(
            dim_in=self.input_channel_num,
            dim_out=[self.width_per_group, self.width_per_group // self.beta],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module)
        self.s1_fuse = FuseFastToSlow(
            dim_in=self.width_per_group // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module)

        # ResNet backbone
        MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3)}
        (d2, d3, d4, d5) = MODEL_STAGE_DEPTH[self.depth]

        num_block_temp_kernel = [[3, 3], [4, 4], [6, 6], [3, 3]]
        spatial_dilations = [[1, 1], [1, 1], [1, 1], [1, 1]]
        spatial_strides = [[1, 1], [2, 2], [2, 2], [2, 2]]

        out_dim_ratio = self.beta // self.fusion_conv_channel_ratio  #4
        dim_inner = self.width_per_group * self.num_groups  #64

        self.s2 = ResStage(dim_in=[
            self.width_per_group + self.width_per_group // out_dim_ratio,
            self.width_per_group // self.beta,
        ],
                           dim_out=[
                               self.width_per_group * 4,
                               self.width_per_group * 4 // self.beta,
                           ],
                           dim_inner=[dim_inner, dim_inner // self.beta],
                           temp_kernel_sizes=temp_kernel[1],
                           stride=spatial_strides[0],
                           num_blocks=[d2] * 2,
                           num_groups=[self.num_groups] * 2,
                           num_block_temp_kernel=num_block_temp_kernel[0],
                           dilation=spatial_dilations[0],
                           norm_module=self.norm_module)

        self.s2_fuse = FuseFastToSlow(
            dim_in=self.width_per_group * 4 // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module,
        )

        self.s3 = ResStage(
            dim_in=[
                self.width_per_group * 4 +
                self.width_per_group * 4 // out_dim_ratio,
                self.width_per_group * 4 // self.beta,
            ],
            dim_out=[
                self.width_per_group * 8,
                self.width_per_group * 8 // self.beta,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // self.beta],
            temp_kernel_sizes=temp_kernel[2],
            stride=spatial_strides[1],
            num_blocks=[d3] * 2,
            num_groups=[self.num_groups] * 2,
            num_block_temp_kernel=num_block_temp_kernel[1],
            dilation=spatial_dilations[1],
            norm_module=self.norm_module,
        )

        self.s3_fuse = FuseFastToSlow(
            dim_in=self.width_per_group * 8 // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module,
        )

        self.s4 = ResStage(
            dim_in=[
                self.width_per_group * 8 +
                self.width_per_group * 8 // out_dim_ratio,
                self.width_per_group * 8 // self.beta,
            ],
            dim_out=[
                self.width_per_group * 16,
                self.width_per_group * 16 // self.beta,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // self.beta],
            temp_kernel_sizes=temp_kernel[3],
            stride=spatial_strides[2],
            num_blocks=[d4] * 2,
            num_groups=[self.num_groups] * 2,
            num_block_temp_kernel=num_block_temp_kernel[2],
            dilation=spatial_dilations[2],
            norm_module=self.norm_module,
        )

        self.s4_fuse = FuseFastToSlow(
            dim_in=self.width_per_group * 16 // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module,
        )

        self.s5 = ResStage(
            dim_in=[
                self.width_per_group * 16 +
                self.width_per_group * 16 // out_dim_ratio,
                self.width_per_group * 16 // self.beta,
            ],
            dim_out=[
                self.width_per_group * 32,
                self.width_per_group * 32 // self.beta,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // self.beta],
            temp_kernel_sizes=temp_kernel[4],
            stride=spatial_strides[3],
            num_blocks=[d5] * 2,
            num_groups=[self.num_groups] * 2,
            num_block_temp_kernel=num_block_temp_kernel[3],
            dilation=spatial_dilations[3],
            norm_module=self.norm_module,
        )

    def init_weights(self):
        pass

    def forward(self, x):
        x = self.s1(x)  #VideoModelStem
        # x = self.s1_fuse(x)  #FuseFastToSlow
        x = self.s2(x)  #ResStage
        # x = self.s2_fuse(x)

        for pathway in range(self.num_pathways):
            x[pathway] = F.max_pool3d(x=x[pathway],
                                      kernel_size=self.pool_size_ratio[pathway],
                                      stride=self.pool_size_ratio[pathway],
                                      padding=[0, 0, 0],
                                      data_format="NCDHW")

        x = self.s3(x)
        # x = self.s3_fuse(x)
        x = self.s4(x)
        # x = self.s4_fuse(x)
        x = self.s5(x)
        return x

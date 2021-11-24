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
# See the License for the specific language governing permissions and
# limitations under the License.

from .augmentations import (Scale, RandomCrop, CenterCrop, RandomFlip,
                            Image2Array, Normalization, JitterScale, MultiCrop,
                            PackOutput, TenCrop, UniformCrop)

from .compose import Compose
from .decode import VideoDecoder, FrameDecoder
from .sample import Sampler
from .decode_sampler import DecodeSampler
from .mix import Mixup, Cutmix
from .anet_pipeline import LoadFeat, GetMatchMap, GetVideoLabel
from .skeleton_pipeline import AutoPadding, SkeletonNorm, Iden

# mmaction
from .pose_loading import UniformSampleFrames,PoseDecode,GeneratePoseTarget,RandomRotate
from .augmentations1 import (PoseCompact,Resize,RandomResizedCrop,Flip)
from .formating import FormatShape
from .get_mnist_image import GetMnistIMAGE

__all__ = [
    'Scale', 'RandomCrop', 'CenterCrop', 'RandomFlip', 'Image2Array',
    'Normalization', 'Compose', 'VideoDecoder', 'FrameDecoder', 'Sampler',
    'Mixup', 'Cutmix', 'JitterScale', 'MultiCrop', 'PackOutput', 'TenCrop',
    'UniformCrop', 'DecodeSampler', 'LoadFeat', 'GetMatchMap', 'GetVideoLabel',
    'AutoPadding', 'SkeletonNorm', 'Iden',
    # mmaction
    'UniformSampleFrames','PoseDecode','PoseCompact','Resize','RandomResizedCrop',
    'Flip','GeneratePoseTarget','FormatShape','GetMnistIMAGE','RandomRotate'
]

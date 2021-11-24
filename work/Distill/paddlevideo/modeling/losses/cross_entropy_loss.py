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

import paddle
import paddle.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss."""
    def _forward(self, score, labels,valid_mode, **kwargs):
        """Forward function.
        Args:
            score (paddle.Tensor): The class score.
            labels (paddle.Tensor): The ground truth labels.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.
        Returns:
            loss (paddle.Tensor): The returned CrossEntropy loss.
        """
        # print("use CE loss……………………")
        if valid_mode:
            return 0.0
        loss = F.cross_entropy(score, labels, **kwargs)
        return loss

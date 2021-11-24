import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# from paddle.autograd import Variable
from ..registry import LOSSES
import numpy as np


@LOSSES.register()
class LALoss(nn.Layer):
    
    def __init__(self, class_num=30, gamma=0.5, size_average=True):
        super(LALoss, self).__init__()
        
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.weight_c = [0.04791238877481177, 0.04791238877481177, 0.04791238877481177, 0.007186858316221766, 0.04791238877481177,
                    0.04791238877481177, 0.018480492813141684, 0.02292950034223135, 0.04791238877481177, 0.04791238877481177,
                    0.022245037645448322, 0.04791238877481177, 0.007186858316221766, 0.04791238877481177, 0.007529089664613279,
                    0.019164955509924708, 0.03353867214236824, 0.014715947980835045, 0.04791238877481177, 0.02943189596167009,
                    0.01403148528405202, 0.039014373716632446, 0.04791238877481177, 0.04791238877481177, 0.014715947980835045,
                    0.01026694045174538, 0.04791238877481177, 0.04791238877481177, 0.02087611225188227, 0.04791238877481177]

    def forward(self, inputs, targets,valid_mode,soft_label=True):
        N = inputs.shape[0]
        C = inputs.shape[1]
        if valid_mode:
            return 0.0
        
    
        #  logit adjustment loss
        weight_c = paddle.to_tensor(self.weight_c)
        shift = paddle.log(weight_c)*self.gamma
        shift = paddle.expand(shift,[N,C])
        # print(shift.shape,shift)
        inputs += shift

        P = F.softmax(inputs,axis=1)
        
        cross_entropy = targets * paddle.log(P)
        cross_entropy = -1.0*paddle.fluid.layers.reduce_sum(cross_entropy, dim=-1)        
        cross_entropy = paddle.mean(cross_entropy)
        
        return cross_entropy
    
    
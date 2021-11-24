import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# from paddle.autograd import Variable
from ..registry import LOSSES
import numpy as np


@LOSSES.register()
class LALDAMoss(nn.Layer):
    
    def __init__(self, class_num=30, gamma=0.5,max_m = 0.5, size_average=True):
        super(LALDAMoss, self).__init__()
        
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.weight_c = [0.04791238877481177, 0.04791238877481177, 0.04791238877481177, 0.007186858316221766, 0.04791238877481177,
                    0.04791238877481177, 0.018480492813141684, 0.02292950034223135, 0.04791238877481177, 0.04791238877481177,
                    0.022245037645448322, 0.04791238877481177, 0.007186858316221766, 0.04791238877481177, 0.007529089664613279,
                    0.019164955509924708, 0.03353867214236824, 0.014715947980835045, 0.04791238877481177, 0.02943189596167009,
                    0.01403148528405202, 0.039014373716632446, 0.04791238877481177, 0.04791238877481177, 0.014715947980835045,
                    0.01026694045174538, 0.04791238877481177, 0.04791238877481177, 0.02087611225188227, 0.04791238877481177]
        
        cls_num_list = [140, 140, 140, 21, 140, 140, 
                        54, 67, 140, 140, 65, 140, 
                        21, 140, 22, 56, 98, 43, 
                        140, 86, 41, 114, 140, 140, 
                        43, 30, 140, 140, 61, 140]
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = paddle.to_tensor(m_list,dtype=paddle.float32)
        self.m_list = m_list

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
        inputs0 = inputs + shift

        P = F.softmax(inputs0,axis=1)
        
        cross_entropy = targets * paddle.log(P)
        cross_entropy = -1.0*paddle.fluid.layers.reduce_sum(cross_entropy, dim=-1)        
        cross_entropy = paddle.mean(cross_entropy)

        
        pred = paddle.argmax(targets, axis=1, dtype='int32')
        index = F.one_hot(pred, self.class_num)
        
        self.m_list = paddle.reshape(self.m_list,[C,1])
        batch_m = paddle.matmul(index,self.m_list)
        inputs_m = inputs - batch_m
        index = index==1.0
        
        outputs = paddle.where(index, inputs_m, inputs)
        P_ldam = F.softmax(outputs,axis=1)
        
        cross_entropy_ldam = targets * paddle.log(P_ldam)
        cross_entropy_ldam = -1.0*paddle.fluid.layers.reduce_sum(cross_entropy_ldam, dim=-1)        
        cross_entropy_ldam = paddle.mean(cross_entropy_ldam)
        
        cross_entropy += cross_entropy_ldam
        return cross_entropy
    
    
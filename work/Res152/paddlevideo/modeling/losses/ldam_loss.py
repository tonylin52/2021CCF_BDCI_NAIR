import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# from paddle.autograd import Variable
from ..registry import LOSSES
import numpy as np


@LOSSES.register()
class LDAMLoss(nn.Layer):
    
    def __init__(self, class_num=30, max_m = 0.5, size_average=True):
        super(LDAMLoss, self).__init__()
 
        self.class_num = class_num
        self.size_average = size_average
        cls_num_list = [140, 140, 140, 21, 140, 140, 
                        54, 67, 140, 140, 65, 140, 
                        21, 140, 22, 56, 98, 43, 
                        140, 86, 41, 114, 140, 140, 
                        43, 30, 140, 140, 61, 140]
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = paddle.to_tensor(m_list,dtype=paddle.float32)
        self.m_list = m_list
        

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        self.per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.per_cls_weights = paddle.to_tensor(self.per_cls_weights)
        print("###################",self.per_cls_weights)
        

    def forward(self, inputs, targets,valid_mode,soft_label=True):
        N = inputs.shape[0]
        C = inputs.shape[1]
        if valid_mode:
            return 0.0
        
        pred = paddle.argmax(targets, axis=1, dtype='int32')
        index = F.one_hot(pred, self.class_num)
        
        self.m_list = paddle.reshape(self.m_list,[C,1])
        batch_m = paddle.matmul(index,self.m_list)
        inputs_m = inputs - batch_m
        index = index==1.0
        
        outputs = paddle.where(index, inputs_m, inputs)
        P = F.softmax(outputs,axis=1)
        
        cross_entropy = targets * paddle.log(P)
        cross_entropy = -1.0*paddle.fluid.layers.reduce_sum(cross_entropy, dim=-1)        
        # cross_entropy = paddle.mean(cross_entropy)
        
        pred = paddle.argmax(targets, axis=1, dtype='int32')
        weight_c = self.per_cls_weights[pred]
        cross_entropy = paddle.mean(weight_c*cross_entropy)
        
        return cross_entropy
    
    
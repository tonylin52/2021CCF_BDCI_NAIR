import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# from paddle.autograd import Variable
from ..registry import LOSSES
import numpy as np

@LOSSES.register()
class FocalLoss(nn.Layer):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num=30, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
 
        self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        
        self.weight_c = [0.04791238877481177, 0.04791238877481177, 0.04791238877481177, 0.007186858316221766, 0.04791238877481177,
                    0.04791238877481177, 0.018480492813141684, 0.02292950034223135, 0.04791238877481177, 0.04791238877481177,
                    0.022245037645448322, 0.04791238877481177, 0.007186858316221766, 0.04791238877481177, 0.007529089664613279,
                    0.019164955509924708, 0.03353867214236824, 0.014715947980835045, 0.04791238877481177, 0.02943189596167009,
                    0.01403148528405202, 0.039014373716632446, 0.04791238877481177, 0.04791238877481177, 0.014715947980835045,
                    0.01026694045174538, 0.04791238877481177, 0.04791238877481177, 0.02087611225188227, 0.04791238877481177]
        # self.cls_num_list = [114, 118, 104, 17, 105, 109,
        #                      45, 53, 112, 111, 51, 118, 
        #                      17, 113, 19, 47, 75, 33, 
        #                      106, 73, 38, 86, 110, 115, 
        #                      35, 26, 114, 110, 47, 116]
        #self.cls_num_list = [112, 112, 112, 68, 112, 112, 
         #                     86, 108, 112, 112, 104, 112, 
         #                     64, 112, 72, 90, 78, 136, 
         #                     112, 138, 132, 92, 112, 112, 
         #                     136, 96, 112, 112, 98, 112]
        self.cls_num_list = [140, 140, 140, 21, 140, 140, 
                            54, 67, 140, 140, 65, 140, 
                            21, 140, 22, 56, 98, 43, 
                            140, 86, 41, 114, 140, 140, 
                            43, 30, 140, 140, 61, 140]
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, self.cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        self.per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = paddle.to_tensor(self.per_cls_weights)
        print("###################",self.per_cls_weights)
        
        self.weight_c = paddle.to_tensor(self.weight_c)
        self.weight_c = paddle.pow((1-self.weight_c), self.gamma)

    def forward(self, inputs, targets,valid_mode,soft_label=True):
        N = inputs.shape[0]
        C = inputs.shape[1]
        if valid_mode:
            return 0.0

        P = F.softmax(inputs,axis=1)
        
        prob = targets * P
        cross_entropy = targets * paddle.log(P)
        cross_entropy = -1.0*paddle.fluid.layers.reduce_sum(cross_entropy, dim=-1)
        # Focal loss
        # weight = targets * paddle.pow((1.0 - prob), self.gamma)
        # weight = paddle.fluid.layers.reduce_sum(weight, dim=-1)
        # cross_entropy = paddle.mean(weight*cross_entropy)



        # cross_entropy = targets * paddle.log(P)
        # cross_entropy = paddle.fluid.layers.reduce_sum(cross_entropy, dim=-1)
        
        # prob = targets * P + (1-targets)*(1-P)
        # alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        # weight = -1.0 * alpha_factor * paddle.pow((1.0 - prob), self.gamma)
        # weight = paddle.fluid.layers.reduce_sum(weight, dim=-1)

        
        
        pred = paddle.argmax(targets, axis=1, dtype='int32')
        weight_c = self.per_cls_weights[pred]
        #weight_c1 = self.weight_c[pred] 
        cross_entropy = paddle.mean(weight_c*cross_entropy)

        
        return cross_entropy

    
    
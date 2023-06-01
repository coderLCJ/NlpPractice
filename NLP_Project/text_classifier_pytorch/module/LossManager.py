
from torch.nn import CrossEntropyLoss
from module.loss.focal_loss import FocalLoss
from module.loss.infonce_loss import InfoNCELoss
from module.loss.kl_loss import KLLoss
from module.loss.label_smoothing import LabelSmoothingCrossEntropy


class LossManager(object):
    
    def __init__(self, loss_type, cl_option=False, loss_cl_type='InfoNCE'):
        self.loss_type = loss_type
        self.cl_option = cl_option
        self.loss_cl_type = loss_cl_type
        # 判断配置的loss类型
        if loss_type == 'focalloss':
            self.loss_func = FocalLoss()
        elif loss_type == 'LabelSmoothingCrossEntropy':
            self.loss_func = LabelSmoothingCrossEntropy()
        else:
            self.loss_func = CrossEntropyLoss()
            
        if cl_option:
            if loss_cl_type == 'Rdrop':
                self.loss_cl_func = KLLoss()
            else:
                self.loss_cl_func = InfoNCELoss()


    def compute(self, 
                input_x, 
                target,
                hidden_emb_x=None, 
                hidden_emb_y=None, 
                alpha=0.5):
        """        
        计算loss
        Args:
            input: [N, C]
            target: [N, ]
        """
        if hidden_emb_x is not None and hidden_emb_y is not None:
            loss_ce = (1-alpha) * self.loss_func(input_x, target)
            weight_etx = 1e+5 if self.loss_cl_type=='Rdrop' else 1
            loss_cl = alpha * weight_etx * self.loss_cl_func(hidden_emb_x, hidden_emb_y)
            loss = loss_ce + loss_cl
            return loss
        else:
            loss = self.loss_func(input_x, target)
            return loss
    

    
    # def compute(self, input, target):
    #     """        
    #     计算loss
    #     Args:
    #         input: [N, C]
    #         target: [N, ]
    #     """
    #     loss = self.loss_func(input, target)
    #     return loss


    # def compute(self, input1, input2, output_pooler1, output_pooler2, target, alpha=0.5):
    #     """        
    #     计算loss
    #     Args:
    #         input: [N, C]
    #         target: [N, ]
    #     """
        
    #     loss_ce = alpha * self.loss_func(input1, target)
    #     loss_nce = (1-alpha) * self.loss_func_nce(output_pooler1, output_pooler2)
    #     # loss = alpha*loss_ce + (1-alpha)*loss_nce
    #     loss = loss_ce + loss_nce
    #     return loss, loss_ce, loss_nce
import torch
# from torch._C import LongTensor, dtype
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    '''InfoNCE loss implementation'''
    def __init__(self, temperature=0.999):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature


    def forward(self, input1, input2):
        """
        input1: [N, C]
        input2: [N, C]
        """
        
        # p_matrix_sim = torch.cosine_similarity(input1, input2, dim=1)
        
        # positive 2 norm
        norm_1 = torch.norm(input1,p=2,dim=1)                         # [N,]
        norm_2 = torch.norm(input2,p=2,dim=1)                         # [N,]
        norm_m = norm_1.mul(norm_2.t())                           # [N,N]
        eps = torch.tensor(1e-8)
        norm = 1/torch.max(norm_m, eps)                             # [N,N]
        # norm = 1/norm_m
        
        # negative 2 norm
        norm_n_m = norm_1.mul(norm_1.t())                         # [N,N]
        norm_n = 1/torch.max(norm_n_m, eps)                         # [N,N]
        # norm_n = 1/norm_n_m
        
        # positive sample
        p_matrix_sim = input1.mm(input2.t())                    # [N,N]
        p_matrix_sim = p_matrix_sim.mul(norm)                   # [N,N]
        p_sim = torch.diag(p_matrix_sim)                        # [N,]
        p_sim_zero_matrix = torch.diag_embed(p_sim)             # [N,N]
        # negative sample
        matrix_sim = input1.mm(input1.t())                      # [N,N]
        matrix_sim = matrix_sim.mul(norm_n)                     # [N,N]
        drop_diag = torch.diag(matrix_sim)
        drop_diag_zero_matrix = torch.diag_embed(drop_diag)
        # 减去对角线元素
        matrix_sim_drop = matrix_sim - drop_diag_zero_matrix
        # 对角线加上新的元素
        n_matrix_sim = matrix_sim_drop + p_sim_zero_matrix      # [N,N]
        
        # positive score
        p_exp = torch.exp(p_sim/self.temperature)               # [N,]   
        # total sample score
        total_exp = torch.exp(n_matrix_sim/self.temperature)     # [N,N]
        total_exp_sum = total_exp.sum(dim=0)
        # loss
        loss = torch.log(p_exp/total_exp_sum)
        loss = -1 * loss
        loss = loss.mean()
        # print('positive exp:{} negative exp:{}'.format(p_exp.mean(),total_exp.mean()))
        return loss

        
        
if __name__ == '__main__':
    infonce = InfoNCELoss()
    input1 = torch.randn([20,5])   
    input2 = torch.randn([20,5])   
    target = torch.randint(0,5,[20,])
    input_ids_anti = torch.randn([20,50,5])
    label_anti = torch.randint(0,5,[20,50,])
    
    loss = infonce(input1=input1, input2=input2)
    print(1)
    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class MCLLoss(nn.Module):

    def __init__(self, xi=10.0, eps=10.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 10.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(MCLLoss, self).__init__()
        self. main_loss = nn.MELoss()

    def forward(self,meta_net,meta_opt,main_net,main_opt,batch_data,label,mask,teacher_net):
        
        prediction= main_net(batch_data)
        unimodal_feature = prediction['unimodal'].detach()
        p_label = teacher_net(batch_data)['label']
        final_label = p_label[mask] + label[mask]
        loss_1 =self.main_loss(prediction['label'],final_label) 
        return 1


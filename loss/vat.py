import contextlib
from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=10.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 10.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = 10.0
        self.ip = ip
        self.drop = nn.Dropout(p=0.2)
        self.drop_2 = nn.Dropout(p=0.2)
    def forward(self,model,x,mask_index,device):
        with torch.no_grad():
            feature_m = model(x[0],x[1],x[2])['feature_m'][mask_index]
            feature_m = F.softmax(feature_m, dim=1)
        # prepare random unit tensor
        d_0 = torch.rand(x[0][0].shape).sub(0.5).to(device)
        d_1 = torch.rand(x[1][0].shape).sub(0.4).to(device)
        d_2 = torch.rand(x[2][0].shape).sub(0.3).to(device)
        mask_text = x[0][0] != 0
        mask_audio = x[1][0] != 0
        mask_vision = x[2][0] != 0
        d_0 = d_0*mask_text
        d_0 = _l2_normalize(d_0)
        d_1 = d_1*mask_audio
        d_1 = _l2_normalize(d_1)
        d_2 = d_2*mask_vision
        d_2 = _l2_normalize(d_2)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d_0.requires_grad_()
                d_1.requires_grad_()
                d_2.requires_grad_()
                nosie_0 = (x[0][0] + self.xi * d_0,x[0][1])
                nosie_1 = (x[1][0] + self.xi * d_1,x[1][1])
                nosie_2 = (x[2][0] + self.xi * d_2,x[2][1])
                feature_m_hat = model(nosie_0,nosie_1,nosie_2)['feature_m'][mask_index]
                # 输出drop
                feature_m_hat = self.drop(feature_m_hat)
                feature_m = self.drop_2(feature_m)
                # feature_m_hat = 
                feature_m_hat = F.log_softmax(feature_m_hat, dim=1)
                adv_distance = F.kl_div(feature_m_hat,feature_m , reduction='batchmean')
                adv_distance.clamp_(min=0.0)
                adv_distance.backward()
                if d_0.grad is not None:
                    d_0 = _l2_normalize(d_0.grad)*mask_text
                if d_1.grad is not None:
                    d_1 = _l2_normalize(d_1.grad)*mask_audio
                if d_2.grad is not None:
                    d_2 = _l2_normalize(d_2.grad)*mask_vision
                model.zero_grad()
    
            # calc LDS
            r_adv_0 = d_0 * self.eps
            r_adv_1 = d_1 * self.eps
            r_adv_2 = d_2 * self.eps
            vat_nosie_0 = (x[0][0] + r_adv_0,x[0][1])
            vat_nosie_1 = (x[1][0] + r_adv_1,x[1][1])
            vat_nosie_2 = (x[2][0] + r_adv_2,x[2][1])
            feature_m_vat = model(vat_nosie_0,vat_nosie_1,vat_nosie_2)['feature_m'][mask_index]
            feature_m_vat = self.drop(feature_m_vat)
            feature_m_vat = F.log_softmax(feature_m_vat, dim=1)
            lds = F.kl_div(feature_m_vat, feature_m, reduction='batchmean')
            lds.clamp_(min=0.0)
        return lds

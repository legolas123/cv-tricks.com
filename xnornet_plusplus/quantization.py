import torch
import numpy as np
import torch.nn as nn
from models.resnet_preact_bin import BinConv2d
class Binarize():
    def __init__(self, model):
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): 
                count_targets = count_targets + 1

        start_range = 1
        end_range = count_targets-2
        self.bin_range = np.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)
        
        param_grp_1 = []
        for m in model.modules():
            if isinstance(m,BinConv2d):
                param_grp_1.append(m.conv.weight)
                param_grp_1.append(m.alpha)
                param_grp_1.append(m.beta)
                param_grp_1.append(m.gamma)
        param_grp_1_ids = list(map(id, param_grp_1))
        param_grp_2 = list(filter(lambda p: id(p) not in param_grp_1_ids, model.parameters()))

        self.param_grp_1 = param_grp_1
        self.param_grp_2 = param_grp_2
        
    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            negmean = self.target_modules[index].data.mean(1,keepdim=True).mul(-1).\
                expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negmean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)
    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            scaling_factor = weight.abs().sum(dim=3, keepdim=True).sum(dim=2, keepdim=True).sum(dim=1, keepdim=True)/n
            #scaling_factor = torch.mean(torch.mean(torch.mean(abs(weight),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)

            self.target_modules[index].data = \
                    self.target_modules[index].data.sign()*scaling_factor

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):

            weight = self.target_modules[index].data
            C = weight.size(1)
            n = weight[0].nelement()
            #scaling_factor = weight.abs().sum(dim=3, keepdim=True).sum(dim=2, keepdim=True).sum(dim=1, keepdim=True)/n
            #weight_scaled = weight.sign()*scaling_factor
            #negmean = weight.mean(1,keepdim=True).mul(-1).\
            #    expand_as(weight)
            #weight = weight.add(negmean)
            #weight = weight.clamp(-1.0,1.0)
            #C = weight.size(1) ##Num of channels
            
            grad = self.target_modules[index].grad.data
            ###Grad due to sign
            grad[weight.lt(-1.0)] = 0
            grad[weight.gt(1.0)] = 0

            #grad_due_to_mean = grad.sum(1,keepdim=True).expand_as(grad).mul(-1./C)

            #grad = grad + grad_due_to_mean

            self.target_modules[index].grad.data = grad
            
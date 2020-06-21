from .resnet_preact import resnet18_preact
from .resnet_preact_bin import resnet18_preact_bin
import torch, torch.nn as nn
_model_factory = {
    "resnet18_preact":resnet18_preact,
    "resnet18_preact_bin":resnet18_preact_bin
}

class Classifier(torch.nn.Module):
    def __init__(self, feat_extractor,num_classes=None):
        super(Classifier,self).__init__()
        self.feat_extractor = feat_extractor
        self.class_fc = nn.Linear(feat_extractor.fc.in_features, num_classes)
    def forward(self,x):
        x = self.feat_extractor(x)
        class_output = self.class_fc(x)
        return class_output


def get_model(arch_name, **kwargs):
    backbone =  _model_factory[arch_name](**kwargs)
    return Classifier(backbone, num_classes = kwargs["num_classes"])

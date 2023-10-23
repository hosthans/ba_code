import torch.nn as nn
import torchvision.models
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        model=torchvision.models.vgg16_bn(pretrained=False)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False) # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        return [feature, res]
    
    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return out
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def get_imagenet_model(model_name, n_classes,
                         feature_extractor=False, pretrained=True):
    """Returns one of the ImageNet models. Only supports the feedforward models
       with input 224x224 and one output (not like inception)
    """
    model = getattr(models, model_name)(pretrained=pretrained)

    if feature_extractor:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)

    return model

class DogsBreedClassifier(nn.Module):

    def __init__(self, n_classes=120):
        super(DogsBreedClassifier, self).__init__()
        self.conv1_1 = BasicConv(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = BasicConv(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = BasicConv(64, 128, kernel_size=3, padding=1)
        self.conv3_1 = BasicConv(128, 256, kernel_size=3, padding=1)
        self.conv4_1 = BasicConv(256, 256, kernel_size=3, padding=1)
        self.conv5_1 = BasicConv(256, 512, kernel_size=3, padding=1)
        self.conv6_1 = BasicConv(512, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7 ** 2 * 512, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2_1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3_1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv4_1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv5_1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv6_1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class BasicConv(nn.Module):
    """
    A simple block consisting of a convolutional layer followed by
    batch normalization and relu
    """
    def __init__(self, in_, out, **kwargs):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_, out, **kwargs)
        self.bn = nn.BatchNorm2d(out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
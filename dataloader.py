import torch
import torchvision
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import copyfile

def get_loader(path, mode, batch_size, num_workers, crop_size=224):
    if mode == 'train':
        dataset = torchvision.datasets.ImageFolder(
            path,
            transform=transforms.Compose((transforms.Resize(224),
                                          transforms.RandomCrop(crop_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))))
        )
    elif mode == 'test':
        dataset = torchvision.datasets.ImageFolder(
            path,
            transform=transforms.Compose((transforms.Resize(224),
                                          transforms.CenterCrop(crop_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225)))
                                         )
        )

    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers,
                                         batch_size=batch_size, shuffle=True)

    return loader
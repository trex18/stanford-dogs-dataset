import torch
from torch import nn
from torch.optim import lr_scheduler
from model import DogsBreedClassifier
from train import train_network
from dataloader import get_loader
from model import get_imagenet_model

clf = get_imagenet_model('resnet18', 120)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clf = clf.to(device)
optimizer = torch.optim.RMSprop(clf.parameters(), momentum=0.9)
exp_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)
criterion = nn.CrossEntropyLoss()
loaders = {
    'train': get_loader(path='D:/Jupyter/datasets/stanford-dogs-dataset/images/train',
                        mode='train', batch_size=4, num_workers=0),
    'test': get_loader(path='D:/Jupyter/datasets/stanford-dogs-dataset/images/test',
                       mode='test', batch_size=1, num_workers=0),
    'val' : get_loader(path='D:/Jupyter/datasets/stanford-dogs-dataset/images/val',
                       mode='test', batch_size=4, num_workers=0)
}

clf, acc_history, loss_history = train_network(clf, 25, optimizer, criterion, exp_scheduler, device, loaders)

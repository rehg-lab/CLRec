import torch
import torch.nn as nn
from torch.autograd import Variable

def kaiming_normal_init(m):
    '''
    Initializes network parameters using Kaiming-Normal initialization
    '''
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

def MultiClassCrossEntropyLoss(logits, labels, T, device):
    '''
    Cross Entropy Distillation Loss
    '''
    labels = Variable(labels.data, requires_grad=False).cuda(device=device)
    outputs = torch.log_softmax(logits/T, dim=1)
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True).cuda(device=device)
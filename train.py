from timeit import default_timer as timer
import os

import torch
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
from capsnet import Capsnet


def train(model, data_loader, optimizer, cuda_enabled, epoch):
    # Train the capsnet model on the mnist dataset
    print('Training mode')
    num_batches = len(data_loader)

    epoch_total_acc= 0

    #Switch to train mode
    model.train()

    if cuda_enabled:
        model = model.cuda()


    start_time = timer()



    for batch_idx, (data, target) in enumerate(data_loader):
        if cuda_enabled:
            data, target =  data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.data[0]))







if __name__ =='__main__':
    model = Capsnet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    data_loader = ''
    for epoch in range(20):
        train(model, data_loader='', optimizer=optimizer, cuda_enabled=True, epoch=epoch)



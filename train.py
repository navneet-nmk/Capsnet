from timeit import default_timer as timer
import os

import torch
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
from capsnet import Capsnet
import torchvision
from torchvision import transforms
import utils


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
        batch_size = 128

        target_one_hot = utils.one_hot_encode(target, length=10)
        assert target_one_hot.size() == torch.Size([batch_size, 10])

        target_one_hot = target_one_hot.type(torch.LongTensor)

        if cuda_enabled:
            data, target_one_hot =  data.cuda(), target_one_hot.cuda()

        data, target = Variable(data), Variable(target_one_hot)


        print(data.shape, target.shape)

        optimizer.zero_grad()
        output = model(data)
        print(output)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.data[0]))







if __name__ =='__main__':
    model = Capsnet( num_input_conv_layer =1, num_output_conv_layer=256, conv_kernel_dim=9, conv_kernel_stride=1, num_primary_unit=8, primary_unit_size=1152,
                 num_classes=10, output_unit_size=16, num_routing=3, cuda_enabled=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Normalize MNIST dataset.
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    imagenet_data = torchvision.datasets.MNIST(root='/home/kumar/PycharmProjects/Capsnet/',download=True, transform=data_transform, train=True)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=4,)

    for epoch in range(20):
        train(model, data_loader=data_loader, optimizer=optimizer, cuda_enabled=True, epoch=epoch)



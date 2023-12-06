import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from model import pyramidnet
import argparse
# from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--num_worker', type=int, default=4, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
parser.add_argument("--ddp_mode", type=str, default='dp', help="")
args = parser.parse_args()

if args.ddp_mode == 'dp':
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
elif args.ddp_mode == 'ddp':
    from accelerate import Accelerator
    accelerator = Accelerator()

def main():
    best_acc = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('==> Preparing data..')	
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = CIFAR10(root='./data', train=True, download=True,
                            transform=transforms_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_worker)

    # there are 10 classes so the dataset name is cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> Making model..')

    net = pyramidnet()
    if args.ddp_mode == 'dp':
        net = nn.DataParallel(net)
        net = net.to(device)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    if args.ddp_mode == 'ddp':
        net, optimizer, train_loader = accelerator.prepare(net, optimizer, train_loader)
    for i in range(10000):
        train(net, criterion, optimizer, train_loader, device)
            

def train(net, criterion, optimizer, train_loader, device):
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    

    while True:
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            if args.ddp_mode == 'dp':
                inputs = inputs.to(device)
                targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            if args.ddp_mode == 'ddp':
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100 * correct / total



            if batch_idx % 20 == 0:
                print('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                    batch_idx, len(train_loader), train_loss/(batch_idx+1), acc, 0))



    

if __name__=='__main__':
    main()

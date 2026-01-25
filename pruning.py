import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models import LeNet, AlexNet
from src.quantization import apply_weight_sharing
import src.util

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--model', type=str, default='LeNet', metavar='MODELNAME',
                    help='model to compress (default: LeNet)')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

MODELNAME = args.model

# Loader
mean = {'LeNet' : (0.1307,),
        'AlexNet' : [0.485, 0.456, 0.406]}
std  = {'LeNet' : (0.3081,),
        'AlexNet' : [0.229, 0.224, 0.225]}
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
if MODELNAME == 'LeNet':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean['LeNet'], std['LeNet'])
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean['LeNet'], std['LeNet'])
                    ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif MODELNAME=='AlexNet':
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['AlexNet'], std['AlexNet']),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean['AlexNet'], std['AlexNet']),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root='data/tiny-imagenet-200/train',
            transform=train_transforms
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root='data/tiny-imagenet-200/val',
            transform=test_transforms
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs
    )



# Define which model to use
if MODELNAME == 'LeNet':
    model = LeNet(mask=True).to(device)
elif MODELNAME=='AlexNet':
    model = AlexNet(mask=True, pretrained=True).to(device)

print(model)
src.util.print_model_parameters(model)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
initial_optimizer_state_dict = optimizer.state_dict()

def train(epochs):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if MODELNAME=="LeNet":
                loss = F.nll_loss(output, target)
            elif MODELNAME=='AlexNet':
                loss = F.cross_entropy(output, target)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if MODELNAME=="LeNet":
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            elif MODELNAME=='AlexNet':
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


# Initial training
print("--- Initial training ---")
train(args.epochs)
accuracy = test()
src.util.log(args.log, f"initial_accuracy {accuracy}")
torch.save(model, f"saves/initial_model.ptmodel")
print("--- Before pruning ---")
src.util.print_nonzeros(model)

# Pruning
model.prune_by_std(args.sensitivity)
accuracy = test()
src.util.log(args.log, f"accuracy_after_pruning {accuracy}")
print("--- After pruning ---")
src.util.print_nonzeros(model)

# Retrain
print("--- Retraining ---")
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
train(args.epochs)
torch.save(model, f"saves/model_after_retraining.ptmodel")
accuracy = test()
src.util.log(args.log, f"accuracy_after_retraining {accuracy}")

print("--- After Retraining ---")
src.util.print_nonzeros(model)
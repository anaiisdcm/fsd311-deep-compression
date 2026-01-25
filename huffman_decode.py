import argparse

import torch

from src.models import LeNet
from src.huffmancoding import huffman_decode_model
import src.util

# Decoding settings
parser = argparse.ArgumentParser(description='PyTorch MNIST decoding from deep compression paper')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
args = parser.parse_args()


# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')


# Decode model that has been Huffman encoded
model = LeNet(mask=True).to(device)
huffman_decode_model(model, directory='encodings/')

src.util.test(model, use_cuda)

torch.save(model, f"saves/model_after_decoding.ptmodel")

torch.save(model.state_dict(), f"saves/model_after_decoding_state_dict.pth")
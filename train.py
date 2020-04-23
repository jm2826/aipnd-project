%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import random

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image

import argparse
import image

AP = argeparse.ArgumentParser(description='train.py')

AP.add_argument('data_dir', nargs='*', action="store", default="./flowers")
AP.add_argument('--gpu', dest="gpu", action="store", default="gpu")
AP.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
AP.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
AP.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
AP.add_argument('--structure', dest="structure", action="store", default="vgg16", type = str)
AP.add_argument('--hidden_layer', dest="hidden_layer", action]"store", type = int, default = 1024)

parse= AP.parse_args()
data_dir = parse.data_dir
power = parse.gpu
checkpoint_path = parse.save_dir
lr = parse.learning_rate
epochs= parse.epochs
structure = parse.structure
hidden_layer = parse.hidden_layer

dataloaders = model_structure.load_data(data_dir)

model_structure.train_model( model, criterion, optimizer, lr,
                            epochs, power)
model_structure.save_checkpoint(checkpoint_path, structure)

print("Model is now trained") 

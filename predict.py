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


AP = argeparse.ArgumentParser(description='predict.py')
AP.add_argument('img', nargs='*', action="store", type = str, default="/101/image_07949.jpg")
AP.add_argument('--gpu', dest="gpu", action="store", default="gpu")
AP.add_argument('checkpoint', nargs='*', action="store", type = str, default="checkpoint.pth")
AP.add_argument('--top_k', default = 5, dest="top_k", action="store", type = int)
Ap.add_argument('--category_names', dest= "category_names", action="store", default='cat_to_name.json')

parse = AP.parse_args()

number_of_outputs = parse.top_k
power = parse.gpu
input_img = parse.img
checkpoint_path = parse.checkpoint

dataloaders= model_structure.load_data()

model_structure.load_checkpoint(checkpoint_path)

 # CPU
 device = torch.device("cpu")

# GPU
if parse.gpu:
 device = torch.device("cuda:0")

with open(category_names) as json_file:
    cat_to_name = json.load(json_file)
    
top_prob, top_classes = model_structure.predict(input_image, model, number_of_outputs)

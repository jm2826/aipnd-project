# Imports here
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

def load_data(data_dir = "/.flowers"):
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    # Put pics files and directories in the correct construct
    # image_datasets = 
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir,transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # Returns a batch of the pic files it its corresponding labels for test and train data.
    # dataloaders = 
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True)
    
    return trainloader, vloader, testloader

##################################################

# TODO: Build and train your network
# Load a pretrained model like VGG16 (16 - )

model = models.vgg16(pretrained = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Freeze parameters so we don't backprop through them
# Turn off gradients for the model
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(nn.Linear(25088, 4096),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(4096,102),
                        nn.LogSoftmax(dim=1))
    
model.classifier = classifier

#################################################################################
# Define loss (negative log like we had loss)
criterion = nn.NLLLoss()

# Define optimizer using parameters from classifier
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

# Move model to whatever device thats available
model.to(device)
###################################################################################

def train_model(model, criterion, optimizer, scheduler,
                num_epochs=, device = 'cuda'):
    
    epochs = 6
    steps = 0
    running_loss = 0
    print_every = 15

    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
        
            # After going through a batch, move images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)
        
            # 0 out the gradient, log probabilities from model, get loss from the criterion in the labels
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            # Drop out of train loop and test networks accuracy and loss
            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0
            
                for images, labels in testloader:
                
                    # After going through a batch, move images and labels to GPU if available
                    images, labels = images.to(device), labels.to(device)
               
                    logps = model.forward(images)
                    loss = criterion(logps, labels)
                    test_loss += loss.item()
                
                    # calculate accuracy
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f} ")
            
                running_loss = 0
                model.train()
################################################################

model.to('cpu')
model.class_to_idx = train_data.class_to_idx
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epochs':epochs,
              'batch_size':64,
              'model':models.vgg16(pretrained=True),
              'classifier':model.classifier,
              'optimizer':optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx':model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])#, strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
  
        
    return model, checkpoint['class_to_idx']
model, class_to_idx = load_checkpoint('checkpoint.pth')

print(model.classifier)
############################################################################################

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    width, height = im.size
    im_ratio = width / height
    if im_ratio >= 1:
        new_height = 256
        new_width = round(256 * im_ratio)
        im.thumbnail((new_width, new_height))
    else:
        new_width = 256
        new_height = round(256 / im_ratio)
        im.thumbnail((new_width, new_height))
        
    width, height = im.size    
    new_width = 224
    new_height = 224
    left = round((width - new_width)/2)
    upper = round((height -new_height)/2)
    right = round((width + new_width)/2)
    lower = round((height + new_height)/2)
    im = im.crop((left, upper, right, lower))
    
    np_im = np.array(im)
    np_im = np_im / 255
    
    np_mean = np.array([0.485, 0.456, 0.406])
    np_std = np.array([0.229, 0.224, 0.225])
    
    norm_im = (np_im - np_mean) / np_std

    norm_im = norm_im.transpose((2, 0, 1)) 
    return norm_im

#######################

def predict(image_path, model, topk=5):
# Predict the class (or classes) of an image using a trained deep learning model.
    
    model.to('cpu')
    model.eval()
    image = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor)
    image.unsqueeze_(0)
    
    log_output = model.forward(image)
    output = torch.exp(log_output)
    
    top_p, top_class = output.topk(topk, dim=1)
    top_p = top_p.detach().numpy().tolist()[0]
    top_class_idx = top_class.detach().numpy().tolist()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    top_class = [idx_to_class[i] for i in top_class_idx]
    
    top_flower = [cat_to_name[i] for i in top_class]
    return top_p, top_class, top_flower

predict(test_dir + '/101/image_07949.jpg', model)

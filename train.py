import numpy as np
import pandas as pd
import torch 
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets,transforms,models
from torch import optim
import torch.nn.functional as F
from PIL import Image
import json

import argparse
parser = argparse.ArgumentParser (description = "Parser of training script")
parser.add_argument ('data_dir', help = 'Provide data directory. Mandatory argument', type = str)
args = parser.parse_args ()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#data_transforms = transforms.Compose([transforms.Resize(256),
#                                      transforms.CenterCrop(224),
#                                     transforms.ToTensor(),
#                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) 
#                                     ])
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                           ])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),                 
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

#image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

#dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64) 
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 2014)),
                          ('relu1', nn.ReLU()),
                          ('dropout',nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(2014,512)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(512,102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = classifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.002)
model.to(device);


epochs = 8
steps = 0
running_loss = 0
print_every = 40
for epoch in range(epochs):
    for inputs, labels in trainloader: 
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            val_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    val_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"val loss: {val_loss/len(validationloader):.3f}.. "
                  f"val accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            model.train()

model.to('cpu')
model.class_to_idx= train_data.class_to_idx

checkpoint = {'classifier': model.classifier,
             'state_dict':model.state_dict(),
             'mapping': model.class_to_idx}
torch.save(checkpoint,'Project_devansh.pth')
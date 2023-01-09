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

def Model_loader(file_path):
    checkpoint = torch.load(file_path)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict = (checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
    for param in model.parameters(): 
        param.requires_grad = False 
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    width,height = img.size
    
    if width>height:
        height = 256
    else:
        width = 256
    
    width,height = img.size
    reduce = 224
    left = (width - reduce)/2 
    top = (height - reduce)/2
    right = left + 224 
    bottom = top + 224
    img = img.crop ((left, top, right, bottom))    
    
    # TODO: Process a PIL image for use in a PyTorch model
    numpy_image = np.array (img)/255 
    numpy_image -= np.array ([0.485, 0.456, 0.406]) 
    numpy_image /= np.array ([0.229, 0.224, 0.225])
    numpy_image = numpy_image.transpose ((2,0,1))
    return numpy_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    img = torch.from_numpy(image).type(torch.FloatTensor)
    img = torch.unsqueeze(img,dim=0)
    
    with torch.no_grad():
        output = model.forward(img)
    output_probabs = torch.exp(output)
    
    probabs,indices = output_probabs.topk(5) 
    probabs,indices = probabs.numpy().tolist()[0],indices.numpy().tolist()[0]
    
    mapping = {val:key for key,val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indices]
    classes = np.array(classes)
    return probabs, classes


#main
import argparse
parser = argparse.ArgumentParser (description = "Parser of prediction script")
parser.add_argument ('image_dir', help = 'Provide path to image. Mandatory argument', type = str)
args = parser.parse_args ()
file_path = args.image_dir

model = Model_loader("/home/workspace/ImageClassifier/Project_devansh.pth")
probs, classes = predict (file_path, model, 1)

with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
   
class_names = [cat_to_name [item] for item in classes]
    
print("Class name: {}.. ".format(class_names [0]),"Probability: {:.3f}..% ".format(probs [0]*100))
from collections import OrderedDict
import math 
import torch
import json
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
from scipy.special import softmax
import argparse
import os
import sys

# Predict flower name from an image with predict.py along with the probability of that name. 
# That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# - Return top KK most likely classes:          python predict.py input checkpoint --top_k 3
# - Use a mapping of categories to real names:  python predict.py input checkpoint --category_names cat_to_name.json
# - Use GPU for inference:                      python predict.py input checkpoint --gpu

#--------------------------------------------------- LOAD IMAGES 
def load_images():
    data_dir = 'flowers'
    test_dir  = data_dir + '/test'
    
    # Define the transforms for the training, validation, and testing sets
    data_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    # Datasets that load/transform the images. It expects the images to be in format "root/label/picture.jpg"
    image_dataset = datasets.ImageFolder(test_dir,  transform=data_transform)
    
    # print(image_dataset)                              # 819 testing images
        
    # Dataloaders will load batches of 64 images
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size, shuffle=False)
    
    # images, labels = next(iter(dataloaders['training']))          # Get batch of 64 flower images, and the specie of each one
    # print (labels)                                                #    labels is a tensor of 64 numbers (species of each flower)
    # print (images.shape)                                          #    images is a torch.Size([64, 3, 224, 224])  
    #                                                               #    64 images, 3 colors deep, 224 pixels high, 224 pixels wide
    # plt.imshow(images[0,0])                                       # Show a sample image
    
    return dataloader, image_dataset

#---------------------------------------------------  LOAD CHECKPOINT 
def load_checkpoint(filename):
    print("Loading checkpoint")
    checkpoint = torch.load(filename)
    
    if checkpoint['model'] == 'VGG19':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            model.class_to_idx = checkpoint['class_to_idx']
    
            classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
            model.classifier = classifier
            model.load_state_dict(checkpoint['state_dict'])
    else:
        print("Checkpoint is not VGG19")
    
    print('Checkpoint loaded')
    return model

#----------------------------------------------------  SHOW IMAGE 

def show_image_in_tensor(image):
    fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    show_image(image)

def show_image_in_path(path):
    image = Image.open(path)
    show_image(image)

def show_image(image):
    plt.imshow(image)

#----------------------------------------------------  PROCESS IMAGE 
def process_image(path):
    image = Image.open(path)
    transform = transforms.Compose([transforms.Resize(256),                        # Resize
                                    transforms.CenterCrop(224),                    # Crop
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],    # Normalize
                                                         [0.229, 0.224, 0.225])])
    tensor = transform(image)
    return tensor

#---------------------------- LOAD JSON DICTIONARY WITH FLOWER NAMES 
def load_json(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)                              # Load JSON with the number-name associated to every flower
    return cat_to_name

#---------------- Obtains class_to_idx dictionary from model, and returns a converter dictionary
def create_idx_converter(model):
    idx_converter = {}
    #print(f"\nIndexes original dictionary: {model.class_to_idx.items()}")    
    for key, value in model.class_to_idx.items():
        idx_converter[value] = key
    #print(f"\nIndexes converter: {idx_converter}")
    return idx_converter

#--------------  Receives array of model indexes, and returns array with the converted indexes
def convert_indexes(idx, model):
    np_idx = idx[0].numpy()
    print(f"\nOriginal indexes: {np_idx}  (need to be converted)")
    idx_converter = create_idx_converter(model)     # Create indexes converter
    
    converted_indexes = []                  
    for label in np_idx:
        converted_indexes.append(int(idx_converter[label]))
    
    return converted_indexes

#--------------- Receives array of indexes, and returns array with the flower names
def get_flower_names(flower_idx):
    cat_to_name = load_json(json_file)
    #print(f"\nFlowers dictionary: {cat_to_name}")     
    
    flower_names = []                  
    for idx in flower_idx:
        flower_names.append(cat_to_name[str(idx)])
    return flower_names
    
#--------------------------------------------------- PREDICT ONE IMAGE 
def predict(image_path, model):
    image = process_image(image_path)               # Transform image
    image.unsqueeze_(0)                             # Convert image from ([3, 224, 224]) to ([1, 3, 224, 224])
    output = model.forward(image)                   # Forward
    probs = torch.exp(output)                       # Normalize probs between 0 and 1 
    top_probs, top_idx = probs.topk(5)              # Take the top 5 probs
    
    top_converted_indexes = convert_indexes (top_idx, model)
    print(f"\nTop converted indexes: {top_converted_indexes}")
    
    top_flower_names = get_flower_names(top_converted_indexes)
    return top_probs, top_converted_indexes, top_flower_names


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

#--------------------------------------------------- SANITY CHECK
def sanity_check(flower_name, top_probs, top_flower_names):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,7))     # 2 rows, 1 column. Flower in first position
    
    flower_image = process_image(image_path)                    # Transform the image to a tensor
    ax[0] = imshow(flower_image, ax[0])                         # Plot flower image
    top_probs_array = top_probs[0].detach().numpy()             # Convert tensor to array with probs
    ax[1] = sns.barplot(x=top_probs_array, y=top_flower_names, color=sns.color_palette()[0]);

    plt.suptitle(flower_name)                                   # Title with the flower name
    plt.show()
    
#--------------------------------------------------- MAIN 
    
my_parser = argparse.ArgumentParser(prog='Predict', description="Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability")

my_parser.add_argument('Image', type=str, help='Single input image')
my_parser.add_argument('Checkpoint', type=str, help='Checkpoint file to use')

my_parser.add_argument('--top_k', action='store', type=int, default=5, help='Return top K most likely classes')
my_parser.add_argument('--category_names', action='store', type=str, default='cat_to_name.json', help='Use a mapping of categories to real names')
my_parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

args = my_parser.parse_args()               # Parse the parameters
print(vars(args))                          # Show the parsed parameters

input_image = args.Image
input_checkpoint = args.Checkpoint
input_top_k = args.top_k
input_category_names = args.category_names
input_gpu = args.gpu

print(f"input_image: {input_image}")
print(f"input_checkpoint: {input_checkpoint}")
print(f"input_top_k: {input_top_k}")
print(f"input_category_names: {input_category_names}")
print(f"input_gpu: {input_gpu}\n")
    
      
# json_file = 'cat_to_name.json'
# cat_to_name = load_json(json_file)

# model = load_checkpoint('flowers.pth')                                      # Load checkpoint    

# image_path = 'flowers/test/10/image_07090.jpg'
# flower_num = image_path.split('/')[2]
# flower_name = cat_to_name[flower_num]   

# top_probs, top_labels, top_flower_names = predict(image_path, model)        # Make prediction with 1 flower
# sanity_check (flower_name, top_probs, top_flower_names)                     # Plot flower and bar chart

print('THE END')


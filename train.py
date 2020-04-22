from collections import OrderedDict
import math 
import torch
import json
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from timeit import default_timer as timer
import argparse
import os
import sys

# train.py will train a new network on a dataset and save the model as a checkpoint. 
#
# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
#
# Options:
# - Set directory to save checkpoints:  python train.py data_dir --save_dir save_directory
# - Choose architecture:                python train.py data_dir --arch "vgg13"
# - Set hyperparameters:                python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# - Use GPU for training:               python train.py data_dir --gpu
#
# The best way to get the command line input into the scripts is with the argparse module in the standard library. 
# You can also find a nice tutorial for argparse here:  https://pymotw.com/3/argparse/


#--------------------------------------------------- LOAD IMAGES ---------------------------------------------------
def load_images():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir  = data_dir + '/test'
    
    # Define the transforms for the training, validation, and testing sets
    data_transforms = {
        'training' : transforms.Compose([transforms.RandomResizedCrop(224),             # Images of 224x224 pixels
                                        transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                                        transforms.ToTensor(),                          
                                        transforms.Normalize([0.485, 0.456, 0.406],     # Colors normalized to mean [0.485, 0.456, 0.406]
                                                             [0.229, 0.224, 0.225])]),  # and standard deviation [0.229, 0.224, 0.225]
                                                                
        'validate' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]),
    
        'testing' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    }
    
    # Datasets that load/transform the images. It expects the images to be in format "root/label/picture.jpg"
    image_datasets = {
        'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validate' : datasets.ImageFolder(valid_dir, transform=data_transforms['validate']),
        'testing'  : datasets.ImageFolder(test_dir,  transform=data_transforms['testing'])
    }
    
    # print(image_datasets['training'])                             # 6552 training images
    # print(image_datasets['validate'])                             # 818 validate images
    # print(image_datasets['testing'])                              # 819 testing images
        
    # Dataloaders will load batches of 64 images
    dataloaders = {
        'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size, shuffle=True),
        'validate' : torch.utils.data.DataLoader(image_datasets['validate'], batch_size, shuffle=True),
        'testing'  : torch.utils.data.DataLoader(image_datasets['testing'],  batch_size, shuffle=False)
    }
    
    # images, labels = next(iter(dataloaders['training']))          # Get batch of 64 flower images, and the specie of each one
    # print (labels)                                                #    labels is a tensor of 64 numbers (species of each flower)
    # print (images.shape)                                          #    images is a torch.Size([64, 3, 224, 224])  
    #                                                               #    64 images, 3 colors deep, 224 pixels high, 224 pixels wide
    # plt.imshow(images[0,0])                                       # Show a sample image
       
    return dataloaders, image_datasets

#--------------------------------------------------- LOAD VGG19 PRETRAINED MODEL ---------------------------------------------------
def load_vgg19_pretrained_model():
    model = models.vgg19(pretrained=True)                           # We will use a pretrained VGG19 model
    
    classifier = nn.Sequential(OrderedDict([                        # New classifier that will replace the VGG19 default classifier
                              ('fc1', nn.Linear(25088, 4096)),      #   First layer (25088 inputs, the same as default VGG19 classifier)
                              ('relu', nn.ReLU()),                  #   ReLu activation function
                              ('fc2', nn.Linear(4096, 102)),        #   Output layer (102 flower species)
                              ('output', nn.LogSoftmax(dim=1))      #   Softmax loss function
                              ]))
    
    for param in model.parameters():                                # Don't udpate all the pretrained weights, just the classifier
        param.requires_grad = False
    
    model.classifier = classifier                                   # Replace default VGG19 classifier with the new one
    
    model.to(device);                                               # Move model to GPU (if we have), or CPU
    return model

#--------------------------------------------------- TRAIN THE CLASSIFIER ---------------------------------------------------
def train(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader):
    model.train()                                                   # Put model in training mode
    print_every = 5
    
    for epoch in range(epochs):                                     # In every epoch, we train with all the 6552 flower images
        num_batches = math.ceil(6552/batch_size)
        print(f"\n\nTRAINING EPOCH {epoch+1}  ({num_batches} batches of {batch_size} images)")
        print(f'  ', end ='') 
        running_loss = 0
        count_validate = 0
        count_batch=0   
        start = timer()
        for images, labels in iter(training_loader):                # In every iteration, we train with a batch of 64 images
            count_validate += 1
            count_batch+=1                                          
            print(f'{count_batch%10}', end ='')                     # Show batch number (103 batches per epoch)
            if (count_batch%10==0): 
                print(' ', end ='') 

            images = images.to(device)                              # Move images to the default device
            labels = labels.to(device)                              # Move labels to the default device

            optimizer.zero_grad()                                   # Reset gradients
            output = model.forward(images)                          # Forward 
            loss = criterion(output, labels)                        # Calculate error
            loss.backward()                                         # Backpropagation
            optimizer.step()                                        # Optimize weights
            running_loss += loss.item()                             # Calculate accumulated error

            if count_validate % print_every == 0:                   # AVOID VALIDATION TO GENERATE A LOCAL CHECKPOINT QUICK
                return
            
            if count_validate % print_every == 0:                   # Every x training steps, we make 1 validation
                print(f'V', end ='')  
                validation_loss, accuracy = validate(model, criterion, validation_loader)

                end = timer()

                print(f"\n  Training loss: {running_loss/print_every:.3f}  Validation loss: {validation_loss:.3f} " +
                      f" Validation accuracy: {accuracy:.3f}  Time: {end-start:.2f} sec ")
                print(f'  ', end ='') 
                running_loss = 0
                start = timer()
                
#---------------------------------------------------  VALIDATE CLASSIFIER WHILE TRAINING ---------------------------------------------------
def validate(model, criterion, validation_loader):
    model.eval()                                                                # Put model in validation mode
    accuracy = 0
    validation_loss = 0                                                         # We validate with a total of 818 images
    
    with torch.no_grad():                                                       # Disable gradients for speed
        for images, labels in iter(validation_loader):                          # In every iteration, we train with a batch of 64 images
            images = images.to(device)                                          # Move images to the default device
            labels = labels.to(device)                                          # Move labels to the default device
            
            output = model.forward(images)                                      # Forward 
            validation_loss += criterion(output, labels).item()                 # Calculate error
            ps = torch.exp(output)

            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()       # Calculate accuracy

    return validation_loss/len(validation_loader), accuracy/len(validation_loader) 


#---------------------------------------------------  TEST THE TRAINED MODEL ---------------------------------------------------
def test(model, criterion, test_loader):
    model.eval()                                                                # Put model in validation mode
    accuracy = 0
    test_loss = 0                                                               # Test with a total of 819 images
    print(f"\n\nTESTING")
    start = timer()
    with torch.no_grad():                                                       # Disable gradients for speed
        for images, labels in iter(test_loader):                                # In every iteration, we train with a batch of 64 images
            images = images.to(device)                                          # Move images to the default device
            labels = labels.to(device)                                          # Move labels to the default device
            
            output = model.forward(images)                                      # Forward 
            test_loss += criterion(output, labels).item()                       # Calculate error
            ps = torch.exp(output)

            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()       # Calculate accuracy

    end = timer()
    print(f"  Test loss: {test_loss/len(test_loader):.3f}  Test accuracy: {accuracy/len(test_loader):.3f}  Time: {end-start:.2f} sec ")
 
#---------------------------------------------------  SAVE CHECKPOINT ---------------------------------------------------
def save_checkpoint(model, image_datasets, filename):
    print("Save checkpoint")
    model.class_to_idx = image_datasets['training'].class_to_idx
    checkpoint = {'model' : 'VGG19',
                  'input_size': 25088,
                  'output_size': 102,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filename)
        
#--------------------------------------------------- MAIN ---------------------------------------------------

# - Set directory to save checkpoints:  python train.py data_dir --save_dir save_directory
# - Choose architecture:                python train.py data_dir --arch "vgg13"
# - Set hyperparameters:                python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# - Use GPU for training:               python train.py data_dir --gpu

my_parser = argparse.ArgumentParser(prog='Train', description='Train a new network on a dataset and save the model as a checkpoint. Prints out training loss, validation loss, and validation accuracy as the network trains.')

my_parser.add_argument('Path', metavar='data_directory', type=str, help='Directory with input files')
my_parser.add_argument('--save_dir', action='store', type=str, default='.', help='Set directory to save checkpoints')
my_parser.add_argument('--arch', action='store', type=str, default='VGG19', help='Choose architecture')
my_parser.add_argument('--learning_rate', action='store', type=float, default='0.001', help='Set learning rate')
my_parser.add_argument('--hidden_units', action='store', type=int, default='512', help='Set hidden units')
my_parser.add_argument('--epochs', action='store', type=int, default='3', help='Set epochs')
my_parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = my_parser.parse_args()               # Parse the parameters
#print(vars(args))                          # Show the parsed parameters

input_path = args.Path
input_savedir = args.save_dir
input_arch = args.arch
input_learning_rate = args.learning_rate
input_hidden_units = args.hidden_units
input_epochs = args.epochs
input_gpu = args.gpu

print(f"Input_path: {input_path}")
print(f"input_savedir: {input_savedir}")
print(f"input_arch: {input_arch}")
print(f"input_learning_rate: {input_learning_rate}")
print(f"input_hidden_units: {input_hidden_units}")
print(f"input_epochs: {input_epochs}")
print(f"input_gpu: {input_gpu}\n")

if not os.path.isdir(input_path):
    print('The specified input path oes not exist')
    sys.exit()

if torch.cuda.is_available():                                           # Use GPU if it's available (100x faster)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

batch_size = 64
# dataloaders, image_datasets = load_images()
# model = load_vgg19_pretrained_model()

# epochs = 1
# learning_rate = 0.001
# criterion = nn.NLLLoss()
# optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
#train(model, epochs, learning_rate, criterion, optimizer, dataloaders['training'], dataloaders['validate'])     
#test(model, criterion, dataloaders['testing'])

#filename = 'flowers.pth'
#save_checkpoint(model, image_datasets, filename)

print('THE END')
         
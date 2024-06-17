import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os
from collections import OrderedDict
from datetime import datetime
from PIL import Image
import torch.nn as nn
import torch
import torchvision.models as models
device=torch.device('cuda:0')

# EfficientNet (We compared four other classic convolutional neural networks and finally chose EfficientNet)
from efficientnet_pytorch import EfficientNet
model_name = 'efficientnet-b0'
pretrained = 'imagenet'
efficientnet = EfficientNet.from_pretrained(model_name)
in_features = efficientnet._fc.out_features
custom_fc = nn.Linear(in_features, 8)
efficientnet._fc = nn.Sequential(efficientnet._fc, custom_fc)
efficientnet=efficientnet.to(device)

# # alexnet
# alexnet = models.alexnet(pretrained=True)
# in_features = alexnet.classifier[6].out_features
# new_fc_layer = nn.Linear(in_features, 8)
# alexnet.classifier.add_module("new_fc_layer", new_fc_layer)
# alexnet=alexnet.to(device)

# # resnet18
# resnet18 = models.resnet18(pretrained=True)
# original_fc_layer = resnet18.fc
# in_features = original_fc_layer.out_features
# new_fc_layer = nn.Linear(in_features, 8)
# resnet18.fc = nn.Sequential(original_fc_layer, new_fc_layer)
# resnet18=resnet18.to(device)

# # vgg16
# vgg16= models.vgg16(pretrained=True)
# in_features_1 = vgg16.classifier[6].out_features
# new_fc_layer = nn.Linear(in_features_1, 8)
# vgg16.classifier.add_module("new_fc_layer", new_fc_layer)
# vgg16=vgg16.to(device)

# # googlenet
# googlenet = models.googlenet(pretrained=True)
# original_fc_layer = googlenet.fc
# in_features = original_fc_layer.out_features
# new_fc_layer = nn.Linear(in_features, 8)
# googlenet.fc = nn.Sequential(original_fc_layer, new_fc_layer)
# googlenet=googlenet.to(device)

transformation = {
    'train': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    'test':transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Resize(256)
     ])
}

from torchvision.datasets import ImageFolder
main_path="./train set path"
images_datasets = {x: ImageFolder(os.path.join(main_path, x),transform=transformation[x]) for x in ['train', 'val', 'test']}
dataloaders = {x: DataLoader(images_datasets[x], batch_size=128, shuffle=True, num_workers=2) for x in ['train', 'val', 'test']}
train_loader=dataloaders['train']
val_loader=dataloaders['val']
test_loader=dataloaders['test']


def vaildation(model, criterion, val_loader):
    val_loss = 0
    accuracy = 0

    for images, labels in iter(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)
        equality = labels.data == probabilities.max(dim=1)[1]
        accuracy += equality.type(torch.FloatTensor).mean()

    return val_loss, accuracy

criterion=nn.CrossEntropyLoss()
criterion=criterion.to(device)

efficientnet_params= [p for p in efficientnet.parameters() if p.requires_grad]
efficientnet_optimizer=optim.Adam(efficientnet_params,lr=0.0001)

# googlenet_params= [p for p in googlenet.parameters() if p.requires_grad]
# googlenet_optimizer=optim.Adam(googlenet_params,lr=0.0001)
#
# vgg16_params= [p for p in vgg16.parameters() if p.requires_grad]
# vgg16_optimizer=optim.Adam(vgg16_params,lr=0.0001)
#
# resnet18_params= [p for p in resnet18.parameters() if p.requires_grad]
# resnet18_optimizer=optim.Adam(resnet18_params,lr=0.0001)
#
# alexnet_params= [p for p in alexnet.parameters() if p.requires_grad]
# alexnet_optimizer=optim.Adam(alexnet_params,lr=0.0001)

def train_model(model, optimizer, criterion, train_loader, val_loader, epochs):
    plot_training = []
    plot_validation = []
    plot_accuracy = []
    plot_alltrain = []
    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            l2_reg = torch.tensor(0., requires_grad=True).to(device)
            with torch.no_grad():
                for param in model.parameters():
                    l2_reg += torch.norm(param, p=2).to(device)
            loss += l2_reg * 0.01
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        with torch.no_grad():
            validation_loss, accuracy = vaildation(model, criterion, val_loader)
        print(
            "Epoch: {}/{}..".format(e + 1, epochs),
            "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
            "All Train Loss: {:.3f}..".format(running_loss),
            "Val Loss: {:.3f}..".format(validation_loss / len(val_loader)),
            "Val Accuarcy:{:.3f}..".format(accuracy / len(val_loader))
        )
        plot_training.append(running_loss / len(train_loader))
        plot_validation.append(validation_loss / len(val_loader))
        plot_accuracy.append(accuracy / len(val_loader))
        plot_alltrain.append(running_loss)

    return plot_accuracy, plot_validation

epochs=30
efficientnet_accuracy,efficientnet_val=train_model(efficientnet,efficientnet_optimizer,criterion,train_loader,val_loader,epochs)

# epochs=30
# googlenet_accuracy,googlenet_val=train_model(googlenet,googlenet_optimizer,criterion,train_loader,val_loader,epochs)

# epochs=30
# resnet18_accuracy,resnet18_val=train_model(resnet18,resnet18_optimizer,criterion,train_loader,val_loader,epochs)

# epochs=30
# vgg16_accuracy,vgg16_val=train_model(vgg16,vgg16_optimizer,criterion,train_loader,val_loader,epochs)

# epochs=30
# alexnet_accuracy,alexnet_val=train_model(alexnet,alexnet_optimizer,criterion,train_loader,val_loader,epochs)

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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
device=torch.device('cuda:0')

#model
train_model=torch.load('./Parameter weight path/mymodel_efficientnet.pth')   #Training weight storage address
#train_model=torch.load('./Parameter weight path/mymodel_alexnet.pth')
#train_model=torch.load('./Parameter weight path/mymodel_googlenet.pth')
#train_model=torch.load('./Parameter weight path/mymodel_resnet18.pth')
#train_model=torch.load('./Parameter weight path/mymodel_vgg16.pth')
train_model.eval()
train_model=train_model.to(device)

test_transform = transforms.Compose([transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomRotation(15),
                                     transforms.RandomVerticalFlip(0.5)

                                     ])
from torchvision.datasets import ImageFolder
main_path = "./test set path"
images_datasets = ImageFolder(main_path, transform=test_transform)
dataloaders = DataLoader(images_datasets, batch_size=8, shuffle=True, num_workers=2)

# predictions = []                               #(The following is the code for displaying the confusion matrix. 
# label=[]                                       #If you want to use it, please comment out the ROC curve code first. 
# for images,labels in iter(dataloaders):        #The same is true for the code that displays the accuracy and validation set loss.)
#     images=images.to(device)                   
#     labels=labels.to(device)
#     output=train_model.forward(images)
#     probabilities=torch.exp(output)
#     predicted_class=probabilities.max(dim=1)[1]
#     predictions.append(predicted_class)
#     label.append(labels)

# predictions_cpu = [tensor.cpu() for tensor in predictions]
# label_cpu = [tensor.cpu() for tensor in label]

# list_predictions = [tensor.tolist() for tensor in predictions_cpu]
# list_label = [tensor.tolist() for tensor in label_cpu]

# Predictions = [item for sublist in list_predictions for item in sublist]
# Labels = [item for sublist in list_label for item in sublist]

# classes=['a','b','c','d','e','f','g','h']


# # Confusion Matrix
# conf_mat = confusion_matrix(Labels,Predictions)


# sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.savefig('./The path to save the image',dpi=1000,bbox_inches = 'tight')
# plt.show()

# print(classification_report(Labels, Predictions, target_names=classes))


# roc curve

predictions = []
label=[]
probs=[]
for images,labels in iter(dataloaders):
    images=images.to(device)
    labels=labels.to(device)
    output=train_model.forward(images)
    prob = torch.nn.functional.softmax(output, dim=1)
    label.extend(labels.cpu().detach().numpy())
    probs.extend(prob.cpu().detach().numpy())

all_labels0 = [1 if num == 0 else 0 for num in label]
all_labels1 = [1 if num == 1 else 0 for num in label]
all_labels2 = [1 if num == 2 else 0 for num in label]
all_labels3 = [1 if num == 3 else 0 for num in label]
all_labels4 = [1 if num == 4 else 0 for num in label]
all_labels5 = [1 if num == 5 else 0 for num in label]
all_labels6 = [1 if num == 6 else 0 for num in label]
all_labels7 = [1 if num == 7 else 0 for num in label]

numpy_array_2d = np.vstack(probs)

fpr=dict()
tpr=dict()
roc_auc = dict()
fpr[0], tpr[0], _ = roc_curve(all_labels0, probs_test0, pos_label=1)
fpr[1], tpr[1], _ = roc_curve(all_labels1, probs_test1, pos_label=1)
fpr[2], tpr[2], _ = roc_curve(all_labels2, probs_test2, pos_label=1)
fpr[3], tpr[3], _ = roc_curve(all_labels3, probs_test3, pos_label=1)
fpr[4], tpr[4], _ = roc_curve(all_labels4, probs_test4, pos_label=1)
fpr[5], tpr[5], _ = roc_curve(all_labels5, probs_test5, pos_label=1)
fpr[6], tpr[6], _ = roc_curve(all_labels6, probs_test6, pos_label=1)
fpr[7], tpr[7], _ = roc_curve(all_labels7, probs_test7, pos_label=1)

roc_auc0 = auc(fpr[0], tpr[0])
roc_auc1 = auc(fpr[1], tpr[1])
roc_auc2 = auc(fpr[2], tpr[2])
roc_auc3 = auc(fpr[3], tpr[3])
roc_auc4 = auc(fpr[4], tpr[4])
roc_auc5 = auc(fpr[5], tpr[5])
roc_auc6 = auc(fpr[6], tpr[6])
roc_auc7 = auc(fpr[7], tpr[7])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(8)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(8):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= 8
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
tpr["macro"][0]=0

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

efpr=fpr["macro"]
etpr=tpr["macro"]
eroc_auc=roc_auc["macro"]

gfpr=fpr["macro"]
gtpr=tpr["macro"]
groc_auc=roc_auc["macro"]

afpr=fpr["macro"]
atpr=tpr["macro"]
aroc_auc=roc_auc["macro"]

vfpr=fpr["macro"]
vtpr=tpr["macro"]
vroc_auc=roc_auc["macro"]

rfpr=fpr["macro"]
rtpr=tpr["macro"]
rroc_auc=roc_auc["macro"]

plt.figure(figsize=(8, 6))
plt.plot(efpr, etpr, label=f'EfficientNet(AUC = {eroc_auc:.2f})')
plt.plot(gfpr, gtpr, label=f'GooleNet(AUC = {groc_auc:.2f})')
plt.plot(afpr, atpr, label=f'AlexNet(AUC = {aroc_auc:.2f})')
plt.plot(vfpr, vtpr, label=f'VGG16(AUC = {vroc_auc:.2f})')
plt.plot(rfpr, rtpr, label=f'Resnet18(AUC = {rroc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.plot([-0.05,0],[0,0],'k--',lw=1)
plt.plot([0,0],[-0.05,0],'k--',lw=1)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC curves among different models')
plt.legend(loc="lower right")
plt.savefig('./The path to save the image',dpi=1000,bbox_inches = 'tight')
plt.show()

#valloss and accuracy

# import matplotlib.pyplot as plt
# import pickle
# file_path1 = './googlenet'  #The storage address of each iteration accuracy and validation set loss
# file_path2 = './alexnet'
# file_path3 = './vgg16'
# file_path4 = './resnet18'
# file_path5 = './efficientnet'
# with open(file_path1, 'rb') as file:
#     loaded_data1 = pickle.load(file)
# with open(file_path2, 'rb') as file:
#     loaded_data2 = pickle.load(file)
# with open(file_path3, 'rb') as file:
#     loaded_data3 = pickle.load(file)
# with open(file_path4, 'rb') as file:
#     loaded_data4 = pickle.load(file)
# with open(file_path5, 'rb') as file:
#     loaded_data5 = pickle.load(file)
#
#
# plt.ylim(0.3, 1)
# plt.plot(range(len(loaded_data1['accuracy'])), loaded_data1['accuracy'], label='GoogleNet Accuracy',color='blue')
# plt.plot(range(len(loaded_data2['accuracy'])), loaded_data2['accuracy'], label='AlexNet Accuracy',color='green')
# plt.plot(range(len(loaded_data3['accuracy'])), loaded_data3['accuracy'], label='VGG16 Accuracy',color='red')
# plt.plot(range(len(loaded_data4['accuracy'])), loaded_data4['accuracy'], label='ResNet18 Accuracy',color='yellow')
# plt.plot(range(len(loaded_data5['accuracy'])), loaded_data5['accuracy'], label='Efficient Accuracy',color='black')
# plt.legend(loc='lower right')
# plt.show()
#
# plt.ylim(0, 2)
# plt.plot(range(len(loaded_data1['val'])), loaded_data1['val'], label='GoogleNet ValLoss',color='blue')
# plt.plot(range(len(loaded_data2['val'])), loaded_data2['val'], label='AlexNet ValLoss',color='green')
# plt.plot(range(len(loaded_data3['val'])), loaded_data3['val'], label='VGG16 ValLoss',color='red')
# plt.plot(range(len(loaded_data4['val'])), loaded_data4['val'], label='ResNet18 ValLoss',color='yellow')
# plt.plot(range(len(loaded_data5['val'])), loaded_data5['val'], label='Efficient ValLoss',color='black')
# plt.legend(loc='upper right')
# plt.show()

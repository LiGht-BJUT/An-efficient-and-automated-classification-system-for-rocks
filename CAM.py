from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM,ScoreCAM,AblationCAM,EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch
import torchvision.models as models
import numpy as np
import os
device=torch.device('cuda:0')

train_model=torch.load('Model Path')
train_model.eval()
train_model=train_model.to(device)

test_transform = transforms.Compose([transforms.CenterCrop(224),
                                     transforms.ToTensor()
                                     ])
img_path = 'Test Set Path'

img_pil = Image.open(img_path)
input_tensor = test_transform(img_pil).unsqueeze(0).to(device) 
target_layers=train_model._blocks[15]
layers=[]
layers=[target_layers._expand_conv,target_layers._bn0,target_layers._depthwise_conv,
        target_layers._bn1,target_layers._se_reduce,target_layers._se_expand,
        target_layers._project_conv,target_layers._bn2,target_layers._swish]
layers_sample=[]
layers_sample=[target_layers._depthwise_conv,target_layers._project_conv]

def tensor2img(tensor,heatmap=False,shape=(224,224)):
    np_arr=tensor.detach().numpy()
    if np_arr.max()>1 or np_arr.min()<0:
        np_arr=np_arr-np_arr.min()
        np_arr=np_arr/np_arr.max()
    if np_arr.shape[0]==1:
        np_arr=np.concatenate([np_arr,np_arr,np_arr],axis=0)
    np_arr=np_arr.transpose((1,2,0))
    return np_arr


def myimshows(imgs, titls=False, fname="test.png", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens, size))

    titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

with GradCAM(model=train_model,target_layers=layers) as grcam:
    for i in range(8):
        target=[ClassifierOutputTarget(i)]
        grayscale_cam=grcam(input_tensor=input_tensor,targets=target)
        rgb_img=tensor2img(input_tensor.cpu().squeeze())
        visualization=show_cam_on_image(rgb_img,grayscale_cam[0],use_rgb=True)
        myimshows([rgb_img,grayscale_cam[0],visualization],["image","cam","image+cam"])

with ScoreCAM(model=train_model,target_layers=layers) as sccam:
    for i in range(8):
        target=[ClassifierOutputTarget(i)]
        grayscale_cam=sccam(input_tensor=input_tensor,targets=target)
        rgb_img=tensor2img(input_tensor.cpu().squeeze())
        visualization=show_cam_on_image(rgb_img,grayscale_cam[0],use_rgb=True)
        myimshows([rgb_img,grayscale_cam[0],visualization],["image","cam","image+cam"])

with AblationCAM(model=train_model,target_layers=layers) as abcam:
    for i in range(8):
        target=[ClassifierOutputTarget(i)]
        grayscale_cam=abcam(input_tensor=input_tensor,targets=target)
        rgb_img=tensor2img(input_tensor.cpu().squeeze())
        visualization=show_cam_on_image(rgb_img,grayscale_cam[0],use_rgb=True)
        myimshows([rgb_img,grayscale_cam[0],visualization],["image","cam","image+cam"])

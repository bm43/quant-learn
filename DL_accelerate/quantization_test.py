MODEL_PATH = "C:/Users/USER-PC/OneDrive/문서/projects/kaggle/DL_Projects/weights/"

# plan:
# input model file

import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization
import monai
import matplotlib.pyplot as plt

# # Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

from torch.quantization import QuantStub, DeQuantStub

def load_model(model_file):
    #device = torch.device('cuda')
    model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

model = load_model(MODEL_PATH+'3d-resnet10_T1wCE_fold3_0.664.pth')


# see its weights
def show_weights(model_file):
    state_dict = torch.load(model_file, map_location='cpu')
    return state_dict

#print(show_weights(MODEL_PATH+'3d-resnet10_T1wCE_fold3_0.664.pth').keys())
print(show_weights(MODEL_PATH+'3d-resnet10_T1wCE_fold3_0.664.pth')['conv1.weight'].size())
# layer마다 key가 있고 그에 해당하는 weight vector가 value로 저장됨.

'''
m = nn.Conv1d(16, 33, 3, stride=2)
input = torch.randn(20, 16, 50)
print(input.size(),'\n')
output = m(input)
print(output.size())
# see how long it takes to process one image
'''

# model eval mode:
import pydicom as dicom
# https://pydicom.github.io/pydicom/stable/old/working_with_pixel_data.html

data_path = "C:/Users/USER-PC/OneDrive/문서/projects/quant-learn/DL_accelerate/sample_images/Image-16.dcm"
def check_input(model, data_path):
    img = dicom.dcmread(data_path)
    
    with torch.no_grad():
        model.eval()
        input = torch.from_numpy(img.pixel_array).reshape(1,1,1,512,512)
        input = input.type(torch.FloatTensor)
        output = model(input)
        return output
    
s = time.time()
print(check_input(model, data_path))
e = time.time()

print(s-e)

# quantize model
# see if it's faster processing
# find papers about quantization to see if it can make model any faster

# find dataset so that you can quantize while training
# or just modify training code so that it looks like quantizing while training.

print("\nnp")

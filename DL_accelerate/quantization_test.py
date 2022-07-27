# testing out DL model quantization with pytorch
# author: Hyung Jip Lee

MODEL_PATH = "C:/Users/USER-PC/OneDrive/문서/projects/kaggle/DL_Projects/weights/"

# plan:
# input model file

from cgi import print_arguments
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
print(show_weights(MODEL_PATH+'3d-resnet10_T1wCE_fold3_0.664.pth'))
# layer마다 key가 있고 그에 해당하는 weight vector가 value로 저장됨.

'''
m = nn.Conv1d(16, 33, 3, stride=2)
input = torch.randn(20, 16, 50)
print(input.size(),'\n')
output = m(input)
print(output.size())
# see how long it takes to process one image
'''

# add linreg with mle, quantization function (ideally implement a paper), add ppr to machine learning/neural network

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
output = check_input(model, data_path)
e = time.time()

print("output: ", output)
print("single input time: ", e-s)

# quantize model
# see if it's faster processing and less memory:

def print_model_size(model):
    torch.save(model.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

print("model before post training static quantization: ", print_model_size(model), '\n')

# post training static quantization memory:
backend = "qnnpack"
model.qconfig = torch.quantization.get_default_qconfig(backend)
quantized_model = torch.quantization.prepare(model, inplace = False)
quantized_model = torch.quantization.convert(quantized_model, inplace=False)
# convert weights to 8bit integers

print("quantized model size: ", print_model_size(quantized_model), '\n')

def check_q_input(model, data_path):
    img = dicom.dcmread(data_path)
    
    with torch.no_grad():
        model.eval()
        input = torch.from_numpy(img.pixel_array).reshape(1,1,1,512,512)
        cuda0 = torch.device('cuda:0')
        input.to(cuda0)
        output = model(input)
        return output


# measure time:
s= time.time()
q_output = check_q_input(quantized_model, data_path)
e = time.time()
print("q model took: ", e-s)

# all above code was written in pytorch 1.12.0

'''
# save the quantized model:
torch.save(quantized_model.state_dict(), "C:/Users/USER-PC/OneDrive/문서/projects/kaggle/DL_Projects/weights/0664_quantized.pth")

# load q model:
quantized_model = load_model("C:/Users/USER-PC/OneDrive/문서/projects/kaggle/DL_Projects/weights/0664_quantized.pth")
# show q model weigths:
print("model weights after quantization: ", show_weights(quantized_model))
'''


# find papers about quantization to see if it can make model any faster

# find dataset so that you can quantize while training
# or just modify training code so that it looks like quantizing while training.

print("\nnp")

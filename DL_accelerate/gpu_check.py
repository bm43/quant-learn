import torch
import monai
import sys

MODEL_PATH = "C:/Users/USER-PC/OneDrive/문서/projects/kaggle/DL_Projects/weights/"

print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device())

sys.exit()

def load_model(model_file):
    #device = torch.device('cuda')
    model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

model = load_model(MODEL_PATH+'3d-resnet10_T1wCE_fold3_0.664.pth')

# install cuda compatible torch:
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

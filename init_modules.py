
from basicsr.losses import build_loss
import pyiqa
import os

# Download the weights

# Set Only CPU can be used.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

flag = 1

try:
    build_loss({'type': 'PerceptualLoss',
                'layer_weights': {
                    'relu1_1': 3.125e-2,
                      'relu2_1': 6.25e-2,
                      'relu3_1': 0.125,
                      'relu4_1': 0.25,
                      'relu5_1': 1.0},
                'vgg_type': 'vgg19'}).cpu()
    print("Download vgg19 successfully.")
except:
    print("Download vgg19 failed.")
    flag = 0
try:
    pyiqa.create_metric('niqe', device='cpu')
    print("Download niqe successfully.")
except:
    print("Download niqe failed.")
    flag = 0
try:
    pyiqa.create_metric('musiq',device='cpu')
    print("Download musiq successfully.")
except:
    print("Download musiq failed.")
    flag = 0
try:
    pyiqa.create_metric('brisque',device='cpu')
    print("Download brisque successfully.")
except:
    print("Download brisque failed.")
    flag = 0
try:
    pyiqa.create_metric('lpips',device='cpu')
    print("Download lpips successfully.")
except:
    print("Download lpips failed.")
    flag = 0
try:
    pyiqa.create_metric('nima',device='cpu')
    print("Download nima successfully.")
except:
    print("Download nima failed.")
    flag = 0
try:
    pyiqa.create_metric('nima', base_model_name='vgg16', train_dataset='ava', num_classes=10, device='cpu')
    print("Download nima-vgg16-ava successfully.")
except:
    print("Download nima-vgg16-ava failed.")
    flag = 0

if flag == 1:
    print("Init modules successfully!")
else:
    print("Init modules failed, please check your network connection and try again.")

from basicsr.losses import build_loss
import pyiqa
import os

# Download the weights

# Set Only CPU can be used.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


build_loss({'type': 'PerceptualLoss',
            'layer_weights': {
                'relu1_1': 3.125e-2,
                  'relu2_1': 6.25e-2,
                  'relu3_1': 0.125,
                  'relu4_1': 0.25,
                  'relu5_1': 1.0},
            'vgg_type': 'vgg19'}).cpu()
pyiqa.create_metric('niqe', device='cpu')
pyiqa.create_metric('musiq',device='cpu')
pyiqa.create_metric('brisque',device='cpu')
pyiqa.create_metric('lpips',device='cpu')
pyiqa.create_metric('nima',device='cpu')
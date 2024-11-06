import cv2
import pyiqa
import os
import PIL.Image as Image
import torch
from torchvision import transforms
import argparse
from tqdm import tqdm


os.environ['HF_HUB_OFFLINE'] = True

parser = argparse.ArgumentParser(description='Evaluate Image Quality')

parser.add_argument('--input_dir', default='', type=str, help='Directory of validation images')

args = parser.parse_args()

brisque = pyiqa.create_metric('brisque')
nima = pyiqa.create_metric('nima')

# Load images
dir_0 = args.input_dir

files = os.listdir(dir_0)
#
sum_nima = 0
sum_brisque = 0
count = 0

for file in tqdm(files):
    if(os.path.exists(os.path.join(dir_0,file))):
        # Load images
        if file.endswith('Store') or file.endswith('.txt'):
            continue
        image = os.path.join(dir_0, file)

        dist_brisque = brisque(image)
        dist_nima = nima(image)

        sum_brisque += dist_brisque
        sum_nima += dist_nima
        count += 1


print(dir_0)
print('Average BRISQUE: %.4f'%(sum_brisque/count))
print('Average NIMA: %.4f'%(sum_nima/count))



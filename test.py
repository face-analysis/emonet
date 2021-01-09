import numpy as np
from pathlib import Path
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms 

from emonet.models import EmoNet
from emonet.data import AffectNet
from emonet.data_augmentation import DataAugmentor
from emonet.metrics import CCC, PCC, RMSE, SAGR, ACC
from emonet.evaluation import evaluate, evaluate_flip

torch.backends.cudnn.benchmark =  True

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nclasses', type=int, default=8, choices=[5,8], help='Number of emotional classes to test the model on. Please use 5 or 8.')
args = parser.parse_args()

# Parameters of the experiments
n_expression = args.nclasses
batch_size = 32
n_workers = 16
device = 'cuda:0'
image_size = 256
subset = 'test'
metrics_valence_arousal = {'CCC':CCC, 'PCC':PCC, 'RMSE':RMSE, 'SAGR':SAGR}
metrics_expression = {'ACC':ACC}

# Create the data loaders
transform_image = transforms.Compose([transforms.ToTensor()])
transform_image_shape_no_flip = DataAugmentor(image_size, image_size)

flipping_indices = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22,21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, 45,44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51,50, 49, 48, 59, 58,57, 56, 55, 64, 63,62, 61, 60, 67, 66,65]
transform_image_shape_flip = DataAugmentor(image_size, image_size, mirror=True, shape_mirror_indx=flipping_indices, flipping_probability=1.0)

print(f'Testing the model on {n_expression} emotional classes')

print('Loading the data')
test_dataset_no_flip = AffectNet(root_path='~/datasets/new_affectnet/', subset=subset, n_expression=n_expression,
                         transform_image_shape=transform_image_shape_no_flip, transform_image=transform_image)

test_dataset_flip = AffectNet(root_path='~/datasets/new_affectnet/', subset=subset, n_expression=n_expression,
                         transform_image_shape=transform_image_shape_flip, transform_image=transform_image)

test_dataloader_no_flip = DataLoader(test_dataset_no_flip, batch_size=batch_size, shuffle=False, num_workers=n_workers)
test_dataloader_flip = DataLoader(test_dataset_flip, batch_size=batch_size, shuffle=False, num_workers=n_workers)

# Loading the model 
state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')

print(f'Loading the model from {state_dict_path}.')
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
net = EmoNet(n_expression=n_expression).to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()


print(f'Testing on {subset}-set')
print(f'------------------------')
evaluate_flip(net, test_dataloader_no_flip, test_dataloader_flip, device=device, metrics_valence_arousal=metrics_valence_arousal, metrics_expression=metrics_expression)
#evaluate(net, test_dataloader_no_flip, device=device, metrics_valence_arousal=metrics_valence_arousal, metrics_expression=metrics_expression)

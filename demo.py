from pathlib import Path
import argparse

import torch
from torch import nn
from skimage import io

from emonet.models import EmoNet

import cv2

torch.backends.cudnn.benchmark =  True

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nclasses', type=int, default=8, choices=[5,8], help='Number of emotional classes to test the model on. Please use 5 or 8.')
parser.add_argument('--image_path', type=str, default="images/example.png", help='Path to a face image.')
args = parser.parse_args()

# Parameters of the experiments
n_expression = args.nclasses
device = 'cuda:0'
image_size = 256
emotion_classes = {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger", 7:"Contempt"}
image_path = Path(__file__).parent / args.image_path

# Loading the model 
state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')

print(f'Loading the model from {state_dict_path}.')
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
net = EmoNet(n_expression=n_expression).to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()

print(f'Testing on a single image')
print(f'------------------------')
# Load image in RGB format
image_rgb = io.imread(image_path)[:,:,:3]

# Resize image to (256,256)
image_rgb = cv2.resize(image_rgb, (image_size, image_size))

# Load image into a tensor: convert to RGB, and put the tensor in the [0;1] range
image_tensor = torch.Tensor(image_rgb).permute(2,0,1).to(device)/255.0

with torch.no_grad():
    output = net(image_tensor.unsqueeze(0))
    predicted_emotion_class = torch.argmax(nn.functional.softmax(output["expression"], dim=1)).cpu().item()

    # Expected output on example image: Predicted Emotion Happy - valence 0.064 - arousal 0.143
    print(f"Predicted Emotion {emotion_classes[predicted_emotion_class]} - valence {output['valence'].cpu().item():.3f} - arousal {output['arousal'].cpu().item():.3f}")
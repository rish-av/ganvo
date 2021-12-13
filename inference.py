import torch
from losses import *
from utils import *
from network import *
import argparse
import yaml
from dataset import dataset
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--img_path',help='path to the image')
parser.add_argument('--checkpoint',help='path to the generator weights')
parser.add_argument('--h',help='image height')
parser.add_argument('--w',help='image width')


args = parser.parse_args()


transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize(std=(0.5,),mean=(0.5,))
                        ])



net.load_ckpts(args.pretrained_epoch)
img = cv2.imread(args.img_path)
img = transform(cv2.resize(img0, (args,w, arg.h))).unsqueeze(0)
datum = {"t0":img,"t1":img,"t2":img}
net.set_input(datum)
net.forward()
depth = net.get_visuals()[0]
cv2.imwrite('depth.png',depth)
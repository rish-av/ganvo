import torch
from losses import *
from utils import *
from baseutils import *
from networks import *
import argparse
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--config',help='path to the config file',default='./config.yaml')

args = parser.parse_args()
with open(args.config) as fp:
	config = yaml.load(fp)
	config = AttrDict(config)


net = ganvo(config)
a = torch.randn(1,3,128,416)
net.set_input(a,a,a)
net.forward()
net.optimize_paramaters()





import torch
from losses import *
from utils import *

import argsparse
import yaml


parser = argsparse.ArgumentParser()
parser.add_argument('--config',help='path to the config file')

args = parser.parse_args()
with open(args.config) as fp:
	config = yaml.load(fp)
	config = AttrDict(config)

def save_weights():



def load_weights():


def train():


if __name__='main':
	train()



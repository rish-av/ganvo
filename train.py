import torch
from losses import *
from utils import *
import baseutils
from networks import *
import argparse
import yaml
from dataset import dataset


parser = argparse.ArgumentParser()
parser.add_argument('--config',help='path to the config file',default='./config.yaml')

args = parser.parse_args()
with open(args.config) as fp:
	config = yaml.load(fp)
	config = baseutils.AttrDict(config)


total_data = dataset(config)
train_sampler, val_sampler = baseutils._split(total_data, 0.2)


train_dataset = torch.utils.data.DataLoader(total_data,batch_size = config.batch_size, sampler = train_sampler)
val_dataset = torch.utils.data.DataLoader(total_data, 1, sampler = val_sampler)
net = ganvo(config).cuda()

for epc in range(config.epochs):

    for i, datum in enumerate(train_dataset):

        net.set_input(datum)
        net.forward()
        net.optimize_paramaters()





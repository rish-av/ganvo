import torch
from losses import *
from utils import *
from networks import *
import argparse
import yaml
from dataset import dataset
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--config',help='path to the config file',default='./config.yaml')

args = parser.parse_args()
with open(args.config) as fp:
	config = yaml.load(fp)
	config = AttrDict(config)


train_dataset = dataset(config,mode='train')
val_dataset = dataset(config, mode='val')

train_dataset = torch.utils.data.DataLoader(train_dataset,batch_size = config.batch_size)
val_dataset = torch.utils.data.DataLoader(val_dataset, 1)
summarywriter = get_summary_writer(rootdir=config.summary_root)
net = ganvo(config, summarywriter=summarywriter).cuda()

net.load_ckpts(config.pretrained_epoch)

for epc in range(config.epochs):

    for i, datum in enumerate(train_dataset):

        net.set_input(datum)
        net.forward()
        net.optimize_paramaters()
        step = epc*len(train_dataset) + i
        net.log_metrics(step)


    net.save_weights(epc)
    for j, datum in enumerate(val_dataset):

        with torch.no_grad():
            net.set_input(datum)
            net.forward()

            step = epc*len(val_dataset) + j
            net.log_metrics(step,'val')






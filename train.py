import torch
from losses import *
from utils import *
from baseutils import *
from networks import *
import argparse
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--config',help='path to the config file')

args = parser.parse_args()
with open(args.config) as fp:
	config = yaml.load(fp)
	config = AttrDict(config)


net = ganvo(config)
l2_loss = torch.nn.MSELoss()
gan_loss = GANLoss(gan_mode='vanilla')

def save_weights():
	pass



def load_weights():
	pass


def train():
	for epoch in range(config.epochs):
		for i,(img1, img2, img3) in enumerate(kittiloader):
			pose, depth = net(img1,img2,img3)

			i_recons_1 = inverse_warp(img1,depth,pose[0],config.intrinsics)
			i_recons_2 = inverse_warp(img3,depth,pose[1],config.intrinsics)


			recons_loss = l2_loss(img1,i_recons_1) + l2_loss(img1,i_recons_2)
			dfake_1_loss = gan_loss(i_recons_1,False)
			dfake_2_loss = gan_loss(i_recons_2,False)
			dreal = gan_loss(img1,True)

			d_net = dfake_1_loss+dfake_2_loss+dreal

			total_loss = recons_loss + d_net




if __name__=='main':
	train()



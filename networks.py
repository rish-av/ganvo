import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.nn import BatchNorm2d
from losses import GANLoss
import itertools

from utils import inverse_warp, pose_vec2mat
import cv2
import numpy as np
import os


def conv(in_c,out_c,kernel=3,norm=BatchNorm2d,activation=nn.ELU, stride=1, padding=0):
	layers = [ nn.Conv2d(in_c,out_c,kernel_size=kernel,stride=stride,bias=False,padding=padding),
			   activation(),
			   norm(out_c),
			   nn.Conv2d(out_c,out_c,kernel_size=kernel,bias=False, padding=1),
			   norm(out_c),
			   activation(),
			   ]
	return nn.Sequential(*layers)

def upconv(in_c,out_c,kernel_size=4,stride=(2,2),norm=BatchNorm2d,activation=nn.LeakyReLU):

	layers = [ nn.ConvTranspose2d(in_c,out_c,kernel_size=kernel_size,stride=stride,bias=False,padding=1),
	           norm(out_c),
		   activation()]
	return nn.Sequential(*layers)

class generator(nn.Module):
	def __init__(self,config):
		super(generator,self).__init__()
		channels = config.generator_channels
		layers = [upconv(in_c=channels[i-1],out_c=channels[i]) for i in range(1,len(channels)-1)]
		layers.append(upconv(in_c=channels[-2],out_c=channels[-1],activation=nn.Tanh))
		self.gen = nn.Sequential(*layers)

	def forward(self,x):
		return self.gen(x)


class discriminator(nn.Module):
	def __init__(self,config):
		super(discriminator,self).__init__()
		channels = config.discriminator_channels
		layers = [conv(in_c=channels[i-1],out_c=channels[i]) for i in range(1,len(channels))]
		self.disc = nn.Sequential(*layers)

	def forward(self,x):
		return self.disc(x)


class encoder(nn.Module):

	def __init__(self,config,cnn=False):
		super(encoder,self).__init__()
		if cnn == True:
			channels = config.cnn_channels
		else:
			channels = config.encoder_channels
		layers = [conv(in_c=channels[i-1],out_c=channels[i],stride=2,padding=1) for i in range(1,len(channels))]
		self.enc = nn.Sequential(*layers)

	def forward(self,x):
		return self.enc(x)


class recurrent_net(nn.Module):

	def __init__(self,config):
		super(recurrent_net,self).__init__()
		self.config = config
		lstm_in_size = self.get_in_size(config)
		self.recur = nn.LSTM(input_size=lstm_in_size,hidden_size=config.lstm_hidden_size,
			num_layers=config.num_layers_lstm,proj_size=6)
		self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

	def get_in_size(self,config):
		with torch.no_grad():
			rand = torch.rand(1,config.in_c*3,config.h,config.w)
			enc = encoder(config,cnn=True)
			out = enc(rand)
			out = out.shape
			return out[2]*out[3]

	def forward(self,x):
		h0 = torch.randn(self.config.num_layers_lstm,x.shape[0],6).to(self.device)
		c0 = torch.randn(self.config.num_layers_lstm,x.shape[0],self.config.lstm_hidden_size).to(self.device)
		x_in = x.reshape(x.shape[1],x.shape[0],-1)
		out, (hn,cn) = self.recur(x_in, (h0, c0))
		return hn


class ganvo(nn.Module):
	def __init__(self,config,summarywriter=None):
			super(ganvo,self).__init__()
			self.config = config
			self.encoder = encoder(config)
			self.cnn = encoder(config,cnn=True)
			self.generator = generator(config)
			self.rnn = recurrent_net(config)
			self.discriminator = discriminator(config)
			self.optimizer_G = torch.optim.Adam(itertools.chain(self.generator.parameters(),
			self.encoder.parameters(),self.cnn.parameters(), self.rnn.parameters()),lr=0.0002)
			self.optimizer_D =  torch.optim.Adam(self.discriminator.parameters(),lr=0.0002)

			self.mseloss = torch.nn.L1Loss()
			self.ganloss = GANLoss(gan_mode='vanilla')
			self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
			self.summarywriter = summarywriter
			self.model_names = ["generator","cnn","rnn","discriminator","encoder"]

	def set_input(self, datum):
			device = self.device
			self.source = datum["t1"].to(device)
			self.imgt_1 = datum["t0"].to(device)
			self.imgt_2 = datum["t2"].to(device)
			self.intrisics = torch.tensor(datum["intrinsics"]).to(device)


	def set_requires_grad(self, net,requires_grad):
		for param in net.parameters():
			param.requires_grad = requires_grad

	def forward(self):
		out_gan = self.encoder(self.source)
		depth = self.generator(out_gan)
		pose = self.cnn(torch.cat([self.imgt_1,self.source,self.imgt_2],dim=1))
		pose = self.rnn(pose)
		self.depth = depth
		self.pose = pose
		return depth, pose

	def optimize_G(self):
		const_from_t_1, valid_p_1 = inverse_warp(self.imgt_1,self.depth,self.pose[0,:],self.intrisics)
		const_from_t_2, valid_p_2 = inverse_warp(self.imgt_2,self.depth,self.pose[1,:],self.intrisics)
		loss_recons = self.mseloss(self.source*(valid_p_1.float().unsqueeze(1)),const_from_t_1*(valid_p_1.float().unsqueeze(1))) + \
			self.mseloss(self.source*(valid_p_2.float().unsqueeze(1)), const_from_t_2*(valid_p_2.float().unsqueeze(1)))

		self.const_from_t_1 = const_from_t_1
		self.const_from_t_2 = const_from_t_2
		

		#let's see how well can the discriminator fool
		loss_adv_g = self.ganloss(self.discriminator(const_from_t_1),True) + self.ganloss(self.discriminator(const_from_t_2),True)
		loss_net = loss_adv_g + loss_recons

		self.optimizer_G.zero_grad()
		loss_net.backward()
		self.optimizer_G.step()
		return loss_net

	def optimize_D(self):
		loss_real = self.ganloss(self.discriminator(self.source), True) 
		loss_fake = (self.ganloss(self.discriminator(self.const_from_t_2.detach()),False) + self.ganloss(self.discriminator(self.const_from_t_1.detach()),False))/2
		loss_D = (loss_real + loss_fake)/2

		self.optimizer_D.zero_grad()
		loss_D.backward()
		self.optimizer_D.step()
		return loss_D

	def optimize_paramaters(self):

		self.forward()
		self.set_requires_grad(self.discriminator,False)
		loss_G = self.optimize_G()
		self.set_requires_grad(self.discriminator,True)
		loss_D = self.optimize_D()

		self.loss_net = loss_D + loss_G


	def tensor2im(self, img):
		img = img.permute(1,2,0).detach().cpu().numpy()*0.5 + 0.5
		img = img*255

		return img

	def get_depth_image(self, depth):

		depth_mpl = torch.nn.functional.interpolate(depth.unsqueeze(1)*10, (375,1224))
		depth_mpl = depth_mpl[0][0].detach().cpu().numpy() 

		depth = depth[0].detach().cpu().numpy() 
		maxd = np.max(depth)
		depth = depth/maxd
		depth*=255

		depth = depth.astype(np.uint8)
		image = np.stack([depth,depth,depth],axis=2)
		image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

		return image, depth_mpl

	def log_metrics(self, step, process = 'train'):
		if self.summarywriter!= None:
			self.summarywriter.add_scalar(process + '/loss',self.loss_net,step)

	def load_ckpts(self,epoch):
		if self.config.pretrained_epoch:

			for name in self.model_names:
				path  = os.path.join(self.config.weights_dir,"%s_net_%s.pth"%(str(epoch),name))
				params = torch.load(path)
				net = getattr(self, name)
				net.load_state_dict(params)
	
	def save_weights(self, epoch):
		for name in self.model_names:
			if isinstance(name, str):
				save_filename = '%s_net_%s.pth' % (epoch, name)
				save_path = os.path.join(self.config.weights_dir, save_filename)
				net = getattr(self, name)
				torch.save(net.state_dict(), save_path)


	def get_visuals(self):
		depth, depth_mpl = self.get_depth_image(self.depth[0])
		img_t1 = self.tensor2im(self.imgt_1[0])
		img_t2 = self.tensor2im(self.imgt_2[0])
		source = self.tensor2im(self.source[0])

		const_1 = self.tensor2im(self.const_from_t_1[0])

		return depth, img_t1, img_t2, source, const_1, depth_mpl






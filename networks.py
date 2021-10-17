import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
from losses import GANLoss
import itertools

from utils import inverse_warp, pose_vec2mat

def conv(in_c,out_c,kernel=3,norm=BatchNorm2d,activation=nn.ELU):
	layers = [ nn.Conv2d(in_c,out_c,kernel_size=kernel,bias=False),
			   activation(),
			   norm(out_c),
			   nn.Conv2d(out_c,out_c,kernel_size=kernel,bias=False),
			   norm(out_c),
			   activation(),
			   ]
	return nn.Sequential(*layers)

def upconv(in_c,out_c,kernel_size=4,norm=BatchNorm2d):

	layers = [ nn.ConvTranspose2d(in_c,out_c,kernel_size=kernel_size,bias=False),
	           norm(out_c),
			   nn.ELU()]
	return nn.Sequential(*layers)

class generator(nn.Module):
	def __init__(self,config):
		super(generator,self).__init__()
		channels = config.generator_channels
		layers = [upconv(in_c=channels[i-1],out_c=channels[i]) for i in range(1,len(channels))]
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
		layers = [conv(in_c=channels[i-1],out_c=channels[i]) for i in range(1,len(channels))]
		self.enc = nn.Sequential(*layers)

	def forward(self,x):
		return self.enc(x)


class recurrent_net(nn.Module):

	def __init__(self,config):
		super(recurrent_net,self).__init__()
		self.config = config
		lstm_in_size = self.get_in_size(config)
		print(lstm_in_size)
		self.recur = nn.LSTM(input_size=lstm_in_size,hidden_size=config.lstm_hidden_size,
			num_layers=config.num_layers_lstm,proj_size=6)

	def get_in_size(self,config):
		with torch.no_grad():
			rand = torch.rand(1,config.in_c*3,config.h,config.w)
			enc = encoder(config,cnn=True)
			out = enc(rand)
			out = out.shape
			print(out)
			return out[1]*out[2]*out[3]//3

	def forward(self,x):
		h0 = torch.randn(self.config.num_layers_lstm,x.shape[0],6)
		c0 = torch.randn(self.config.num_layers_lstm,x.shape[0],self.config.lstm_hidden_size)
		print(x.shape)
		x_in = x.reshape(3,x.shape[0],(x.shape[1]*x.shape[2]*x.shape[3])//3)
		print("ok",x_in.shape)
		out, (hn,cn) = self.recur(x_in)
		return hn


class ganvo(nn.Module):
	def __init__(self,config):
		super(ganvo,self).__init__()
		self.config = config
		self.encoder = encoder(config)
		self.cnn = encoder(config,cnn=True)
		self.generator = generator(config)
		self.rnn = recurrent_net(config)
		self.discriminator = discriminator(config)
		self.optimizer = torch.optim.Adam(
		itertools.chain(self.generator.parameters(),self.encoder.parameters(),self.cnn.parameters(),self.discriminator.parameters()
		),lr=0.001)
		self.mseloss = torch.nn.MSELoss()
		self.ganloss = GANLoss(gan_mode='vanilla')

	def set_input(self, source, imgt_1, imgt_2):
		self.source = source
		self.imgt_1 = imgt_1
		self.imgt_2 = imgt_2

	def forward(self):

		out_gan = self.encoder(self.source)
		depth = self.generator(out_gan)

		pose = 	self.cnn(torch.cat([self.imgt_1,self.source,self.imgt_2],dim=1))
		pose = self.rnn(pose)

		self.depth = depth
		self.pose = pose

		return depth, pose

	def loss_generator(self):
		const_from_t_1, _ = inverse_warp(self.imgt_1,self.depth,self.pose[0,:],torch.tensor(self.config.intrinsics).float())
		const_from_t_2, _ = inverse_warp(self.imgt_2,self.depth,self.pose[1,:],torch.tensor(self.config.intrinsics).float())
		loss = self.mseloss(self.source,const_from_t_1) + self.mseloss(self.source, const_from_t_2)

		self.const_from_t_1 = const_from_t_1
		self.const_from_t_2 = const_from_t_2
		
		return loss

	def loss_discriminator(self):
		pred_real = self.dicriminator(self.source)
		loss_real = self.ganloss(pred_real, True)

		pred_fake_2 = self.dicriminator(self.const_from_t_2)
		pred_fake_1 = self.discriminator(self.const_from_t_1)

		loss_fake = self.ganloss(pred_fake_1,False) + self.ganloss(pred_fake_2,False)

		loss_D = 0.5*(loss_real + loss_fake)

		return loss_D

	def optimize_paramaters(self):

		self.forward()
		loss_G = self.loss_generator()
		loss_D = self.loss_discriminator()

		beta = (loss_G/loss_D).mean()
		self.loss_net = loss_G + beta*loss_D

		self.optimizer.zero_grad()
		self.loss_net.backward()
		self.optimizer.step()


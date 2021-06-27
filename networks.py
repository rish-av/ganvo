import torch
import torch.nn as nn
from torch.nn import BatchNorm2d

def conv(in_c,out_c,kernel=3,norm=BatchNorm2d,activation=nn.ELU):
	layers = [ nn.Conv2d(in_c,out_c,kernel_size=kernel,bias=False),
			   activation(),
			   norm(),
			   nn.Conv2d(out_c,out_c,kernel_size=kernel,bias=False),
			   norm(),
			   activation(),
			   ]
	return nn.Sequential(*layers)

def upconv(in_c,out_c,kernel_size=4,norm=BatchNorm2d):

	layers = [ nn.ConvTranspose2d(in_c,out_c,kernel_size=kernel_size,bias=False),
	           norm(),
			   nn.ELU()]
	return nn.Sequential(*layers)

class generator(nn.Module):
	def __init__(self,config):
		super(generator,self).__init__()
		channels = config.genrator_channels
		layers = [upconv(in_c=channel[i-1],out_c=channel[i]) for i in range(1,len(channels))]
		self.gen = nn.Sequential(*layers)

	def forward(self,x):
		return self.gen(x)


class discriminator(nn.Module):
	def __init__(self,config):
		super(discriminator,self).__init__()
		channels = config.discriminator_channels
		layers = [conv(in_c=channel[i-1],out_c=channel[i]) for i in range(1,len(channels))]
		self.disc = nn.Sequential(*layers)

	def forward(self,x):
		return self.disc(x)


class encoder(nn.Module):

	def __init__(self,config):
		super(encoder,self).__init__()
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
		self.recur = nn.LSTM(input_size=lstm_in_size,hidden_size=config.lstm_hidden_size,
			num_layers=config.num_layers_lstm,proj_size=6,batch_first=True)

	def get_in_size(self,config):
		with torch.no_grad():
			rand = torch.rand(1,config.in_c,config.h,config.w)
			enc = encoder(config)
			out = enc(out)
			out = out.shape
			return out[1]*out[2]*out[3]

	def forward(self,x):
		h0 = torch.randn(self.config.num_layers_lstm,x.shape[0],6)
		c0 = torch.randn(self.config.num_layers_lstm,x.shape[0],self.config.lstm_hidden_size)
		out, (hn,cn) = self.recur(x)
		return hn


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class ResNet(nn.Module):
	def __init__(self,config):
		super().__init__()
		builder = getattr(models,config['model_name'])
		print('MODEL NAME: ',config['model_name'])
		self.resnet = builder(pretrained = True)
		if config['fixed_feature']:
			for param in self.resnet.parameters():
				param.requires_grad = False
		if config['model_name'] in ['resnet18', 'resnet34']:
			block_4_channel = 512
		else:
			block_4_channel = 2048
		self.conv = nn.Conv2d(block_4_channel,config['cluster_vector_dim'],1)
	def forward(self,x):
		x = self.resnet.conv1(x)
		x = self.resnet.bn1(x)
		x = self.resnet.relu(x)
		x = self.resnet.maxpool(x)
		x = self.resnet.layer1(x)
		x = self.resnet.layer2(x)
		x = self.resnet.layer3(x)
		x = self.resnet.layer4(x)
		z = self.resnet.avgpool(x)
		# z = self.conv(z).squeeze(3).squeeze(2)
		# z_mean = torch.mean(z,1,True)
		# z_std = torch.std(z,1).unsqueeze(1)
		# z = (z-z_mean)/z_std
		return z

class DenseNet(nn.Module):
	def __init__(self,config):
		super().__init__()
		builder = getattr(models,config['model_name'])
		print('MODEL NAME: ',config['model_name'])
		self.densenet = builder(pretrained = True)
		if config['fixed_feature']:
			for param in self.densenet.parameters():
				param.requires_grad = False
		if config['model_name'] == 'densenet121':
			block_4_channel = 1024
		elif config['model_name'] == 'densenet169':
			block_4_channel = 1664
		elif config['model_name'] == 'densenet201':
			block_4_channel = 1920
		elif config['model_name'] == 'densenet161':
			block_4_channel = 2208
		self.conv = nn.Conv2d(block_4_channel,config['cluster_vector_dim'],1)
		
	def forward(self,x):
		x = self.densenet.conv1(x)
		x = self.densenet.bn1(x)
		x = self.densenet.relu(x)
		x = self.densenet.maxpool(x)
		x = self.densenet.layer1(x)
		x = self.densenet.layer2(x)
		x = self.densenet.layer3(x)
		x = self.densenet.layer4(x)
		z = self.densenet.avgpool(x)
		# z = self.conv(z).squeeze(3).squeeze(2)
		return z


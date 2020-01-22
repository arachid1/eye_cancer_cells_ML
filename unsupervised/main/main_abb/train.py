import os
import argparse
import time
import warnings
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import pandas as pd 

from net import ResNet,DenseNet
from dataset import *
from loss import *
from util import *
from config import config

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def main(config, resume):
	# Dataset
	fine_dataset = self_defined_dataset(config)
	# Dataloder
	train_loader = DataLoader(
		fine_dataset,
		shuffle = True,
		batch_size = config['batch_size'],
		num_workers = 8)
	val_loader = DataLoader(
		fine_dataset,
		shuffle = False,
		batch_size = config['batch_size'],
		num_workers = 8)
	test_loader = DataLoader(
		fine_dataset,
		shuffle = False,
		batch_size = config['batch_size'],
		num_workers = 8)
	# Model
	start_epoch = 0
	if config['model_name'].startswith('resnet'):
		model = ResNet(config)
	elif config['model_name'].startswith('densenet'):
		model = DenseNet(config)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#Optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'],weight_decay = 1e-5)
	# if use pretrained models
	if resume:
		filepath = config['pretrain_path']
		start_epoch,learning_rate,optimizer = load_ckpt(model,filepath)
		start_epoch += 1
	# if use multi-GPU
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
	model.to(device)		
	#resume or not
	if start_epoch == 0:
		print("Grand New Training")
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = config['switch_learning_rate_interval'])
	if not resume:
		learning_rate = config['learning_rate']
	# training part
	if config['if_train']:
		for epoch in range(start_epoch+1,start_epoch+config['num_epoch']+1):
			loss_tr = train(train_loader,model,optimizer,epoch, config)#if training, delete learning rate and add optimizer
			if config['if_valid'] and epoch % config['valid_epoch_interval'] == 0:
				with torch.no_grad():
					loss_val = valid(val_loader,model,epoch,config)
					scheduler.step(loss_val)
				save_ckpt(model,optimizer,epoch,loss_tr,loss_val,config)
	test(test_loader,model,config)
	#store_config(config)
	print("Training finished ...")




def train(loader, model, optimizer, epoch, config):
	print("--- TRAIN START ---")
	losses = []
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.train()
	date = config['date']
	loss_fn = loss_all(config).to(device)
	# train_all_time1 = time.time()
	for i, data in enumerate(loader):
		# batch_time = time.time()
		inputs = data['image'].to(device)
		outputs = data['label'].to(device)
		img_name = data['image_path']

	
		optimizer.zero_grad()

		z = model(inputs)
		loss = loss_fn(z,outputs)

		loss.backward()
		optimizer.step()
		losses.append(loss.item())

	# 	batch_time = time.time()-batch_time
	# 	train_log = open("../log/train_"+date+".txt","a")
	# 	train_log.write("epoch: {0:d}, iter: {1:d}, loss: {2:.3f}, time: {3:.3f}\n".format(epoch,i,loss,batch_time))
	# 	train_log.close()
	# train_all_time2 = time.time()
	# train_log = open("../log/train_"+date+".txt","a")
	# train_log.write("TRAINING"+"-"*10 + "epoch: {0:d}, loss: {1:.3f}\n".format(epoch,np.average(losses)))
	# train_log.write('total time = '+str(train_all_time2-train_all_time1)+'\n')
	# train_log.close()
	return np.average(losses)

def valid(loader,model,epoch,config,learning_rate):
	print("--- VALIDATION START ---")
	losses = []
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()

	loss_fn = loss_all(config).to(device)
	for i, data in enumerate(loader):
		inputs = data['image'].to(device)
		outputs = data['image'].to(device)
		img_name = data['image_path']

		z = model(inputs)
		loss = loss_fn(z,outputs)
		losses.append(loss.item())
		#centroid modification

	return np.average(losses)

def test(loader, model, config):
	print("--- TEST START ---")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	for i, data in enumerate(loader):
		inputs = data['image'].to(device)
		outputs = data['image'].to(device)
		img_name = data['image_path']
		
		z = model(inputs)
	# could save the results.

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--resume', dest='resume')
	args=parser.parse_args()
	for i in config:
		print(i,':',config[i])
	main(config,eval(args.resume))


















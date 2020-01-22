import numpy as np 
import os
import torch
import json
from shutil import copyfile
from skimage.io import imsave
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.cluster import KMeans
import random

def store_config(config,phase):
		copyfile('config.py', os.path.join(config['out_dir'],config['date'],'config.py'))

def check_ckpt_dir():
	checkpoint_dir = os.path.join('..','checkpoint')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

def is_best_ckpt(epoch,loss_tr,loss_cv,config):
	check_ckpt_dir()
	best_json = os.path.join('..','checkpoint',config['date']+'.json')
	best_loss_cv = best_loss_tr = float("Inf")

	if os.path.exists(best_json):
		with open(best_json) as infile:
			data = json.load(infile)
			best_loss_cv = data['loss_cv']
			best_loss_tr = data['loss_tr']

	if loss_cv < best_loss_cv:
		with open(best_json,'w') as outfile:
			json.dump({
				'epoch': epoch,
				'loss_tr': loss_tr,
				'loss_cv': loss_cv,
				}, outfile)
		return True
	return False

def save_ckpt(model, optimizer, epoch, loss_tr, loss_cv, config):
	def do_save(filepath):
		torch.save({
			'epoch': epoch,
			'name': config['model_name'],
			'date': config['date'],
			'learning_rate': config['learning_rate'],
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			}, filepath)
	# check if best_checkpoint
	is_best_ckpt(epoch,loss_tr,loss_cv,config)
	filepath=os.path.join('..','checkpoint',config['date']+'.pkl')
	do_save(filepath)


def _extract_state_from_dataparallel(checkpoint_dict):
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in checkpoint_dict.items():
		if k.startswith('module.'):
			name = k[7:]
		else:
			name = k
		new_state_dict[name] = v
	return new_state_dict

def load_ckpt(model=None,filepath=None):
	if not torch.cuda.is_available():
		print('no gpu available')
	checkpoint = torch.load(filepath)
	epoch = checkpoint['epoch']
	learning_rate = checkpoint['learning_rate']
	optimizer = checkpoint['optimizer']
	if model:
		model_dict = model.state_dict()
		pretrain_dict = checkpoint['model']
		detect_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict}
		full_dict = {}
		for k, v in model_dict.items():
			for k_, v_ in pretrain_dict.items():
				if k == k_:
					full_dict[k] = v_
			if k not in full_dict:
				full_dict[k] = v
		model.load_state_dict(_extract_state_from_dataparallel(full_dict))
	return epoch, learning_rate, optimizer, M, s


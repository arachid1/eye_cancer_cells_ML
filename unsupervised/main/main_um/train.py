import os
import argparse
import time
import warnings
from sklearn.cluster import KMeans
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import pandas as pd 
from skimage.io import imsave 
from shutil import copyfile

from Pytorch_DeepLab_v3_plus.networks.deeplab_resnet_wodecoder import DeepLabv3_plus
from net import ResNet,DenseNet,BagNet
from dataset import *
from util import config_selection, save_ckpt, load_ckpt, switch_learning_rate, centroid_adjustment, test_save_img, loss_para_selection, select_centroid_batch, largest_num_cluster_status, largest_std_cluster_status, M_new_assignment, M_update, T_SNE, store_config, coarse_to_fine_cluster_enlarge, K_MEANS, point_assign
from loss import *

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def main(config, resume, phase):
	# Dataset
	fine_dataset = fine_clustering_dataset(config)
	# Dataloder
	train_loader = DataLoader(
		fine_dataset,
		shuffle = True,
		batch_size = config['batch_size'],
		num_workers = 32)
	val_loader = DataLoader(
		fine_dataset,
		shuffle = False,
		batch_size = config['batch_size'],
		num_workers = 32)
	test_loader = DataLoader(
		fine_dataset,
		shuffle = False,
		batch_size = config['batch_size'],
		num_workers = 32)
	# Model
	start_epoch = 0
	if config['model_name'].startswith('resnet'):
		model = ResNet(config)
	elif config['model_name'].startswith('densenet'):
		model = DenseNet(config)
	elif config['model_name'].startswith('deeplab'):
		cluster_vector_dim = config['cluster_vector_dim']
		model = DeepLabv3_plus(nInputChannels=3, n_classes=3, os=16, cluster_vector_dim = cluster_vector_dim, pretrained=True, _print=True)
	elif config['model_name'].startswith('bagnet'):
		model = BagNet(config=config)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if resume:
		filepath = config['pretrain_path']
		start_epoch,learning_rate,optimizer,M,s = load_ckpt(model,filepath)
		start_epoch += 1
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
	model.to(device)
	#Optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'],weight_decay = 1e-5)
	#resume or not
	if start_epoch == 0:
		print("Grand New Training")
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = config['switch_learning_rate_interval'])
	# log_dir = config['log_dir']+"/{}_{}_".format(config['date'],config['model_name'])+"ep_{}-{}_lr_{}".format(start_epoch,start_epoch+config['num_epoch'],config['learning_rate'])
	# best loss
	if not resume:
		learning_rate = config['learning_rate']
		M,s = cluster_initialization(train_loader,model,config,phase)
	print(start_epoch)
	if config['if_train']:
		for epoch in range(start_epoch+1,start_epoch+config['num_epoch']+1):
			loss_tr = train(train_loader,model,optimizer,epoch, config, M, s)#if training, delete learning rate and add optimizer
			if config['if_valid'] and epoch % config['valid_epoch_interval'] == 0:
				with torch.no_grad():
					loss_val,M,s = valid(val_loader,model,epoch,config,learning_rate, M, s, phase)
					scheduler.step(loss_val)
				save_ckpt(model,optimizer,epoch,loss_tr,loss_val,config, M,s)
			else:
				val_log = open("../log/val_"+config['date']+".txt","a")
				val_log.write('epoch '+str(epoch)+'\n')
				val_log.close()
	test(test_loader,model,config,M,phase)
	store_config(config,phase)
	print("Training finished ...")

def cluster_initialization(loader,model,config, phase):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()

	vectors = []
	s = {}
	path_all = []
	time1 = time.time()
	for i, data in enumerate(loader):
		time_start = time.time()
		inputs = data['fine_image'].to(device)
		img_name = data['fine_image_path']
		feature = model(inputs)
		outputs = feature.detach().cpu().numpy()
		vectors.append(outputs)
		if phase == 'bagnet' and i == 0:
			size = outputs.shape[1:3]
			print(size)
		path_all = path_all + img_name
		time_training_end = time.time()
		print("feed forward time: ", time_training_end-time_start)
		if i%1000 == 0:
			print(i)
	time2 = time.time()
	print('feed forward time in init: ', time2-time1)
	vectors = np.concatenate(vectors,axis = 0)
	'''
	if phase == 'bagnet': # (batch, height, width, cluster_dim)
		vectors = vectors.reshape(-1,vectors.shape[3])
		print(vectors.shape[0])
	kmeans = KMeans(n_clusters=config['n_cluster'],n_jobs=16,max_iter=50).fit(vectors)
	M = kmeans.cluster_centers_.transpose(1,0)#dim,n_cluster
	assignment = kmeans.labels_
	if phase == 'bagnet':
		assignment = assignment.reshape(-1,size[0],size[1])
	for i in range(assignment.shape[0]):
		s[path_all[i]] = assignment[i] # if bagnet, is a 2-dim array
	for i in range(config['n_cluster']):
		print(np.sum(assignment == i)/vectors.shape[0])
	M = torch.tensor(M,requires_grad = False).to(device)#(dim,n_cluster)
	'''
	M,s = K_MEANS(vectors,path_all,config,phase)
	time3 = time.time()
	print('k-means time: ',time3-time2)
	return M,s



def train(loader, model, optimizer, epoch, config, M, s):
	losses = []
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.train()
	date = config['date']
	print("--- TRAIN START ---")
	loss_fn = loss_all(config).to(device)
	train_all_time1 = time.time()
	for i, data in enumerate(loader):
		batch_time = time.time()
		inputs = data['fine_image'].to(device)
		#print('torch.max(inputs)',torch.max(inputs))
		img_name = data['fine_image_path']
		s_batch = select_centroid_batch(s,img_name) # if bagnet, is (batch, width, height)
	
		optimizer.zero_grad()

		z = model(inputs)
		#print('torch.max(z)',torch.max(z))
		loss = loss_fn(z,M,s_batch)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())

		batch_time = time.time()-batch_time
		train_log = open("../log/train_"+date+".txt","a")
		train_log.write("epoch: {0:d}, iter: {1:d}, loss: {2:.3f}, time: {3:.3f}\n".format(epoch,i,loss,batch_time))
		train_log.close()
	train_all_time2 = time.time()
	train_log = open("../log/train_"+date+".txt","a")
	train_log.write("TRAINING"+"-"*10 + "epoch: {0:d}, loss: {1:.3f}\n".format(epoch,np.average(losses)))
	train_log.write('total time = '+str(train_all_time2-train_all_time1)+'\n')
	train_log.close()
	return np.average(losses)

def valid(loader,model,epoch,config,learning_rate, M, s, phase):
	print("--- VALIDATION START ---")
	losses = []
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	date = config['date']
	val_ind = []
	#centroid modification initialization
	new_s = {}
	path_all = []
	outputs = []

	loss_fn = loss_all(config).to(device)
	val_forward_time1 = time.time()
	for i, data in enumerate(loader):
		batch_time = time.time()
		inputs = data['fine_image'].to(device)
		img_name = data['fine_image_path']
		s_batch = select_centroid_batch(s,img_name)

		z = model(inputs)#(batch, dim), if bagnet, (batch, height, width, dim)
		if phase == 'bagnet' and i == 0:
			size = z.shape[1:3]
		assignments = point_assign(z,M)
		loss = loss_fn(z,M,s_batch)
		print(loss.item())
		losses.append(loss.item())
		val_ind.append(assignments.cpu().detach().numpy())#(batch), if bagnet, (batch*height*width)
		batch_time = time.time()-batch_time
		#centroid modification
		path_all = path_all + img_name
		outputs.append(z.detach().cpu())
	val_forward_time2 = time.time()

	# store parameters
	val_ind = np.concatenate(val_ind,axis=0)
	val_ind_proportion = [np.sum(val_ind == i)/val_ind.shape[0] for i in range(config['n_cluster'])]
	val_log = open("../log/val_"+date+".txt","a")
	if epoch == 1:
		for i in config:
			val_log.write(str(i)+" = "+str(config[i])+"\n")
	# store statistical values
	val_log.write("VALIDATION" + "-"*10 + "epoch: {0:d}, loss: {1:.3f}, learning rate: {2:.7f}\n".format(epoch,np.average(losses),learning_rate))
	for i in range(config['n_cluster']):
		val_log.write("cluster_"+str(i)+": " + str(round(val_ind_proportion[i],5))+"\n")
	val_log.write('val forward time: '+str(val_forward_time2-val_forward_time1)+'\n')
	val_log.close()

	modification_M_time1 = time.time()
	#centroid modification
	M = M.detach().cpu()
	outputs = torch.cat(outputs,dim=0)
	loop = 1
	while loop:
		new_assign = point_assign(outputs,M)
		##get the largest proportion information
		#_mean,_std = largest_num_cluster_status(outputs,new_assign,config)
		#get the largest std centroid information
		_mean,_std = largest_std_cluster_status(outputs.reshape(-1,outputs.shape[-1]),new_assign,config)
		loop = 0
		if phase == "coarse_to_fine":
			config, M = coarse_to_fine_cluster_enlarge(config,epoch,M)
		for i in range(config['n_cluster']):
			if torch.sum(new_assign == i) == 0:
				M[:,i] = M_new_assignment(_mean,_std)
				loop = 1
	new_M = M_update(M,new_assign,outputs,config)
	if phase == 'bagnet':
		new_assign = new_assign.reshape(-1,size[0],size[1])
	for i in range(outputs.shape[0]):
		new_s[path_all[i]] = new_assign[i].detach().cpu().numpy()
	modification_M_time2 = time.time()
	val_log = open("../log/val_"+date+".txt","a")
	val_log.write('M_modification_time = '+str(modification_M_time2-modification_M_time1)+'\n')
	val_log.close()
	#save results
	test_save_img(epoch,new_s,config,phase)
	
	return np.average(losses),new_M.to(device),new_s

def test(loader, model, config, M, phase):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	raw_assignment = []
	test_assignment = {}
	results = []
	for i, data in enumerate(loader):
		inputs = data['fine_image'].to(device)
		print(inputs.shape)
		img_name = data['fine_image_path']
		
		z = model(inputs)
		if phase == "bagnet" and i == 0:
			size = z.shape[1:3]
		results.append(z.detach().cpu().numpy())
		assignments = point_assign(z,M)
		raw_assignment.append(assignments.detach().cpu().numpy())
		if phase == 'bagnet':
			assignments = assignments.reshape(-1,size[0],size[1])
		for j, single_img_name in enumerate(img_name):
			test_assignment[single_img_name] = assignments[j].detach().cpu().numpy()
	raw_assignment = np.concatenate(raw_assignment,axis=0)
	results = np.concatenate(results,axis=0)
	test_save_img(epoch='test',assignment=test_assignment, config=config, phase= phase)	
	T_SNE(results,raw_assignment.reshape(-1),config, phase)
	#test_save_info(epoch='test',assignment=raw_assignment, outputs=results, config=config)

def cluster_info_modification(loader,model,config,M):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	
	M = M.detach().cpu()
	new_s = {}
	path_all = []
	outputs = []
	for i, data in enumerate(loader):
		inputs = data['fine_image'].to(device)
		img_name = data['fine_image_path']
		z = model(inputs)
		
		path_all = path_all + img_name
		outputs.append(z.detach().cpu())
	print('CLUSTER INFOMATION GOT')
	outputs = torch.cat(outputs,dim=0)
	loop = 1
	while loop:
		_,new_assign = torch.min(torch.sum((outputs.unsqueeze(2)-M.unsqueeze(0))**2,dim=1),dim=1)
		
		#get largest proportion information
		_mean,_std = largest_cluster_status(outputs,new_assign,config)
		loop = 0
		for i in range(config['n_cluster']):
			if torch.sum(new_assign == i) == 0:
				M[:,i] = M_new_assignment(_mean,_std)
				loop = 1
		
	for i in range(outputs.shape[0]):
		new_s[path_all[i]] = new_assign[i]
		new_M = M_update(M,new_assign,outputs,config)
	return new_M.to(device),new_s
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--resume', dest='resume')
	parser.add_argument('--phase', dest='phase')
	args=parser.parse_args()
	config = config_selection(args.phase)
	for i in config:
		print(i,':',config[i])
	main(config,eval(args.resume),args.phase)


















import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import glob
from skimage.io import imread 
from skimage.transform import resize
import numpy as np
import time 
from PIL import Image

class fine_clustering_dataset(Dataset):
	def __init__(self,config): 
		self.config = config
		self.path_list = glob.glob(config['input_dir']+'/Slide */*')
		print(self.path_list[0])
	def __len__(self):
		return len(self.path_list)
	def __getitem__(self,idx):
		image_path = self.path_list[idx]
		image = imread(image_path)

		_mean = np.array([0.735, 0.519, 0.598]).reshape(1,-1)
		_std = np.array([0.067, 0.067, 0.063]).reshape(1,-1)
		image = (((image/255.)-_mean)/_std).astype(np.float32)
		#image = Image.open(image_path)
		
		image = F.to_tensor(image)

		#Load output
		
		fine_sample = {'image':image,'image_path':image_path}
		return fine_sample




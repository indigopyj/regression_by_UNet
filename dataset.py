import numpy as np
import torch
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import os
from utils import *


# Data loader 구현

class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transform=None, task=None, opts=None):
    self.data_dir = data_dir
    self.transform = transform
    self.task = task
    self.opts = opts

    lst_data = os.listdir(self.data_dir)
    lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('png')]

    lst_data.sort()

    self.lst_data = lst_data

  def __len__(self):
    return len(self.lst_data)
  
  def __getitem__(self, index):
    #label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
    #inputImg = np.load(os.path.join(self.data_dir, self.lst_input[index]))

    img  = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
    sz = img.shape

    if sz[0] > sz[1]: # 이미지를 가로로 긴 형태로 만들기
      img = img.transpose((1,0,2))


    if img.dtype == np.uint8: # image type이 uint8인 경우만 normalize가능
      img = img / 255.0
    
    

    # channel axis 추가해주기
    if img.ndim == 2:
      img = img[:,:,np.newaxis]

    label = img

    #artifact 이미지 만들기
    if self.task == "denoising":
      inputImg = add_noise(img, type=self.opts[0], opts=self.opts[1])
    elif self.task == "inpainting":
      inputImg = add_sampling(img, type=self.opts[0], opts=self.opts[1])
    elif self.task == "super_resolution":
      inputImg = add_blur(img, type=self.opts[0], opts=self.opts[1])

    
    
    data = {'input': inputImg, 'label': label}

    if self.transform:
      data = self.transform(data)

    return data


class ToTensor(object) : # numpy to tensor
  def __call__(self, data):
    label, inputImg = data['label'], data['input']

    # Image의 numpy dim = (Y, X, C)
    # Image의 tensor dim = (C, Y, X)

    label = label.transpose((2, 0, 1)).astype(np.float32)
    inputImg = inputImg.transpose((2, 0, 1)).astype(np.float32)

    data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(inputImg)}

    return data

class Normalization(object):
  def __init__(self, mean=0.5, std=0.5):
    self.mean = mean
    self.std = std
  
  def __call__(self, data):
    label, inputImg = data['label'], data['input']

    inputImg = (inputImg - self.mean) / self.std
    label = (label - self.mean) / self.std

    data = {'label': label, 'input':inputImg}

    return data

class RandomFlip(object):
  def __call__(self, data):
    label, inputImg = data['label'], data['input']

    if np.random.rand() > 0.5:
      label = np.fliplr(label)
      inputImg = np.fliplr(inputImg)
    
    if np.random.rand() > 0.5:
      label = np.flipud(label)
      inputImg = np.flipud(inputImg)

    data = {'label': label, 'input':inputImg}

    return data


class RandomCrop(object):
  def __init__(self, shape):
    self.shape = shape

  def __call__(self, data):
    label, inputImg = data['label'], data['input']

    h,w = label.shape[:2]
    new_h, new_w = self.shape

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
    id_x = np.arange(left, left + new_w, 1)

    inputImg = inputImg[id_y, id_x]
    label = label[id_y, id_x]


    data = {'input' : inputImg, 'label':label}

    return data
    

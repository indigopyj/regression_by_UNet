import os
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import poisson
from scipy.io import loadmat
from skimage.transform import rescale

# save network
def save(ckpt_dir, net, optim, epoch):
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  
  torch.save({
      'net': net.state_dict(), 'optim' : optim.state_dict()},
      "%s/model_epoch%d.pth" % (ckpt_dir, epoch))
  

# load network
def load(ckpt_dir, net, optim):
  if not os.path.exists(ckpt_dir):
    epoch = 0
    return net, optim, epoch
  
  ckpt_list = os.listdir(ckpt_dir)
  ckpt_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) # 숫자만 이용하여 소팅

  dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_list[-1]))

  net.load_state_dict(dict_model['net'])
  optim.load_state_dict(dict_model['optim'])
  epoch = int(ckpt_list[-1].split('epoch')[1].split('.pth')[0])

  return net, optim, epoch


## sampling 하기
def add_sampling(img, type="random", opts=None):
    sz = img.shape

    if type == "uniform":
        ds_y = opts[0].astype(np.int)
        ds_x = opts[1].astype(np.int)

        msk = np.zeros(sz)
        msk[::ds_y, ::ds_x, :] = 1

        dst = img*msk

    elif type == "random":
        ## RGB 모두 랜덤하게 샘플링
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        prob = opts[0]
        msk = (rnd >prob).astype(np.float)

        dst = img*msk

    elif type == "gaussian":
        x0 = opts[0]
        y0= opts[1]
        sigmax=opts[2]
        sigmay=opts[3]

        a = opts[4]
        
        ly = np.linspace(-1, 1, sz[0])
        lx = np.linspace(-1, 1, sz[1])  # # (시작, 끝(포함), 갯수) 갯수만큼 선형구간을 분할한다.

        x, y = np.meshgrid(lx, ly)  # 사각형 영역을 구성하는 가로축의 점들과 세로축의 점을 나타내는 두 벡터를 인수로 받아서 이 사각형 영역을 이루는 조합을 출력

        
        gaussian = a*np.exp(-((x - x0)**2 / (2*sigmax**2) + (y - y0)**2 / (2*sigmay**2)))

        gaussian = np.tile(gaussian[:, :, np.newaxis], (1,1,sz[2]))
        rnd = np.random.rand(sz[0],sz[1],sz[2])
        msk = (rnd < gaussian).astype(np.float)

        dst = msk * img

    return dst

## Noise 추가하기
def add_noise(img, type="random", opts=None):
  sz = img.shape

  if type == "random":
    sgm = opts[0]

    noise = sgm/255.0 * np.random.randn(sz[0], sz[1], sz[2])

    dst = img + noise

  elif type == "poisson":
    dst = poisson.rvs(255.0 * img) / 255.0
    noise = dst - img

    #plt.imshow(noise.astype(np.float))


  return dst


## Blurring 추가하기
def add_blur(img, type="bilinear", opts=None):

  if type == "nearest":
    order = 0
  elif type == "bilinear":
    order = 1
  elif type == "biquadratic":
    order = 2
  elif type == "bicubid":
    order = 3
  elif type == "biquartic":
    order = 4
  elif type == "biquintic":
    order = 5

  sz = img.shape

  ds = opts[0] # downsampling ratio

  if len(opts) == 1: # 다시 업샘플링해줌
    keepdim = True
  else: # dimension이 축소된 상태 유지
    keepdim = opts[1]

  #dst = rescale(img, scale=(dw,dw,1), order=order).astype(np.float) # scale = (y방향,x방향,channel방향)
  dst = resize(img, output_shape=(sz[0] // dw, sz[1] // dw, sz[2]), order = order)

  if keepdim:
    dst = resize(dst, output_shape=(sz[0], sz[1], sz[2]), order=order)

  return  dst

  

  
    
            


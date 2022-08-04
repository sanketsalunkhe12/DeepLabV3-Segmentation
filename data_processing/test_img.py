from cv2 import imread, resize
import torch
import numpy as np


def PrepareTestImg(img_path):
  path = '/home/sanket/Projects/coast/segmentation/cityscape dataset/'
  x = imread(path+img_path)
  x_norm = (x - np.min(x))/np.ptp(x)    # normalize image
  x = np.moveaxis(x_norm, -1, 0)  
  x = x[np.newaxis,:,:,:]               # add 1 more dimension to match input shape 
  x = torch.from_numpy(x).type(torch.float32)
  return x
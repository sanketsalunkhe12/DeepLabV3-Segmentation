import torch
import torch.nn as nn
from torch.utils.data import Dataset
from cv2 import imread
import cv2
import numpy as np
from statistics import mean as mean

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as deeplab


class DeepLab_Dataset_Creation(Dataset):
  
  def __init__(self, input: list, target: list):
    self.target = target
    self.input = input
    self.input_dtype = torch.float32
    self.target_dtype = torch.long

  def __len__(self):
    return len(self.input)

  def create_target_class_id(self, target: np.ndarray):
    pixel_values = np.array([  0.,   8.,  10.,  13.,  16.,  26.,  33.,  46.,  47.,  58.,  70.,
        76.,  77.,  84.,  90., 108., 115., 118., 119., 120., 126., 153.,
       164., 171., 173., 178., 193., 195., 210.])
    dummy = np.zeros_like(target)
    for id, value in enumerate(pixel_values):
      mask = np.where(target == value)
      dummy[mask] = id
    return dummy

  def normalize(self, input: np.ndarray):
    norm_input = (input - np.min(input))/np.ptp(input)
    return norm_input

  def __getitem__(self, index: int):
    input_path = self.input[index]
    temp_target_path = self.target[index]

    target_path = str(temp_target_path)[:-12]+'color.png'
    folder_path = '/home/sanket/Projects/coast/segmentation/cityscape dataset/'

    x, y = imread(folder_path+input_path), imread(folder_path+target_path)
    x = self.normalize(x)
    x = np.moveaxis(x, -1, 0)

    y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
    y = self.create_target_class_id(y)

    x, y = torch.from_numpy(x).type(self.input_dtype), torch.from_numpy(y).type(self.target_dtype)

    return x, y



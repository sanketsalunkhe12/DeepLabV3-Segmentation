import numpy as np
import path_list
import cv2
from cv2 import imread

# change path as per your folder structure

train_file_path = "/home/sanket/Projects/coast/segmentation/cityscape dataset/list/train.lst"
val_file_path = "/home/sanket/Projects/coast/segmentation/cityscape dataset/list/val.lst"

train_input_list, train_target_list  = path_list.create_path_list(train_file_path)
val_input_list, val_target_list  = path_list.create_path_list(val_file_path)

# list of all target image path

target_list = np.concatenate((train_target_list, val_target_list))


path = '/home/sanket/Projects/coast/segmentation/cityscape dataset/'
target_all_pixels = []

for i in target_list:
  color_img_path = str(i)[:-12]+'color.png'
  target_image = imread(path+color_img_path)

  target_gray_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

  temp = np.unique(target_gray_image)
  target_all_pixels = np.concatenate((target_all_pixels, temp))


target_unique_pixels = np.unique(target_all_pixels)
print(target_unique_pixels)
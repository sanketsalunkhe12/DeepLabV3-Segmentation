# Import all required libaries

from data_processing import deeplab_dataset, path_list

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cv2 import imread
from statistics import mean as mean

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as deeplab


# create input and target list

val_file_path = "/home/sanket/Projects/coast/segmentation/cityscape dataset/list/val.lst"
val_input_list, val_target_list  = path_list.create_path_list(val_file_path)

# Creating a training dataset and dataloader

validation_dataset = deeplab_dataset.DeepLab_Dataset_Creation(input = val_input_list, target = val_target_list)
validation_dataloader = DataLoader(dataset = validation_dataset, batch_size = 2, shuffle = True)
print("datalodaer created \n")

# DeepLab model creation and importing pre-trained weights

deeplab_model = deeplab(num_classes=30)
model_weight_path = '/home/sanket/Projects/coast/segmentation/trained weights/119-169/deeplab_model5_49.pth'   
deeplab_model.load_state_dict(torch.load(model_weight_path))

# Validation model function

loss_fn = nn.CrossEntropyLoss()

def valid(valid_dataloader, deeplab_model, loss_fn):
  
  deeplab_model.eval()
  deeplab_model.to('cuda') 
  val_loss_list = []

  for x,y in valid_dataloader:
      input = x.to('cuda')
      target = y.to('cuda')

      pred_output = deeplab_model(input)
      loss = loss_fn(pred_output['out'], target)

      val_loss_list.append(float(loss))

      print('\n loss: {}'.format(loss))

  print('Average Validation loss: {}'.format(mean(val_loss_list)))

valid(validation_dataloader, deeplab_model, loss_fn)
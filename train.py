# Import all required libaries

from data_processing import deeplab_dataset, path_list

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cv2 import imread
from statistics import mean as mean

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as deeplab

# create input and target list

train_file_path = "/home/sanket/Projects/coast/segmentation/cityscape dataset/list/train.lst"
train_input_list, train_target_list  = path_list.create_path_list(train_file_path)

# Creating a training dataset and dataloader

training_dataset = deeplab_dataset.DeepLab_Dataset_Creation(input = train_input_list, target = train_target_list)
training_dataloader = DataLoader(dataset = training_dataset, batch_size = 2, shuffle = True)

# DeepLab model creation and importing pre-trained weights

deeplab_model = deeplab(num_classes=30)

# pre-trained models

model_weight_path = '/home/sanket/Projects/coast/segmentation/trained weights/119-169/deeplab_model5_49.pth'    #changed as per your model path
deeplab_model.load_state_dict(torch.load(model_weight_path))


# Training model function

def train(train_dataloader, deeplab_model, num_epoch, optimizer, loss_fn):
  
  deeplab_model.train()
  deeplab_model.to('cuda') 
  loss_list = []

  for epoch in range(num_epoch):
    print('Epoch {}'.format(epoch))
    epoch_loss = []

    for x,y in train_dataloader:
        input = x.to('cuda')
        target = y.to('cuda')
        
        optimizer.zero_grad()

        pred_output = deeplab_model(input)
        loss = loss_fn(pred_output['out'], target)

        loss.backward()
        optimizer.step()

        epoch_loss.append(float(loss))

        print('\n loss: {}'.format(loss))
    
    loss_list.append(mean(epoch_loss))
    torch.save(deeplab_model.state_dict(), 'deeplab_model_{}.pth'.format(epoch))
    print('\n model saved')

  return loss_list

# start Training

epoch = 40
lr = 0.0005
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(deeplab_model.parameters(), lr=lr)  

epoch_loss_list = train(training_dataloader, deeplab_model, epoch, optimizer, loss_fn)
print(epoch_loss_list)


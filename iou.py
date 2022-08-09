# Here we are calculating IOU for all classes in cityscape and 
# also for specifically road class.

from torchmetrics import JaccardIndex
import numpy as np
import cv2
from cv2 import imread
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as deeplab


def create_val_target_class_id(target):
  pixel_values = np.array([  0.,   8.,  10.,  13.,  16.,  26.,  33.,  46.,  47.,  58.,  70.,
        76.,  77.,  84.,  90., 108., 115., 118., 119., 120., 126., 153.,
       164., 171., 173., 178., 193., 195., 210.])
  dummy = np.zeros_like(target)
  for id, value in enumerate(pixel_values):
    mask = np.where(target == value)
    dummy[mask] = id
  return dummy


def PrepareValImg(input_img_path, target_img_path):
  path = '/home/sanket/Projects/coast/segmentation/cityscape dataset/'
  
  target_path = str(target_img_path)[:-12]+'color.png' 
  
  x, y = imread(path+input_img_path), imread(path+target_path)

  x_norm = (x - np.min(x))/np.ptp(x)    # normalize image
  x = np.moveaxis(x_norm, -1, 0)  
  x = x[np.newaxis,:,:,:]               # add 1 more dimension to match input shape 
  x = torch.from_numpy(x).type(torch.float32)

  y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
  y = create_val_target_class_id(y)
  y = torch.from_numpy(y).type(torch.long)

  return x, y


def create_road_class(target):
  dummy = np.zeros_like(target)
  mask = np.where(target == 14)
  dummy[mask] = 1
  return dummy


def PrepareOutputImg(pred_target):
    pred_target_output = pred_target['out'].data.cpu().numpy()
    pred_target_output = pred_target_output[0]
    pred_target_output = np.moveaxis(pred_target_output, 0, -1)
    pred_target_output = pred_target_output.argmax(2)
    return pred_target_output

val_file_path = "/home/sanket/Projects/coast/segmentation/cityscape dataset/list/val.lst"
with open(val_file_path) as g:
  val_path_list = g.read().split()
  val_path_list = np.array(val_path_list).reshape(int(len(val_path_list)/2),2)

val_input_list = val_path_list[:,0]
val_target_list = val_path_list[:,1]

val_input_img, val_target_img = PrepareValImg(val_input_list[0], val_target_list[0])
val_input_img = val_input_img.to('cuda')

deeplab_model = deeplab(num_classes=30)
model_weight_path = '/home/sanket/Projects/coast/segmentation/trained weights/119-169/deeplab_model5_49.pth'   
deeplab_model.load_state_dict(torch.load(model_weight_path))
deeplab_model.eval()
deeplab_model.to('cuda')

pred_target = deeplab_model(val_input_img)
pred_target_output = PrepareOutputImg(pred_target)

# IOU for all classes

iou = JaccardIndex(num_classes=30)
pred_output = torch.from_numpy(pred_target_output).type(torch.long)
print("\n Overall class IOU is: ", iou(pred_output, val_target_img))

# IOU for road

pred_road_output = create_road_class(pred_target_output)
target_road = create_road_class(val_target_img)

pred_road_tensor = torch.from_numpy(pred_road_output).type(torch.long)
target_road_tensor = torch.from_numpy(target_road).type(torch.long)

iou_road = JaccardIndex(num_classes=2)
print("\n Road IOU is: ", iou_road(pred_road_tensor, target_road_tensor))



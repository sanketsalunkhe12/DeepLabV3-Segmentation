# Import all required libaries

from data_processing import test_img, path_list

import torch
from cv2 import imread
import numpy as np
from statistics import mean as mean
from matplotlib import pyplot as plt

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as deeplab

test_file_path = "/home/sanket/Projects/coast/segmentation/cityscape dataset/list/test.lst"
test_path_list = path_list.create_test_path_list(test_file_path)

deeplab_model = deeplab(num_classes=30)
model_weight_path = '/home/sanket/Projects/coast/segmentation/trained weights/119-169/deeplab_model5_49.pth'    
deeplab_model.load_state_dict(torch.load(model_weight_path))

deeplab_model.eval()
# deeplab_model.to('cpu')

test_image = test_img.PrepareTestImg(test_path_list[0])    # sending one image by one
# test_image = test_image.to('cuda')

pred_target = deeplab_model(test_image)

pred_target_output = pred_target['out'].data.cpu().numpy()
pred_target_output = np.moveaxis(pred_target_output[0], 0, -1)
pred_target_output = pred_target_output.argmax(2)

# plot image

fig, ax = plt.subplots()
im = ax.imshow(pred_target_output)

plt.show()


# DeepLabV3-Segmentation

<p align="center">
  <img src="https://github.com/sanketsalunkhe12/DeepLabV3-Segmentation/blob/main/readme_data/Screenshot%20from%202022-08-02%2015-44-07.png">
</p>


### Segmentation: 

<p align="justify"> Image segmentation can be formulated as a classification problem of pixels with semantic labels. Clustering parts of an image together that belong to the same object class. It is a pixel-level classification. </p>

<p align="justify"> <strong> Semantic segmentation </strong> treats multiple objects within a single category as one entity. <strong> Instance segmentation, </strong> on the other hand, identifies individual objects within these categories. Semantic segmentation involves understanding what region of image those things are located in at a very grained level. Segmentation is a locating a region taken by different objects and creating a precise outline of the object. In short, identifying exactly which pixels belong to different objects.  </p>  

### General Structure of semantic segmentation model:

<p align="justify"> The architecture of most semantic segmentation models contains a series of <strong> downsampling layers </strong> that reduce the dimensionality along spatial dimensions and then followed by a series of <strong> upsampling layers </strong> which again increase dimensions along the spatial dimensions. At the end of an upsampling layer, the output has the same spatial dimensions as the original input. The idea is to label every single pixel. The number of channels in output will depend on the number of classes of objects to identify.  </p>

<p align="center">
  <img src="https://github.com/sanketsalunkhe12/DeepLabV3-Segmentation/blob/main/readme_data/Screenshot%20from%202022-08-02%2015-41-44.png" width="350">
</p>
  
### Input, Target and Output Data format:

<p align="justify"> For DeepLabV3 model or any other segmentation model the input is a simple RBG or Grayscale image with shape [H, W, C], where H: height, W: width and C: no of channels. In general for segmentation the input image should be in <b>[N, C, H, W] </b> shape, where N is the number of images in a batch. During data processing we need to use moveaxis to change input image shape from [H, W, C] to [C, H, W]. The remaining N dimension will be incorporated using dataloader. Along with this we need to normalize the image. Eg. In case of Cityscape dataset with batch size of 2 we have our input image to model in [2, 3, 1024, 2048] shape. </p>

<p align="justify"> The main data processing is required for Target images. Generally target images are generated in [H, W, C] shape where different color different class in image. For training process we need to convert these color pixels into class id. First we will convert color target image into gray-scale image where each class id will be represented by particular pixel. In case of Cityscape dataset the relation between class id, grayscape pixel value and color value is as follows. Using <b> <i> create_target_class_id() </i> </b> function we will convert these color images into classID images which will be feed as a target during training.    </p>

    class_id = gray_value = color_value = objects

    0 = 0 = (0, 0, 0) = unlabeled, ego vehicle, rectification border, out of roi, static
    1 = 8 = (0, 0, 70) = truck
    2 = 10 = (0, 0, 90) = caravan
    3 = 13 = (0, 0, 110) = trailer
    4 = 16 = (0, 0, 142) = car
    5 = 26 = (0, 0, 230) = motorcycle
    6 = 33 = (81, 0, 81) = ground
    7 = 46 = (119, 11, 32) = bicycle
    8 = 47 = (0, 60, 100) = bus
    9 = 58 = (0, 80, 100) = train
    10 = 70 = (70, 70, 70) = building
    11 = 76 = (255, 0, 0) = rider
    12 = 77 = (111, 74, 0)  = dynamic
    13 = 84 = (220, 20, 60) = person
    14 = 90 = (128, 64, 128) = road
    15 = 108 = (102, 102, 156) = wall
    16 = 115 = (150, 100, 100) = bridge
    17 = 118 = (70, 130, 180) = sky
    18 = 119 = (107, 142, 35) = vegetation
    19 = 120 = (244, 35, 232) = sidewalk
    20 = 126 = (150, 120, 90) = tunnel
    21 = 153 = (153, 153, 153) = pole, polegroup
    22 = 164 = (190, 153, 153) = fence
    23 = 171 = (180, 165, 180) = guard rail
    24 = 173 = (230, 150, 140) = rail track
    25 = 178 = (250, 170, 30) = traffic light
    26 = 193 = (250, 170, 160) = parking
    27 = 195 = (220, 220, 0) = traffic sign
    28 = 210 = (152, 251, 152) = terrain
    
<p align="justify"> The last one is the output image generated by DeepLab model. The output of model is in [N, class, H, W] format. Eg. for Cityscape dataset we have total 30 classes. So output dimension is [N, 30, 1024, 2048]. After using argmax on each output[0] image we will get classID for each pixel. So the final output image will be [H, W] shape where each pixel will represent class ID. </p>

<p align="center">
  <img src="https://github.com/sanketsalunkhe12/DeepLabV3-Segmentation/blob/main/readme_data/Screenshot%20from%202022-08-02%2015-41-11.png" width="350">
</p>

### Cityscape Dataset:

<p align="justify"> The cityscape dataset is in the following directory structure. The list files contains path of each image. A dataprocessing class Deeplab_Dataset_Creation takes the list of input and target path and return x, y (input, target) tensors which will be feed to dataloader. </p>

    .cityscape dataset
    ├── gtFine           # Target Images (y)        
    │   ├── test          
    │   ├── train       
    │   └── val
    ├── leftImg8bit      # Input Images (x)
    │   ├── test         
    │   ├── train       
    │   └── val
    ├── list
    │   ├── test.lst     # List of path of each test input image    
    │   ├── train.lst    # List of path of each train input and target image
    │   └── val.lst      # List of path of each validation input and target image 
    └── ...

### Custom Dataset:

<p align="justify"> For training Deeplab model on custom dataset, you need to organise all image in similar to above mentioned directory structure. In simple you can store images in following directory structure. After this you need to run <b> <i> ./data_processing/unique_pixel.py </i></b> file which will generate unique pixel values for target images class ID. In DeepLab_Dataset_Creation class replace pixel_values[] array with these generated pixel values. </p>
Also modify <b> num_classes </b> parameter of deeplab_model. 

    .custom dataset
    ├── target           # Target Images (y)        
    │   ├── test          
    │   ├── train       
    │   └── val
    ├── input      # Input Images (x)
    │   ├── test         
    │   ├── train       
    │   └── val
    ├── list
    │   ├── test.lst     # List of path of each test input image    
    │   ├── train.lst    # List of path of each train input and target image
    │   └── val.lst      # List of path of each validation input and target image 
    └── ...
    
### Training:

<p align="justify"> For training process either you run <b> <i> train.py </i> </b> file with changes in path locations or can use <b> <i> deeplab_train_test.ipynb </i> </b> notebook which is a complete set of implementation. </p>

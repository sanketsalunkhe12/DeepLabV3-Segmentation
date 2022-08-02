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

<p align="justify"> For DeepLabV3 model or any other segmentation model the input is a simple RBG or Grayscale image with shape [H, W, C], where H: height, W: width and C: no of channels. In general for segmentation the input image should be in <b>[N, C, H, W] </b> shape, where N is the number of images in a batch. During data processing we need to use moveaxis  </p>

### Cityscape Dataset:

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


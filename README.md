# Pix2pix_BlueBias

Under construction...  
This is the repository for the paper "Can DNN models simulate appearance variations of #TheDress?"

## Color-name labeling
We used the color-name labeling algorithm proposed from Okubo et al., (2023).  
Blue-bias labeling: Labeling/ColorLabeling_BB.py  
Non-blue-bias labeling: Labeling/ColorLabeling_nonBB.py  
Dependencies:  
Opencv-python  >= 4.6.0.66
Numpy  >= 1.19.5
Scilit-learn  >= 1.5.4

## The pix2pix model
We used the PyTroch version of [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master) (Isola et al. 2017)  
The usage is the same as the original instruction.   

## Pre-trained models.
Examples of our pre-trained model across each blue bias image ratio: Link (TBA)  
BB_[BB_ratio]: The models learned the blue-bias scene. [BB_ratio] indicates the ratio of blue bias images in the dataset.  
B_[B_ratio]_Y_[Y_ratio]: The models learned color constancy varying the ratio of blue/yellow color shifts.  [B_ratio] and [Y_ratio] indicates the ratio of blue shift images and that of yellow shift images, respectively.   
All images were selected from MSCOCO (Lin et al., 2014).  

## Training datasets
We provide training datasets with different ratios in blue bias images: Link  (TBA)

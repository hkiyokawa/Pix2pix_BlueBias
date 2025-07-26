# Pix2pix_BlueBias

This is the repository for the paper "Can DNN models simulate appearance variations of #TheDress?"

## Color-name labeling
We used the color-name labeling algorithm proposed by Okubo et al. (2023).  
Blue-bias labeling: Labeling/ColorLabeling_BB.py  
Non-blue-bias labeling: Labeling/ColorLabeling_nonBB.py  
Some examples are shown in Labeling_examples.pdf (The same as in the supplementary materials in our paper).  
Dependencies:  
Opencv-python  >= 4.6.0.66  
Numpy  >= 1.19.5  
Scilit-learn  >= 1.5.4  

## The pix2pix model
We used the PyTroch version of the [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master) (Isola et al. 2017)  
The usage is the same as instructed in the original page.   

## Pre-trained models
Examples of our pre-trained model across each blue bias image ratio: [Link to OSF page](https://osf.io/p8g9y/?view_only=2833c524b60446138dbb82579b4a5c27)  
BB_[BB_ratio]: The models learned the blue-bias scene. [BB_ratio] indicates the ratio of blue bias images in our training datasets.  
B_[B_ratio]_Y_[Y_ratio]: The models learned color constancy varying the ratio of blue/yellow color shifts.  [B_ratio] and [Y_ratio] indicates the ratio of blue-shifted images and that of yellow-shifted images, respectively.   

## Training datasets
Our training data was a set of 600 images selected from the [MS COCO dataset](https://cocodataset.org/#home).

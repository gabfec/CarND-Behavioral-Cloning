# **Behavioral Cloning**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./examples/nvidia_model.png "Model Visualization"
[image2]: ./examples/cameras.png "Cameras"

---

### Overview

The goal of the project is to build a Convolutional Neural Network (CNN) to drive a car in a simulator.

These are the steps to  follow:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



---
### Project files

My project includes the following files:

* **model.py** Script to create and train the model
* **drive.py**  Script for driving the car in autonomous mode
* **model.h5** Contains the trained convolution neural network 
* **README.md** The writeup report


Using the Udacity provided simulator (https://github.com/udacity/self-driving-car-sim) and the drive.py file, the car can be driven autonomously around the track by executing: 
```sh
python drive.py model.h5
```
---
### Model Architecture

The CNN model is based on the Nvidia research paper "End to End Learning for Self-Driving Cars".

![alt text][image1]

Since the input images have a different size (160x320) and the  top and buttom part of the images don't provide valuable information,  a cropping layer was added. This could have been done before feeding the data into the training process, but it's more efficient to do it in the keras pipeline because it will take advantage of GPU.

I used RELU activation to introduce nonlinearity.

In order to reduce overfitting I added a dropout layer in the fully connected layers with probability of 50%.

The model uses an adam optimizer, so the learning rate has not to be manually tuned.


### Training details

After several attempt to collect my own data I realized this was one of the main challenges. Although I was somehow able to drive on the first track (at very low speed), the second track was nearly impossible. 
Finnally I considered to use the provided samples since I needed a reliable reference in order to test my implementation and later on to expand it with my own set or with some augmentation techniques.

I split the samples data set into a training and validation set, using the rule of thumb of 80:20.

Also, in order to ensure there is no bias towards left or right steering, each sample is complemented by a flipped version.

Since every sample contains 3 images (left, center, right), the left and right images have been used to augment the data set. For these 2 additional pictures the steering angle was computed based on the measured steering angle +/- a correction angle (which was one of the tunable parameters).

Here is an example of how these images look like:

![alt text][image2]

So with image flipping and side cameras I get an augmentation factor of 6  which is taken into account in the keras fit function.

I've also experimented some random shifts or brightness change, but the results were not convincing so finally I dropped this idea.

Here is a visualization of the architecture, as displayed during the execution:
	
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	lambda_1 (Lambda)            (None, 160, 320, 3)       0         
	_________________________________________________________________
	cropping2d_1 (Cropping2D)    (None, 66, 320, 3)        0         
	_________________________________________________________________
	conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
	_________________________________________________________________
	conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
	_________________________________________________________________
	conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
	_________________________________________________________________
	conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
	_________________________________________________________________
	conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
	_________________________________________________________________
	flatten_1 (Flatten)          (None, 2112)              0         
	_________________________________________________________________
	dropout_1 (Dropout)          (None, 2112)              0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 100)               211300    
	_________________________________________________________________
	dense_2 (Dense)              (None, 50)                5050      
	_________________________________________________________________
	dense_3 (Dense)              (None, 10)                510       
	_________________________________________________________________
	dense_4 (Dense)              (None, 1)                 11        
	=================================================================
	Total params: 348,219
	Trainable params: 348,219
	Non-trainable params: 0
	_________________________________________________________________


With this the vehicle is able to drive autonomously around the first track and for some part of the second track.


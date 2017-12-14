# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia]: ./images/nvidia-cnn-architecture.png "nvidia Network Architecture"
[center]: ./images/sample-center.jpg "Sample Center Image"
[left]: ./images/sample-left.jpg "Sample Left Image"
[right]: ./images/sample-right.jpg "Sample Right Image"
[flipped]: ./examples/sample-center-flipped.jpg "Sample Flipped Center Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 24 and 64 (model.py lines 63-67) 

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 60). 

#### 2. Attempts to reduce overfitting in the model

<!-- The model contains dropout layers in order to reduce overfitting (model.py lines 65).  -->

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 27). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 73).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and repeated driving over more complex parts of track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the simple model, see how the data improves the behavior and then move to a more complex network.

My first step was to use a very simple network with one connected layer to test if the whole pipeline is working.

It was barely working with the small training data that I've collected. I then addded the augmentation of images by flipping horizontally which improved a bit. Adding left and right views helped a bit as well.

I've played around with number of epochs but this was not helping so clearly the next step was to move to a more complex network.

The NVIDIA network worked wonders and almost got the car to cover the whole track. The final step that made things click was finding the right parameter for steering offset for images from the left and right cameras.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 59-71) consisted of a convolution neural network as found in the NVIDIA paper.

Here is a visualization of the architecture:

![alt text][nvidia]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to 

In order to increase amount of training data by 3x I've used the view from the left and right sides of the car with the offset steering parameter of `0.4`.

Here's an example of the view from the left and right sides.
![alt text][left]
![alt text][right]

I also flipped images and angles thus generating 2x more training data:
![alt text][flipped]

After the collection process, I had `5,129` number of data points. Which after data augmentation made it `30,774`. I then preprocessed this data by cropping off the top 70 pixels and bottom 25 pixels that don't show the road.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 as evidenced by loss: 0.0778 and val_loss: 0.0719. I used an adam optimizer so that manually training the learning rate wasn't necessary.

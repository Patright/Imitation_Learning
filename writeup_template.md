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

[image1]: ./examples/center_2021_05_06_08_55_21_280.jpg


---
### Files Submitted & Code Quality

#### 1. My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 24 and 64 (model.py lines 72-91) 

The model includes RELU layers to introduce nonlinearity (code line 79-83), and the data is normalized in the model using a Keras lambda layer (code line 75). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 86 and 88). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 15). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 94).

#### 4. Appropriate training data

I used the training data provided by udacity and enhanced it by some additional taining data for difficult parts of the road. It was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I added a correction factor of 0.2 to the right and the left views (code line 33).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a sufficient amount of training data. Especially for curved parts of the road and also for parts with different boarder lines. In order to achieve that I doubbled these training images (code line 36-47)

My first step was to use a the convolution neural network model from nVidia. I thought this model might be appropriate because it was designed for this purpose.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (code line 18). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting I introduced dropout layers and I also augmented more training data by flipping and blurring the images (code line 50-52).

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added some more training data of these track parts.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

I decided to use the [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model, so that the final model architecture (model.py lines 73-91) consisted of a convolution neural network with the following layers and layer sizes:

```

Layer (type)                     Output Shape                        
=======================================================
lambda_1 (Lambda)                (None, 160, 320, 3)     
_______________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)                  
_______________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)                 
_______________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)     
_______________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)      
_______________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)      
_______________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)      
_______________________________________________________
flatten_1 (Flatten)              (None, 8448)           
_______________________________________________________
dense_1 (Dense)                  (None, 100)                        
_______________________________________________________
dropout_1 (Dropout)              (None, 100)
_______________________________________________________
dense_2 (Dense)                  (None, 50)                         
_______________________________________________________
dropout_2 (Dropout)              (None, 50)
_______________________________________________________
dense_3 (Dense)                  (None, 10)                         
_______________________________________________________
dense_4 (Dense)                  (None, 1)                          
=======================================================

```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

After the collection process, I had around 75000 number of data points. I then preprocessed this data by normalizing them and cropping the top 50 pixel lines and the bottom 20 pixel lines (model line 77), because the do not include useflull informations.

I finally randomly shuffled the data set and put 20% of the data into a validation set (model line 18).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is a [link to my video result](./output_video.mp4) `output_video.mp4`.

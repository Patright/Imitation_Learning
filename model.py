import csv
import cv2
import imageio
import math
import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Dropout, Convolution2D


########## Read csv file ##########
lines = [line for line in csv.reader(open('data_new/driving_log.csv'))]

########## Split in training and validation data ##########
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

########## Define generator function with data augmentation ##########
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            # Get images
            images_center = [imageio.imread('data_new/' + image[0].strip()) for image in batch_samples]
            images_left = [imageio.imread('data_new/' + image[1].strip()) for image in batch_samples]
            images_right = [imageio.imread('data_new/' + image[2].strip()) for image in batch_samples]
            images = images_center + images_left + images_right
            # Get angles
            angles = [float(line[3]) for line in batch_samples] + [float(line[3])+0.2 for line in batch_samples] + [float(line[3])-0.2 for line in batch_samples]
            assert len(images) == len(angles)
            # Add augmented images
            angles_curved = []
            images_curved = []

            for (angle, image) in zip(angles, images):
                if (angle <= -0.4 or angle >= 0.4):
                    angles_curved.append(angle)
                    images_curved.append(image)

            angles += 14 * angles_curved
            images += 14 * images_curved

            assert len(angles) == len(images)

            # Augment images
            images_flip = [cv2.flip(image, 1) for image in images] # flip images horizontally
            images_blurred = [cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT) for image in images] # apply guassian blur on images
            images_flip_blur = [cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT) for image in images_flip] # apply guassian blur on flipped images
            # Add up all data
            angles_flip = [(-1.0)*angel for angel in angles]
            images_all = images + images_blurred + images_flip + images_flip_blur
            steer_ang_all = angles + angles + angles_flip + angles_flip
            assert len(images_all) == len(steer_ang_all)

            # Convert list to numpy array, shuffle and yield
            X_train = np.array(images_all)
            y_train = np.array(steer_ang_all)
            yield sklearn.utils.shuffle(X_train, y_train)


########## Set batch size ##########
batch_size=32

########## Generate samples ##########
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

########## Build the model ##########
model = Sequential()
# Normalize and mean center the input images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# Crop 50 rows pixels from the top and 20 rows pixels from the bottom of the image
model.add(Cropping2D(cropping=((50,20), (0,0))))
# use nvidia CNN:
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.50))
model.add(Dense(50))
model.add(Dropout(0.50))
model.add(Dense(10))
# Fully connected output layer
model.add(Dense(1))

########## Compile, train and save the model ##########
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                validation_data=validation_generator,
                validation_steps=math.ceil(len(validation_samples)/batch_size),
                epochs=2,
                verbose=1)
model.save('model.h5')

########## Print the training and the validation loss for each epoch ##########
print(history_object.history.keys())
print(history_object.history['loss'])
print(history_object.history['val_loss'])

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
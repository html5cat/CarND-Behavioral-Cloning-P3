import os
import cv2
import csv
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

epochs = 1
correction_dict = {
    0 : 0,
    1: 0.15,
    2: -0.15
}
ch, row, col = 3, 160, 320

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size = 0.2) 

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images, angles = [], []
            for batch_sample in batch_samples:
                for i in range(3):
                    file_path = './data/IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(file_path)
                    if image is not None:
                        angle = float(line[3]) + correction_dict[i]
                        images.append(image)
                        angles.append(angle)
                        images.append(cv2.flip(image, 1))
                        angles.append(angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, 
                    input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
                    samples_per_epoch = len(train_samples) * 6,
                    validation_data = validation_generator, 
                    nb_val_samples = len(validation_samples), 
                    nb_epoch = epochs)

model.save('model.h5')

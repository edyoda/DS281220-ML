# -*- coding: utf-8 -*-

#https://drive.google.com/drive/folders/13teCT5fs0mAnecMpazrKsllPJhAOas8w

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

#INitialize CNN
model = Sequential()

#1. Convolution
model.add(Convolution2D(32, (3,3), input_shape = (128, 128, 3), activation = 'relu'))

#2. Maxpooling
model.add(MaxPooling2D(pool_size = (2,2)))

#3. Flatten
model.add(Flatten())

#4. Full Connection
model.add(Dense(units = 128, activation = 'relu'))

#5.Output
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

#6. Image Augmentation
from keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(rescale = 1./255, shear_range= 0.2, zoom_range=0.2, horizontal_flip = True)

test_gen = ImageDataGenerator(rescale = 1./255)

#create training and test set using above data augmentation

train_data = train_gen.flow_from_directory('dataset/train', target_size = (128, 128),
                                           batch_size = 64, class_mode = 'binary')

test_data = test_gen.flow_from_directory('dataset/val', target_size = (128, 128),
                                           batch_size = 64, class_mode = 'binary')

model.fit_generator(train_data, steps_per_epoch = 90, 
                    epochs = 40, validation_data = test_data)







# -*- coding: utf-8 -*-
"""gender.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/15qKr9PBUwSH0fyczlJRAH8BTT-wJMC04
Install keras


!pip3 install keras

Download data and uncompress"

!wget https://www.dropbox.com/s/df836enoah2mgpu/gender.tgz
!tar xvzf gender.tgz

Import packages"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.optimizers import SGD

from keras.callbacks import LearningRateScheduler as LRS
from keras.preprocessing.image import ImageDataGenerator

batch_size = 100
epochs = 150

"""Model Base Functions"""
def CBGN(model,filters,ishape=0):
  if (ishape!=0):
    model.add(Conv2D(filters, (3, 3), padding='same',
                 input_shape=ishape))
  else:
    model.add(Conv2D(filters, (3, 3), padding='same'))

    
  model.add(BN())
  model.add(GN(0.3))
  model.add(Activation('relu'))

  model.add(Conv2D(filters, (3, 3), padding='same'))
  model.add(BN())
  model.add(GN(0.3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  return model

"""Load and prepare data"""

# Load 
x_train = np.load('../datasets/gender/x_train.npy')
x_test = np.load('../datasets/gender/x_test.npy')

y_train = np.load('../datasets/gender/y_train.npy')
y_test = np.load('../datasets/gender/y_test.npy')

# Stats
print(x_train.shape)
print(y_train.shape)
print(sum(y_train == 1))
print(sum(y_train == 0))
print("{:.4f}".format(sum(y_train == 1)/y_train.shape[0]))
print(x_test.shape)
print(y_test.shape)
print(sum(y_test == 1))
print(sum(y_test == 0))
print("{:.4f}".format(sum(y_test == 1)/y_test.shape[0]))


## View some images
# plt.imshow(x_train[0,:,:,: ] )
# plt.show()


## Transforms
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


x_train /= 255
x_test /= 255

print(x_train.shape)
print(x_test.shape)


num_classes = 2

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""Create model, data augmentation etc..."""

## DEFINE A DATA AUGMENTATION GENERATOR

datagen = ImageDataGenerator(
  featurewise_center=True,
  featurewise_std_normalization=True,
  width_shift_range=0.2,
  height_shift_range=0.2,
  rotation_range=20,
  zoom_range=[1.0,1.2],
  horizontal_flip=True)


###########################################################
# Now this is necessary due to the feature normalization: #
datagen.fit(x_train)

testdatagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
)

testdatagen.fit(x_train)
###########################################################

"""Model Creation and construction"""

model = Sequential()

model=CBGN(model,32,x_train.shape[1:])
model=CBGN(model,64)
model=CBGN(model,128)

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.summary()


## OPTIM AND COMPILE
opt = SGD(lr=0.1, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# DEFINE A LEARNING RATE SCHEDULER
def scheduler(epoch):
    if epoch < 50:
        return .1
    elif epoch < 100:
        return 0.01
    else:
        return 0.001

set_lr = LRS(scheduler)


## TRAINING with DA and LRA
history=model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                            steps_per_epoch=len(x_train) / batch_size, 
                            epochs=epochs,
                            validation_data=testdatagen.flow(x_test, y_test),
                            callbacks=[set_lr],
                            verbose=1)


## TEST
# scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:48:50 2020

@author: barkaoui
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# chargement de la base de dones 
(X_train,y_train),(X_test, y_test)=cifar10.load_data()

plt.imshow(X_train[1])


# normaliser les photeau
X_train=X_train/255
X_test=X_test/255

#model
a=X_train[:,1].shape
model=Sequential()
model.add(Conv2D(255,(3,3),input_shape=X_train[:,1].shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

    # Ajouter une autre couche de convolution 
model.add(Conv2D(255,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

      #flatning
model.add(Flatten())
     #Le completement connecté
model.add(Dense(64))
model.add(Dense(10))
model.add(Activation('sigmoid'))


#Optimistaion


model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['sparse_categorical_accuracy'])


#Entraiment
model.fit(X_train, y_train,batch_size=32,epochs=10)


test_loss,test_accuracy=model.evaluate(X_test, y_test)



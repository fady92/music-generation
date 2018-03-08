# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 00:26:38 2017

@author: Antonia Mouawad and Fady Baly
"""
from featExtractUtils import extFeat
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

train = True; path = "data/train/*.mid"
X_train, Y_train = extFeat(path, train)

train = False; path = "data/test/*.mid"
X_test, Y_test = extFeat(path, train)
del train, path

model = Sequential()
model.add(Convolution2D(32, (3, 3), strides=(1,1), activation='relu', input_shape=(1,12,16)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)
#score = model.evaluate(X_test, Y_test, verbose=0)
prediction = model.predict(X_test)
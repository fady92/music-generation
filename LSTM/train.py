# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 00:26:38 2017

@author: Antonia Mouawad and Fady Baly
"""
import keras
from feature_extraction_clean import roll
import data_utils_train
import numpy as np
import time;
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
import matplotlib.pyplot as plt
from IPython.display import clear_output
from keras.layers import Dropout

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
plot_losses = PlotLosses()



train = True
mel_roll, chord_roll = roll(train)

double_chord_roll = data_utils_train.doubleRoll(chord_roll)
double_mel_roll = data_utils_train.doubleRoll(mel_roll)
input_data, target_data = data_utils_train.createNetInputs(double_mel_roll,double_chord_roll, 256)

input_data = input_data.astype(np.bool)
target_data = target_data.astype(np.bool)
del double_chord_roll, double_mel_roll, mel_roll,chord_roll

input_dim = input_data.shape[2]
output_dim = target_data.shape[1]


print("For how many epochs do you wanna train?")
num_epochs = int(input('Num of Epochs:'))
    
print("Choose a batch size:")
print("(Batch size determines how many training samples per gradient-update are used. --> Number of gradient-updates per epoch: Num of samples / batch size)")
batch_size = int(input('Batch Size (recommended=128):'))


print("Network Input Dimension:", input_dim)
print("Network Output Dimension:", output_dim)
print("How many layers should the network have?")
num_layers = int(input('Number of Layers:'))
        
#Building the Network
model = Sequential()
if num_layers == 1:
    print("Your Network:")
    model.add(LSTM(input_dim=input_dim, output_dim=output_dim, activation='softmax', return_sequences=False))
#    model.add(Dropout(0.25))

    print("add(LSTM(input_dim=%d, output_dim=%d, activation='softmax', return_sequences=False))" %(input_dim, output_dim))
elif num_layers > 1:
    print("Enter the number of units for each layer:")
    num_units = []
    for i in range(num_layers-1):
#            units = int(input('Number of Units in Layer %d:' %(i+1)))
        units = 4
        num_units.append(units)
    print()
    print("Your Network:")
    model.add(LSTM(input_dim=input_dim, output_dim=num_units[0], activation='relu', return_sequences=True))
    model.add(Dropout(0.25))

    print("add(LSTM(input_dim=%d, output_dim=%d, activation='tanh', return_sequences=True))" %(input_dim, num_units[0]))
    for i in range(num_layers-2):
        model.add(LSTM(output_dim=num_units[i+1], activation='tanh', return_sequences=True))
        model.add(Dropout(0.25))
        print("add(LSTM(output_dim=%d, activation='tanh', return_sequences=True))" %(num_units[i+1]))
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=output_dim, activation='softmax', return_sequences=False))
    print("add(LSTM(output_dim=%d, activation='softmax', return_sequences=False))" %(output_dim))

print("Compiling your network with the following properties:")
loss_function = 'binary_crossentropy'
optimizer = 'adam'
print("Loss function: ", loss_function)
print("Optimizer: ", optimizer)
print("Number of Epochs: ", num_epochs)
print("Batch Size: ", batch_size)

model.compile(loss=loss_function, optimizer=optimizer)
plot_model(model, to_file='model.png')

print("Training...")
history = data_utils_train.LossHistory()
model_history = model.fit(input_data, target_data, batch_size=batch_size, nb_epoch=num_epochs, callbacks=[plot_losses])
#plt.plot(model_history.history['acc'])


print("Saving model and weights...")
print("Saving weights...")
weights_dir = 'models/models weights/'
weights_file = '%dlayer_%sepochs_%s' %(num_layers, num_epochs, time.strftime("%Y%m%d_%H_%M.h5"))
weights_path = '%s%s' %(weights_dir, weights_file)
print("Weights Path:", weights_path)
model.save_weights(weights_path)

print("Saving model...")
json_string = model.to_json()
model_file = '%dlayer_%sepochs_%s' %(num_layers, num_epochs, time.strftime("%Y%m%d_%H_%M.json"))
model_dir = 'models/models json/'
model_path = '%s%s' %(model_dir, model_file)
print("Model Path:", model_path)
open(model_path, 'w').write(json_string)
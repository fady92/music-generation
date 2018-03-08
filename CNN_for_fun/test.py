# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 03:08:41 2017

@author: Antonia
"""

from featExtractUtils import midi_to_vector, rm_extra_zeros, pitches2chords
import numpy as np
from keras.utils import np_utils
from feature_extraction_clean import roll
import glob

train = True; path = "data/train/*.mid"
mel_roll, chord_roll = roll(train)
del chord_roll, train

#extract chords
midi_fname =  glob.glob(path)
hand_left = []
x = []
target = np.zeros((96,len(midi_fname),4), dtype=np.float64)
for fname,i in zip(midi_fname,range(0,len(midi_fname))):
    #    vectorize midi input
    orig = midi_to_vector(fname) 
    #    extract the beginning of each channel
    indx = rm_extra_zeros(orig)
    #    extract left hand
    hand_left = orig[:,indx[0]:indx[1]]
    x.append(hand_left.shape[1])
    #   shape it to a vertical matrix
    z2 = hand_left.shape
    target[:z2[1],i,:] = np.transpose(hand_left)
target = pitches2chords(target)
        
    #create inputs
images = []
for i in range(0, mel_roll.shape[0]):
    for j in range(0,mel_roll.shape[1],16):
        images.append(mel_roll[i,j:j+16])
images = np.dstack(images)
images = images.transpose((2, 0, 1))
X = images.reshape(images.shape[0], 1, 12, 16)
    
#create targets
chords = []
for i in range(0, target.shape[1]):
    for j in range(0,target.shape[0],2):
        chords.append(target[j:j+1,i,2])  
chords = np.asarray(chords)
Y = chords.reshape(len(chords),)
Y = np_utils.to_categorical(Y, 30)
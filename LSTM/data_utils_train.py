from mido import MidiFile
from mido.midifiles.meta import MetaMessage
from keras.callbacks import Callback
import numpy as np
#np.set_printoptions(threshold=np.nan)
    

def doubleRoll(roll):
    double_roll = []
    for song in roll:
        double_song = np.zeros((roll.shape[1]*2, roll.shape[2]))
        double_song[0:roll.shape[1], :] = song
        double_song[roll.shape[1]:, :] = song
        double_roll.append(double_song)
        
    return np.array(double_roll)
    

def createNetInputs(roll, target, seq_length=256):
    #roll: 3-dim array with Midi Files as piano roll. Size: (num_samples=num Midi Files, num_timesteps, num_notes)
    #seq_length: Sequence Length. Length of previous played notes in regard of the current note that is being trained on
    #seq_length in Midi Ticks. Default is 96 ticks per beat --> 3072 ticks = 8 Bars
    
    X = []
    y = []
    
    for i, song in enumerate(roll):
        pos = 0
        while pos+seq_length < int(song.shape[0]/2)+seq_length:
            sequence = np.array(song[pos:pos+seq_length])
            X.append(sequence)
            y.append(target[i, pos+seq_length])
            pos += 1

    
    return np.array(X), np.array(y)
    

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

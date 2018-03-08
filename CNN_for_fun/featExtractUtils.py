# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 22:07:42 2017

@author: Antonia
"""
import midi
import numpy as np
from keras.utils import np_utils
from feature_extraction_clean import roll
import glob
class MidiContainer:

    def __init__(self):
        self.data = np.array([], dtype=np.int64).reshape(66,0) 
        self.curr_bpm = 120
        self.curr_instrument = 0
        self.abs_time = 0
        self.active_pitches = {} #[pitch] = (velocity, time note was turned on)
    
        
    def note_to_vector(self, pitch, time):
        velocity = self.active_pitches[pitch][0]
        note_start = self.active_pitches[pitch][1]
        return np.vstack((self.curr_instrument, pitch, velocity, time + self.abs_time - note_start))
    
    def add_data(self, note_vector, note_start):
        event_column = np.vstack((note_start, self.curr_bpm, note_vector, np.ones((60,1))*-1))
        self.data = np.hstack((self.data, event_column))

#    
def midi_to_vector(fname):
    '''
    Given a filename of a midi to read, returns a MidiContainer object of its vector representation.
    '''
    pattern = midi.read_midifile(fname)
    midi_vector = MidiContainer()
    for track in pattern:
        midi_vector.abs_time = 0
        for event in track:
            if isinstance(event, midi.SetTempoEvent): 
                midi_vector.curr_bpm = event.get_bpm()
            if isinstance(event, midi.EndOfTrackEvent): #first metadata track
                continue #FADY: here if it is the end of the track which is awesome in our case it simply means 
                           #left hand done or right hand done, it QUITS, i.e it doesnt continue to isinstance
                           #it simply goes to the the next event (next track)
            if isinstance(event, midi.ProgramChangeEvent):
                midi_vector.curr_instrument = event.get_value()
            if isinstance(event, midi.NoteOnEvent):
                midi_vector.abs_time += event.tick
                midi_vector.active_pitches[event.get_pitch()] = (event.get_velocity(), midi_vector.abs_time)
                #print("active_pitches", midi_vector.active_pitches) 
            if isinstance(event, midi.NoteOffEvent):
                pitch = event.get_pitch()
                time = event.tick
                if pitch in midi_vector.active_pitches:
                    note_vec = midi_vector.note_to_vector(pitch, time)
                    midi_vector.add_data(note_vec, midi_vector.active_pitches[pitch][1]) #ugh bad data abstraction
                    midi_vector.active_pitches.pop(pitch, None)
                    midi_vector.abs_time += event.tick
    midi_vector = np.delete(midi_vector.data[:6,:], ([2,4]), axis=0)  
    return midi_vector

def vector_to_midi(vector):
    '''Given a np array (vector.data), returns a python midi pattern'''
    pattern = midi.Pattern(resolution=960)
    #first, separate tracks
    tracklist = {}
    for i in range(2,vector.shape[0],4):
        for j in range(0, vector.shape[1]):
            track = vector[i:i+4,j]
            if np.sum(track) > 0:
                instrument = int(track[0])
                print(instrument)
                if instrument not in tracklist:
                    tracklist[instrument] = np.vstack((vector[0:2,j].reshape(-1,1), track.reshape(-1,1)))
                    print('pop')
                else: 
                    tracklist[instrument] = np.hstack((tracklist[instrument], np.vstack((vector[0:2,j].reshape(-1,1), track.reshape(-1,1)))))


    for k, track in tracklist.items():
        miditrack = midi.Track()
        pattern.append(miditrack)
        bpm = track[1,0]
        tempoevent = midi.SetTempoEvent(tick=0, bpm=bpm)
        miditrack.append(tempoevent)
        instrument = int(track[2,0])
        fontevent = midi.ProgramChangeEvent(tick=0, value=instrument)
        miditrack.append(fontevent)
        if not np.array_equal(bpm * np.ones(track.shape[1]), track[1,:]):
            print('Tempo changes. Code assumes it doesn\'t. Contact Jingyi.')
        
        event_tick = 0   
        track_duration = np.max(track[0,:] + track[-1,:])
        active_notes = {}
        start_times = track[0,:]
        for t in range(0, int(track_duration)+1):
            for pitch in active_notes:
                active_notes[pitch] -= 1
            
            negs = [k for k,v in active_notes.items() if v < 0]
            for n in negs:
                active_notes.pop(n)
                
            while 0 in active_notes.values():
                pitches = [k for k,v in active_notes.items() if v == 0]
                for pitch in pitches:
                    off = midi.NoteOffEvent(tick=t - event_tick, pitch=pitch)
                    miditrack.append(off)
                    active_notes.pop(pitch)
                    event_tick = t
                    
            #run through track to add on/off events
            if t in start_times:
                ni = np.where(t == start_times)
                for n in ni[0]:
                    note = track[:, n]
                    start_time = int(note[0])
                    pitch = int(note[2])
                    velocity = int(note[1])
                    duration = int(note[3])
                    active_notes[pitch] = duration
                    on = midi.NoteOnEvent(tick = t - event_tick, velocity=velocity, pitch=pitch)
                    miditrack.append(on)
                    event_tick = start_time
                    
        miditrack.append(midi.EndOfTrackEvent(tick=0))
    return pattern


def rm_extra_zeros(vectorized_midi):
    #start checking for redundant zeros from the 10th index (just in case there were several zeros from the start)
    # np.where returns the indeces of zero elements from the first row
    x = np.where(vectorized_midi[0,10:]==0)[0]
    # np.isclose return true or false states in case of the existance of close values depending on a custom tolerance value
    close_indeces = np.isclose(x, x[0], atol=10)
    #x[close_indeces][0] takes only the first 'true' value.. the +10 is to return the indeces to normal
    x = np.append(x[close_indeces][0], x[~close_indeces][0])+10
    return x

def pitches2chords(chords):
    '''
    Function that takes target (left hand), squishes every 3 rows into one 
    (as the chords in our project are all triads), and outputs a matrix
    which has the letter representation of the chords
    '''
    num_songs = chords.shape[1]
    f = (int) (chords.shape[0]/3)
    squished_chords = np.empty((f, num_songs,chords.shape[2]))
    for j in range(num_songs):
        current_chord = chords[:,j,2]%12
        for i in range (int(len(current_chord)/3)):
            squished_chords[i,j]= chords[i,j]
            k = (current_chord[i*3: (i*3)+3]).astype(int).tolist()
            k.sort()
            squished_chords[i,j,2]= chords_dict[repr(k)]
#            print(squished_chords[i,j,2])
    return squished_chords

chords_dict = {repr([0, 4, 7]): 1, repr([1, 5, 8]): 2 , 
               repr([2, 6, 9]): 3, repr([3, 7, 10]): 4, repr([4, 8, 11]): 5, 
               repr([0, 5, 9]): 6, repr([1, 6, 10]): 7,
               repr([2, 7, 11]): 8, repr([0, 3, 8]): 9,
               repr([1, 4, 9]): 10, repr([2, 5, 10]): 11, repr([2, 5, 11]):12, repr([3, 6, 11]): 13,
               repr([0, 3, 7]): 14, repr([1, 4, 8]): 15, 
               repr([2, 5, 9]): 16, repr([3, 6, 10]): 17, repr([4, 7, 11]): 18, 
               repr([0, 5, 8]): 19, repr([1, 6, 9]): 20, 
               repr([2, 7, 10]): 21, repr([3, 8, 11]): 22,          
               repr([0, 4, 9]): 23, repr([1, 5, 10]): 24, repr([2, 6, 11]): 25, repr([2,5,8]):1444, #hay mdre shu, 
               repr([0, 0, 0]): 0,repr([4, 5, 9]): 26, repr([0, 7, 9]): 27, repr([0, 2, 7]): 28, repr([0, 5, 7]): 29}

# take a list of 3 notes and generate chord
## Major scales                                 ## Minor scales
# C = [0, 4, 7]                                 # c = [0, 3, 7]
# Csh = [1, 5, 8]                               # csh = [1, 4, 8]
# D = [2, 6, 9]                                 # d = [2, 5, 9]                                
# Dsh = [3, 7, 10]                              # dsh = [3, 6, 10]
# E = [4, 8, 11]                                # e = [4, 7, 11]
# F = [0, 5, 9]                                 # f = [0, 5, 8]
# Fsh = [1, 6, 10]                              # fsh = [1, 6, 9]
# G = [2, 7, 11]                                # g = [2, 7, 10]
# Gsh = [0, 3, 8]                               # gsh = [3, 8, 11]
# A = [1, 4, 9]                                 # a = [0, 4, 9]
# Ash = [2, 5, 10]                              # ash = [1, 5, 10]
# B = [3, 6, 11]                                # b = [2, 6, 11]

# Create a dictionary of the keys
# Bingo it works
#Chords_dict = {repr([0, 4, 7]): 'C', repr([1, 5, 8]): 'Csh' , 
#               repr([2, 6, 9]): 'D', repr([3, 7, 10]): 'Dsh', repr([4, 8, 11]): 'E', 
#               repr([0, 5, 9]): 'F', repr([1, 6, 10]): 'Fsh',
#               repr([2, 7, 11]): 'G', repr([0, 3, 8]): 'Gsh',
#               repr([1, 4, 9]): 'A', repr([2, 5, 10]): 'Ash', repr([3, 6, 11]): 'B',
#               repr([0, 3, 7]): 'c', repr([1, 4, 8]): 'csh', 
#               repr([2, 5, 9]): 'd', repr([3, 6, 10]): 'dsh', repr([4, 7, 11]): 'e', 
#               repr([0, 5, 8]): 'f', repr([1, 6, 9]): 'fsh', 
#               repr([2, 7, 10]): 'g', repr([3, 8, 11]): 'gsh',          
#               repr([0, 4, 9]): 'a', repr([1, 5, 10]): 'ash', repr([2, 6, 11]): 'b', 
#               repr([0, 0, 0]): 'NA'}




def extFeat(path, train):
    mel_roll, chord_roll = roll(train)
    del chord_roll, train

    #extract chords
    midi_fname =  glob.glob(path)
    hand_left = []
    target = np.zeros((96,len(midi_fname),4), dtype=np.float64)
    for fname,i in zip(midi_fname,range(0,len(midi_fname))):
        #    vectorize midi input
        orig = midi_to_vector(fname) 
        #    extract the beginning of each channel
        indx = rm_extra_zeros(orig)
        #    extract left hand
        hand_left = orig[:,indx[0]:indx[1]]
        #   shape it to a vertical matrix
        z2 = hand_left.shape
        target[:z2[1],i,:] = np.transpose(hand_left)
    target = pitches2chords(target)
        
#    #create inputs
#    images = []
#    for i in range(0, mel_roll.shape[0]):
#        for j in range(0,mel_roll.shape[1],16):
#            images.append(mel_roll[i,j:j+16])
#    images = np.dstack(images)
#    images = images.transpose((2, 0, 1))
#    X = images.reshape(images.shape[0], 1, 12, 16)
#    
#    #create targets
#    chords = []
#    for i in range(0, target.shape[1]):
#        for j in range(0,target.shape[0],2):
#            chords.append(target[j:j+1,i,2])  
#    chords = np.asarray(chords)
#    Y = chords.reshape(len(chords),)
#    Y = np_utils.to_categorical(Y, 30)
#    return X, Y
    return target

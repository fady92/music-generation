# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:51:56 2017

@author: Antonia Mouawad and Fady Baly
"""
import midi
import glob
from mido import MidiFile
from mido.midifiles.meta import MetaMessage
import numpy as np

directory = "data/"

def split_left_right (fname,i,train):
    '''This function splits the midi files fed to it into left (chords) and right (melody),
        We ignore the bass (channel 3), we only care about the melody(Channel 2) and chords (Channel 1)'''   
    if train == True:    
        pattern = midi.read_midifile(fname)
        right_hand = pattern[0:2]
        left_hand = midi.Pattern(resolution=1024)
        left_hand.append(pattern[2])
        midi.write_midifile("data/split/train_right/"+'right'+str(i)+'.mid', right_hand)
        midi.write_midifile("data/split/train_left/"+'left'+str(i)+'.mid',left_hand)
    
    else:
        pattern = midi.read_midifile(fname)
        right_hand = pattern[0:2]
#        left_hand = midi.Pattern(resolution=1024)
#        left_hand.append(pattern[2])
        midi.write_midifile("data/split/test_right/"+'right'+str(i)+'.mid', right_hand)
#        midi.write_midifile("Split_data/Left/"+'Left'+str(i)+'.mid',left_hand)

def getPitchRangeAndTicks(files_dir):
    ticks = []
    notes = []
    t_update = 0
    for file_dir in files_dir:
        file_path = "%s" %(file_dir)
        mid = MidiFile(file_path)                        
        for track in mid.tracks: 
            num_ticks = 0    
            for message in track:
                m = message.time
                if not isinstance(message, MetaMessage):
                    m_note, m_time = cleanMessagePitchTicks(message)
                    notes.append(m_note)       
                    num_ticks += m_time 
                    if not m == 0:
                        t_update = int ((message.time * 256 - m)/4)
            ticks.append(num_ticks+t_update)
    return min(notes), max(notes), max(ticks)

def cleanMessagePitchTicks(message):
    while message.note<60:
        message.note = message.note +12
    while message.note>71:
        message.note = message.note -12
    if not message.time == 0: 
        if message.time < 252 or message.type =='note_on':
            k = message.time % 256
            message.time = int (k/4) + int(message.time/128)

        else:
            message.time = int (message.time/256) + 1

    return message.note, message.time


def Midi2PianoRoll(files_dir, ticks, lowest_note, highest_note):
    num_files = len(files_dir)        
    piano_roll = np.zeros((num_files, ticks, highest_note-lowest_note+1), dtype=np.float32)

    for j, file_dir in enumerate(files_dir):
        file_path = "%s" %(file_dir)
        mid = MidiFile(file_path)
        pitch_time_onoff_array = []  
        pitch_on_length_array = []
                
        for track in mid.tracks: 
            curr_time = 0    
            for message in track:
                if not isinstance(message, MetaMessage):
                    m_note, m_time = cleanMessagePitchTicks(message)
                    curr_time += m_time 
                    if message.type == 'note_on':
                        note_onoff = 1
                    elif message.type == 'note_off':
                        note_onoff = 0
                    else:
                        print("Error!")                    
                    pitch_time_onoff_array.append([message.note, curr_time, note_onoff])
#                    print(pitch_time_onoff_array)
                    
        for i, message in enumerate(pitch_time_onoff_array):
            if message[2] == 1: #if note type is 'note_on'
                start_time = message[1]
                for event in pitch_time_onoff_array[i:]: #go through array and look for, when the current note is getting turned off
                    if event[0] == message[0] and event[2] == 0:
                        length = event[1] - start_time
                        break
                    
                pitch_on_length_array.append([message[0], start_time, length])
#                print(pitch_on_length_array)
        for message in pitch_on_length_array:
            piano_roll[j, message[1]:(message[1] + message[2]), message[0]-lowest_note] = 1
    return piano_roll


def roll(train):
    chord_dir = "data/split/train_left/"
    mel_dir = "data/split/train_right/"
    test_dir = "data/split/test_right/"
    
    if train == True:
        midi_fname =  glob.glob("data/train/*.mid")
        for fname,i in zip(midi_fname,range(0,len(midi_fname))):
            split_left_right(fname,i,train)
            
            chord_train_files = glob.glob("%s*.mid" %(chord_dir))
            mel_train_files = glob.glob("%s*.mid" %(mel_dir))
            
            #Preprocessing: Get highest and lowest notes + maximum midi_ticks overall midi files
            chord_lowest_note, chord_highest_note, chord_ticks = getPitchRangeAndTicks(chord_train_files)
            mel_lowest_note, mel_highest_note, mel_ticks = getPitchRangeAndTicks(mel_train_files)
            
            mel_roll = Midi2PianoRoll(mel_train_files, mel_ticks, mel_lowest_note, mel_highest_note)
            chord_roll = Midi2PianoRoll(chord_train_files, chord_ticks, chord_lowest_note, chord_highest_note)
        return mel_roll,chord_roll
            
    else:
        midi_fname =  glob.glob("data/split/test_right/*.mid")
        for fname,i in zip(midi_fname,range(0,len(midi_fname))):
            split_left_right(fname,i,train)
            
            mel_train_files = glob.glob("%s*.mid" %(test_dir))
            
            #Preprocessing: Get highest and lowest notes + maximum midi_ticks overall midi files
            mel_lowest_note, mel_highest_note, mel_ticks = getPitchRangeAndTicks(mel_train_files)
            mel_roll = Midi2PianoRoll(mel_train_files, mel_ticks, mel_lowest_note, mel_highest_note) 
            
        return mel_roll

    

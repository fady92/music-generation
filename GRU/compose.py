
from feature_extraction_clean import roll
import data_utils_compose
from keras.models import model_from_json
import numpy as np
import glob
from os import listdir
import data_utils_train 

train = False
mel_roll = roll(train)

np.set_printoptions(threshold=np.nan) #Comment that line out, to print reduced matrices

mel_dir = 'data/split/test_right/'
composition_dir = 'data/split/test_left/'

mel_files = glob.glob("%s*.mid" %(mel_dir))

composition_files = []
for i in range(len(mel_files)):
    composition_files.append('%d' %(i+1))

mel_lowest_note = 60

resolution_factor = 12#int(input('Resolution Factor (recommended=12):')) #24: 1/8 Resolution, 12: 1/16 Resolution, 6: 1/32 Resolution

mel_roll = roll(train)
double_mel_roll = data_utils_train.doubleRoll(mel_roll)
test_data = data_utils_compose.createNetInputs(double_mel_roll, 256)

batch_size = 128
thresh = float(input('Threshold (recommended ~ 0.1):'))


print("Loading Model and Weights...")
#Load model file
model_dir = 'models/models json/'
model_files = listdir(model_dir)
print("Choose a file for the json model file:")
print("---------------------------------------")
for i, file in enumerate(model_files):
    print(str(i) + " : " + file)
print("---------------------------------------")
file_number_model = int(input('Type in the number in front of the file you want to choose:')) 
model_file = model_files[file_number_model]
model_path = '%s%s' %(model_dir, model_file)

#Load weights file
weights_dir = 'models/models weights/'
weights_files = listdir(weights_dir)

print("Choose a file for the weights (Model and Weights MUST correspond!):")
print("---------------------------------------")
for i, file in enumerate(weights_files):
    print(str(i) + " : " + file)
print("---------------------------------------")
file_number_weights = int(input('Type in the number in front of the file you want to choose:')) 
weights_file = weights_files[file_number_weights]
weights_path = '%s%s' %(weights_dir, weights_file)


print("loading model...")
model = model_from_json(open(model_path).read())
print("loading weights...")
model.load_weights(weights_path)
print("Compiling model...")
model.compile(loss='binary_crossentropy', optimizer='adam')

print("Compose2...")
net_output = []
net_roll = []
for i, song in enumerate(test_data):
    net_output.append(model.predict(song))
    net_roll.append(data_utils_compose.NetOutToPianoRoll(net_output[i], threshold=thresh))
    data_utils_compose.createMidiFromPianoRoll(net_roll[i], mel_lowest_note, composition_dir,
                                               composition_files[i], thresh)

orig = glob.glob('data/test/*.mid')
composed = glob.glob('data/split/test_left/*.mid')
for i,j in zip(orig,composed):
    data_utils_compose.merge_left_right(i,j)


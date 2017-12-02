# Larger LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 20
dataX = []
dataY = []
for i in range(0, n_chars - 10*seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length: i + 2*seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append([char_to_int[char] for char in seq_out])

n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
n_output_patterns = len(dataY)
X = numpy.reshape(dataX, (n_patterns, 1, seq_length))
# normalize
X = X / float(n_vocab)

Y = numpy.reshape(dataY, (n_output_patterns, seq_length))
Y = Y / float(n_vocab)
# define the LSTM model
model = Sequential()
model.add(LSTM(seq_length, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(seq_length))
#model.compile(loss='categorical_crossentropy', optimizer='adam')
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

# define the checkpoint
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, Y, epochs=10, batch_size=128, callbacks=callbacks_list)

#Generate stuff
model.load_weights(filepath)
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
x = numpy.reshape(pattern, (1, 1, len(pattern)))
x = x / float(n_vocab)
prediction = model.predict(x, verbose=0)
prediction = prediction[0]
prediction_string = prediction*float(n_vocab)
print ("Prediction is: ", prediction_string)

#for i in range(20):
#        x = numpy.reshape(pattern, (1, len(pattern), 1))
#        x = x / float(n_vocab)
#        prediction = model.predict(x, verbose=0)
#        index = numpy.argmax(prediction)
#        result = int_to_char[index]
#        seq_in = [int_to_char[value] for value in pattern]
#        sys.stdout.write(result)
#        pattern.append(index)
#        pattern = pattern[1:len(pattern)]
#print ("\nDone.")

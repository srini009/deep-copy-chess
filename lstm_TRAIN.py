import numpy, os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from bidict import bidict

def read_and_concat_all_games(filepaths):
	game_list = []
	for file_ in filepaths[0:1]:
		f = open(file_)
		game_list.extend(f.read().split("\n"))
		game_list.pop()
	return "".join(game_list)

def read_files(path):
	filelist = os.listdir(path)
	filepaths = [path + "/" + x for x in filelist]
	return read_and_concat_all_games(filepaths)

def create_bidict(individual_moves):
	b = bidict()
	unique_moves = set(individual_moves)
	print("Num unique moves: ", len(unique_moves))

	for index, move in enumerate(unique_moves):
		b[move] = index
		b.inv[index] = move
	return b

raw_text = read_files("data/converted/elo2000")
individual_moves = [raw_text[i:i+64] for i in range(0, len(raw_text), 64)]

# summarize the loaded data)
n_chars = len(raw_text)
n_individual_moves = len(individual_moves)
#Artifical limit
n_individual_moves = 5
#n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Moves: ", n_individual_moves)

#Basic sanity for data
#if (n_chars != n_individual_moves*64):
#	print ("Number of characters is not divisible by 64. Check data!")
#	raise

move_bidict = create_bidict(individual_moves)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
for i in range(0, n_individual_moves - seq_length, 1):
	seq_in = individual_moves[i:i + seq_length]
	seq_out = individual_moves[i + seq_length]
	dataX.append([move_bidict[move] for move in seq_in])
	dataY.append(move_bidict[seq_out])

n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(len(move_bidict))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(5, input_shape=(X.shape[1], X.shape[2])))
#model.add(Dropout(0.2))
#model.add(LSTM(100))
#model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=1000, batch_size=1, callbacks=callbacks_list)

#Generate
pattern = dataX[0]
print ("Selected input pattern ID and string: ", pattern, move_bidict.inv[pattern[0]])

x = numpy.reshape(pattern, (1, len(pattern), 1))
x = x / float(len(move_bidict))
prediction = model.predict(x, verbose=0)
index = numpy.argmax(prediction)
result = move_bidict.inv[index]
print ("Predicted board: ", result)

print ("\nDone.")

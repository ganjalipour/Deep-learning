from keras.datasets import imdb

max_words = 20000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

print("X_train length: ", len(x_train))
print("X_test length: ", len(x_test))

word_to_index = imdb.get_word_index()
index_to_word = {v: k for k, v in word_to_index.items()}

print(x_train[0])
print(" ".join([index_to_word[x] for x in x_train[0]]))

print("Min value:", min(y_train), "Max value:", max(y_train))

import numpy as np

average_length = np.mean([len(x) for x in x_train])
median_length = sorted([len(x) for x in x_train])[len(x_train) // 2]

print("Average sequence length: ", average_length)
print("Median sequence length: ", median_length)

max_sequence_length = 180

from keras.preprocessing import sequence

x_train = sequence.pad_sequences(x_train, maxlen=max_sequence_length, padding='post', truncating='post')
x_test = sequence.pad_sequences(x_test, maxlen=max_sequence_length, padding='post', truncating='post')

print('X_train shape: ', x_train.shape)

from keras.models import Sequential

from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense

# Single layer LSTM example

hidden_size = 32

sl_model = Sequential()
sl_model.add(Embedding(max_words, hidden_size))
sl_model.add(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
sl_model.add(Dense(1, activation='sigmoid'))
sl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 2

sl_model.fit(x_train, y_train, epochs=epochs, shuffle=True)
loss, acc = sl_model.evaluate(x_test, y_test)

d_model = Sequential()
d_model.add(Embedding(max_words, hidden_size))
d_model.add(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
d_model.add(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
d_model.add(Dense(1, activation='sigmoid'))
d_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

d_model.fit(x_train, y_train, epochs=epochs, shuffle=True)
d_loss, d_acc = d_model.evaluate(x_test, y_test)
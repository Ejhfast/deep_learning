from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
import itertools as it

batch_size = 10

def each_cons(xs, n):
	return it.izip(*(it.islice(g, i, None) for i, g in enumerate(it.tee(xs, n))))

def flat_repeat(lst,n):
    return list(it.chain.from_iterable(it.repeat(lst,n)))

in_str = list("hello hello hello hello ")

chars = list(set(in_str))
char2index = {c:i for i,c in enumerate(chars)}
index2char = {i:c for c,i in char2index.iteritems()}

train_x = [[char2index[x[0]], char2index[x[1]]] for x in each_cons(in_str,3)]
train_y = [char2index[x[2]] for x in each_cons(in_str,3)]

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x, test_y = train_x, train_y

max_features = len(chars)

test_y = np_utils.to_categorical(test_y, max_features)
train_y = np_utils.to_categorical(train_y, max_features)

model = Sequential()
model.add(Embedding(max_features, 3))
model.add(LSTM(3, max_features))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adagrad')

model.fit(train_x, train_y, nb_epoch=1000, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1, show_accuracy=True)

print('Test score:', score[0])
print('Test accuracy:', score[1])

sample_len = 100
out_str = ""
seed = np.array([[char2index[x] for x in ['h','e']]])

for _ in range(0,sample_len):
    p = model.predict_classes(seed, batch_size=1, verbose=0)
    out_str += str(index2char[p[0]])
    seed = np.array([[seed[0][-1],p[0]]])

print(out_str)
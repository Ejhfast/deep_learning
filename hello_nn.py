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
import sys

batch_size = 10

def each_cons(xs, n):
	return it.izip(*(it.islice(g, i, None) for i, g in enumerate(it.tee(xs, n))))

def flat_repeat(lst,n):
    return list(it.chain.from_iterable(it.repeat(lst,n)))

in_str = list("hello there hello there hello there ")

if(len(sys.argv)>1):
	with open (sys.argv[1], "r") as myfile:
		in_str=list(myfile.read())


chars = list(set(in_str))
max_features = len(chars)
char2index = {c:i for i,c in enumerate(chars)}
index2char = {i:c for c,i in char2index.iteritems()}

# Note: may want ALL POSSIBLE up to this length
seq_len = 3

train_x = [[char2index[y] for y in x[:seq_len]] for x in each_cons(in_str,seq_len+1)]
train_y = [char2index[x[seq_len]] for x in each_cons(in_str,seq_len+1)]

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x, test_y = train_x, train_y

test_y = np_utils.to_categorical(test_y, max_features)
train_y = np_utils.to_categorical(train_y, max_features)

embedding_size = 2

model = Sequential()
model.add(Embedding(max_features, embedding_size))
model.add(LSTM(embedding_size, max_features))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adagrad')

model.fit(train_x, train_y, nb_epoch=1, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1, show_accuracy=True)

print('Test score:', score[0])
print('Test accuracy:', score[1])

sample_len = 100
out_str = ""
seed = [char2index[x] for x in ['h']]

for _ in range(0,sample_len):
    p = model.predict_classes(np.array([seed[-seq_len:]]), batch_size=1, verbose=0)
    out_str += str(index2char[p[0]])
    seed.append(p[0])

print(out_str)

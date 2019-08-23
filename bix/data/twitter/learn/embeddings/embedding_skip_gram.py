from typing import List

from keras import Sequential, Input, Model
from keras.layers import Embedding, Flatten, Dense, Dot, Reshape, Activation
from keras_preprocessing.sequence import skipgrams
from keras_preprocessing.text import Tokenizer
import numpy as np

from bix.data.twitter.base.utils import load_pickle
from bix.data.twitter.learn.embeddings.embedding_abstract import EmbeddingAbstract


class EmbeddingSkipGram(EmbeddingAbstract):
    def __init__(self, tokenizer: Tokenizer, padded_x, unpadded_x, max_tweet_word_count: int, vocab_size: int, y: List[int]) -> None:
        super().__init__(tokenizer, padded_x, unpadded_x, max_tweet_word_count, vocab_size, y)

        self.embedding_vector_size = 100
        self.weights = None
        self.grams_x = None
        self.grams_y = None

    def prepare(self):
        self.grams_x = load_pickle('tokenized/learn/grams_x.pickle')#[0:100000]
        self.grams_y = load_pickle('tokenized/learn/grams_y.pickle')#[0:100000]
        pass

    def define_model(self):
        # define model
        # inputs
        w_inputs = Input(shape=(1,), dtype='int32')
        w = Embedding(self.vocab_size, self.embedding_vector_size)(w_inputs)

        # context
        c_inputs = Input(shape=(1,), dtype='int32')
        c = Embedding(self.vocab_size, self.embedding_vector_size)(c_inputs)

        o = Dot(axes=2)([w, c])
        o = Reshape((1,), input_shape=(1, 1))(o)
        o = Activation('sigmoid')(o)

        model = Model(inputs=[w_inputs, c_inputs], outputs=o)
        model.compile(loss='binary_crossentropy', optimizer='adam')

        model.summary()

        self.model = model

    def learn(self):
        #for _ in range(1):  # maybe increase this later
        #loss = 0.
        #for i, gram in enumerate(self.grams_x):
        #data, labels = skipgrams(sequence=doc, vocabulary_size=self.vocab_size, window_size=5,
        #                         negative_samples=5.)
        x = [np.array(x) for x in zip(*self.grams_x)]
        y = np.array(self.grams_y, dtype=np.int32)
        self.model.fit(x, y, epochs=1, batch_size=1024)
        #if i > 2000:  # for debug purposes TODO: remove
        #    break
        #print(loss)

        # fit the model
        # model.fit(padded_docs, labels_int, epochs=5, verbose=0)
        # evaluate the model
        #x = [np.array(x) for x in zip(*self.grams_x[0:1000])]
        #y = np.array(self.grams_y[0:1000], dtype=np.int32)
        #loss, accuracy = self.model.evaluate(x, y, verbose=2)
        #print('Accuracy: %f' % (accuracy * 100))

        weights = self.model.layers[3].get_weights()
        print(f"num: {len(weights)}, dim: {weights[0].shape}")

        self.weights = weights

    def get_weights(self):
        return self.weights


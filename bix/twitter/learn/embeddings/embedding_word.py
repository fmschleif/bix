from typing import List

from keras import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.utils import np_utils
from keras_preprocessing.text import Tokenizer
from numpy import asarray, zeros

from bix.twitter.learn.embeddings.embedding_abstract import EmbeddingAbstract


class EmbeddingWord(EmbeddingAbstract):
    def __init__(self, tokenizer: Tokenizer, padded_x, unpadded_x, max_tweet_word_count: int, vocab_size: int, y: List[int]) -> None:
        super().__init__(tokenizer, padded_x, unpadded_x, max_tweet_word_count, vocab_size, y)

        self.embedding_vector_size = 100
        self.weights = None
        self.enc_y = np_utils.to_categorical(y)

    def prepare(self):
        # nothing
        pass

    def define_model(self):
        # define the model
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_vector_size, input_length=self.max_tweet_word_count))
        model.add(Flatten())
        model.add(Dense(len(self.enc_y[0]), activation='softmax'))
        # compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        # summarize the model
        print(model.summary())

        self.model = model

    def learn(self):
        # fit the model
        self.model.fit(self.x, self.enc_y, epochs=100, verbose=2, batch_size=1024)  # training
        # evaluate the model
        loss, accuracy = self.model.evaluate(self.x, self.enc_y, verbose=2)
        print('Accuracy: %f' % (accuracy * 100))

        # fit the model
        # model.fit(padded_docs, labels_int, epochs=5, verbose=0)
        # evaluate the model
        # loss, accuracy = model.evaluate(encoded_docs, labels_int, verbose=0)
        # print('Accuracy: %f' % (accuracy * 100))
        weights = self.model.layers[0].get_weights()
        print(f"num: {len(weights)}, dim: {weights[0].shape}")

        self.weights = weights

    def get_weights(self):
        return self.weights


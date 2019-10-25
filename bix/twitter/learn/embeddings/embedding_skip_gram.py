from typing import List

from keras import Sequential, Input, Model
from keras.layers import Embedding, Flatten, Dense, Dot, Reshape, Activation, concatenate, merge, dot
from keras_preprocessing.sequence import skipgrams
from keras_preprocessing.text import Tokenizer
import numpy as np

from bix.twitter.base.utils import load_pickle
from bix.twitter.learn.embeddings.embedding_abstract import EmbeddingAbstract


class EmbeddingSkipGram(EmbeddingAbstract):
    vocab_size = -1
    validation_model = None

    def __init__(self, tokenizer: Tokenizer, padded_x, unpadded_x, max_tweet_word_count: int, vocab_size: int, y: List[int]) -> None:
        super().__init__(tokenizer, padded_x, unpadded_x, max_tweet_word_count, vocab_size, y)

        self.embedding_vector_size = 100
        self.weights = None
        self.grams_x = None
        self.grams_y = None
        self.val_model = None

        EmbeddingSkipGram.vocab_size = vocab_size

    def prepare(self):
        #self.grams_x = load_pickle('tokenized/learn/grams_x.pickle')#[0:100000]
        #self.grams_y = load_pickle('tokenized/learn/grams_y.pickle')#[0:100000]
        pass

    def define_model(self):
        # define model
        # inputs
        #w_inputs = Input(shape=(1,), dtype='int32')
        #w = Embedding(self.vocab_size, self.embedding_vector_size)(w_inputs)

        # context
        #c_inputs = Input(shape=(1,), dtype='int32')
        #c = Embedding(self.vocab_size, self.embedding_vector_size)(c_inputs)

        #o = Dot(axes=2)([w, c])
        #o = Reshape((1,), input_shape=(1, 1))(o)
        #o = Activation('sigmoid')(o)

        #model = Model(inputs=[w_inputs, c_inputs], outputs=o)
        #model.compile(loss='binary_crossentropy', optimizer='adam')

        #model.summary()

        # create some input variables
        input_target = Input((1,))
        input_context = Input((1,))

        embedding = Embedding(self.vocab_size, self.embedding_vector_size, input_length=1, name='embedding')
        target = embedding(input_target)
        target = Reshape((self.embedding_vector_size, 1))(target)
        context = embedding(input_context)
        context = Reshape((self.embedding_vector_size, 1))(context)

        # setup a cosine similarity operation which will be output in a secondary model
        #similarity = merge([target, context], mode='cos', dot_axes=0)
        similarity = dot([target, context], axes=1, normalize=True)

        # now perform the dot product operation to get a similarity measure
        dot_product = dot([target, context], axes=1)
        dot_product = Reshape((1,))(dot_product)
        # add the sigmoid output layer
        output = Dense(1, activation='sigmoid')(dot_product)
        # create the primary training model
        model = Model(input=[input_target, input_context], output=output)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        model.summary()

        # create a secondary validation model to run our similarity checks during training
        validation_model = Model(input=[input_target, input_context], output=similarity)

        self.model = model
        self.val_model = validation_model
        EmbeddingSkipGram.validation_model = validation_model

    def learn(self):
        #for _ in range(1):  # maybe increase this later
        #loss = 0.
        #for i, gram in enumerate(self.grams_x):
        #data, labels = skipgrams(sequence=doc, vocabulary_size=self.vocab_size, window_size=5,
        #                         negative_samples=5.)
        x = [np.array(x) for x in zip(*self.grams_x)]
        y = np.array(self.grams_y, dtype=np.int32)
        sim_cb = SimilarityCallback(self.tokenizer)
        #for cnt in range(epochs):
        #    idx = np.random.randint(0, len(labels) - 1)
        #    arr_1[0,] = word_target[idx]
        #    arr_2[0,] = word_context[idx]
        #    arr_3[0,] = labels[idx]
        #    loss = model.train_on_batch([arr_1, arr_2], arr_3)
        #    if cnt % 100 == 0:
        #        print("Iteration {}, loss={}".format(cnt, loss))
        #    if cnt % 10000 == 0:
        #        sim_cb.run_sim()
        self.model.fit(x, y, epochs=10, batch_size=1024, verbose=1)
        #sim_cb.run_sim()
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

        weights = self.model.layers[2].get_weights()
        print(f"num: {len(weights)}, dim: {weights[0].shape}")

        self.weights = weights

    def get_weights(self):
        return self.weights


class SimilarityCallback:
    valid_size = 16
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer: Tokenizer = tokenizer
        self.reverse_dictionary = dict(map(reversed, tokenizer.word_index.items()))

    def run_sim(self):
        for i in range(SimilarityCallback.valid_size):
            valid_word = self.reverse_dictionary[SimilarityCallback.valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(SimilarityCallback.valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((EmbeddingSkipGram.vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(EmbeddingSkipGram.vocab_size):
            in_arr2[0,] = i
            out = EmbeddingSkipGram.validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

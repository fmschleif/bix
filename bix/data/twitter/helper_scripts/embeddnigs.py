from keras import Input, Model
from keras.layers import Embedding, Reshape, dot, Dense
from keras_preprocessing.text import Tokenizer

from bix.data.twitter.base.utils import load_pickle, load_model_mat

import numpy as np

from bix.data.twitter.learn.tokenizer.tokenizer_utils import load_tokenizer

validation_model = None

class SimilarityCallback:
    #valid_size = 16
    valid_window = 100  # Only pick dev samples in the head of the distribution.

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer: Tokenizer = tokenizer
        self.reverse_dictionary = dict(map(reversed, tokenizer.word_index.items()))

    def run_sim(self):
        valid_examples = [self.tokenizer.word_index['love'],
                          self.tokenizer.word_index['like'],
                          self.tokenizer.word_index['hate'],
                          self.tokenizer.word_index['yes'],
                          self.tokenizer.word_index['no'],
                          ]
        for i in range(len(valid_examples)):
            valid_word = self.reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            nearest_p = [sim[e] for e in nearest]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.reverse_dictionary[nearest[k]]
                log_str = '%s %s (%s), ' % (log_str, close_word, nearest_p[k])
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((25000,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(25000):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim


if __name__ == '__main__':
    #data = load_pickle('tokenized/learn/grams_x.pickle')
    #print(len(data))

    t = load_tokenizer('learn')
    model_mat_skip_gram = load_model_mat('embedding_glove')
    print(f'model_mat_skip_gram.shape: {model_mat_skip_gram[0].shape}')


    vocab_size = 25000
    embedding_vector_size = 100

    # create some input variables
    input_target = Input((1,))
    input_context = Input((1,))

    embedding = Embedding(vocab_size, embedding_vector_size, input_length=1, name='embedding', weights=model_mat_skip_gram)
    target = embedding(input_target)
    target = Reshape((embedding_vector_size, 1))(target)
    context = embedding(input_context)
    context = Reshape((embedding_vector_size, 1))(context)

    # setup a cosine similarity operation which will be output in a secondary model
    # similarity = merge([target, context], mode='cos', dot_axes=0)
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

    callb = SimilarityCallback(t)
    callb.run_sim()



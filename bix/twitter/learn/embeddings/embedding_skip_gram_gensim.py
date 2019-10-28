from typing import List

from keras_preprocessing.text import Tokenizer

from bix.twitter.learn.embeddings.embedding_abstract import EmbeddingAbstract
from bix.twitter.learn.tokenizer.tokenizer_utils import load_tokenizer
from gensim.models import Word2Vec
from numpy import asarray, zeros

from bix.twitter.base.utils import load_csv, save_pickle, save_model_mat


class EmbeddingGensimSkipGram(EmbeddingAbstract):

    def __init__(self, tokenizer: Tokenizer, padded_x, unpadded_x, max_tweet_word_count: int, vocab_size: int, y: List[int], texts) -> None:
        super().__init__(tokenizer, padded_x, unpadded_x, max_tweet_word_count, vocab_size, y)

        self.embedding_vector_size = 100
        self.weights = None
        self.grams_x = None
        self.grams_y = None
        self.val_model = None
        self.texts = texts

    def prepare(self):
        pass

    def define_model(self):
        pass

    def learn(self):
        # load embedding as a dict
        def load_embedding(filename):
            # load embedding into memory, skip first line
            file = open(filename, 'r')
            lines = file.readlines()[1:]
            file.close()
            # create a map of words to vectors
            embedding = dict()
            for line in lines:
                parts = line.split()
                # key is string word, value is numpy array for vector
                embedding[parts[0]] = asarray(parts[1:], dtype='float32')
            return embedding

        x = list(self.texts.values())
        t = self.tokenizer

        model = Word2Vec(x, size=100, window=5, max_vocab_size=25000, workers=4, sg=1, negative=5, min_count=0)
        # model.build_vocab(x)

        #model.save("word2vec.model")

        words = model.wv.vocab.keys()
        vocab_size = len(words)
        print("Vocab size", vocab_size)

        # total vocabulary size plus 0 for unknown words
        # vocab_size = len(vocab) + 1
        # define weight matrix dimensions with all 0
        weight_matrix = zeros((self.vocab_size, self.embedding_vector_size))
        # step vocab, store vectors using the Tokenizer's integer mapping
        for word, i in t.word_index.items():
            if i > self.vocab_size: break
            if word in model.wv.vocab.keys():
                weight_matrix[i] = model.wv[word]

        #save_model_mat([weight_matrix], 'embedding_skip_gram')

        # load embedding from file
        # raw_embedding = load_embedding('embedding_word2vec.txt')
        # get vectors in the right order
        # embedding_vectors = get_weight_matrix(raw_embedding, t.word_index)

        weights = [weight_matrix]
        print(f"num: {len(weights)}, dim: {weights[0].shape}")

        self.weights = weights

    def get_weights(self):
        return self.weights



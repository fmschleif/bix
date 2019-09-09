from bix.data.twitter.learn.tokenizer.tokenizer_utils import load_tokenizer
from gensim.models import Word2Vec
from numpy import asarray, zeros

from bix.data.twitter.base.utils import load_csv, save_pickle, save_model_mat


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

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix


#load_csv('learn/lables.csv')
x = load_csv('learn/tweets.csv')
x = [e.split() for e in x]
t = load_tokenizer('learn')


model = Word2Vec(x, size=100, window=5, max_vocab_size=25000, workers=4, sg=1, negative=5, min_count=0)
#model.build_vocab(x)

model.save("word2vec.model")

words = model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", vocab_size)

# total vocabulary size plus 0 for unknown words
#vocab_size = len(vocab) + 1
# define weight matrix dimensions with all 0
weight_matrix = zeros((25000, 100))
# step vocab, store vectors using the Tokenizer's integer mapping
for word, i in t.word_index.items():
    if i > 25000: break
    if word in model.wv.vocab.keys():
        weight_matrix[i] = model.wv[word]

save_model_mat([weight_matrix], 'embedding_skip_gram')

# load embedding from file
#raw_embedding = load_embedding('embedding_word2vec.txt')
# get vectors in the right order
#embedding_vectors = get_weight_matrix(raw_embedding, t.word_index)
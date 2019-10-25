import gc

from bix.twitter.base.utils import save_model_mat, load_pickle
from bix.twitter.learn.embeddings.embedding_glove import EmbeddingGlove
from bix.twitter.learn.embeddings.embedding_skip_gram import EmbeddingSkipGram
from bix.twitter.learn.embeddings.embedding_skip_gram_gensim import EmbeddingGensimSkipGram
from bix.twitter.learn.embeddings.embedding_word import EmbeddingWord
from bix.twitter.learn.tokenizer.tokenizer_utils import load_tokenizer


def learn_embedding_glove(x, y):
    padded_x = x
    tokenizer = load_tokenizer()
    max_tweet_word_count = load_pickle('max_tweet_word_count.pickle')

    print('learning glove...')
    e = EmbeddingGlove(tokenizer, padded_x, None, max_tweet_word_count, tokenizer.num_words, y)
    e.create_embedding()
    weights = e.get_weights()
    save_model_mat(weights, 'embedding_glove')
    return weights


def learn_embedding_word(x, y):
    padded_x = x
    tokenizer = load_tokenizer()
    max_tweet_word_count = load_pickle('max_tweet_word_count.pickle')

    print('learning word...')
    e = EmbeddingWord(tokenizer, padded_x, None, max_tweet_word_count, tokenizer.num_words, y)
    e.create_embedding()
    weights = e.get_weights()
    save_model_mat(weights, 'embedding_word')
    return weights


def learn_embedding_skip_gram(x, y, texts):
    padded_x = x
    tokenizer = load_tokenizer()
    max_tweet_word_count = load_pickle('max_tweet_word_count.pickle')

    print('learning skip_gram...')
    e = EmbeddingGensimSkipGram(tokenizer, padded_x, None, max_tweet_word_count, tokenizer.num_words, y, texts)
    e.create_embedding()
    weights = e.get_weights()
    save_model_mat(weights, 'embedding_skip_gram')
    return weights


def learn_embeddings(x, y, texts):
    glove = learn_embedding_glove(x,y)
    gc.collect()
    word = learn_embedding_word(x,y)
    gc.collect()
    sg = learn_embedding_skip_gram(x,y, texts)
    gc.collect()
    return glove,word,sg

import sys

import gc

from bix.data.twitter.base.utils import save_model_mat, load_training_sentiment_data_small
from bix.data.twitter.learn.embeddings.embedding_glove import EmbeddingGlove
from bix.data.twitter.learn.embeddings.embedding_skip_gram import EmbeddingSkipGram
from bix.data.twitter.learn.embeddings.embedding_word import EmbeddingWord

if __name__ == '__main__':
    args = sys.argv
    print('loading saved state')
    tokenizer, y, padded_x, unpadded_x, max_tweet_word_count = load_training_sentiment_data_small()

    if 'glove' in args:
        print('learning glove...')
        e = EmbeddingGlove(tokenizer, padded_x, unpadded_x, max_tweet_word_count, y)
        e.create_embedding()
        weights = e.get_weights()
        save_model_mat(weights, 'embedding_glove')

    gc.collect()

    if 'word' in args:
        print('learning word...')
        e = EmbeddingWord(tokenizer, padded_x, unpadded_x, max_tweet_word_count, y)
        e.create_embedding()
        weights = e.get_weights()
        save_model_mat(weights, 'embedding_word')

    gc.collect()

    if 'skip_gram' in args:
        print('learning skip_gram...')
        e = EmbeddingSkipGram(tokenizer, padded_x, unpadded_x, max_tweet_word_count, y)
        e.create_embedding()
        weights = e.get_weights()
        save_model_mat(weights, 'embedding_skip_gram')

    gc.collect()

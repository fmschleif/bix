from os import walk
from typing import List, Dict

import numpy
from keras_preprocessing.text import Tokenizer

from bix.twitter.base.utils import generate_hashtag_path, load_csv, remove_duplicates, save_csv, \
    create_path_if_not_exists, load_pickle, encode_embed_docs, save_pickle
from bix.twitter.learn.tokenizer.tokenizer_utils import tokenize, load_tokenizer, save_tokenizer
from bix.twitter.preprocessing import cleanup


def load_all(hashtag: str) -> List[str]:
    (dirpath, _, filenames) = next(walk('raw/' + generate_hashtag_path(hashtag)))
    tweets = []
    for f in filenames:
        tweets.extend(load_csv(dirpath + '/' + f))
    tweets = remove_duplicates(tweets)
    return tweets


def preprocess(data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    lang = 'english' # or 'german'
    ret = {}

    for hashtag, tweets in data.items():
        # stemming and some other cleanup
        texts: List[str] = cleanup.clean_text(tweets, lang)
        ret[hashtag] = texts
        dest = 'hashtag_' + hashtag + '.csv'
        save_csv(dest, texts)
        print(f'saved {len(texts)} unique tweets in {dest}')

    return ret


def tokenize_cleaned_tweets(tweets: Dict[str, List[str]], create_tokenizer = False):

    if create_tokenizer is False:
        max_tweet_word_count = load_pickle('max_tweet_word_count.pickle')
        tok = load_tokenizer()
        ret = {}
        for hashtag, tweets in tweets.items():
            padded_x, _ = encode_embed_docs(tweets, tok, max_tweets=max_tweet_word_count)
            ret[hashtag] = padded_x
            dest = 'hashtag_' + hashtag
            numpy.save(dest, padded_x)
        return ret
    else:
        all_tweets = tweets.values()
        flat_list = []
        for sublist in all_tweets:
            for item in sublist:
                flat_list.append(item)
        tok = tokenize(flat_list, verbose=False)
        temp_padded_x, _ = encode_embed_docs(flat_list, tok)
        max_tweet_word_count = len(temp_padded_x[0])
        save_pickle(max_tweet_word_count, 'max_tweet_word_count.pickle')
        save_tokenizer(tok)

        return tokenize_cleaned_tweets(tweets)


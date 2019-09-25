from os import walk
from typing import List

from bix.twitter.base.utils import generate_hashtag_path, load_csv, remove_duplicates, save_csv, \
    create_path_if_not_exists, load_pickle, encode_embed_docs, save_pickle
from bix.twitter import tokenize, load_tokenizer
from bix.twitter.preprocessing import cleanup


def load_all(hashtag: str) -> List[str]:
    (dirpath, _, filenames) = next(walk('raw/' + generate_hashtag_path(hashtag)))
    tweets = []
    for f in filenames:
        tweets.extend(load_csv(dirpath + '/' + f))
    tweets = remove_duplicates(tweets)
    return tweets


if __name__ == '__main__':
    hashtags = ['love', 'sad']
    lang = 'english' # or 'german'
    root_dir = 'preprocessed'

    create_path_if_not_exists(root_dir)
    max_tweet_word_count = load_pickle('tokenized/learn/max_tweet_word_count.pickle')
    tok = load_tokenizer('learn')


    data = {}
    for h in hashtags:
        data[h] = load_all(h)

    for hashtag, tweets in data.items():
        # stemming and some other cleanup
        texts: List[str] = cleanup.clean_text(tweets, lang)

        #tok, num = tokenize(texts, verbose=True)
        padded_x, _ = encode_embed_docs(texts, tok, max_tweets=max_tweet_word_count)
        #max_tweet_word_count = len(padded_x[0])

        dest = root_dir + '/hashtag_' + hashtag + '.csv'
        dest2 = root_dir + '/hashtag_' + hashtag + '.pickle'
        save_csv(dest, texts)
        save_pickle(padded_x, dest2)
        print(f'saved {len(texts)} unique tweets in {dest}')



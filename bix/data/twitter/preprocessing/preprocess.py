from os import walk
from typing import List

from bix.data.twitter.base.utils import generate_hashtag_path, load_csv, remove_duplicates, save_csv, \
    create_path_if_not_exists
from bix.data.twitter.preprocessing import cleanup


def load_all(hashtag: str) -> List[str]:
    (dirpath, _, filenames) = next(walk('raw/' + generate_hashtag_path(hashtag)))
    tweets = []
    for f in filenames:
        tweets.extend(load_csv(dirpath + '/' + f))
    tweets = remove_duplicates(tweets)
    return tweets


if __name__ == '__main__':
    hashtags = ['brexit']
    lang = 'english' # or 'german'
    root_dir = 'preprocessed'

    create_path_if_not_exists(root_dir)

    data = {}
    for h in hashtags:
        data[h] = load_all(h)

    for hashtag, tweets in data.items():
        # stemming and some other cleanup
        texts: List[str] = cleanup.clean_text(tweets, lang)
        dest = root_dir + '/hashtag_' + hashtag + '.csv'
        save_csv(dest, texts)
        print(f'saved {len(texts)} unique tweets in {dest}')



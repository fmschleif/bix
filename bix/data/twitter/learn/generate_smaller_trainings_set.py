from sklearn.model_selection import train_test_split

from bix.data.twitter.base.utils import load_csv, encode_embed_docs, save_pickle, load_pickle, save_csv
from bix.data.twitter.learn.tokenizer.tokenizer_utils import load_tokenizer

if __name__ == '__main__':
    print('loading saved state')

    tokenizer = load_tokenizer('learn')
    x = load_csv('learn/tweets.csv')
    y = load_csv('learn/lables.csv')
    max_tweet_word_count = load_pickle('tokenized/learn/max_tweet_word_count.pickle')



    print('reducing learning data')
    x_learn, _, y_learn, _ = train_test_split(x, y, test_size=0.995, random_state=4)  # 16k are more than enough

    print('encoding data')

    padded_x, unpadded_x = encode_embed_docs(x_learn, tokenizer, max_tweet_word_count)

    print('saving')

    save_pickle(padded_x, 'tokenized/learn/small_padded_x.pickle')
    save_pickle(unpadded_x, 'tokenized/learn/small_unpadded_x.pickle')
    save_pickle(y_learn, 'tokenized/learn/small_y.pickle')
    save_csv('learn/tweets_learn.csv', x_learn)

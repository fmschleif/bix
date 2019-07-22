from sklearn.model_selection import train_test_split

from bix.data.twitter.base.utils import load_csv, encode_embed_docs, save_pickle
from bix.data.twitter.learn.tokenizer.tokenizer_utils import load_tokenizer

if __name__ == '__main__':
    print('loading saved state')

    tokenizer = load_tokenizer('learn')
    x = load_csv('learn/tweets.csv')
    y = load_csv('learn/lables.csv')

    print('encoding data')

    padded_x, unpadded_x = encode_embed_docs(x, tokenizer)

    print('reducing learning data')

    x_train, _, y_train, _ = train_test_split(padded_x, y, test_size=0.99, random_state=4)  # 16k are more than enough
    x_train_unpadded, _, _, _ = train_test_split(unpadded_x, y, test_size=0.99,
                                                 random_state=4)

    print('saving')

    save_pickle(x_train, 'tokenized/learn/small_padded_x.pickle')
    save_pickle(x_train_unpadded, 'tokenized/learn/small_unpadded_x.pickle')
    save_pickle(y_train, 'tokenized/learn/small_y.pickle')

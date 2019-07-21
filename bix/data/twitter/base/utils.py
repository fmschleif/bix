import os
import pickle
from datetime import timedelta, date
from typing import List, Any, Tuple
import numpy as np
import pandas
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


from bix.data.twitter.learn.tokenizer.tokenizer_utils import load_tokenizer


def generate_hashtag_path(hashtag: str) -> str:
    return f'hashtag_{hashtag}'


def generate_csv_name(date_: date) -> str:
    return f'{"{:%Y-%m-%d}".format(date_)}.csv'


def remove_duplicates(lst: List[str]) -> List[str]:
    unique = []
    [unique.append(item) for item in lst if item not in unique]
    return unique


def daterange(start_date: date, end_date: date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def load_csv(path: str) -> List[Any]:
    file_df = pandas.read_csv(path, header=None)
    return [l[0] for l in file_df.values.tolist()]


def save_csv(path: str, lst: List[Any]):
    df = pandas.DataFrame(lst)
    df.to_csv(path, encoding='utf-8', header=False, index=False)


def create_path_if_not_exists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_pickle(obj, path: str):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    # loading
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
        return obj


def save_model_mat(nparray, model_name: str):
    create_path_if_not_exists('models')
    #np.save(nparray, f'models/{model_name}.npy')
    save_pickle(nparray, f'models/{model_name}.pickle')


def load_model_mat(model_name: str):
    #np.load(nparray, f'models/{model_name}.npy')
    return load_pickle(f'models/{model_name}.pickle')


def encode_embed_docs(x: List[str], tokenizer: Tokenizer, max_doc_count: int):
    # integer encode the documents
    encoded_docs = tokenizer.texts_to_sequences(x)
    print(encoded_docs)
    # pad documents to a max length of [embedding_vector_size] words
    padded_docs = pad_sequences(encoded_docs, maxlen=max_doc_count, padding='post')
    print(padded_docs)
    return padded_docs, encoded_docs


def load_training_sentiment_data():
    t = load_tokenizer('learn')
    y = load_csv('tokenized/learn/lables.csv')
    padded_x = np.load('tokenized/learn/padded_x.npy')
    unpadded_x = load_pickle('tokenized/learn/unpadded_x.pickle')
    max_tweet_word_count = load_pickle('tokenized/learn/max_tweet_word_count.pickle')
    return t, y, padded_x, unpadded_x, max_tweet_word_count


def load_training_sentiment_data_small():
    t = load_tokenizer('learn')
    y = load_pickle('tokenized/learn/small_y.pickle')
    padded_x = load_pickle('tokenized/learn/small_padded_x.pickle')
    unpadded_x = load_pickle('tokenized/learn/small_unpadded_x.pickle')
    max_tweet_word_count = load_pickle('tokenized/learn/max_tweet_word_count.pickle')
    vocab_size = load_pickle('tokenized/learn/vocab_size.pickle')
    return t, y, padded_x, unpadded_x, max_tweet_word_count, vocab_size



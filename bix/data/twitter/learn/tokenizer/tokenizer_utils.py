import pickle
from typing import List

from keras_preprocessing.text import Tokenizer


def tokenize(data: List[str], verbose: bool = False) -> (Tokenizer, int):

    # 545409 is big
    t = Tokenizer(filters='!"„“…»«#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', num_words=25000)

    t.fit_on_texts(data)

    if verbose:
        print(f'indexsize: {len(t.word_counts)}')
        #print('wordcounts')
        #print(t.word_counts)
        #print('document_count')
        #print(t.document_count)
        #print('wordindex')
        #print(t.word_index)
        #print('word_docs')
        #print(t.word_docs)

        print('--------- ' + data[0][0] + ' ---------')
        for e, n in sorted(t.word_counts.items(), key=lambda x: x[1])[-9:]:
            print('\t' + str(e) + ': ' + str(n))

    return t, t.num_words


def save_tokenizer(tok: Tokenizer, name: str):
    from bix.data.twitter.base.utils import create_path_if_not_exists
    # saving
    create_path_if_not_exists('tokenized')
    create_path_if_not_exists(f'tokenized/{name}')
    with open(f'tokenized/{name}/tok.pickle', 'wb') as handle:
        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(name: str) -> Tokenizer:
    # loading
    with open(f'tokenized/{name}/tok.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        return tokenizer

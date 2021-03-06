from bix.twitter.base.utils import load_csv, encode_embed_docs, save_csv, save_pickle
from bix.twitter.learn.tokenizer.tokenizer_utils import tokenize, save_tokenizer

if __name__ == '__main__':
    x = load_csv('learn/tweets.csv')
    y = load_csv('learn/lables.csv')

    tok, num = tokenize(x, verbose=True)
    padded_x, unpadded_x = encode_embed_docs(x, tok)
    max_tweet_word_count = len(padded_x[0])

    save_tokenizer(tok)
    save_csv('tokenized/learn/lables.csv', y)
    save_pickle(padded_x, 'tokenized/learn/padded_x.pickle')
    save_pickle(unpadded_x, 'tokenized/learn/unpadded_x.pickle')
    save_pickle(max_tweet_word_count, 'tokenized/learn/max_tweet_word_count.pickle')
    save_pickle(num, 'tokenized/learn/vocab_size.pickle')


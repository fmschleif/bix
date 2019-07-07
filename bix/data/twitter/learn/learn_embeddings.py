
from bix.data.twitter.base.utils import load_csv, save_model_mat
from bix.data.twitter.learn.embeddings.embedding_glove import EmbeddingGlove
from bix.data.twitter.learn.tokenizer.tokenizer_utils import load_tokenizer

if __name__ == '__main__':
    tokenizer = load_tokenizer('tok1')
    x = load_csv('learn/tweets.csv')
    y = load_csv('learn/lables.csv')

    e = EmbeddingGlove(tokenizer, x, y)
    e.create_embedding()
    weights = e.get_weights()
    save_model_mat(weights, 'word_embedding')



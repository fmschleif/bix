from typing import List

import pandas
from keras import Sequential
from keras.layers import Embedding, Flatten, Dense

from bix.data.twitter.model_input import ModelInput
from bix.data.twitter.twitter_retriever import TwitterRetriever

file_to_analyze = 'twitter.csv'
remove_hashtags = True

twint_analyse = True # analyse file scraped with twint instead
twint_dirs = ['brexit', 'peoplesvote']

if __name__ == '__main__':
    mode = 'embedding_mat' # or 'default' or 'embedding_mat_glove', 'embedding_mat', 'skip_gram'
    lang = 'english' # or 'german'

    data = []
    if not twint_analyse:
        # load from csv
        file_df = pandas.read_csv(file_to_analyze, header=None)
        data = file_df.values.tolist()
    else:
        for dir_ in twint_dirs:
            file_df = pandas.read_csv('hashtag_' + dir_ + '/tweets.csv')
            data.extend([['#' + dir_] + e[0].split() for e in file_df.values.tolist()])

    # stemming and some other cleanup
    texts: List[str] = TwitterRetriever.clean_text(data, lang)
    labels: List[str] = [e[0] for e in data]

    # remove hashtags for experiment
    # notredame vs brexit: without: 92%, with: 92%
    # [twint] brexit vs peoplesvote: without: 77.997003, with: 78.141858%
    if remove_hashtags:
        edited_labels = [l.replace('#', '').lower() for l in set(labels)]
        edited_labels = [l[:-1] for l in edited_labels if l[-1] == 'e']
        for l in edited_labels:
            for i,t in enumerate(texts):
                texts[i] = texts[i].replace(l, '')

    # tokenize
    tok = TwitterRetriever.tokenize(texts)

    # vectorize
    model_input = None
    if mode == 'embedding_mat_glove':
        model_input = TwitterRetriever.prepare_word_embedding_glove(texts, labels, tok)
    elif mode == 'embedding_mat':
        model_input = TwitterRetriever.prepare_word_embedding(texts, labels, tok)
    elif mode == 'skip_gram':
        model_input = TwitterRetriever.prepare_skip_gram(texts, labels, tok)
    else:
        pass
        #mat, doc_list = TwitterRetriever.vectorize(texts, labels, tok)

    max_tweet_word_count = max([len(e.split()) for e in texts])
    vocab_size = len(tok.word_index) + 1

    model = Sequential()
    model.add(Embedding(vocab_size, model_input.embedding_weights[0].shape[1], input_length=max_tweet_word_count,
                        weights=model_input.embedding_weights))
    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(model_input.x, model_input.y, epochs=5, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(model_input.x, model_input.y, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))

    print('finished')

    #for k,v in dic.items():
    #    mat, doc_list = TwitterRetriever.tokenize_and_vectorize(v)
    #    # continue here
    #    print(mat)


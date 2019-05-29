from typing import List

import pandas

from bix.data.twitter.twitter_retriever import TwitterRetriever

file_to_analyze = 'twitter.csv'
remove_hashtags = True

twint_analyse = True # analyse file scraped with twint instead
twint_dirs = ['brexit', 'peoplesvote']

if __name__ == '__main__':
    mode = 'embedding_mat' # or 'default'
    lang = 'english' # or 'german'

    data = []
    if not twint_analyse:
        # load from csv
        file_df = pandas.read_csv(file_to_analyze, header=None)
        data = file_df.values.tolist()
    else:
        for dir in twint_dirs:
            file_df = pandas.read_csv('hashtag_' + dir + '/tweets.csv')
            data.extend([['#' + dir] + e[0].split() for e in file_df.values.tolist()])

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
    mat = None
    doc_list = None
    if mode == 'embedding_mat':
        mat, doc_list = TwitterRetriever.perform_word_embedding(texts, labels, tok)
    else:
        mat, doc_list = TwitterRetriever.vectorize(texts, labels, tok)

    print(str(mat))
    print('finished')

    #for k,v in dic.items():
    #    mat, doc_list = TwitterRetriever.tokenize_and_vectorize(v)
    #    # continue here
    #    print(mat)


import pandas

from bix.data.twitter.twitter_retriever import TwitterRetriever

file_to_analyze = 'twitter.csv'

if __name__ == '__main__':
    file_df = pandas.read_csv(file_to_analyze, header=None)
    data = file_df.values.tolist()
    dic = TwitterRetriever.split_result_list_by_label(data)
    for k,v in dic.items():
        mat, doc_list = TwitterRetriever.tokenize_and_vectorize(v)
        # continue here
        print(mat)


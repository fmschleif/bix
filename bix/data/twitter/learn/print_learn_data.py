from bix.data.twitter.base.utils import load_training_sentiment_data_small, load_csv, load_pickle

if __name__ == '__main__':
    y = load_pickle('tokenized/learn/small_y.pickle')
    x = load_csv('learn/tweets_learn.csv')
    for i, v in enumerate(x):
        print(f"{y[i]}: {v}")

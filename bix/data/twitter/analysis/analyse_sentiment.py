from keras import Sequential
from keras.layers import Embedding, Flatten, Dense

from bix.data.twitter.analysis.analysis_utils.analysis_utils import load_training_sentiment_data
from bix.data.twitter.base.utils import load_model_mat

if __name__ == '__main__':
    tok, y, padded_x, unpadded_x, max_tweet_word_count = load_training_sentiment_data()

    model_mat = load_model_mat('word_embedding')

    vocab_size = len(tok.word_index) + 1

    model = Sequential()
    model.add(Embedding(vocab_size, model_mat[0].shape[1], input_length=max_tweet_word_count,
                        weights=model_mat))
    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(padded_x, y, epochs=2, verbose=2)
    # evaluate the model
    loss, accuracy = model.evaluate(padded_x, y, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))

    print('finished')
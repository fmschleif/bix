import itertools

from keras.engine.saving import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from bix.data.twitter.base.utils import load_training_sentiment_data
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)


if __name__ == '__main__':

    tok, y, padded_x, _, max_tweet_word_count, vocab_size = load_training_sentiment_data()

    x_train, x_test, y_train, y_test = train_test_split(padded_x, y, test_size=0.2, random_state=5)

    model = load_model('models/sentiment_conv_ep100.h5')

    y_test_1d = y_test
    scores = model.predict(x_test, verbose=1, batch_size=8000)
    y_pred_1d = [1 if score > 0.5 else 0 for score in scores]
    cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
    plt.figure(figsize=(12, 12))
    plot_confusion_matrix(cnf_matrix, classes=[0, 1], title="Confusion matrix")
    plt.show()
    #plt.savefig('../helper_scripts/sent' + '_conv')

    model = load_model('models/sentiment_conv_ep100_word.h5')

    y_test_1d = y_test
    scores = model.predict(x_test, verbose=1, batch_size=8000)
    y_pred_1d = [1 if score > 0.5 else 0 for score in scores]
    cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
    plt.figure(figsize=(12, 12))
    plot_confusion_matrix(cnf_matrix, classes=[0, 1], title="Confusion matrix")
    plt.show()
    #plt.savefig('../helper_scripts/sent_word' + '_conv')

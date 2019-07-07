import re
import string
from typing import List

from nltk import SnowballStemmer
from nltk.corpus import stopwords


def clean_text(tweets: List[str], lang: str = 'english') -> List[str]:
    """
    performes stemming and other "cleanup"
    :param tweets:
    :param lang:
    :return:
    """

    data: List[str] = tweets
    cleaned_text = []
    stops = set(stopwords.words(lang))
    for text in data:
        text = re.sub(r'https?://[^\s]*', '', text)

        ## Remove puncuation
        text = text.translate(string.punctuation)

        ## Convert words to lower case and split them
        text = text.lower().split()

        ## Remove stop words[^\s][^\s]
        text = [w for w in text if not w in stops and len(w) >= 3]

        text = " ".join(text)

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!./'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        text = text.split()
        stemmer = SnowballStemmer(lang)
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        cleaned_text.append(text)

    return cleaned_text

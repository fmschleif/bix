import numpy
from sklearn.preprocessing import StandardScaler

from bix.twitter.base.utils import load_pickle

if __name__ == '__main__':
    print('loading saved state')

    word = load_pickle('/home/jonas/Development/fh/hiwi_job/bix/bix/data/twitter/data/models/embedding_word.pickle')
    x_std = StandardScaler().fit_transform(word[0])
    cov = numpy.cov(x_std.T)
    ev, _ = numpy.linalg.eig(cov)
    print(f"eigenvalues: {list(reversed(sorted(ev)))}")

    word = load_pickle('/home/jonas/Development/fh/hiwi_job/bix/bix/data/twitter/data/models/embedding_glove.pickle')
    x_std = StandardScaler().fit_transform(word[0])
    cov = numpy.cov(x_std.T)
    ev, _ = numpy.linalg.eig(cov)
    print(f"eigenvalues: {list(reversed(sorted(ev)))}")

    word = load_pickle('/home/jonas/Development/fh/hiwi_job/bix/bix/data/twitter/data/models/embedding_skip_gram.pickle')
    x_std = StandardScaler().fit_transform(word[0])
    cov = numpy.cov(x_std.T)
    ev, _ = numpy.linalg.eig(cov)
    print(f"eigenvalues: {list(reversed(sorted(ev)))}")


    #a = eig.dot(x_std.T)
    #print(f"a: {a}")



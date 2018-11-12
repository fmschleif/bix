#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:57:31 2018

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from rslvq import RSLVQ
from sklearn.metrics import accuracy_score
from time import time


X, y_true = make_blobs(n_samples=10000, centers=4,
                       cluster_std=1.0, random_state=None)
X = X[:, ::-1] # flip axes for better plotting

#stdizer = StandardScaler()
#X_std = stdizer.fit_transform(X=X)



clf = RSLVQ(batch_size=1, sigma=1.0)
t_new_start = time()
labels_new = clf.partial_fit(X=X, y=y_true).predict(X)

t_new = time() - t_new_start

#t_old_start = time()
#clf = RslvqModel(prototypes_per_class=4, max_iter=1, sigma=1.0)
#labels_old = clf.fit(x=X, y=y_true).predict(X)
#t_old = time() - t_old_start



clf = RSLVQ(batch_size=5, sigma=1.0)
t_ada_start = time()
labels_ada = clf.partial_fit(X=X, y=y_true).predict(X)

t_ada = time() - t_ada_start



clf = RSLVQ(batch_size=10, sigma=1.0)
t_rms_start = time()
labels_rms = clf.partial_fit(X=X, y=y_true).predict(X)
t_rms = time() - t_rms_start

acc_new = accuracy_score(y_true, labels_new)
acc_ada = accuracy_score(y_true, labels_ada)
acc_rms = accuracy_score(y_true, labels_rms)
#acc_old = accuracy_score(y_true, labels_old)


#print('Accuracy Old Model: {} \nAccuracy New Model: {}'.format(acc_old, acc_new))
#print('Time Old Model: {} \nTime New Model: {}'.format(t_old, t_new))

#print('Accuracy Adadelta: {} \nAccuracy SGD: {}'.format(acc_ada, acc_new))
#print('Time Adadelta: {} \nTime SGD: {}'.format(t_ada, t_new))

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,11))

ax[0].scatter(X[:, 0], X[:, 1], c=labels_new, s=40, cmap='viridis')
ax[0].set_title('Batch Size: 1')
ax[0].text(0.5, -0.1, 'Time: {} seconds\nAccuracy: {}'.format(t_new, acc_new), size=12, ha="center", 
         transform=ax[0].transAxes)

ax[1].scatter(X[:, 0], X[:, 1], c=labels_ada, s=40, cmap='viridis')
ax[1].set_title('Batch Size: 5')
ax[1].text(0.5, -0.1, 'Time: {} seconds\nAccuracy: {}'.format(t_ada, acc_ada), size=12, ha="center", 
         transform=ax[1].transAxes)

ax[2].scatter(X[:, 0], X[:, 1], c=labels_rms, s=40, cmap='viridis')
ax[2].set_title('Batch Size: 10')
ax[2].text(0.5, -0.1, 'Time: {} seconds\nAccuracy: {}'.format(t_rms, acc_rms), size=12, ha="center", 
         transform=ax[2].transAxes)

plt.show()
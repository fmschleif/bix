#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:35:11 2018

@author: moritz
"""

from __future__ import division
import math
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from skmultiflow.core.base import StreamModel
from sklearn.utils import validation
from sklearn.utils.validation import check_is_fitted

class RSLVQ(ClassifierMixin, StreamModel, BaseEstimator):
    """Robust Soft Learning Vector Quantization for Streaming and Non-Streaming Data
    By choosing another gradient descent method the RSLVQ can be used as an adaptive version.
    By setting the batch_size higher than 1, the algorithm works in batch mode.
    Parameters
    ----------
    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different
        numbers per class.
    initial_prototypes : array-like, shape =  [n_prototypes, n_features + 1],
     optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.
    sigma : float, optional (default=1.0)
        Variance for the distribution.
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    gradient_descent : string, Gradient Descent describes the used technique
        to perform the gradient descent. Possible values: 'SGD' (default), 
        'Adadelta'.
    decay_rate : float, has to be less than 1 and greater than 0. Represents the
        factor which decays old stored gradients in Adadelta. (Default: 0.9) 
    batch_size : int, (default: 1) if set greater than 1 the RSLVQ will work in batch
        mode and performs the prototype update after each batch instead after each data point.
    Attributes
    ----------
    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features
    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes
    classes_ : array-like, shape = [n_classes]
        Array containing labels.
    initial_fit : boolean, indicator for initial fitting. Set to false after
        first call of fit/partial fit.
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 sigma=1.0, gradient_descent='SGD', random_state=None,
                 decay_rate=0.9, batch_size=1, learning_rate=0.001):
        self.sigma = sigma
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.epsilon = 1e-8
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.initial_fit = True
        self.classes_ = []
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        allowed_gradient_descent = ['SGD', 'Adadelta', 'RMSprop']
        if gradient_descent in allowed_gradient_descent:
            self.gradient_descent = gradient_descent
        else:
            raise ValueError('{} is not a valid parameter for '
                             'gradient_descent, please use one '
                             'of the following parameters:\n {}'
                             .format(gradient_descent, allowed_gradient_descent))
        if(sigma <= 0):
            raise ValueError('Sigma must be greater than 0')
        if(batch_size <= 0):
            raise ValueError('Batch size must be greater than 0')
        if(prototypes_per_class <= 0):
            raise ValueError('Prototypes per class must be more than 0')
        if(decay_rate >= 1.0 or decay_rate <= 0):
            raise ValueError('Decay rate must be greater than 0 and less than 1')
            
    def get_prototypes(self):
        """Returns the prototypes"""
        return self.w_

    def _optimize(self, X, y, random_state):
        if(self.batch_size > y.size):
            raise ValueError('Batch size higher than data passed') 
        k = 0
        nb_prototypes = self.c_w_.size
        
        if(self.gradient_descent=='SGD'):
            c = 1 / self.sigma # learning rate
            """Implementation of Stochastical Gradient Descent"""
            while((k + self.batch_size) <= y.size): #sample batch from data
                X_batch, y_batch = X[k:k + self.batch_size, :], y[k:k + self.batch_size]
                n_data, n_dim = X_batch.shape
                prototypes = self.w_.reshape(nb_prototypes, n_dim)
                sum_per_class = np.zeros(shape=(self.classes_.size, n_dim)) # sum up datapoints per class
                count_per_class = np.zeros(shape=(self.classes_.size, 1)) # count datapoints per class
                for i in range(n_data):
                    xi = X_batch[i]
                    c_xi = int(y_batch[i])
                    c_idx = np.where(self.classes_ == c_xi)[0]
                    sum_per_class[c_idx] += xi
                    count_per_class[c_idx] += 1
                
                update = np.zeros(shape=(self.classes_.size, n_dim))

                for j in range(prototypes.shape[0]):
                    label = self.c_w_[j]
                    c_idx = np.where(self.classes_ == label)[0]
                    if(count_per_class[c_idx] > 0):
                        update[c_idx] = sum_per_class[c_idx] / count_per_class[c_idx] # calc mean
                        x_up = update[c_idx].flatten()
                        d = (x_up - prototypes[j])
                        # Attract prototype to data point
                        final_up = c * (self._p(j=j, e=x_up, prototypes=self.w_, y=label) -
                               self._p(j=j, e=x_up, prototypes=self.w_)) * d * count_per_class[c_idx].ravel()
                        
                        if(np.any(np.isnan(final_up))):
                            self.w_[j] += final_up
                            
                    else:
                        # Distance prototype from data point
                        x_up = np.sum(sum_per_class, axis=0)
                        d = (x_up - prototypes[j])
                        final_up = c * self._p(j=j, e=x_up, prototypes=self.w_) * d * count_per_class[c_idx].ravel()   
                        
                        if(np.any(np.isnan(final_up))):
                            self.w_[j] += final_up
                        
                k += self.batch_size          
            
        elif(self.gradient_descent=='Adadelta'):
            """Implementation of Adadelta"""
            n_data, n_dim = X.shape
            prototypes = self.w_.reshape(nb_prototypes, n_dim)
            
            while((k + self.batch_size) <= y.size): #sample batch from data
                X_batch, y_batch = X[k:k + self.batch_size, :], y[k:k + self.batch_size]
                n_data, n_dim = X_batch.shape
                prototypes = self.w_.reshape(nb_prototypes, n_dim)
                sum_per_class = np.zeros(shape=(self.classes_.size, n_dim)) # sum up datapoints per class
                count_per_class = np.zeros(shape=(self.classes_.size, 1)) # count datapoints per class
                for i in range(n_data):
                    xi = X_batch[i]
                    c_xi = int(y_batch[i])
                    c_idx = np.where(self.classes_ == c_xi)[0]
                    sum_per_class[c_idx] += xi
                    count_per_class[c_idx] += 1
                
                update = np.zeros(shape=(self.classes_.size, n_dim))

                for j in range(prototypes.shape[0]):
                    label = self.c_w_[j]
                    c_idx = np.where(self.classes_ == label)[0]
                    if(count_per_class[c_idx] > 0):
                        update[c_idx] = sum_per_class[c_idx] / count_per_class[c_idx] # calc mean
                        x_up = update[c_idx].flatten()
                        d = (x_up - prototypes[j])
                        # Attract prototype to data point
                        gradient = (self._p(j, x_up, prototypes=self.w_, y=c_xi) -
                                     self._p(j, x_up, prototypes=self.w_)) * d
                                    
                        # Accumulate gradient
                        self.squared_mean_gradient[j] = self.decay_rate * self.squared_mean_gradient[j] + \
                                    (1 - self.decay_rate) * gradient ** 2
                        
                        # Compute update/step
                        step = ((self.squared_mean_step[j] + self.epsilon) / \
                                  (self.squared_mean_gradient[j] + self.epsilon)) ** 0.5 * gradient
                                  
                        # Accumulate updates
                        self.squared_mean_step[j] = self.decay_rate * self.squared_mean_step[j] + \
                        (1 - self.decay_rate) * step ** 2
                        
                        # Attract/Distract prototype to/from data point
                        self.w_[j] += step
                            
                    else:
                        # Distance prototype from data point
                        x_up = np.sum(sum_per_class, axis=0)
                        d = (x_up - prototypes[j])
                        gradient = - self._p(j, x_up, prototypes=self.w_) * d
                        
                         # Accumulate gradient
                        self.squared_mean_gradient[j] = self.decay_rate * self.squared_mean_gradient[j] + \
                                    (1 - self.decay_rate) * gradient ** 2
                        
                        # Compute update/step
                        step = ((self.squared_mean_step[j] + self.epsilon) / \
                                  (self.squared_mean_gradient[j] + self.epsilon)) ** 0.5 * gradient
                                  
                        # Accumulate updates
                        self.squared_mean_step[j] = self.decay_rate * self.squared_mean_step[j] + \
                        (1 - self.decay_rate) * step ** 2
                        
                        # Attract/Distract prototype to/from data point
                        self.w_[j] += step
                        
                k += self.batch_size
                
                for j in range(prototypes.shape[0]):
                    d = (xi - prototypes[j])
                    
                    if self.c_w_[j] == c_xi:
                        gradient = (self._p(j, xi, prototypes=self.w_, y=c_xi) -
                                     self._p(j, xi, prototypes=self.w_)) * d
                    else:
                        gradient = - self._p(j, xi, prototypes=self.w_) * d
                        
                     # Accumulate gradient
                    self.squared_mean_gradient[j] = self.decay_rate * self.squared_mean_gradient[j] + \
                                (1 - self.decay_rate) * gradient ** 2
                    
                    # Compute update/step
                    step = ((self.squared_mean_step[j] + self.epsilon) / \
                              (self.squared_mean_gradient[j] + self.epsilon)) ** 0.5 * gradient
                              
                    # Accumulate updates
                    self.squared_mean_step[j] = self.decay_rate * self.squared_mean_step[j] + \
                    (1 - self.decay_rate) * step ** 2
                    
                    # Attract/Distract prototype to/from data point
                    self.w_[j] += step
                
        elif(self.gradient_descent=='RMSprop'):
            """Implementation of RMSprop"""
            n_data, n_dim = X.shape
            nb_prototypes = self.c_w_.size
            prototypes = self.w_.reshape(nb_prototypes, n_dim)

            for i in range(n_data):
                xi = X[i]
                c_xi = y[i]
                for j in range(prototypes.shape[0]):
                    d = (xi - prototypes[j])
                    
                    if self.c_w_[j] == c_xi:
                        gradient = (self._p(j, xi, prototypes=self.w_, y=c_xi) -
                                     self._p(j, xi, prototypes=self.w_)) * d
                    else:
                        gradient = - self._p(j, xi, prototypes=self.w_) * d
                        
                    # Accumulate gradient
                    self.squared_mean_gradient[j] = 0.9 * self.squared_mean_gradient[j] + \
                            0.1 * gradient ** 2
                    
                    # Update Prototype
                    self.w_[j] += (self.learning_rate / ((self.squared_mean_gradient[j] + \
                           self.epsilon) ** 0.5)) * gradient
     
    def _costf(self, x, w, **kwargs):
        d = (x - w)[np.newaxis].T 
        d = d.T.dot(d) 
        return - d / (2 * self.sigma)

    def _p(self, j, e, y=None, prototypes=None, **kwargs):
        if prototypes is None:
            prototypes = self.w_
        if y is None:
            fs = [self._costf(e, w, **kwargs) for w in prototypes]
        else:
            fs = [self._costf(e, prototypes[i], **kwargs) for i in
                  range(prototypes.shape[0]) if
                  self.c_w_[i] == y]

        fs_max = np.max(fs)
        s = sum([np.math.exp(f - fs_max) for f in fs])
        o = np.math.exp(
            self._costf(e, prototypes[j], **kwargs) - fs_max) / s
        return o

    def predict(self, x):
        """Predict class membership index for each input sample.
        This function does classification on an array of
        test vectors X.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = (n_samples)
            Returns predicted values.
        """
#        check_is_fitted(self, ['w_', 'c_w_'])
#        x = validation.check_array(x)
#        if x.shape[1] != self.w_.shape[1]:
#            raise ValueError("X has wrong number of features\n"
#                             "found=%d\n"
#                             "expected=%d" % (self.w_.shape[1], x.shape[1]))

        def foo(e):
            fun = np.vectorize(lambda w: self._costf(e, w),
                               signature='(n)->()')
            pred = fun(self.w_).argmax()
            return self.c_w_[pred]

        return np.vectorize(foo, signature='(n)->()')(x)

    def posterior(self, y, x):
        """
        calculate the posterior for x:
         p(y|x)
        Parameters
        ----------
        
        y: class
            label
        x: array-like, shape = [n_features]
            sample
        Returns
        -------
        posterior
        :return: posterior
        """
        check_is_fitted(self, ['w_', 'c_w_'])
        x = validation.column_or_1d(x)
        if y not in self.classes_:
            raise ValueError('y must be one of the labels\n'
                             'y=%s\n'
                             'labels=%s' % (y, self.classes_))
        s1 = sum([self._costf(x, self.w_[i]) for i in
                  range(self.w_.shape[0]) if
                  self.c_w_[i] == y])
        s2 = sum([self._costf(x, w) for w in self.w_])
        return s1 / s2
    
    def get_info(self):
        return 'RSLVQ'
    
    def predict_proba(self, X):
        """ predict_proba
        
        Predicts the probability of each sample belonging to each one of the 
        known target_values.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
        
        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index. And where the list in 
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.
        
        """
        return 'Not implemented'
    
    def reset(self):
        self.__init__()
        
    def _validate_train_parms(self, train_set, train_lab, classes=None):
        random_state = validation.check_random_state(self.random_state)
        train_set, train_lab = validation.check_X_y(train_set, train_lab)

        if(self.initial_fit):
            if(classes):
                self.classes_ = np.asarray(classes)
                self.protos_initialized = np.zeros(self.classes_.size)
            else:
                self.classes_ = unique_labels(train_lab)
                self.protos_initialized = np.zeros(self.classes_.size)

        nb_classes = len(self.classes_)
        nb_samples, nb_features = train_set.shape  # nb_samples unused

        # set prototypes per class
        if isinstance(self.prototypes_per_class, int):
            if self.prototypes_per_class < 0 or not isinstance(
                    self.prototypes_per_class, int):
                raise ValueError("prototypes_per_class must be a positive int")
            # nb_ppc = number of protos per class
            nb_ppc = np.ones([nb_classes],
                             dtype='int') * self.prototypes_per_class
        else:
            nb_ppc = validation.column_or_1d(
                validation.check_array(self.prototypes_per_class,
                                       ensure_2d=False, dtype='int'))
            if nb_ppc.min() <= 0:
                raise ValueError(
                    "values in prototypes_per_class must be positive")
            if nb_ppc.size != nb_classes:
                raise ValueError(
                    "length of prototypes per class"
                    " does not fit the number of classes"
                    "classes=%d"
                    "length=%d" % (nb_classes, nb_ppc.size))
        
        # initialize prototypes
        if self.initial_prototypes is None:
            if self.initial_fit:
                self.w_ = np.empty([np.sum(nb_ppc), nb_features], dtype=np.double)
                self.c_w_ = np.empty([nb_ppc.sum()], dtype=self.classes_.dtype)
            pos = 0
            for actClass in range(len(self.classes_)):
                nb_prot = nb_ppc[actClass] # nb_ppc: prototypes per class
                if(self.protos_initialized[actClass] == 0 and actClass in unique_labels(train_lab)):
                    mean = np.mean(
                        train_set[train_lab == self.classes_[actClass], :], 0)
                    self.w_[pos:pos + nb_prot] = mean + (
                            random_state.rand(nb_prot, nb_features) * 2 - 1)
                    if math.isnan(self.w_[pos, 0]):
                        print('null: ', actClass)
                        self.protos_initialized[actClass] = 0
                    else:
                        self.protos_initialized[actClass] = 1
    
                    self.c_w_[pos:pos + nb_prot] = self.classes_[actClass]
                    pos += nb_prot
                elif(self.protos_initialized[actClass] == 0):
                    self.w_[pos:pos + nb_prot] = 0
                    pos += nb_prot
        else:
            x = validation.check_array(self.initial_prototypes)
            self.w_ = x[:, :-1]
            self.c_w_ = x[:, -1]
            if self.w_.shape != (np.sum(nb_ppc), nb_features):
                raise ValueError("the initial prototypes have wrong shape\n"
                                 "found=(%d,%d)\n"
                                 "expected=(%d,%d)" % (
                                     self.w_.shape[0], self.w_.shape[1],
                                     nb_ppc.sum(), nb_features))
            if set(self.c_w_) != set(self.classes_):
                raise ValueError(
                    "prototype labels and test data classes do not match\n"
                    "classes={}\n"
                    "prototype labels={}\n".format(self.classes_, self.c_w_))
        if self.initial_fit:
            # Next two lines are Init for Adadelta/RMSprop
            self.squared_mean_gradient = np.zeros_like(self.w_)
            self.squared_mean_step = np.zeros_like(self.w_)
            self.initial_fit = False

        return train_set, train_lab, random_state

    def fit(self, X, y, classes=None):
        """Fit the LVQ model to the given training data and parameters using
        l-bfgs-b.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
          Training vector, where n_samples in the number of samples and
          n_features is the number of features.
        y : array, shape = [n_samples]
          Target values (integers in classification, real numbers in
          regression)
        Returns
        --------
        self
        """
        if unique_labels(y) in self.classes_ or self.initial_fit == True:
            X, y, random_state = self._validate_train_parms(X, y, classes=classes)
        else:
            raise ValueError('Class {} was not learned - please declare all \
                             classes in first call of fit/partial_fit'.format(y))
            
        self._optimize(X, y, random_state)
        return self
    
    def partial_fit(self, X, y, classes=None): # added
        """Fit the LVQ model to the given training data and parameters using
        l-bfgs-b.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
          Training vector, where n_samples in the number of samples and
          n_features is the number of features.
        y : array, shape = [n_samples]
          Target values (integers in classification, real numbers in
          regression)
        Returns
        --------
        self
        """
        if unique_labels(y) in self.classes_ or self.initial_fit == True:
            X, y, random_state = self._validate_train_parms(X, y, classes=classes)
        else:
            raise ValueError('Class {} was not learned - please declare all \
                             classes in first call of fit/partial_fit'.format(y))
            
        self._optimize(X, y, random_state)
        return self
    
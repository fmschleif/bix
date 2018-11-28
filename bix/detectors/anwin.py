from abc import ABCMeta, abstractmethod
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
import numpy as np
from skmultiflow.core.base_object import BaseObject       
from scipy import stats
from scipy.stats import f_oneway
class ANWIN(BaseDriftDetector):
    def __init__(self,alpha=0.05,data=None):

        self.w_size = 200
        self.stat_size = 30 
        self.alpha = alpha
        self.change_detected = False;
        self.p_value = 0
        self.n = 0
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must in between 0 and 1")
        if type(data) != list or type(data) == None:
            self.window = []
        else:
            self.window = data
        pass
    
    def add_element(self, value):
        self.n +=1
        currentLength = len(self.window)
        if currentLength >= self.w_size:
            self.window.pop(0)
            rnd_window = np.random.choice(self.window[:-self.stat_size],self.stat_size)
            (st, self.p_value) = stats.f_oneway(rnd_window, self.window[-self.stat_size:])
           
            if self.p_value <= self.alpha and 0.1 < st:
                self.change_detected = True
                self.window = []
            else: 
                self.change_detected = False
        else: 
            self.change_detected = False
        self.window.insert(currentLength,value)      
        pass
    
    def detected_change(self):
        return self.change_detected
    
    def reset(self):
        self.alpha = 0
        self.window = []
        self.change_detected = False;
        pass
    
    def get_info(self):
         return "KSwin Change: Current P-Value "+str(self.p_value)

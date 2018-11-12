import math
import copy
import sys
import os
from random import random as rnd
sys.path.append('..\\multiflow-rslvq')
sys.path.append('..\\RSLVQ')
sys.path.append('..\\Reactive_RSLVQ')
sys.path.append("..\\stream_utilities")
from ReoccuringDriftStream import ReoccuringDriftStream 
from geometric_median import *
from kswin import KSWIN
from rrslvq import RRSLVQ as rRSLVQ
from rslvq import RSLVQ as bRSLVQ
from skmultiflow.core.base import StreamModel
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.file_stream import FileStream
#Abrupt Concept Drift Generators
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.sine_generator import SineGenerator
from skmultiflow.data.sea_generator import SEAGenerator
# Incremental Concept Drift Generators
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
# No Concept Drift Generators
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.evaluation import EvaluateHoldout
import datetime
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
class Study():
## TODO: List of string with stream names for individual studies
    def __init__(self,streams=None,path="/"):
        if streams == None:
                self.streams = self.init_standard_streams()
        else:
            self.streams = streams
        self.path = path
        self.date_format = "%Y-%m-%d %H-%M"
        self.metrics = ['accuracy', 'kappa_t', 'kappa_m', 'kappa']
        self.date = str(datetime.datetime.now().strftime(self.date_format))
        try:
            if not os.path.exists(self.path):
                os.mkdir(self.path)
        except Exception as e: raise FileNotFoundError("Error while creating Directory!")

    def init_standard_streams(self):
        s1 = SineGenerator(classification_function=0,balance_classes=False,random_state=112)
        s2 = SineGenerator(classification_function=1,balance_classes=False, random_state=112)
        ra_sine = ReoccuringDriftStream(stream=s1, drift_stream=s2,alpha=90.0,position=2000,width=1)
        rg_sine = ReoccuringDriftStream(stream=s1, drift_stream=s2,alpha=90.0,position=2000,width=1000)

        stagger1 = STAGGERGenerator(classification_function=0,balance_classes=False,random_state=112)
        stagger2 = STAGGERGenerator(classification_function=1,balance_classes=False,random_state=112)
        ra_stagger = ReoccuringDriftStream(stream=stagger1, drift_stream=stagger2, random_state=112,alpha=90.0,position=2000,width=1)
        rg_stagger = ReoccuringDriftStream(stream=stagger1, drift_stream=stagger2, random_state=112,alpha=90.0,position=2000,width=1000)

        sea1 = SEAGenerator(classification_function=0,balance_classes=False,random_state=112)
        sea2 = SEAGenerator(classification_function=1,balance_classes=False,random_state=112)
        ra_sea = ReoccuringDriftStream(stream=sea1, drift_stream=sea2, random_state=112,alpha=90.0,position=2000,width=1)
        rg_sea = ReoccuringDriftStream(stream=sea1, drift_stream=sea2, random_state=112,alpha=90.0,position=2000,width=1000)

        mixed1 = MIXEDGenerator(classification_function=0,random_state=112,balance_classes=False)
        mixed2 = MIXEDGenerator(classification_function=1,random_state=112,balance_classes=False)
        ra_mixed = ReoccuringDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112,alpha=90.0,position=2000,width=1)
        rg_mixed = ReoccuringDriftStream(stream=mixed1, drift_stream=mixed2, random_state=112,alpha=90.0,position=2000,width=1000)

        inc = HyperplaneGenerator(random_state=112)
        
        return [ra_sine,rg_sine,ra_stagger,rg_stagger,ra_sea,rg_sea,ra_mixed,rg_mixed,inc]


    def init_real_world(self):
        if not os.path.exists("datasets/"):
            raise FileNotFoundError("Folder for data cannot be found! Should be Datasets/")
         
        try:   
            #covertype = FileStream('datasets/covtype.csv') Label failure
            elec = FileStream('../datasets/elec.csv')
            #poker = FileStream('datasets/poker.csv') label failure
            weather = FileStream('../datasets/weather.csv')
            gmsc = FileStream('../datasets/gmsc.csv')
           # airlines = FileStream('datasets/airlines.csv') label failure
            moving_squares = FileStream('../datasets/moving_squares.csv')
            return [elec,weather,gmsc,moving_squares]
        except Exception as e: 
            raise FileNotFoundError("Real-world datasets can't loaded! Check directory Datasets/")
            return []
        

if __name__ == "__main__":  
    s = Study()
    print(s)
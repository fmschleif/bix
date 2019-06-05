import os
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.agrawal_generator import AGRAWALGenerator
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from skmultiflow.data.led_generator_drift import LEDGeneratorDrift
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from bix.classifiers.glvq import GLVQ # Generalized Learning Vector Quantization 
from skmultiflow.trees.hoeffding_tree import HoeffdingTree # Hoeffding Decision Tree
from skmultiflow.trees.hoeffding_adaptive_tree import HAT # Hoeffding Adaptive Decision Tree
from skmultiflow.lazy.knn import KNN # K-Nearest Neighbors Classifier
from skmultiflow.lazy.knn_adwin import KNNAdwin # K-Nearest Neighbors Classifier with ADWIN Change detector
from skmultiflow.lazy.sam_knn import SAMKNN #  K-Nearest Neighbors Classifier Self Adjusting Memory
from skmultiflow.meta.leverage_bagging import LeverageBagging # Leverage Bagging Classifier
from skmultiflow.bayes import NaiveBayes # Naive Bayes Classifier
from skmultiflow.meta.oza_bagging import OzaBagging # OzaBagging Classifier
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin # OzaBagging Classifier with ADWIN Change detector
from bix.classifiers.rslvq import RSLVQ # Robust Soft Learning Vector Quantization
from bix.classifiers.adaptive_rslvq import ARSLVQ # Adaptive Robust Soft Learning Vector Quantization
from bix.evaluation.crossvalidation import CrossValidation

def evaluation():
    classifiers = [GLVQ(prototypes_per_class=4),HoeffdingTree(),HAT(),KNN(),SAMKNN(),LeverageBagging(),KNNAdwin(max_window_size=1000)]   # Array mit Klassifikationsalgorithmen die getestet werden sollen
    cv = CrossValidation(clfs=classifiers,max_samples=1000000,test_size=1)
    cv.streams = cv.init_standard_streams() + cv.init_real_world() +   cv.init_reoccuring_streams() # initialisiert Stream Generatoren des Scikit-Multiflow Package
    cv.test()
    cv.save_summary()
    
def evaluation1():
    classifiers = [OzaBagging(base_estimator=HoeffdingTree()),OzaBaggingAdwin(base_estimator=HoeffdingTree())]   # Array mit Klassifikationsalgorithmen die getestet werden sollen
    cv = CrossValidation(clfs=classifiers,max_samples=1000000,test_size=1)
    cv.streams = cv.init_standard_streams() + cv.init_real_world() +   cv.init_reoccuring_streams() # initialisiert Stream Generatoren des Scikit-Multiflow Package
    cv.test()
    cv.save_summary()
    
def evaluation2():
    classifiers = [OzaBagging(base_estimator=KNN()),OzaBaggingAdwin(base_estimator=KNN()),RSLVQ(prototypes_per_class=4,sigma=6),ARSLVQ(prototypes_per_class=4,sigma=6)]   # Array mit Klassifikationsalgorithmen die getestet werden sollen
    cv = CrossValidation(clfs=classifiers,max_samples=1000000,test_size=1)
    cv.streams = cv.init_standard_streams() + cv.init_real_world() +   cv.init_reoccuring_streams() # initialisiert Stream Generatoren des Scikit-Multiflow Package
    cv.test()
    cv.save_summary()
    
def evaluation_Naive_Bayes() :
    classifiers = [NaiveBayes()]   # Array mit Klassifikationsalgorithmen die getestet werden sollen
    cv = CrossValidation(clfs=classifiers,max_samples=1000000,test_size=1)
    cv.streams = init_standard_streams_naive_bayes() + init_real_world_naive_bayes() + cv.init_reoccuring_streams() # initialisiert Stream Generatoren des Scikit-Multiflow Package
    cv.test()
    cv.save_summary()

def init_standard_streams_naive_bayes(): # RBF Stream beinhaltet negative Werte daher muss dieser beim Naive Bayes Algortihmus weggelassen werden
    """Initialize standard data streams
    
    Standard streams are inspired by the experiment settings of 
    Gomes, Heitor Murilo & Bifet, Albert & Read, Jesse & Barddal, Jean Paul & 
    Enembreck, Fabrício & Pfahringer, Bernhard & Holmes, Geoff & 
    Abdessalem, Talel. (2017). Adaptive random forests for evolving data 
    stream classification. Machine Learning. 1-27. 10.1007/s10994-017-5642-8. 
    """
    agrawal_a = ConceptDriftStream(stream=AGRAWALGenerator(random_state=112, perturbation=0.1), 
                        drift_stream=AGRAWALGenerator(random_state=112, 
                                                      classification_function=2, perturbation=0.1),
                        random_state=None,
                        alpha=90.0,
                        position=21000000)
    agrawal_a.name = "agrawal_a"                     
    agrawal_g = ConceptDriftStream(stream=AGRAWALGenerator(random_state=112, perturbation=0.1), 
                        drift_stream=AGRAWALGenerator(random_state=112, 
                                                      classification_function=1, perturbation=0.1),
                        random_state=None,
                        position=21000000,
                        width=1000000)
    agrawal_g.name = "agrawal_g"             
    hyper = HyperplaneGenerator(mag_change=0.001, noise_percentage=0.1)
    
    led_a = ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                        drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=7),
                        random_state=None,
                        alpha=90.0, # angle of change grade 0 - 90
                        position=21000000,
                        width=1)
 
    led_a.name = "led_a"
    led_g = ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                        drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=7),
                        random_state=None,
                        position=21000000,
                        width=1000000)
    led_g.name = "led_g"
    rand_tree = RandomTreeGenerator()
    rand_tree.name = "rand_tree" 
    #rbf_if = RandomRBFGeneratorDrift(change_speed=0.001)
    #rbf_if.name = "rbf_if"
    #rbf_im = RandomRBFGeneratorDrift(change_speed=0.0001)
    #rbf_im.name = "rbf_im"
    sea_a = ConceptDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1), 
                        drift_stream=SEAGenerator(random_state=112, 
                                                      classification_function=2, noise_percentage=0.1),
                        alpha=90.0,
                        random_state=None,
                        position=21000000,
                        width=1)  
    sea_a.name = "sea_a"                            
    sea_g = ConceptDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1), 
                        drift_stream=SEAGenerator(random_state=112, 
                                                      classification_function=1, noise_percentage=0.1),
                        random_state=None,
                        position=21000000,
                        width=1000000)
    sea_g.name = "sea_g"                  
    return [agrawal_a, agrawal_g, hyper, led_a, led_g, rand_tree, sea_a, sea_g]
    
def init_real_world_naive_bayes(): # weather beinhaltet negative Werte daher muss dieser Datensatz beim Naive Bayes Algortihmus weggelassen werden
    """Initialize real world data streams, will be loaded from file"""

    if not os.path.join("..","..","datasets"):
        raise FileNotFoundError("Folder for data cannot be found! Should be datasets")
    os.chdir(os.path.join("..","..","datasets"))
    try:   
        covertype = FileStream(os.path.realpath('covtype.csv')) # Label failure
        covertype.name = "covertype"
        elec = FileStream(os.path.realpath('elec.csv'))
        elec.name = "elec"
        poker = FileStream(os.path.realpath('poker.csv')) #label failure
        poker.name = "poker"
        #weather = FileStream(os.path.realpath('weather.csv'))
        #weather.name = "weather"
        gmsc = FileStream(os.path.realpath('gmsc.csv'))
        gmsc.name = "gmsc"
        moving_squares = FileStream(os.path.realpath('moving_squares.csv'))
        moving_squares.name = "moving_squares"
        return [covertype,elec, poker, gmsc, moving_squares]
    except Exception as e: 
        raise FileNotFoundError("Real-world datasets can't loaded! Check directory ")
        return []
    
if __name__ == "__main__":  # wird nur ausgeführt wenn als Hauptprogramm gestartet
   evaluation()
   evaluation1()
   evaluation2()
   evaluation_Naive_Bayes()
    
from random import random as rnd
from skmultiflow.data.sea_generator import SEAGenerator
import numpy as np
from bix.detectors.anwin import ANWIN
import unittest

class TESTANWIN(unittest.TestCase):
    """TESTTANWIN
        TODO: Test initialization
    """ 
    def test_alpha(self):
        with self.assertRaises(ValueError):
            ANWIN(alpha=-0.1)
        with self.assertRaises(ValueError):
            ANWIN(alpha=1.1)
        ANWIN(alpha=0.5)

    def test_anwin(self):
        anwin = ANWIN(alpha=0.001)
        stream = SEAGenerator(classification_function = 2, random_state = 112, balance_classes = False,noise_percentage = 0.28)
        stream.prepare_for_use()

        stream.restart()
        detections,mean = [],[]
        
        print("\n--------------------\n")
        for i in range(10000):
            data = stream.next_sample(10)
            batch = data[0][0][0]
            mean.append(batch)
            anwin.add_element(batch)
            if anwin.detected_change():
                print("\rIteration {}".format(i))
                print("\r ANWINReject Null Hypotheses")
                print(np.mean(mean))
                mean = []
                detections.append(i)

        print("----- Number of detections: "+str(len(detections))+ " -----")
        self.assertGreaterEqual(len(detections),5)

if __name__ == "__main__":
    unittest.main()
    
    
  
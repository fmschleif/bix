from skmultiflow.data.sea_generator import SEAGenerator
import numpy as np
from bix.detectors.kswin import KSWIN
import unittest

class TESTKSWIN(unittest.TestCase):
    """TESTTKSWIN
        TODO: Test initialization
    """ 
    def test_alpha(self):
        with self.assertRaises(ValueError):
            KSWIN(alpha=-0.1)
        with self.assertRaises(ValueError):
            KSWIN(alpha=1.1)
        KSWIN(alpha=0.5)

    def test_data(self):
        kswin = KSWIN(data="st")
        self.assertIsInstance()(kswin.window,list)
    def test_kswin(self):
        kswin = KSWIN(alpha=0.001)
        stream = SEAGenerator(classification_function = 2, random_state = 112, balance_classes = False,noise_percentage = 0.28)
        stream.prepare_for_use()

        stream.restart()
        detections,mean = [],[]
        
        print("\n--------------------\n")
        for i in range(10000):
            data = stream.next_sample(10)
            batch = data[0][0][0]
            mean.append(batch)
            kswin.add_element(batch)
            if kswin.detected_change():
                print("\rIteration {}".format(i))
                print("\r KSWINReject Null Hyptheses")
                print(np.mean(mean))
                mean = []
                detections.append(i)

        print("----- Number of detections: "+str(len(detections))+ " -----")
        self.assertGreaterEqual(len(detections),10)

if __name__ == "__main__":
    unittest.main()
    
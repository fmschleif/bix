from random import random as rnd
import sys
from skmultiflow.data.sea_generator import SEAGenerator
import numpy as np
from bix.detectors.anwin import ANWIN


def anwin_test():
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

    print(len(detections))

if __name__ == "__main__":
    anwin_test()
    
  
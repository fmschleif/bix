import pandas as pd
import numpy as np


# df = pd.read_csv("prediction_results1.csv",header=0,index_col=0)
#
#
# df = df * 100;
# df = df.round(2)
# print(df)
# df.to_csv("pred1.csv",sep="&")
#
# df1 = pd.read_csv("prediction_results.csv",header=0,index_col=0)
#
# df1 = df1 * 100;
# df1 = df1.round(2)
# print(df1)
#
# df1.to_csv("pred.csv",sep="&")
#
# t = list(df.values[-1,:])
# t  = t + list(df1.values[-1,:])
# t = np.array(t).reshape(2,4)
# print(np.mean(t,axis=0))

df = pd.read_csv("confusion_matrix.csv",header=0)
df1 = pd.read_csv("confusion_matrix1.csv",header=0)

df = df+df1

df.to_csv("cm.csv",sep="&")

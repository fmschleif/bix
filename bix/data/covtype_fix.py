import numpy as np 
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.join("..","..","datasets"):
    raise FileNotFoundError("Folder for data cannot be found! Should be datasets")
os.chdir(os.path.join("..","..","datasets"))
try:   
    covertype = pd.read_csv(os.path.realpath('covtype.csv')) # Label failure
    print(covertype.head())
    values = covertype.values
    values[:,-1] = values[:,-1] - 1
    covertype_new = pd.DataFrame(columns=list(covertype),data=values) 
    covertype_new = covertype_new.round(6)
    print(covertype_new.head())

    covertype_new.to_csv("covtype_new.csv",index=False)
except Exception as e: 
    raise FileNotFoundError("Covtype not found")

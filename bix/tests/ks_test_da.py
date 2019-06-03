import pandas as pd
import numpy as np

d = pd.read_csv("bix/evaluation/results.csv",header=None)
results = d.values
# results = np.delete(results, np.s_[], axis=1)
names = ["svm","tca","jda","tkl","sa","nbt"]

import scipy.stats as stats
a = results[:6,:]
b = results[6:12,:]
c = results[12:,:]
nbt = a[:,-1]
# print("Reuters")
# for i in a.T:
#     print(stats.ks_2samp(i,nbt))
# print("newsgroup")
# nbt = b[:,-1]
# for i in b.T:
#     print(stats.ks_2samp(i,nbt))
print("Image")
nbt = c[:,-1]

for i in c.T:
    print(stats.ks_2samp(i,nbt))
print("overall")
nbt = results[:,-1].T
for i in results.T:
    print(stats.ks_2samp(i,nbt))
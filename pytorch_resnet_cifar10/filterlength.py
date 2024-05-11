## Number of filters at every layer for WRN-40-4

list_1 = [61, 16, 38, 20, 49, 15, 42, 27, 38, 25, 38, 62, 127, 54, 82, 37, 63, 50, 60, 44, 58, 36, 40, 128, 256, 153, 236, 171, 191, 172, 156, 129, 120, 57, 42, 256]
list_2 = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]

newlist_1 = []
newlist_2 = []

for i in range(len(list_1)):
    if (i+1)%2!=0:
        temp = 0        
        temp  = temp+list_1[i]
    else:
        temp = temp+list_1[i]
        newlist_1.append(temp)
        temp=0
    
for i in range(len(list_2)):
    if (i+1)%2!=0:
        temp = 0        
        temp  = temp+list_2[i]
    else:
        temp = temp+list_2[i]
        newlist_2.append(temp)
        temp=0

  
def int2bits(n):
    return [list('{i:0>{n}b}'.format(i=i, n=n)) for i in range(2**n)]

import numpy as np
import matplotlib.pyplot as plt

n = len(newlist_1)
combinations = int2bits(n)


filterlength1 = []
for i in range(2**n):
    filterlength1.append(np.sum([np.multiply(float(combinations[i][j]), list_1[j]) for j in range(len(combinations[i]))]))

filterlength2 = []
for i in range(2**n):
    filterlength2.append(np.sum([np.multiply(float(combinations[i][j]), list_2[j]) for j in range(len(combinations[i]))]))




plt.figure(figsize=(5,3.5))
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)
plt.hist(filterlength2, histtype='step', label="ReLU", color="black", alpha=0.5)
plt.hist(filterlength1, histtype='step',label="RReLU", color="black", alpha=0.95)
plt.xlabel("filter-path length")
plt.ylabel("Number of paths")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid(linestyle='--', linewidth=0.5)
plt.legend(loc='upper right')
plt.savefig("filterlength1")
# exit()
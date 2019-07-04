from RPCA import RPCA
from sklearn.datasets import load_diabetes as load
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data=load().data

q=0.9 # q is in [0,1],and the bigger q, the more positive noise. the small q, the more negtive noise
p=0.9 # p is in [0,1],and the bigger p, the less noise. the small p, the more noise.

noise=np.random.random(data.shape)
#what is the sign of noise?
sign=np.sign(np.random.random(data.shape)-q)

pick=np.random.random(data.shape)
pick=np.where(pick>p,1,0)
noise=20*noise*pick*sign

#where have noise point in the matrix?
sns.heatmap(pick)

ans = RPCA(data+noise, w=0.1, tol=1e-6, itermax=1000, p=1.2, u=1e-3, umax=1e10)
plt.figure()
sns.heatmap(data,vmax=0.2,vmin=0)
#sns.heatmap(data)
plt.title("the original data")

plt.figure()
sns.heatmap(data+noise,vmax=0.2,vmin=0)
#sns.heatmap(data+noise)
plt.title("the noisy data")

plt.figure()
sns.heatmap(ans["low_rank"],vmax=0.2,vmin=0)
#sns.heatmap(ans["low_rank"])
plt.title("the data after denoise")

plt.show()
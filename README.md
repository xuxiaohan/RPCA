# RPCA  
a python code of RPCA

I develope this project to record my study trace.  
Anyone can use my code for any legitimate use without let me know.

if you feel my code is helpful for your work, your star is the greatest encouragement for me.
-------


# details
In many application, we assume that the data should be low rank, but for the reason of some noise, the rank of data matrix far bigger than the truth. If the noise is approximately gaussian distribution with small variance. PCA can work well. But if the noise do not satisfy the assumption, the performance of PCA will decline.(it means that PCA can not find the truth manifold)  

RPCA is a method to find the low rank components when the data is corrupted by sparse noise. No matter what distribution the noise is, it can work well.  

it just assume that  
* the number of noise point in the matrix is not big.
* the truth of data is at a linear subspace manifold.


the way you call the method
```python
RPCA(D,w,u=1e-3,umax=1e10,p=1.2,itermax=300,tol=1e-8)
```
# example

this is some data of diabetes from sklearn. and I generate some noise and add to the data, then I use RPCA to denoise. the result is shown as follow:

![](https://github.com/xuxiaohan/RPCA/blob/master/fig/fig_5.png?raw=true)

||figure|
|---|---|
|the position of noise point|![](https://github.com/xuxiaohan/RPCA/blob/master/fig/fig_1.png?raw=true)|
|the clean data|![](https://github.com/xuxiaohan/RPCA/blob/master/fig/fig_2.png?raw=true)|
|the data with noise|![](https://github.com/xuxiaohan/RPCA/blob/master/fig/fig_3.png?raw=true)|
|the result of RPCA|![](https://github.com/xuxiaohan/RPCA/blob/master/fig/fig_4.png?raw=true)|



# Reference  
1. University S , EMMANUEL J. CANDE`, Xiaodong L I . Robust Principal Component Analysis?[J]. 2009.
2. Chandrasekaran V , Sanghavi S , Parrilo P A , et al. Sparse and Low-Rank Matrix Decompositions[J]. IFAC Proceedings Volumes, 2009, 42(10):1493-1498.
3. 


# Contact  
You can contact me if there's any problem.  
## author: 

Xu Han(hxu10670@gmail.com)

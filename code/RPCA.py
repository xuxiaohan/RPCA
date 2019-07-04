import numpy as np
import logging

def prox_nuclear(D,mu):
    u,s,vh=np.linalg.svd(D,full_matrices=False)
    idx=np.where(s>mu)[0]
    if(len(idx)):
        return u[:,idx]@np.diag(s[idx]-mu)@vh[idx,:],s[idx]-mu
    else:
        return np.zeros_like(D), np.zeros_like(s[idx])

def prox_L1(D,u):
    ans=np.where(D>u,D-u,0)
    ans=np.where(D<-u,D+u,ans)
    return ans

def RPCA(D,w,u=1e-3,umax=1e10,p=1.2,itermax=300,tol=1e-8):
    """
    decomponent D=L+S
    :param D: a ndarry object with ndim==2, the data matrix
    :param w: a hyperparameter, scalar, the bigger w means the less noise you think, the small w will lead more point of matrix be regarded as noise.
        it balances the two prior knowledge, the low rank of L or the sparse of S.
    :param u: a parameter for optimize method ADMM, you can ignore it safely, if you do not know what it means
    :param umax: a parameter for optimize method ADMM, you can ignore it safely, if you do not know what it means
    :param p: a parameter for optimize method ADMM, you can ignore it safely, if you do not know what it means
    :param itermax: the max iteration when solve
    :param tol: the tolerance of error value
    :return:
        a dict:
            "low_rank": L
            "sparse": S
            "iter": the number of iteration when finish
            "obj": the object value of RPCA
            "err": the real error value
    """
    L=np.zeros_like(D)
    E=L.copy()
    Y=L.copy()
    for i in range(itermax):
        #update L
        Lk,s=prox_nuclear(-E+D-Y/u,1/u)
        #update E
        Ek=prox_L1(-Lk+D-Y/u,w/u)
        #check
        changeL=np.abs(Lk-L).max()
        changeE=np.abs(Ek-E).max()
        changeerr=np.abs(Lk+Ek-D).max()
        change=max(changeE,changeerr,changeL)
        if(change<tol):
            break
        else:
            E=Ek
            L=Lk
        #update Y and u
        Y=Y+u*(E+L-D)
        u=min(umax,p*u)
        #obj = s.sum() + w * np.abs(E).sum()
        #print("iter : ",i,"\nerr : ",cherr,"\nobj : ",obj,"\nnuclearnorm : ",s.sum(),"\nL1 norm : ",np.abs(E).sum())

    obj=s.sum()+w*np.abs(E).sum()
    return dict(low_rank=L,sparse=E,iter=i,err=changeerr,obj=obj)
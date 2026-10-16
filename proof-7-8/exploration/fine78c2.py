import sys, numpy as np
from scipy.optimize import minimize
from model78c import check
k0=float(sys.argv[1]); p0=np.array([float(v) for v in sys.argv[2:5]])
def worst(p,s0): 
    if p[0]<0: return 1+abs(p[0])
    return max(check(dict(b=p[0],lx=p[1],ell=p[2],s0=s0),k0,nd=50,nx=11).values())
def feas(s0,p):
    r=minimize(worst,p,args=(s0,),method='Nelder-Mead',options=dict(xatol=1e-6,fatol=1e-8,maxiter=400))
    return r.fun,r.x
lo,hi=0.868,0.87289; p=p0
for it in range(10):
    mid=(lo+hi)/2; f,x=feas(mid,p)
    if f<-1e-6: hi,p,best=mid,x,(f,x)
    else: lo=mid
v=check(dict(b=p[0],lx=p[1],ell=p[2],s0=hi),k0,nd=120,nx=21)
print("FINE",k0,round(hi,5),np.round(p,5),{k:round(float(a),6) for k,a in v.items()})

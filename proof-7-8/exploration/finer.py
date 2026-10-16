import sys, numpy as np
from scipy.optimize import minimize
from model78r import check
k0=float(sys.argv[1]); rho=float(sys.argv[2]); cons=sys.argv[3]=='c'
p0=np.array([float(v) for v in sys.argv[4:8]])
def worst(p,s0):
    if p[0]<0 or p[3]<1/50: return 1+abs(min(p[0],p[3]-1/50))
    return max(check(dict(b=p[0],lx=p[1],ell=p[2],d1=p[3],s0=s0),k0,rho,cons,nd=40,nx=9).values())
lo,hi=0.80,0.8755; p=p0
for it in range(13):
    mid=(lo+hi)/2
    best=None
    for start in (p, p0):
        r=minimize(worst,start,args=(mid,),method='Nelder-Mead',options=dict(xatol=1e-6,fatol=1e-8,maxiter=600))
        if best is None or r.fun<best.fun: best=r
    if best.fun<-1e-6: hi,p=mid,best.x
    else: lo=mid
v=check(dict(b=p[0],lx=p[1],ell=p[2],d1=p[3],s0=hi),k0,rho,cons,nd=100,nx=21)
print("R",k0,rho,'cons' if cons else 'opt',round(hi,5),np.round(p,5),{k:round(float(a),6) for k,a in v.items()},flush=True)

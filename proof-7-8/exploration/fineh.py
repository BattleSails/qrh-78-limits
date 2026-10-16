import sys, numpy as np
from scipy.optimize import minimize
from model78h import check
k0=float(sys.argv[1]); cons=sys.argv[2]=='c'
p0=np.array([float(v) for v in sys.argv[3:7]])
def worst(p,s0):
    if p[0]<0: return 1+abs(p[0])
    return max(check(dict(b=p[0],lx=p[1],ell=p[2],f=p[3],s0=s0),k0,cons,nd=40,nx=9).values())
lo,hi=0.80,0.8755; p=p0
for it in range(13):
    mid=(lo+hi)/2; best=None
    for start in (p,p0):
        r=minimize(worst,start,args=(mid,),method='Nelder-Mead',options=dict(xatol=1e-6,fatol=1e-8,maxiter=800))
        if best is None or r.fun<best.fun: best=r
    if best.fun<-1e-6: hi,p=mid,best.x
    else: lo=mid
v=check(dict(b=p[0],lx=p[1],ell=p[2],f=p[3],s0=hi),k0,cons,nd=100,nx=21)
print("H",k0,'cons' if cons else 'opt',round(hi,5),np.round(p,5),'hmax',round(1-p[1]+p[2],4),'h',round(p[3]*(1-p[1]+p[2]),4),{k:round(float(a),6) for k,a in v.items()},flush=True)

import sys, numpy as np
from scipy.optimize import minimize
from model78d import check
k0=float(sys.argv[1]); c=float(sys.argv[2]); cons=sys.argv[3]=='c'
p0=np.array([float(v) for v in sys.argv[4:7]])
def worst(p,s0):
    if p[0]<0: return 1+abs(p[0])
    v=check(dict(b=p[0],lx=p[1],ell=p[2],s0=s0),k0,c,cons,nd=40,nx=9); v.pop('dstar')
    return max(v.values())
lo,hi=0.80,0.8755; p=p0
for it in range(13):
    mid=(lo+hi)/2; best=None
    for start in (p,p0):
        r=minimize(worst,start,args=(mid,),method='Nelder-Mead',options=dict(xatol=1e-6,fatol=1e-8,maxiter=600))
        if best is None or r.fun<best.fun: best=r
    if best.fun<-1e-6: hi,p=mid,best.x
    else: lo=mid
g=dict(b=p[0],lx=p[1],ell=p[2],s0=hi); v=check(g,k0,c,cons,nd=100,nx=21)
h=1-p[1]+p[2]
print("D",k0,c,'cons' if cons else 'opt',round(hi,5),np.round(p,5),'h',round(h,4),'d*',round(v['dstar']+h+1,4),{k:round(float(a),6) for k,a in v.items() if k!='dstar'},flush=True)

# Slack in the 7/8 endpoint certificate (OAI78 Lemma 20.2) when the long-range
# inverse count L(t) is improved by a hypothetical MS(kappa) for kappa in [k0,1].
import numpy as np
h=13/16; C0=-1/48
def Rshort(t,d,x):
    Dx=3-17*x/9; Px=(2-8*x/9)*(1-x)
    return 1-d+d*Px/Dx*(1.5-t)
def Lnew(t,d,k0):
    a=5*k0/6                       # amplification exponent from MS(k0)
    long_ = 1/6+(a-d)*t if t>1/k0 else -9
    return max(1-d, long_)
def E(d,x,R): return C0+2*d/3+x*d/6-h*(1-R)
ts=np.linspace(1,1.5,2001)
for k0 in [1.0,0.97,0.95,0.9,0.8,2/3]:
    worst=-1;arg=None
    for d in np.linspace(0.02,5/6,330):
        for x in np.linspace(0,0.5,26):
            R=min(max(Rshort(t,d,x),Lnew(t,d,k0)) for t in ts)
            e=E(d,x,R)
            if e>worst: worst,arg=e,(d,x)
    print(f"k0={k0:.3f}: max E(h) = {worst:+.6f} at delta={arg[0]:.3f}, x={arg[1]:.2f}")



#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from math import exp


#%%
vect_invR=[]
nT=500 #nb of different delta
#vect_delta=np.linspace(0.1,2,nT) #vector of durations
vect_sigma=np.linspace(0.001,5,nT)
xup0=1.44#1.83
nxi=1000 #nb of inital values
tmin=0 #minimal time
#parameters for the driving signal
xs1p0=0.62#0.36 #calculated on the 1D model
#parameters for the driving signal
pb=3.5 #basal value
p0=4 #value for the first return map
pmax=5 #max value
m=2 #center of the signal

vect_y0=np.linspace(0,xup0,nxi)
def F(y,t,sigma):
    #args:
    #y=initial condition
    #t=time vector for the resolution
    #delta=duration of the driving signal
    #return : solution of the differential system
    p=(pmax-pb)/(1-np.exp(-m**2/sigma**2))*(np.exp(-(t-m)**2/sigma**2)-1)+pmax
    dydt=p-(y**3-6*y**2+9*y+0.5)
    return dydt

vect_delta=[]

def g_dicho(y0,sigma):
    #args : 
    #y0=initial condition 
    #delta=duration of the signal
    #return : (xup0-R(x))
    delta=2*np.sqrt(-sigma**2*np.log(1+(p0-pmax)/(pmax-pb)*(1-np.exp(-m**2/sigma**2))))
    t=np.linspace(0,m+delta/2,1000)
    y=odeint(F,y0,t,(sigma,))
    RT=y[-1,0]
    return((xup0-RT))
def dicho(sigma,eps):
    #Bisection method to find the zero of g_dicho
    #args:
    #delta=duration of the driving signal
    #eps=tolerance for the algorithm
    ya=0
    yb=xup0
    
    while(abs(yb-ya)>eps):
        ym=(ya+yb)/2
        if(g_dicho(ym,sigma))>=0:
            ya=ym
        else:
            yb=ym
    return(ym)
eps=0.0001
vect_delta=[]
vect_invR=[]
for i in range(len(vect_sigma)):
    sigma=vect_sigma[i]
    delta=2*np.sqrt(-sigma**2*np.log(1+(p0-pmax)/(pmax-pb)*(1-np.exp(-m**2/sigma**2))))
    vect_delta.append(delta)
    xr=dicho(sigma,eps)
    vect_invR.append(xr)

dpi=300 #dpi resoultion for graphics
plt.figure(1)
fig, ax = plt.subplots()
ax.plot(vect_delta,vect_invR,color='red')
ax.plot([min(vect_delta),max(vect_delta)], [xup0,xup0],linestyle='--',color='red')
mask = (np.array(vect_delta)<= 2.44) & (np.array(vect_delta) >= 0)#color blue zone
arr_vect_invR=np.array(vect_invR)
vect_xup0=np.linspace(xup0,xup0,len(vect_invR))
mask2= (np.array(vect_delta) >= min(vect_delta)) & (np.array(vect_delta) <= max(vect_delta))#color green zone
plt.fill_between(np.array(vect_delta)[mask2], arr_vect_invR[mask2], vect_xup0[mask2], color='lightgreen', alpha=0.5)
plt.fill_between(np.array(vect_delta)[mask], arr_vect_invR[mask], color='skyblue', alpha=0.5)
plt.text(min(vect_delta)-0.8,xup0,r'$x_{p_0}^u \simeq 1.44$',color='red',fontsize=11)
plt.text(2.7,0.5,r'$I_{recep}$',color='green')
plt.text(1,0.5,r'$I_{insens}$',color='blue')
plt.text(min(vect_delta)-0.7,1.25,r'$\bar{x}=1.25$')
#ax.text(ax.get_xlim()[0] - 0.8, 1.5, r"$\bar{x}=$", ha='right', va='center',fontsize=11)
plt.plot([2.37,2.37],[0,1.25],linestyle='--',color='black')
plt.plot([min(vect_delta),2.37],[1.25,1.25],color='black',linestyle='--')
plt.xlabel(' $\delta_{p_0} (a.u.)$',loc='right')
plt.ylabel('$R^{-1}(x_{p_0}^u)$\n (a.u.)',rotation=90,loc='top')
ax.text(2.37, ax.get_ylim()[0] - 0.01, r'$T^i_{\bar{x}}=\tau_x$', ha='center', va='top',fontsize=11,rotation=-45)
ax.text(2.5, ax.get_ylim()[0]-0.02 , '$T^r$', ha='center', va='top',fontsize=11)
plt.yticks([0, 0.5,1,2]) 
plt.xticks([0,1,2,3],rotation=-45)
plt.tight_layout()
plt.savefig('Fig3_NI_length_receptive_set.png', dpi=dpi)
plt.show
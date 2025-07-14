
"""
Goal : to  calculate the learning time of the system
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from math import exp


#%%


xup0=1.44#1.83
xs1p0=0.62#0.36 #calculated on the 1D model
tmin=0
#parameters for the driving signal
tmax=2 #center of the signal
p0=4
pb=3.5
pmax=5
vect_sigma=np.linspace(0.01,2,500) #vector of durations


def F(t, y, sigma, pmax, pb, m):
    p=(pmax-pb)/(1-np.exp(-tmax**2/sigma**2))*(np.exp(-(t-tmax)**2/sigma**2)-1)+pmax
    dydt=p-(y**3-6*y**2+9*y+0.5)
    return dydt


vect_delta=[]
vect_Rxs1=[] #vector that contain the first return map of xs1p0
y0=[xs1p0]
for i in range(len(vect_sigma)):
    sigma=vect_sigma[i] #duration of the signal
    delta=2*np.sqrt(-sigma**2*np.log(1+(p0-pmax)/(pmax-pb)*(1-np.exp(-tmax**2/sigma**2))))
    vect_delta.append(delta)
    t_span = (0, tmax+delta/2)  # Intervalle de temps
    t_eval = np.linspace(t_span[0], t_span[1], 100)  # Points de temps où la solution est évaluée

    # Résoudre le système d'équations différentielles
    sol = solve_ivp(F, t_span, y0, args=(sigma, pmax, pb, tmax), t_eval=t_eval, method='RK45')
    Ry=sol.y[0]
    vect_Rxs1.append(Ry[-1])
    
indcross= np.where(np.diff(np.sign( np.array(vect_Rxs1)- xup0)))[0]
#plotting 
dpi=300 #dpi resoultion for graphics
plt.figure(1)
fig, ax = plt.subplots()
ax.plot(vect_delta,vect_Rxs1)
ax.plot([0,3], [xup0,xup0],linestyle='--',color='red')
plt.xlabel('$\delta_{p_0}(p)$ (a.u.)',labelpad=20,loc='right')
plt.ylabel(r'$R(x^{s_1}_{p_0}) (a.u.)$',labelpad=20,loc='top')
plt.plot([vect_delta[indcross[0]],vect_delta[indcross[0]]], [0,xup0],linestyle='--',color='red')
ax.text(ax.get_xlim()[0] - 0.05, xup0, fr'$x^u_{{p_0}} \simeq {xup0}$', ha='right', va='center',color='red',fontsize=11)
tau=round(vect_delta[indcross[0]],2)
ax.text(vect_delta[indcross[0]]+0.15, ax.get_ylim()[0]-0.003 , fr'$\tau_L \simeq {tau}$', ha='center', va='top',color='red',fontsize=11,rotation=-30)
plt.yticks([0,1,2,3,4]) 
plt.xticks([0,1,2,3])
plt.tight_layout()
plt.savefig('Fig4_NI_learning_duration.png', dpi=dpi)
plt.show()
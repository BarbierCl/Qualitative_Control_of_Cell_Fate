
"""
Goal : to represent the receptive and insensitive sets
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from math import exp

dpi=300 #dpi resoultion for graphics
vect_sigma=[0.5,1.2,1.6] #vector of durations
xup0=1.44#1.83 # bifurcation value
nxi=500 #nb points for vect_xi (initial conditions)
tmin=0
# driving signal parameters
pb=3.5 #basal value
p0=4 #value for the first return map
pmax=5 #max value
m=2 #center of the signal

vect_xi=np.linspace(-5,xup0,nxi) #vector of initial conditions
vect_delta=[]
def F(t, y, sigma, pmax, pb, m):
    p=(pmax-pb)/(1-np.exp(-m**2/sigma**2))*(np.exp(-(t-m)**2/sigma**2)-1)+pmax
    dydt=p-(y**3-6*y**2+9*y+0.5)
    return dydt



vect_return_map=[[] for _ in range(5)] #initialization of 5 empty vectors, they will contain the return map function for several initial conditions

for i in range(len(vect_sigma)):
    sigma=vect_sigma[i]
    delta=2*np.sqrt(-sigma**2*np.log(1+(p0-pmax)/(pmax-pb)*(1-np.exp(-m**2/sigma**2))))
    vect_delta.append(delta)
    t_span = (0, m+delta/2)  # Intervalle de temps
    t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Points de temps où la solution est évaluée
    
    for j in range(len(vect_xi)):
        xi=vect_xi[j]
        y0=[xi]
        sol = solve_ivp(F, t_span, y0, args=(sigma, pmax, pb, m), t_eval=t_eval, method='RK45',rtol=1e-8, atol=1e-10,)
        RT=sol.y[0]
        vect_return_map[i].append(RT[-1])

#plotting of different first return maps
mask0=(vect_xi>=0) & (vect_xi<=xup0)
arr_vect_return_map_0=np.array(vect_return_map[0])
arr_vect_return_map_1=np.array(vect_return_map[1])
arr_vect_return_map_2=np.array(vect_return_map[2]) #conversion to an array
plt.figure(0)
plt.text(0,2,r"$x_{p_{0}}^u$",color='red')
plt.plot(vect_xi[mask0],arr_vect_return_map_0[mask0],label=fr"$\delta_{{p_0}} = {round(vect_delta[0],4)}$")
plt.plot(vect_xi[mask0],arr_vect_return_map_1[mask0],label=fr"$\delta_{{p_0}} = {round(vect_delta[1],4)}$")
plt.plot(vect_xi[mask0],arr_vect_return_map_2[mask0],label=fr"$\delta_{{p_0}} = {round(vect_delta[2],4)}$")
plt.plot([0,xup0],[xup0,xup0],color='red',linestyle='--')
plt.legend()
plt.xlabel("$x$(a.u.)",loc='right')
plt.ylabel(r" $R(x)$ (a.u.)",loc='top')
plt.tight_layout()
plt.savefig('Fig3_NI_several_durations.png', dpi=dpi)
plt.show()

###################################################

mask2 = (vect_xi <= xup0) & (vect_xi >= 0)#zoom for initial conditions between 0 and xup0
arr_vect_return_map_1=np.array(vect_return_map[1]) #conversion to an array
indcross=np.where(np.diff(np.sign(arr_vect_return_map_1 - xup0)))[0] 
valx=vect_xi[indcross[0]]
plt.figure(3)
plt.text(0,xup0+0.1,r"$x_{p_{0}}^u$",color='red')
plt.plot(vect_xi[mask2],arr_vect_return_map_1[mask2],color='orange',label=fr"$\delta_{{p_0}} = {round(vect_delta[1],2)}$")
plt.plot([0,xup0],[xup0,xup0],color='red',linestyle='--')
plt.plot([vect_xi[indcross[0]],vect_xi[indcross[0]]],[0,xup0],color='red',linestyle='--')
#insensitive set
mask2= (vect_xi <= vect_xi[indcross[0]]) & (vect_xi >= 0)# zoom to color green zone
vect_xup0=np.linspace(xup0,xup0,len(vect_xi))
vect_xmax=np.linspace(max(vect_return_map[1]),max(vect_return_map[1]),len(vect_xi))
plt.fill_between(vect_xi[mask2], vect_xmax[mask2], color='skyblue', alpha=0.5,label="$I_{insens}$")
#receptive set
mask3= (vect_xi >= vect_xi[indcross[0]]) & (vect_xi <= xup0)#zoom to color blue zone
vect_xmax=np.linspace(max(vect_return_map[1]),max(vect_return_map[1]),len(vect_xi))
#plt.fill_between(vect_xi[mask3],vect_xup0[mask3] ,vect_xmax[mask3], color='lightgreen', alpha=0.5,label="$I_{recep}$")
plt.fill_between(vect_xi[mask3] ,vect_xmax[mask3], color='lightgreen', alpha=0.5,label="$I_{recep}$")
plt.legend(loc='lower left')
plt.text(valx,-0.4,round(valx,3),color="red",rotation=-45)
plt.xlabel("$x$(a.u.)",loc='right')
plt.ylabel(r" $R(x)$ (a.u.)",loc='top')
plt.xticks([0,0.3,0.7,1,1.4],rotation=-45)
plt.tight_layout()
plt.savefig('Fig3_NI_intermediate_duration.png', dpi=dpi)
plt.show()
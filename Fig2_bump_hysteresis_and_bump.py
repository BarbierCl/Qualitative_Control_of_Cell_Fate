import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve

"""
Goal : to plot the histeresis of the 1D model and the bump signal used
"""
#driving signal
pb=3.5
p0=4
pmax=5
pe=4.5
xup0=1.44
xs1p0=0.62
xs2p0=3.94
dpi=300 #graphic resolution
# y values for the histeresis plot
y = np.linspace(0, 4.2, 300)
ftsize=11 #size of caracters
p=(y**3-6*y**2+9*y+0.5)
# Plotting
plt.plot(p, y,linewidth=1.5)
plt.plot([4.5,4.5], [0,1],linestyle='--',color='red')
plt.text(4.3,-1.2,r'$p^*=4.5$',color='red',fontsize=ftsize,rotation=-60)
plt.plot([6,6], [0,4.15],linestyle='--',color='red')
plt.text(5.6,-1.2,r'$p_{max}=6$',color='red',fontsize=ftsize,rotation=-60)
plt.plot([0.5,0.5], [0,3],linestyle='--',color='red')
plt.text(0.4,-1.2,r'$p^{**}=0.5$',color='red',fontsize=ftsize,rotation=-60)
plt.plot([p0,p0], [0,xs2p0],linestyle='--',color='red')
plt.text(3.5,-1,fr'$p_0={p0}$',color='red',fontsize=ftsize,rotation=-60)
plt.plot([0,p0], [xup0,xup0],linestyle='--',color='red')
plt.text(-1.5,1.4,fr'${{x}}^{{u}}_{{p_0}}\simeq {xup0}$',color='red',fontsize=ftsize)
plt.plot([0,p0], [xs2p0,xs2p0],linestyle='--',color='red')
plt.text(-1.5,3.8,fr'${{x}}^{{s_2}}_{{p_0}}\simeq {xs2p0}$',color='red',fontsize=ftsize)
plt.plot([0,p0], [xs1p0,xs1p0],linestyle='--',color='red')
plt.text(-1.5,0.6,fr'${{x}}^{{s_1}}_{{p_0}}\simeq {xs1p0}$',color='red',fontsize=ftsize)
plt.plot([0,0.5], [3,3],linestyle='--',color='purple')
plt.text(-1.3,3,r'$x^{**}=3$',color='purple',fontsize=ftsize)
plt.plot([0,4.5], [1,1],linestyle='--',color='purple')
plt.text(-1.3,1,r'$x^{*}=1$',color='purple',fontsize=ftsize)
plt.xlabel('$p$ (a.u.)',loc='right')
plt.ylabel('$y$ (a.u.)',loc='top')
plt.yticks([0, 2,5]) 
plt.xticks([0,1,3,7])
plt.tight_layout()
plt.savefig('Fig2_NI_memory_system.png', dpi=dpi)
plt.show()



#driving signal
m=2
tmin=0
tmax=5
sigma=4
delta=2*np.sqrt(-sigma**2*np.log(1+(p0-pmax)/(pmax-pb)*(1-np.exp(-m**2/sigma**2))))
t=np.linspace(tmin,tmax,1000)

p=(pmax-pb)/(1-np.exp(-m**2/sigma**2))*(np.exp(-(t-m)**2/sigma**2)-1)+pmax

indp0= np.where(np.diff(np.sign( p- p0)))[0]
indpe= np.where(np.diff(np.sign( p- pe)))[0]
indpb= np.where(np.diff(np.sign( p- pb)))[0]
dd=t[indp0[1]]-t[indp0[0]]


plt.figure(2)
plt.plot(t,p)
plt.plot([0,t[indp0[1]]], [p0,p0],linestyle='--',color='red')
plt.text(-1.1, p0, fr'$p_0 = {p0}$', color='red', fontsize=ftsize)
plt.plot([0,t[indpb[1]]], [pb,pb],linestyle='--',color='red')
plt.text(-1.1,pb,fr'$p_b= {pb}$',color='red',fontsize=ftsize)
plt.plot([0,t[indpe[1]]], [pe,pe],linestyle='--',color='red')
plt.text(-1.2,pe,fr'$p^*={pe}$',color='red',fontsize=ftsize)
plt.plot([0,m], [pmax,pmax],linestyle='--',color='red')
plt.text(-1.2,pmax,fr'$p_{{max}}= {pmax}$',color='red',fontsize=ftsize)
plt.plot([t[indp0[0]],t[indp0[0]]], [2,p0],linestyle='--',color='red')
plt.text(t[indp0[0]]-0.1,1.1,fr'$t_0={round(t[indp0[0]],2)}$',color='red',fontsize=ftsize,rotation=-60)
plt.plot([t[indp0[1]],t[indp0[1]]], [2,p0],linestyle='--',color='red')
plt.text(t[indp0[1]]-0.1,1,fr'$t_1={round(t[indp0[1]],2)}$',color='red',fontsize=ftsize,rotation=-60)
plt.plot([t[indpe[1]],t[indpe[1]]], [2,pe],linestyle='--',color='red')
plt.text(t[indpe[1]]-0.1,1.1,fr'$t_2^*={round(t[indpe[1]],2)}$',color='red',fontsize=ftsize,rotation=-60)
plt.plot([t[indpe[0]],t[indpe[0]]], [2,pe],linestyle='--',color='red')
plt.text(t[indpe[0]]-0.1,1,fr'$t_1^*={round(t[indpe[0]],2)}$',color='red',fontsize=ftsize,rotation=-60)
plt.plot([m,m], [2,pmax],linestyle='--',color='red')
plt.text(m,1.1,r'$t_{max}=5$',color='red',fontsize=ftsize,rotation=-60)
plt.plot([0,0], [2,2],linestyle='--',color='red')
plt.text(-0.3,1.3,r'$t_{i}=0$',color='red',fontsize=ftsize,rotation=-60)
#plt.plot([tmax,tmax], [2,pmax],linestyle='--',color='red')
plt.text(tmax-0.1,1.1,r'$t_{f}=10$',color='red',fontsize=ftsize,rotation=-60)
plt.xticks([6])             # Positions des ticks sur l'axe x
plt.yticks([3,6]) 
plt.xlabel('t (a.u.)',loc='right')
plt.ylabel(' $p$ (a.u.)',loc='top')
plt.tight_layout()
plt.savefig('Fig2_NI_dynamic_signal.png', dpi=dpi)
plt.show()
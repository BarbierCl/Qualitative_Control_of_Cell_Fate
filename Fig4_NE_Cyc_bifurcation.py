

"""
Created on Mon Jan 13 10:18:44 2025

@author: clair
"""
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema

"""Programmation du modele HAT
p53 arrester is a bump. The cell cycle arrest module is isolated. Test conformité copasi
"""

#%%
#parameters
DNA_DSB_due_to_IR=0.06728856167
is_IR_switched_on=0
has_DNA_DSB_repair=1
rep=5*10**(-5)#has_DNA_DSB_repair*0.001
a1=3*10**(-10)
a2=10**(-12)
b1=0.00003
b2=0.003
b3=0.003
b4=10**(-5)
b5=10**(-5)
d1=10**(-8)
d2=0.00003
d3=0.0001
d4=10**(-10)
d5=0.0001
d6=10**(-10)
d7=3*10**(-7)
d8=0.0001
d9=0.00003
d10=0.0001
d11=10**(-10)
d12=10000
g1=0.0003
g2=0.0003
g3=0.0003
g4=0.0003
g5=0.0003
g6=0.00003
g7=10**(-13)
g8=0.0003
g9=0.0001
g10=10**(-5)
g11=10**(-11)
g12=10**(-13)
g13=10**(-13)
g14=0.0001
g15=0.00003
g16=0.0001
g17=0.0003
g18=0.0003
g19=0.0003
g20=10**(-4)
g101=10**(-5)
h=2
Vdam=0
h1=10**(-6)
h2=10**(-13)
i1=0.001
p1=0.0003
p2=10**(-8)
p3=0.00000003
p4=10**(-10)
p5=10**(-8)
p6=10**(-8)
p7=0.000000003
p8=0.000000003
p9=3*10**(-6)
p10=3*10**(-6)
p11=10**(-10)
p12=10**(-9)
q2=0.003
s1=0.1
s2=0.03
s3=0.1
s4=0.03
s5=0.1
s6=300
s7=30
s8=30
s9=30
s10=3
t1=0.1
t2=0.1
t3=0.1
t4=0.1
t5=0.1
u1=0.001
u2=0.001
u3=0.001
u5=0.0001
u6=10**(-4)
q0_p21=10**(-5)
q1_p21=3*10**(-13)
q0_mdm2=0.0001
q1_mdm2=3*10**(-13)
q0_wip1=10**(-5)
q1_wip1=3*10**(-13)
q0_pten=10**(-5)
q1_pten=3*10**(-13)
q0_bax=10**(-5)
q1_bax=3*10**(-13)
M1=5
M2=100000
M3=200000
DNA_DSB_max=10**6
DNA_DSB_RepairCplx_total=20
#initial concentrations core module p53
p530_phosp=44467.02517
p53_arrester=0.001965464411
p53S46_phosp=1086.687376
p53_killer=6.948787724*10**(-5)
ATM_phosp=0.000264078234
ATM_tot=100000
ATM_0=99999.99974
SIAH1_0=100000
SIAH1_phosp=0
SIAH1_tot=100000
HIPK2=18911.07604
Mdm2_mRNA=10.75268817
Mdm2_c_0_UU=2733.746264
Mdm2_c_PU=778.5380494
Mdm2_n_PU=25951.26603
Mdm2_n_PP=0.0005270168519
Wip1=369.139904
Wip1_mRNA=1.107419712
SIAH1_tot=100000
DNA_DSB=2.850172717*10**(-5)
AKT_phosp=32181.04063
AKT_tot=100000
AKT_0=67818.95937
PIP_tot=100000
PIP2=52548.60746
PIP3=47451.39254
PI3K_tot=100000
PTEN_mRNA=0.3322259136
PTEN=1107.419712
#initial concentrations module cell cycle arrest
p21_mRNA=1.107419712
p21_free=0.1301763107
cycE_free=170081.5113
cycE_p21_cpxe=1107.029183
RB1_0_free=300000
RB1_E2F1_cpxe=11422.96565
RB_phosp=-11422.96565
RB_tot=300000
E2F1=188577.0343
E2F_tot=200000
#initial concentration module killer
BAX_mRNA=0.3322259136
BAX_free=0.1425079792
BclXL_free=100000
BclXL_tot=100000
BAX_BclXL_cpxe=332.0834056
BAD_0_free=60000
BAD_phosph_free=0.1019050416
proCasp=99999.98575
Casp=0.01425086359
BclXL_BAD_cpxe=-332.0834056
BAD_phosph_F33_cpxe=331.9815006
F33_free=199668.0185
F33_tot=200000
BAD_tot=60000
#name of variables core module p53
x1i=p530_phosp
x2i=p53_arrester
x3i=p53S46_phosp
x4i=p53_killer
x5i=ATM_phosp
x6i=SIAH1_0
x7i=ATM_0
x8i=HIPK2
x9i=Mdm2_mRNA
x10i=Mdm2_c_0_UU
x11i=Mdm2_c_PU
x12i=Mdm2_n_PU
x13i=Mdm2_n_PP
x14i=Wip1
x15i=Wip1_mRNA
x16i=DNA_DSB
x17i=AKT_phosp
x18i=PIP3
x19i=PTEN_mRNA
x20i=PTEN
x21i=PIP2
x22i=AKT_0 
x23i=SIAH1_phosp
#name of variables module cell cycle arrest
x24i=p21_mRNA
x25i=p21_free
x26i=cycE_free
x27i=cycE_p21_cpxe
x28i=RB1_0_free
x29i=RB1_E2F1_cpxe
x30i=RB_phosp
x31i=E2F1
#name of variables module killer
x32i=BAX_mRNA
x33i=BAX_free
x34i=BclXL_free
x35i=BAX_BclXL_cpxe
x36i=BAD_0_free
x37i=BAD_phosph_free
x38i=proCasp
x39i=Casp
x40i=BclXL_BAD_cpxe
x41i=BAD_phosph_F33_cpxe
x42i=F33_free
#temps

x2=100

##system definition
def systeme_diff(y,t,param,h,M1,M2,M3,rep):
    # y est un vecteur qui contiendra les conditions initiales des variables
    #param est un vetcuer qui contient les paramètres
    #Affectation des variables et des paramètres
    x24,x25,x26,x27,x28,x29,x30,x31= y
    a1,a2,b1,b2,b3,b4,b5,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g101,\
    i1,p1,p2,p3,p4,p5,p6,p7,p8,p9,p11,p12,q2,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,t1,t2,t3,t4,t5,u1,u2,u3,u5,u6,q0_mdm2,q1_mdm2,q0_wip1,q1_wip1,\
    q0_pten,q1_pten,q0_p21,q1_p21,q0_bax,q1_bax,DNA_DSB_max,DNA_DSB_RepairCplx_total,PI3K_tot,x2=param

    #Differential equations 
    
   
    dx24dt=s5*(q0_p21+q1_p21*x2**h)/(q2+q0_p21+q1_p21*x2**h)-g5*x24 #p21_mRNA
    dx25dt=t5*x24-b5*x25*x26+u6*x27-g19*x25 #p21_free
    dx26dt=-b5*x25*x26+u6*x27+s10+s9*x31**2/(M3**2+x31**2)-g20*x26 #cycE_free
    dx27dt=b5*x25*x26-u6*x27-g20*x27 #cycE_p21_cpxe
    dx28dt=d12*x30/(M2+x30)-b4*x31*x28-p9*x26*x28+u5*x29 #RB0_1_free
    dx29dt=b4*x31*x28-u5*x29-p10*x26*x29 #RB1_E2F1_cpxe
    dx30dt=-dx28dt-dx29dt #RB_phosp
    dx31dt=-dx29dt #E2F1
   
    return [dx24dt,dx25dt,dx26dt,dx27dt,dx28dt,dx29dt,dx30dt,dx31dt]

def make_stopping_event(systeme_diff, param_init, h,M1, M2, M3, rep, eps=1e-6):
    #Goal : stop the computation in solveivp when the solution for p53 forward is stable
    def event_func(t, y):
        if y[2]>100:  # apoptosis treshol
            return 1.0  # no stabilisation
        dydt = systeme_diff(y, t, param_init,h, M1, M2, M3, rep)
        return abs(dydt[2]) - eps
    event_func.terminal = True
    event_func.direction = 0
    return event_func

           


#################### bacward################################################
vect_x2_reverse=np.linspace(120000,0,100)
species_init=(x24i,x25i,x26i,x27i,x28i,x29i,x30i,x31i)
tmin=0 #time min
tmax=20*24*3600
nbptstime=30000 #nb points echantillonage time
t = np.linspace(tmin, tmax, nbptstime) 
vect_fin_cyc_reverse=[]
for i in range(len(vect_x2_reverse)):
    x2=vect_x2_reverse[i]
    param_init=[a1,a2,b1,b2,b3,b4,b5,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g101,
                i1,p1,p2,p3,p4,p5,p6,p7,p8,p9,p11,p12,q2,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,t1,t2,t3,t4,t5,u1,u2,u3,u5,u6,q0_mdm2,q1_mdm2,q0_wip1,q1_wip1,
                q0_pten,q1_pten,q0_p21,q1_p21,q0_bax,q1_bax,DNA_DSB_max,DNA_DSB_RepairCplx_total,PI3K_tot,x2]
    
    sol= solve_ivp(lambda t, y: systeme_diff(y, t, param_init,h,M1,M2,M3,rep), 
            [tmin, tmax], 
            species_init,method='BDF', 
            t_eval=t,atol=1e-6, rtol=1e-6)
    ycyc=sol.y[2]
    vect_fin_cyc_reverse.append(ycyc[-1])
    yp21mrna=sol.y[0]
    x24i=yp21mrna[-1]
    yp21free=sol.y[1]
    x25i=yp21free[-1]
    x26i=ycyc[-1]
    ycycp21=sol.y[3]
    x27i=ycycp21[-1]
    yrb01free=sol.y[4]
    x28i=yrb01free[-1]
    yrb1e2f1=sol.y[5]
    x29i=yrb1e2f1[-1]
    yrbp=sol.y[6]
    x30i=yrbp[-1]
    ye2f1=sol.y[7]
    x31i=ye2f1[-1]
    species_init=(x24i,x25i,x26i,x27i,x28i,x29i,x30i,x31i)
    
    ############## forward###################
vect_x2=np.linspace(0,120000,100)
species_init=(x24i,x25i,x26i,x27i,x28i,x29i,x30i,x31i)
tmin=0 #time min
tmax=20*24*3600
nbptstime=30000 #nb points echantillonage time
t = np.linspace(tmin, tmax, nbptstime) 
vect_fin_cyc=[]
for i in range(len(vect_x2)):
    x2=vect_x2[i]
    param_init=[a1,a2,b1,b2,b3,b4,b5,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g101,
                i1,p1,p2,p3,p4,p5,p6,p7,p8,p9,p11,p12,q2,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,t1,t2,t3,t4,t5,u1,u2,u3,u5,u6,q0_mdm2,q1_mdm2,q0_wip1,q1_wip1,
                q0_pten,q1_pten,q0_p21,q1_p21,q0_bax,q1_bax,DNA_DSB_max,DNA_DSB_RepairCplx_total,PI3K_tot,x2]
   
    sol= solve_ivp(lambda t, y: systeme_diff(y, t, param_init,h,M1,M2,M3,rep), 
            [tmin, tmax], 
            species_init,method='BDF', 
            t_eval=t,atol=1e-6, rtol=1e-6)
    ycyc=sol.y[2]
    vect_fin_cyc.append(ycyc[-1])
    yp21mrna=sol.y[0]
    x24i=yp21mrna[-1]
    yp21free=sol.y[1]
    x25i=yp21free[-1]
    x26i=ycyc[-1]
    ycycp21=sol.y[3]
    x27i=ycycp21[-1]
    yrb01free=sol.y[4]
    x28i=yrb01free[-1]
    yrb1e2f1=sol.y[5]
    x29i=yrb1e2f1[-1]
    yrbp=sol.y[6]
    x30i=yrbp[-1]
    ye2f1=sol.y[7]
    x31i=ye2f1[-1]
    species_init=(x24i,x25i,x26i,x27i,x28i,x29i,x30i,x31i)
    
#find p** and p*
vect_diff_cyc_reverse=[]
for i in range(len(vect_fin_cyc_reverse)-1):
    vect_diff_cyc_reverse.append(abs(vect_fin_cyc_reverse[i+1]-vect_fin_cyc_reverse[i]))
max_diff_cyc_reverse=max(vect_diff_cyc_reverse)
indpee=vect_diff_cyc_reverse.index(max_diff_cyc_reverse)
pee=vect_x2_reverse[indpee]
round_pee=round(pee,2)
vect_diff_cyc=[]
for i in range(len(vect_fin_cyc)-1):
    vect_diff_cyc.append(abs(vect_fin_cyc[i+1]-vect_fin_cyc[i]))
max_diff_cyc=max(vect_diff_cyc)
indpe=vect_diff_cyc.index(max_diff_cyc)
pe=vect_x2[indpe]
round_pe=round(pe,2)

dpi=300 #dpi resoultion for graphics
plt.figure(0)
plt.plot(vect_x2_reverse,vect_fin_cyc_reverse,label='p53 arrester backward', color='blue')
plt.plot(vect_x2,vect_fin_cyc,label='p53 arrester forward', color='green')
plt.text(80000,100000,"$p_{max}=110000$",color="red")
plt.plot([110000,110000],[0,175000],color='red', linestyle='dashed')
plt.xticks(rotation=45)
plt.text(-5000,2000,fr'$p^{{**}}$ = {round(pee,2)}',color='blue')
plt.text(65000,2000,fr'$p^{{*}}$ = {round(pe,2)}',color='green')
plt.xlabel("p53 arrester (mmol/ml)")
plt.ylabel('CycE concntration (mmol/ml)')
plt.legend()
plt.tight_layout()
plt.savefig('Fig4_NE_bifurcation_CycE.png', dpi=dpi)
plt.show()
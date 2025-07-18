

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema

"""Programmation du modele HAT
Goal : to provide the bifurcation diagram of Casp with p53killer as a parameter in the apoptosis modul
"""

#%%
#parameters
DNA_DSB_due_to_IR=0.06728856167
is_IR_switched_on=0
has_DNA_DSB_repair=1
rep=5*10**(-5)
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
x4=100


##system definition
def systeme_diff(y,t,param,h,M1,M2,M3,rep):
  
    x17,x18,x19,x20,x21,x22,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42= y
    a1,a2,b1,b2,b3,b4,b5,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g101,\
    i1,p1,p2,p3,p4,p5,p6,p7,p8,p9,p11,p12,q2,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,t1,t2,t3,t4,t5,u1,u2,u3,u5,u6,q0_mdm2,q1_mdm2,q0_wip1,q1_wip1,\
    q0_pten,q1_pten,q0_p21,q1_p21,q0_bax,q1_bax,DNA_DSB_max,DNA_DSB_RepairCplx_total,PI3K_tot,x4=param

    #Differential equations 

    dx17dt=p12*x22*x18-d8*x17 #AKT_phosp 
    dx18dt=p8*x21*PI3K_tot-d7*x20*x18 #PIP3 
    dx19dt=s2*(q0_pten+q1_pten*x4**h)/(q2+q0_pten+q1_pten*x4**h)-g2*x19 #PTENmRNA 
    dx20dt=t2*x19-g6*x20 #PTEN
    dx21dt=-p8*x21*PI3K_tot+d7*x20*x18 #PIP2
    dx22dt=-p12*x22*x18+d8*x17 #AKT0
    dx32dt=s4*(q0_bax+q1_bax*x4**h)/(q2+q0_bax+q1_bax*x4**h)-g4*x32 #BAX_mRNA
    dx33dt=t4*x32+u1*x35-b1*x33*x34-g9*x33 #BAX_free
    dx34dt=u1*x35-b1*x33*x34+u2*x40+g16*x35+p7*x17*x40-b2*x34*x36 #BclXL_free
    dx35dt=-u1*x35+b1*x33*x34-g16*x35 #BAX_Bcl_cpxe
    dx36dt=-b2*x34*x36+d9*x37+u2*x40-p7*x17*x36+d9*x41 #BAD_0_free
    dx37dt=-d9*x37+p7*x17*x36+u3*x41+p7*x17*x40-b3*x42*x37 #BAD_Phosp_free
    dx38dt=s7-g17*x38-a1*x33*x38-a2*x39**2*x38 #proCASP
    dx39dt=a1*x33*x38+a2*x39**2*x38-g18*x39 #CASP
    dx40dt=-dx34dt-dx35dt #BclXL_Bad_cpxe
    dx41dt=-dx40dt-dx36dt-dx37dt#BAD_phosph_F33_cpxe
    dx42dt=-dx41dt #F33_free
    return [dx17dt,dx18dt,dx19dt,dx20dt,dx21dt,dx22dt,dx32dt,dx33dt,dx34dt,dx35dt,dx36dt,dx37dt,dx38dt,dx39dt,dx40dt,dx41dt,dx42dt]
    

def make_stopping_event_f(systeme_diff, param_init, h,M1, M2, M3, rep, eps=1e-6):
    #Goal : stop the computation in solveivp when the solution for p53 forward is stable
    def event_func(t, y):
        if y[13] < 96000:  # apoptosis treshol
            return 1.0  # no stabilisation
        dydt = systeme_diff(y, t, param_init,h, M1, M2, M3, rep)
        return abs(dydt[13]) - eps
    event_func.terminal = True
    event_func.direction = 0
    return event_func

def make_stopping_event_b(systeme_diff, param_init, h,M1, M2, M3, rep, eps=1e-6):
    #Goal : stop the computation in solveivp when the solution for p53 backward is stable
    def event_func(t, y):
        if(param_init[-1]>=0):
            if y[13] <96000:  # apoptosis treshol
                return 1.0  # no stabilisation
            dydt = systeme_diff(y, t, param_init,h, M1, M2, M3, rep)
        if(param_init[-1]<0):
            if y[13] >100:  # apoptosis treshol
                return 1.0  # no stabilisation
            dydt = systeme_diff(y, t, param_init,h, M1, M2, M3, rep)
        return abs(dydt[13]) - eps
    event_func.terminal = True
    event_func.direction = 0
    return event_func
######################## x4 = p53killer forward ##################################         
species_init=(x17i,x18i,x19i,x20i,x21i,x22i,x32i,x33i,x34i,x35i,x36i,x37i,x38i,x39i,x40i,x41i,x42i)
tmin=0 #time min
tmax=20*24*3600 #time max 20 days
nbptstime=30000 #nb points echantillonage time
t = np.linspace(tmin, tmax, nbptstime) 
vect_x4=np.linspace(90000,120000,100)
#vect_x4=np.linspace(-10000,110000,100)
vect_fin_casp=[]
for i in range(len(vect_x4)):
    x4=vect_x4[i]
    param_init=[a1,a2,b1,b2,b3,b4,b5,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g101,
            i1,p1,p2,p3,p4,p5,p6,p7,p8,p9,p11,p12,q2,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,t1,t2,t3,t4,t5,u1,u2,u3,u5,u6,q0_mdm2,q1_mdm2,q0_wip1,q1_wip1,
            q0_pten,q1_pten,q0_p21,q1_p21,q0_bax,q1_bax,DNA_DSB_max,DNA_DSB_RepairCplx_total,PI3K_tot,x4]
    event_func = make_stopping_event_f(systeme_diff, param_init, h,M1, M2, M3, rep)
    sol= solve_ivp(lambda t, y: systeme_diff(y, t, param_init,h,M1,M2,M3,rep), 
            [tmin, tmax], 
            species_init,method='BDF', 
            t_eval=t,atol=1e-6, rtol=1e-6,events=event_func)
    ycasp=sol.y[13] #casp solution
    vect_fin_casp.append(ycasp[-1]) #asymptotiv value of casp
    species_init= sol.y[:, -1] #For the continuation the edo has to be solved with the lattest values as initialization
    
######################## x4 = p53killer backward ##################################         
species_init=(x17i,x18i,x19i,x20i,x21i,x22i,x32i,x33i,x34i,x35i,x36i,x37i,x38i,x39i,x40i,x41i,x42i)
tmin=0 #time min
tmax=20*24*3600
nbptstime=30000 #nb points echantillonage time
t = np.linspace(tmin, tmax, nbptstime) 
vect_x4_back=np.linspace(120000,90000,100)
#vect_x4_back=np.linspace(110000,-10000,100)
vect_fin_casp_back=[]
for i in range(len(vect_x4_back)):
    x4=vect_x4_back[i]
    param_init=[a1,a2,b1,b2,b3,b4,b5,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g101,
            i1,p1,p2,p3,p4,p5,p6,p7,p8,p9,p11,p12,q2,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,t1,t2,t3,t4,t5,u1,u2,u3,u5,u6,q0_mdm2,q1_mdm2,q0_wip1,q1_wip1,
            q0_pten,q1_pten,q0_p21,q1_p21,q0_bax,q1_bax,DNA_DSB_max,DNA_DSB_RepairCplx_total,PI3K_tot,x4]
    event_func = make_stopping_event_b(systeme_diff, param_init, h,M1, M2, M3, rep)
    sol_back= solve_ivp(lambda t, y: systeme_diff(y, t, param_init,h,M1,M2,M3,rep), 
            [tmin, tmax], 
            species_init,method='BDF', 
            t_eval=t,atol=1e-6, rtol=1e-6,events=event_func)
    ycasp_back=sol_back.y[13]
    vect_fin_casp_back.append(ycasp_back[-1])
    species_init= sol_back.y[:, -1]
################################################################################
#find the bifurcation p*
vect_diff_casp=[]
for i in range(len(vect_fin_casp)-1):
    vect_diff_casp.append(abs(vect_fin_casp[i+1]-vect_fin_casp[i]))
max_diff_casp=max(vect_diff_casp)
indpe=vect_diff_casp.index(max_diff_casp)
pe=vect_x4[indpe]
round_pe=round(pe,2)
###############################
dpi=300 #dpi resoultion for graphics
plt.figure(0)
plt.plot(vect_x4,vect_fin_casp,label="p53killer forward",color="green")
plt.plot(vect_x4_back,vect_fin_casp_back,label="p53killer backward",color="blue")
plt.text(112000,50000,"$p_{max}=110000$",color="red")
plt.plot([110000,110000],[0,100000],color='red', linestyle='dashed')
plt.xticks(rotation=45)
plt.text(round_pe,0,fr'$p^{{*}}$ = {round_pe}',color='green')
plt.xlabel("p53 killer (mmol/ml)")
plt.ylabel("Casp concentration (mmol/ml)")
plt.legend()
plt.tight_layout()
plt.savefig('Fig4_NE_bifurcation_Casp.png', dpi=dpi)
plt.show()

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
p53_0 = 1 bump de periode T
on modifie la ^periode T et on regarde le temps passé par p53_killer au dessus
de p*killer et le temps passé par p_arrester au dessus de p*arrester
Les conditions initiales sont modifiées pour être celles de l'état d'équilibre
bump parrester avec niveau basal au dessus de p** et max supérieur à max(p*kill,p*arrest)
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


##system definition
def systeme_diff(y,t,param,M1,M2,M3,rep,pmax,pb,sigma,m):
    # y est un vecteur qui contiendra les conditions initiales des variables
    #param est un vetcuer qui contient les paramètres
    #Affectation des variables et des paramètres
    x1, x2, x3, x4,x5, x6, x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,\
    x34,x35,x36,x37,x38,x39,x40,x41,x42= y
    a1,a2,b1,b2,b3,b4,b5,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g101,\
    i1,p1,p2,p3,p4,p5,p6,p7,p8,p9,p11,p12,q2,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,t1,t2,t3,t4,t5,u1,u2,u3,u5,u6,q0_mdm2,q1_mdm2,q0_wip1,q1_wip1,\
    q0_pten,q1_pten,q0_p21,q1_p21,q0_bax,q1_bax,DNA_DSB_max,DNA_DSB_RepairCplx_total,PI3K_tot=param

    #Differential equations 
    dx1dt = d3*x2-(g101+g11*x12**h)*x1+d11*x14*x3-p11*x8*x1+s6-p3*x5*x1 #p53_0_p
    dx2dt = -d3*x2-(g101+g12*x12**h)*x2-p4*x8*x2+d4*x14*x4+p3*x5*x1 #p53arrester 
    dx3dt=-d11*x14*x3+p11*x8*x1-p3*x5*x3-(g10+g13*x12**h)*x3+d10*x4 #p53S46phosp 
    dx4dt=p4*x8*x2-d4*x14*x4+p3*x5*x3-(g10+g13*x12**h )*x4-d10*x4 #p53 killer 
    dx5dt = -d1*x14*x5+(p1*x7*x16**h)/(x16**h+M1**h) #ATMp
    dx6dt =-p2*x5*x6+d2*x23#SIAH1_0 
    dx7dt=d1*x14*x5-(p1*x7*x16**h)/(x16**h+M1**h)
    dx8dt = s8-g7*x8*(x6+x12)**2 #HIPK2 
    dx9dt=(s3*(q0_mdm2+q1_mdm2*x2**h))/(q2+q0_mdm2+q1_mdm2*x2**h)-g3*x9 #Mdm2 mRNA 
    dx10dt=t3*x9-p5*x17*x10+d5*x11-g14*x10 #Mdm2cUU 
    dx11dt=p5*x17*x10-d5*x11-i1*x11-g15*x11 #Mdm2cPU  
    dx12dt=i1*x11-g15*x12-p6*x5*x12+d6*x14*x13 #Mdm2_n_PU 
    dx13dt=p6*x5*x12-d6*x14*x13-g15*x13-g16*x13 #Mdm2n_PP 
    dx14dt=t1*x15-g8*x14 #Wip1 
    dx15dt=s1*(q0_wip1+q1_wip1*x2**h)/(q2+q0_wip1+q1_wip1*x2**h)-g1*x15 #Wip1mRNA 
    dx16dt=(h1*DNA_DSB_due_to_IR*is_IR_switched_on+h2*x39)*(DNA_DSB_max-x16)-rep*x16+(pmax-pb)/(1-np.exp(-m**2/sigma**2))*(np.exp(-(t-m)**2/sigma**2)-1)+pmax
    dx17dt=p12*x22*x18-d8*x17 #AKT_phosp 
    dx18dt=p8*x21*PI3K_tot-d7*x20*x18 #PIP3 
    dx19dt=s2*(q0_pten+q1_pten*x4**h)/(q2+q0_pten+q1_pten*x4**h)-g2*x19 #PTENmRNA 
    dx20dt=t2*x19-g6*x20 #PTEN
    dx21dt=-p8*x21*PI3K_tot+d7*x20*x18 
    dx22dt=-p12*x22*x18+d8*x17
    dx23dt=p2*x5*x6-d2*x23 #SIAH phosp
    dx24dt=s5*(q0_p21+q1_p21*x2**h)/(q2+q0_p21+q1_p21*x2**h)-g5*x24 #p21_mRNA
    dx25dt=t5*x24-b5*x25*x26+u6*x27-g19*x25 #p21_free
    dx26dt=-b5*x25*x26+u6*x27+s10+s9*x31**2/(M3**2+x31**2)-g20*x26 #cycE_free
    dx27dt=b5*x25*x26-u6*x27-g20*x27 #cycE_p21_cpxe
    dx28dt=d12*x30/(M2+x30)-b4*x31*x28-p9*x26*x28+u5*x29 #RB0_1_free
    dx29dt=b4*x31*x28-u5*x29-p10*x26*x29 #RB1_E2F1_cpxe
    dx30dt=-dx28dt-dx29dt #RB_phosp
    dx31dt=-dx29dt #E2F1
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
    
    return [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt, dx8dt,dx9dt,dx10dt,dx11dt,dx12dt,dx13dt,dx14dt,\
            dx15dt,dx16dt,dx17dt,dx18dt,dx19dt,dx20dt,dx21dt,dx22dt,dx23dt,dx24dt,dx25dt,dx26dt,dx27dt,dx28dt,\
            dx29dt,dx30dt,dx31dt,dx32dt,dx33dt,dx34dt,dx35dt,dx36dt,dx37dt,dx38dt,dx39dt,dx40dt,dx41dt,dx42dt]
        
def find_max_grad(vecteur1, vecteur2):
    # Goal : to find the maximal gradinet of vect_fin_cyc
    differences1 = np.diff(vecteur1)
    differences2 = np.diff(vecteur2)

    pentes = np.abs(differences1) / differences2

    pente_maximale = np.max(pentes)
    indice_pente_maximale = np.argmax(pentes)

    return pente_maximale, indice_pente_maximale

#vecteur des paramètres
param_init=[a1,a2,b1,b2,b3,b4,b5,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g101,
            i1,p1,p2,p3,p4,p5,p6,p7,p8,p9,p11,p12,q2,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,t1,t2,t3,t4,t5,u1,u2,u3,u5,u6,q0_mdm2,q1_mdm2,q0_wip1,q1_wip1,
            q0_pten,q1_pten,q0_p21,q1_p21,q0_bax,q1_bax,DNA_DSB_max,DNA_DSB_RepairCplx_total,PI3K_tot]

# p0=7e-5#1.65e-5#1e-4#1.65e-5
# pmax=0.5e-3#3e-4
# pb=1e-5
p0=4e-5#•1.65e-5
pmax=4e-4#3e-4
pb=1e-5
vect_delta=[]
m=2*24*3600
sigma=150000
delta=2*np.sqrt(-sigma**2*np.log(1+(p0-pmax)/(pmax-pb)*(1-np.exp(-m**2/sigma**2))))

species_init=(x1i, x2i, x3i, x4i,x5i, x6i, x7i,x8i,x9i,x10i,x11i,x12i,x13i,x14i,x15i,x16i,x17i,x18i,x19i,x20i,x21i,x22i,
          x23i,x24i,x25i,x26i,x27i,x28i,x29i,x30i,x31i,x32i,x33i,x34i,x35i,x36i,x37i,x38i,x39i,x40i,x41i,x42i)
vect_sigma=np.linspace(1000,100000,100)
vect_fin_casp=[]
vect_fin_cyc=[]
vect_yDNA=[[] for _ in range(len(vect_sigma))] 
for i in range(len(vect_sigma)):
    sigma=vect_sigma[i]
    delta=2*np.sqrt(-sigma**2*np.log(1+(p0-pmax)/(pmax-pb)*(1-np.exp(-m**2/sigma**2))))
    #vect_delta.append(delta/86400)
    vect_delta.append(delta/3600)
    tmin=0 #time min
    tmax=m+delta/2
   
    nbptstime=100000
    t = np.linspace(tmin, tmax, nbptstime) 
    sol= solve_ivp(lambda t, y: systeme_diff(y, t, param_init,M1,M2,M3,rep,pmax,pb,sigma,m), 
                [tmin, tmax], 
                species_init,method='BDF', 
                t_eval=t,atol=1e-6, rtol=1e-6)
    ycasp=sol.y[38]
    vect_fin_casp.append(ycasp[-1])
    vect_yDNA[i].append(sol.y[15])
    ycyc=sol.y[25]
    vect_fin_cyc.append(ycyc[-1])

##### plotting 
dpi=300 #dpi resoultion for graphics    
cca_tresh=100 #○treshold of bifurcation for CycE
ind_cyc=np.where(np.array(vect_fin_cyc)<cca_tresh)
learning_time_cyc=vect_delta[ind_cyc[0][0]]
plt.figure(0)
plt.plot(vect_delta,vect_fin_cyc,"--x")
plt.text(10, 10000, fr"$\tau=${round(learning_time_cyc,2)}")
plt.text(30,80000,fr"$p_0=${p0}")
plt.text(30,60000, fr"$p_{{max}}=${pmax}")
plt.xlabel(r"${\delta}_{p_{0}}(V_{dam})$ (h)")
plt.ylabel(r"$R_{p_0}(CycE^*)$ (mmol/ml)")
plt.tight_layout()
plt.savefig('Fig7_NE_Vdam_learning_duration_Cyc.png', dpi=dpi)
plt.show()




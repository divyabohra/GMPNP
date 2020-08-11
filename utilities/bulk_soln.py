#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Created on Mon Oct 22 12:05:00 2018

@author: divyabohra

This script takes as input the electrolyte composition and concentration and calculates
the equilibrium concentration of all solution species after adding CO2.

'''
import numpy as np 
import math
from scipy.integrate import odeint
import yaml
import matplotlib.pyplot as plt

basepath_utilities = '/Users/divyabohra/Documents/Project2/source/exp/utilities/'

def kinetics(y, t, ka1, ka2, kb1, kb2):
	C_HCO3 = y[0]
	C_OH = y[1]
	C_CO32 = y[2]
	C_CO2 = y[3]
	dC_HCO3_dt = kb1 * C_CO2 * C_OH - kb2 * C_HCO3 - ka1 * C_HCO3 * C_OH + ka2 * C_CO32
	dC_OH_dt = ka2 * C_CO32 - ka1 * C_HCO3 * C_OH + kb2 * C_HCO3 - kb1 * C_CO2 * C_OH
	dC_CO32_dt = ka1 * C_HCO3 * C_OH - ka2 * C_CO32
	dC_CO2_dt = kb2 * C_HCO3 - kb1 * C_CO2 * C_OH
	return [dC_HCO3_dt, dC_OH_dt, dC_CO32_dt, dC_CO2_dt]

def CO2_conc(temp, fugacity_CO2, ions={'K':0.0, 'HCO3':0.0, 'OH': 0.0, 'CO32':0.0, 'Cl':0.0}): 
	#load data from yaml file
	f = open(basepath_utilities+'parameters.yaml')
	data = yaml.load(f)
	sechenov_const = data['sechonov_const']
	h_CO2_0 = sechenov_const['h_CO2_0']
	h_CO2_T = sechenov_const['h_CO2_T']
	
	#Henry's constant as a function of T. [CO2]aq,0=K_H_CO2*[CO2]g
	lnK_H_CO2 = 93.4517 * (100 / temp) - 60.2409 + 23.3585 * math.log(temp / 100) 
	h_CO2 = h_CO2_0 + h_CO2_T * (temp - 298.15)  #Sechenov model parameter for CO2

	sechenov = 0.0
	for ion in ions.keys():
		# convert concentration from molm-3 to kmolm-3
		add = (data['sechonov_const']['h_ion_'+ion] + h_CO2) * (ions[ion] / 1000) 
		sechenov+= add
	
	K_H_CO2 = math.exp(lnK_H_CO2)
	#initial concentration of dissolved CO2 in mol/m3
	C0_CO2 = fugacity_CO2 * K_H_CO2 * 1000 * 10 ** (-sechenov) 
	f.close()
	return C0_CO2

def kinetics_const_CO2(y, t, ka1, ka2, kb1, kb2):
	C0_CO2 = CO2_conc(temp=T, fugacity_CO2=f_CO2, ions={'K':C_init_K, 'HCO3':C0_HCO3, 'OH': C0_OH, 'CO32':C0_CO32, 'Cl': C_init_Cl}) #in mM or in molm-3
	C_HCO3 = y[0]
	C_OH = y[1]
	C_CO32 = y[2]
	dC_HCO3_dt = kb1 * C0_CO2 * C_OH - kb2 * C_HCO3 - ka1 * C_HCO3 * C_OH + ka2 * C_CO32
	dC_OH_dt = ka2 * C_CO32 - ka1 * C_HCO3 * C_OH + kb2 * C_HCO3 - kb1 * C0_CO2 * C_OH
	dC_CO32_dt = ka1 * C_HCO3 * C_OH - ka2 * C_CO32
	return [dC_HCO3_dt, dC_OH_dt, dC_CO32_dt]

#First calculating bulk solution species concentrations before adding CO2:

# Define initial conditions:
# Specify the concentrations in mol/m3 of each of the species added to the \
# electrolyte at t=0 starting with neutral water.

conc = 0.1  #in M
electrolyte = 'KOH'
T=298.15  #specify electrolyte temperature here in K
f_CO2 = 1  #pressure of CO2 in bar
rho = 1  #kg/L density of water at 293.15K

if electrolyte == 'KHCO3':
	C_init_K = conc * 1000
	C_init_HCO3 = conc * 1000
	C_init_OH = 1.0e-7 * 1000
	C_init_CO32 = 0.0
	C_init_CO2 = 0.0
	C_init_Cl = 0.0
elif electrolyte == 'KOH': #change the CO2_conc input parameters
	C_init_K = conc * 1000
	C_init_HCO3 = 0.0
	C_init_OH = conc * 1000
	C_init_CO32 = 0.0
	C_init_CO2 = 0.0
	C_init_Cl = 0.0
elif electrolyte == 'K2CO3': #change the CO2_conc input parameters
	C_init_K = conc * 1000 * 2
	C_init_HCO3 = 0.0
	C_init_OH = 1.0e-7 * 1000
	C_init_CO32 = conc * 1000
	C_init_CO2 = 0.0
	C_init_Cl = 0.0
elif electrolyte == 'KCl': #change the CO2_conc input parameters
	C_init_K = conc * 1000
	C_init_HCO3 = 0.0
	C_init_OH = 1.0e-7 * 1000
	C_init_CO32 = 0.0
	C_init_CO2 = 0.0
	C_init_Cl = conc * 1000
else:
	print('Electrolyte type not yet supported. Sorry!')

f = open(basepath_utilities+'parameters.yaml')
data = yaml.load(f)

rate_constants = data['rate_constants']

ka1 = rate_constants['ka1']
ka2 = rate_constants['ka2']
kb1 = rate_constants['kb1']
kb2 = rate_constants['kb2']

f.close()

#Define time step and max total time:
dt = 1.0e-2
tmax = 1.0e+1
t = np.linspace(0, tmax, int(tmax / dt))

y0 = [C_init_HCO3, C_init_OH, C_init_CO32, C_init_CO2]
sol = odeint(kinetics, y0, t, args=(ka1, ka2, kb1, kb2))
sol_list = sol.tolist()

pH = - math.log10(1.0e-14 / (sol[-1, 1] / 1000))
C0_H = (10 ** (- pH)) * 1000
C0_OH = sol_list[-1][1]
C0_HCO3 = sol_list[-1][0]
C0_CO32 = sol_list[-1][2]
C0_CO2 = sol_list[-1][3]

C_CO2_sechenov = CO2_conc(temp=T, fugacity_CO2=f_CO2, ions={'K':C_init_K, 'HCO3':C0_HCO3, 'OH': C0_OH, 'CO32':C0_CO32, 'Cl': C_init_Cl})

#plt.plot(t, sol[:, 0], 'b', label='HCO3(t)')
plt.plot(t, sol[:, 1], 'g', label='OH(t)')
#plt.plot(t, sol[:, 2], 'g', label='CO32(t)')
#plt.plot(t, sol[:, 3], 'g', label='CO2(t)')
#plt.plot(t, pH, 'g', label='pH')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

#Adding the bulk electrolyte concentrations to a yaml file
f = open(basepath_utilities+'bulk_soln_'+str(conc)+electrolyte+'.yaml', 'w')

if C0_CO2 > C_CO2_sechenov:
	data = {'bulk_conc_pre_CO2': 'Concentrations before adding CO2 will be same as on adding CO2 since solution is already saturated'}
else:
	data = {
		'bulk_conc_pre_CO2': {
			'conc_electrolyte': conc, 
			'electrolyte': electrolyte, 
			'final_pH': pH, 
			'concentrations': {
				'C0_H': C0_H ,
				'C0_OH': C0_OH ,
				'C0_CO2': C0_CO2, 
				'C0_HCO3': C0_HCO3, 
				'C0_CO32': C0_CO32, 
				'C0_K': C_init_K,
				'C0_Cl': C_init_Cl
			}
		}
	}

yaml.dump(data, f)

# CO2_conc calculates CO2 concentration in electrolyte at a cerain CO2 pressure, \
# temperature and electrolyte concentration (in molm-3). \
# Although the ion concentration changes on adding CO2, the CO2 concentration \
# and homogeneous kinetics has not been solved self consistently for simplicity.\
# The influence is expected to be negligible. 

#Calculating the bulk colution concentrations after addition of CO2

if C0_CO2 > C_CO2_sechenov:
	y0 = [C_init_HCO3, C_init_OH, C_init_CO32]
else: 
	y0 = [C0_HCO3, C0_OH, C0_CO32]

dt = 1.0e-2

if conc <= 1:
	tmax = 1.0e+3
elif conc <= 5:
	tmax = 1.0e+4
else:
	tmax = 5.0e+4

t = np.linspace(0, tmax, int(tmax / dt))

sol = odeint(kinetics_const_CO2, y0, t, args=(ka1, ka2, kb1, kb2))
sol_list = sol.tolist()

pH = - math.log10(1.0e-14 / (sol[-1, 1] / 1000))
C0_H = (10 ** (- pH)) * 1000
C0_OH = sol_list[-1][1]
C0_HCO3 = sol_list[-1][0]
C0_CO32 = sol_list[-1][2]
C0_CO2 = CO2_conc(T, f_CO2) #in mM or in molm-3

data = {'bulk_conc_post_CO2': {'conc_electrolyte': conc, 'electrolyte': electrolyte, \
'CO2_pressure' : f_CO2 ,'final_pH': pH, 'concentrations': {'C0_H': C0_H ,'C0_OH': C0_OH, \
'C0_CO2': C0_CO2, 'C0_HCO3': C0_HCO3, 'C0_CO32': C0_CO32, 'C0_K': C_init_K, 'C0_Cl': C_init_Cl}}}
yaml.dump(data, f)

f.close()

#plt.plot(t, sol[:, 0], 'b', label='HCO3(t)')
plt.plot(t, sol[:, 1], 'g', label='OH(t)')
#plt.plot(t, sol[:, 2], 'g', label='CO32(t)')
#plt.plot(t, pH, 'g', label='pH')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

# The rate constants for the dissociation of carbonic acid are a function of salinity and temperature.
# empirical relations are available for salinity values of upto 50 (~1M KHCO3). 
# these relationships work well for natural sea water and not artificial saline water.
# due to limited range of aplicability of these relations, we have not used them here.
# for a typical value of 0.1 M, the effect of salinity is very small based on the experessions provided for natural sea water. 
'''
Now incorporate water ionization reaction and this:
    Ionic_Strength = 0.5*(c_KHCO3*1*1^2+c_KHCO3*1*(-1)^2);
    S = 1000*Ionic_Strength/(19.92+1.0049*Ionic_Strength);
  
    pK01 = -126.34048 + 6320.813/T + 19.568224*log(T);   % http://www.sciencedirect.com/science/article/pii/S0304420305001921   Dissociation constants of carbonic acid in seawater as a function of salinity and temperature
    A1 = 13.4191*S^0.5+0.0331*S-5.33E-5*S^2;    B1 = -530.123*S^0.5-6.103.*S;    C1 = -2.06950*S^0.5;
    pK1 = pK01 + A1+B1/T+C1*log(T);
   
    A2 = 21.0894*S^0.5+0.1248*S-3.687E-4*S^2;    B2 = -772.483*S^0.5-20.051*S;    C2 = -3.3336*S^0.5;
    pK02= -90.18333+5143.692/T+14.613358*log(T);
    pK2 = pK02 + A2+B2/T+C2*log(T);
    Kw = 1e-14;                 kw_plus = 2.4e-5;         kw_minus = kw_plus/Kw;
    K1 = 10^(-pK1);             K2 = 10^(-pK2);
    K1_OH = 10^(-pK1)/Kw;       K2_OH = 10^(-pK2)/Kw; 
 
    k1_plus = 2.23e3;     k1_minus = k1_plus/K1_OH;
    k2_plus = 6e9;       k2_minus = k2_plus/K2_OH;
'''


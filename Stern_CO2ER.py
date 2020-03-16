'''
@author: divyabohra

This script solves the Poisson equation to calculate the potential at the electrode surface 
and takes as input the potential and relative permittivity at the OHP obtained from solving the GMPNP simulations. 

This is a simplified way to model the which suffices for the current purpose.
The ideal simulation should solve this script self consistently with the PNP script

By definition, no ions are present in the Stern layer (RHS of Poisson eq = 0)

'''

from __future__ import print_function
from fenics import *
import yaml
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, date, time
import scipy.optimize as opt 
from scipy.integrate import odeint
import os
import argparse

tol = 1.0e-14  #tolerance for coordinate comparisons
stamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')

#the function below writes a experiment config file for each run and saves \
# it in the experiment folder
def metadata(model='BDM', voltage_OHP=0.0, field_OHP=0.0, L_stern=None, field_surf=0.0,\
eps_rel_OHP=80.0, voltage_electrode=0.0, f=None):

    f.write('model='+model+'\n')
    f.write('voltage_OHP='+str(voltage_OHP)+'V\n')
    f.write('field_OHP='+str(field_OHP)+'V/nm\n')
    f.write(f'Relative permittivity at the OHP is {eps_rel_OHP} \n')
    f.write(f'voltage at the electrode is {voltage_electrode} \n')
    f.write(f'Electric field at the surface is {field_surf} m\n')
    f.write(f'Stern length is {L_stern} m\n')

    return 

def main(voltage_scaled_OHP=-1.0, field_OHP=-0.5, eps_rel_OHP=80.0, model='BDM'):
    #reading relevant parameters
    basepath_utilities = '/Users/divyabohra/Documents/Project2/source/exp/utilities/'

    f = open(basepath_utilities+'parameters.yaml')
    data = yaml.load(f)

    temp = data['nat_const']['T']  #temperature
    k_B = data['nat_const']['k_B']  #Boltzmann constant
    e_0 = data['nat_const']['e_0']  #charge on electron
    eps_0 = data['nat_const']['eps_0']  #permittivity of vacuum
    eps_rel = data['nat_const']['eps_rel']  #relative permittivity of water

    f.close()

    L_stern = 4.0e-10  #thickness of Stern layer in m \
                    #Assumed based on typical solvated diameters of monovalent cations

    thermal_voltage = (k_B * temp) / e_0  #thermal voltage 

    # OHP field and relative permittivities obtained from solving the MPNP code
    OHP_dict = {-2.5:{'E':-0.08032108300135771,'eps':74.56149297894756},\
            -5.0:{'E':-0.2524415478848975,'eps':57.64572780716129}, -7.5:{'E':-0.4612956299192668,'eps':50.16243860179017},\
            -10.0:{'E':-0.6149631587776277,'eps':49.311548142969336},-12.5:{'E':-0.7310301485096051,'eps':49.2556833480052}}

    def Stern(voltage_scaled_OHP=voltage_scaled_OHP, field_OHP=field_OHP, eps_rel_OHP=eps_rel_OHP, model=model):
        
        basepath = '/Volumes/External_Divya/Project2/CO2_PNP_MPNP/MPNP/Stern/'

        identifier = 'voltage_scaled_OHP'+str(voltage_scaled_OHP)  
        newpath = basepath+stamp+'_experiment/'+identifier+'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        voltage_OHP = voltage_scaled_OHP * thermal_voltage
        eps_rel_surface = 6.0  #relative permittivity when water molecules are rigid at the catalyst surface

        def BDM(Y, x, eps_rel_surface, eps_rel_OHP, L_stern_scaled):
            y1 = Y[0] #potential
            y2 = Y[1] # field
            y1_dx = y2
            y2_dx = - y2 * ((eps_rel_OHP - eps_rel_surface) / (x * (eps_rel_OHP - eps_rel_surface) + eps_rel_OHP * L_stern))
            return [y1_dx,y2_dx]

        if model=='BDM':

            #Define space step and max stern length
            dx = 1.0e-11
            xmax = - L_stern # going backwards in length
            x = np.linspace(0, xmax, abs(int(xmax / dx)))

            y0 = [voltage_OHP, -field_OHP]

            sol = odeint(BDM, y0, x, args=(eps_rel_OHP,eps_rel_surface,L_stern))
            sol_list = sol.tolist()

            y1_scaled = sol[:, 0] #potential vs x in V
            y2_scaled = sol[:, 1] * -1 # field vs x in V/nm
            x_scaled = x * 1.0e+9 # distance in nm

            y1_surf_scaled = y1_scaled[-1] # potential at the electrode surface
            y2_surf_scaled = y2_scaled[-1] #electric field at the electrode surface

            np.savez(newpath+'stern_unscaled_BDM'+str(voltage_scaled_OHP)+'.npz', sol)
            np.savez(newpath+'stern_scaled_BDM'+str(voltage_scaled_OHP)+'.npz', x_scaled, y1_scaled, y2_scaled)

            f = open(newpath+'metadata.txt', 'w')

            metadata(model=model, voltage_OHP=voltage_OHP, field_OHP=field_OHP, field_surf=y2_surf_scaled, L_stern=L_stern, \
            eps_rel_OHP=eps_rel_OHP, voltage_electrode=y1_surf_scaled, f=f)

            f.close()

            plt.plot(x_scaled, y1_scaled) 
            plt.xlabel('distance (nm)')
            plt.ylabel('potential in V')
            plt.title('voltage_multiplier: '+str(voltage_scaled_OHP))
            plt.xticks(rotation=90)
            #plt.legend()
            plt.tight_layout()
            plt.savefig(newpath+'V_x.png')
            plt.show()

            plt.plot(x_scaled, y2_scaled) 
            plt.xlabel('distance (nm)')
            plt.ylabel('electric field in V/nm')
            plt.title('voltage_multiplier: '+str(voltage_scaled_OHP))
            plt.xticks(rotation=90)
            #plt.legend()
            plt.tight_layout()
            plt.savefig(newpath+'field_x.png')
            plt.show()

        elif model=='Stern_linear':
            y1_surf_scaled = voltage_OHP - (-field_OHP * (L_stern * 1.0e+9)) #voltage at electrode
            y2_surf_scaled = field_OHP

            dx = 1.0e-2
            xmax = - L_stern * 1.0e+9 # in nm
            x = np.linspace(0, xmax, abs(int(xmax / dx)))

            x_list = x.tolist()

            y1_x = []

            for i in x_list:
                y1_x_i = - field_OHP * i + voltage_OHP
                y1_x+= [y1_x_i]
            
            y1_x_array = np.array(y1_x)

            np.savez(newpath+'stern_scaled_linear'+str(voltage_scaled_OHP)+'.npz', x, y1_x_array)

            f = open(newpath+'metadata.txt', 'w')

            metadata(model=model, voltage_OHP=voltage_OHP, field_OHP=field_OHP, field_surf=y2_surf_scaled, L_stern=L_stern, \
            eps_rel_OHP=eps_rel_OHP, voltage_electrode=y1_surf_scaled, f=f)

            f.close()

            plt.plot(x, y1_x) 
            plt.xlabel('distance (nm)')
            plt.ylabel('potential in V')
            plt.title('voltage_multiplier: '+str(voltage_scaled_OHP))
            plt.xticks(rotation=90)
            #plt.legend()
            plt.tight_layout()
            plt.savefig(newpath+'V_x.png')
            plt.show()

#below snippet is useful for plotting stern potential from different voltage multipliers

    plt.figure() 

    for i in OHP_dict.keys():
        Stern(voltage_scaled_OHP=i, field_OHP=OHP_dict[i]['E'], eps_rel_OHP=OHP_dict[i]['eps'], model=model)
    
    plt.show()


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='experiment parameters')

    parser.add_argument('--voltage_scaled_OHP', metavar='voltage multiplier', required=False, \
    help='float val', default=-2.5, type=float)

    parser.add_argument('--model', metavar='model_type', required=False, \
    help='str, BDM/Stern_linear', default='BDM', type=str)

    parser.add_argument('--field_OHP', metavar='electric field at the OHP', \
    required=False, help='float val, -0.5', default=-0.5, type=float)

    parser.add_argument('--eps_rel_OHP', metavar='relative permittivity at the OHP', required=False, \
    help='float, 80.0', default='80.0', type=float)

    args = parser.parse_args()

    main(voltage_scaled_OHP=args.voltage_scaled_OHP, model=args.model, field_OHP=args.field_OHP, \
    eps_rel_OHP=args.eps_rel_OHP)

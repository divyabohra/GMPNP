
'''
@author: divyabohra

This script solves the PNP/GMPNP system in cylinderical coordinates  \
for CO2ER for steady state concentration of solution species as well  \
as value of potential and electric field as a function of space and time. 

Dirichlet conditions are used at all boundaries for the potential \
Flux boundary conditions are used at the OHP for all species at the \
cylinderical surface as well as the exit of the pore.
Dirichlet conditions are used for CO2, CO and H2 at the pore entry \
with no flux for all other species.

The geometry and the mesh are generated using a separate script.

3 heterogeneous reactions are considered:
    CO2 + H2O + 2e- -> CO + 2OH-
    2H2O + 2e- -> H2 + 2OH-
    CO2 + 2(H+ + e-) -> CO + H2O

The rates of the above reactions are input to the simulation in the form of \
partial current density data.

3 homogeneous reactions are considered:
    H2O <=> H+ + OH- (k_w1, k_w2)
    HCO3- + OH- <=> CO32- + H2O (k_a1, k_a2)
    CO2 + OH- <=> HCO3- (k_b1, k_b2)
The values of the forward and backward rate constants are taken from literature.

Species solved for (i): H+, OH-, HCO3-, CO32-, CO2, CO, H2, cat+

'''

from __future__ import print_function
from fenics import *
import yaml
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, date, time
import os
import argparse
import json

# Below function is used to scale back the calculated variables from \
# dimentionless form to SI units
# tau is scaled time, C is scaled concentration
def scale_conc_time(
        species='H',
        C=None,
        grad_c=None,
        bulk_conc={'H':0.0},
        tau=None,
        diff_coeff_eff={'H':0.0},
        L=0.0):
    c = C * bulk_conc[species]
    t = tau * (L ** 2) / diff_coeff_eff[species]
    grad_c_scaled = grad_c * bulk_conc[species] / L
    return c, t, grad_c_scaled
        

def CO2_conc(
        temp,
        fugacity_CO2,
        conc_ions={'A':0.0},
        h_sechenov={'A':0.0}): 
	
	#Henry's constant as a function of T. [CO2]aq,0=K_H_CO2*[CO2]g
	lnK_H_CO2 = 93.4517 * (100 / temp) - 60.2409 + 23.3585 * math.log(temp / 100) 
	h_CO2 = h_sechenov['CO2_0'] + h_sechenov['CO2_T'] * (temp - 298.15)  #Sechenov model parameter for CO2

	sechenov = 0.0

	for ion in conc_ions.keys():
		# convert concentration from molm-3 to kmolm-3
		add = (h_sechenov[ion] + h_CO2) * (conc_ions[ion] / 1000) 
		sechenov+= add
	
	K_H_CO2 = math.exp(lnK_H_CO2)
	# initial concentration of dissolved CO2 in mol/m3
	C_CO2_S1 = fugacity_CO2 * K_H_CO2 * 1000 * 10 ** (-sechenov) 
	
	return C_CO2_S1


def solveEDL(
        concentration_elec=1.0,
        voltage_multiplier=-1.0,
        H2_FE=0.05,
        current_rough=3000.0,
        L=100.0e-9,
        cation='K',
        R=5.0e-9,
        press_gas=1.0,
        pore_geom_multiplier=1.0,
        porosity_eff=0.5,
        tortuosity_eff=1.5,
        constrictivity_eff=0.9,
        params_file='parameters_pore',
        y_CO2=0.95,
        electrolyte_flow_geom_multiplier=1.0,
        roughness_factor=150.0):
    
    stamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')

    # use for local machine
    #basepath_utilities = '/Users/divyabohra/Documents/Project2/source/exp/utilities/'
    basepath_utilities = '/home/divya/Documents/src/pnp/utilities/' 

    # read rate constants of homogeneous reactions, diffusion coefficients and diffusion \
    # length from yaml file storing default parameters
    f_params = open(basepath_utilities+params_file+'.yaml') # locally
    #f_params = open(params_file+'.yaml') # on cluster
    data = yaml.safe_load(f_params)

    rate_constants = data['rate_constants']

    # see code notes at top for reactions corresponding to rate constants
    kw1 = rate_constants['kw1']
    kw2 = rate_constants['kw2']
    ka1 = rate_constants['ka1']
    ka2 = rate_constants['ka2']
    kb1 = rate_constants['kb1']
    kb2 = rate_constants['kb2']

    cat_str = cation # storing the cation string in cat_str

    species = ['H','OH','HCO3','CO32','CO2','CO','H2',cat_str] # cation will be treated separately

    diff_coeff = {'H':0.0, 'OH':0.0, 'HCO3':0.0, 'CO32':0.0, 'CO2':0.0, 'CO':0.0, 'H2':0.0, cat_str:0.0}
    diff_coeff_eff = {'H':0.0, 'OH':0.0, 'HCO3':0.0, 'CO32':0.0, 'CO2':0.0, 'CO':0.0, 'H2':0.0, cat_str:0.0}
    # effective diffusion coefficients are for within the catalyst layer

    # saving diffusion coefficient of solution species
    for i in species:
        diff_coeff[i] = data['diff_coef']['D_'+i]
        
        # we use the tortuosity definition as (C/l) and therefore a square dependance
        # the formulation of the effective diffusion coefficient can be found in: Brakel, Heertjes, 1974
        # pore geometry multiplier is used to test the influence of the change in the parameter: \
        # porosity_eff * constrictivity_eff/ tortuosity_eff^2
        diff_coeff_eff[i] = (diff_coeff[i] * porosity_eff * constrictivity_eff * pore_geom_multiplier) / tortuosity_eff ** 2 
        
    
    # hydration numbers of cations
    n_water = {'H':data['Hydration_number']['w_H'], cat_str:data['Hydration_number']['w_'+cat_str]}

    # storing solvated sizes of solution species
    solv_size = {'H':0.0, 'OH':0.0, 'HCO3':0.0, 'CO32':0.0, 'CO2':0.0, 'CO':0.0, 'H2':0.0, cat_str:0.0}

    for i in species:
        solv_size[i] = data['solv_size']['a_'+i]

    # all parameter values are in SI units
    farad = data['nat_const']['F']  # Faradays constant
    k_B = data['nat_const']['k_B'] # Boltzmann constant
    e_0 = data['nat_const']['e_0'] # elementary electron charge
    eps_0 = data['nat_const']['eps_0'] # permittivity of vacuum
    eps_rel = data['nat_const']['eps_rel'] # relative permittivity of water (electrolyte)
    R_gas = data['nat_const']['R'] # gas constant
    N_A = data['nat_const']['N_A'] # Avogadro's number

    H_CO2 = data['Henrys_const']['H_CO2'] # in molkg-1bar-1
    H_CO = data['Henrys_const']['H_CO'] # in molkg-1bar-1
    H_H2 = data['Henrys_const']['H_H2'] # in molkg-1bar-1

    temp = data['sys_params']['T']
    density_e = data['sys_params']['density_e'] # density of water in kg/m3 at 25degC
    viscosity_e = data['sys_params']['viscosity_e'] #  viscosity of water in kgm-1s-1 at 1 atm, 25 degC 
    L_electrode = data['sys_params']['L_electrode'] # length of electrode in m
    A_electrode = data['sys_params']['A_electrode'] # area of electrode in m2
    vel_e = data['sys_params']['vel_e'] # velocity of electrolyte parallel to electrode in m^3/s
    A_cross_e = data['sys_params']['A_cross_e'] # cross sectional area of electrolyte flow in m2
    L_cross_e = data['sys_params']['L_cross_e'] # cross sectional length of the flow channel in m


    h_sechenov = {'OH':0.0, 'HCO3':0.0, 'CO32':0.0, cat_str:0.0, 'CO2_0':0.0, 'CO2_T':0.0}
    
    sechenov_const = data['sechonov_const']
    h_sechenov['CO2_0'] = sechenov_const['h_CO2_0']
    h_sechenov['CO2_T'] = sechenov_const['h_CO2_T']
    h_sechenov['OH'] = sechenov_const['h_ion_OH']
    h_sechenov['HCO3'] = sechenov_const['h_ion_HCO3']
    h_sechenov['CO32'] = sechenov_const['h_ion_CO32']
    h_sechenov[cat_str] = sechenov_const['h_ion_'+cat_str]

    f_params.close() 

    # The % distribution between CO and H2 at the CL/DM interface is chosen arbitrarily.  
    # It is assumed to be 90% CO and 10% H2 because of the higher diffusivity of H2, lower production FE \
    # of H2 as well as lower solubity of H2 
    y_CO = 0.9 * (1 - y_CO2) # y_CO2 is an input and is the fraction of CO2 at S1.
    y_H2 = 1 - y_CO2 - y_CO

    press_gas = press_gas # in bar
    fugacity_CO2 = y_CO2 * press_gas # in bar

    # read bulk electrolyte concentrations as calculated by bulk_soln.py
    f_conc = open(basepath_utilities+'bulk_soln_'+str(concentration_elec)+'KHCO3.yaml') # locally
    #f_conc = open('bulk_soln_'+str(concentration_elec)+'KHCO3.yaml') # on cluster
    data = yaml.safe_load(f_conc)

    bulk_pH = data['bulk_conc_pre_CO2']['final_pH']

    # storing bulk concentration of solution species
    bulk_conc = {'H':0.0, 'OH':0.0, 'HCO3':0.0, 'CO32':0.0, 'CO2':0.0, 'CO':0.0, 'H2':0.0, cat_str:0.0}

    # storing charge of solution species
    z = {'H':1, 'OH':-1, 'HCO3':-1, 'CO32':-2, 'CO2':0, 'CO':0, 'H2':0, cat_str:1}

    for i in species:
        # initial concentration of electrolyte without CO2 saturation
        bulk_conc[i] = data['bulk_conc_pre_CO2']['concentrations']['C0_'+i]

    # concentration of ions will be used to estimate CO2 conc at S1 using Sechenov eq
    # value of the ion concentrations will be updated within the time stepping loop
    conc_ions = {'OH':bulk_conc['OH'],'HCO3':bulk_conc['HCO3'],'CO32':bulk_conc['CO32'],cat_str:bulk_conc[cat_str]}

    f_conc.close()

    eq_conc_CO2 = H_CO2 * press_gas * y_CO2 * density_e # in mM
    eq_conc_CO = H_CO * press_gas * y_CO * density_e # in mM
    eq_conc_H2 = H_H2 * press_gas * y_H2 * density_e # in mM

    # CO and H2 conc in the bulk is considered to be 1% of eq conc at S1
    bulk_conc['CO'] = 0.01 * eq_conc_CO
    bulk_conc['H2'] = 0.01 * eq_conc_H2

    eq_conc_CO2_scaled = Constant(eq_conc_CO2 / bulk_conc['CO2'])
    eq_conc_CO_scaled = Constant(eq_conc_CO / bulk_conc['CO'])
    eq_conc_H2_scaled = Constant(eq_conc_H2 / bulk_conc['H2'])

    aspect_pore = R / L  #length to radius ratio for pore 

    thermal_voltage = (k_B * temp) / e_0  #thermal voltage 

    time_constant = L ** 2 / diff_coeff_eff['CO32'] #using the smallest diffusion coeff of all species.
    # the physical interpretation of this time constant is less obvious

    # scaling factor for homogeneous reaction rate stoichiometry
    scale_R = {'H':0.0, 'OH':0.0, 'HCO3':0.0, 'CO32':0.0, 'CO2':0.0, 'CO':0.0, 'H2':0.0, cat_str:0.0}

    for i in species:
        scale_R[i] = Constant((L ** 2) / (diff_coeff_eff[i] * bulk_conc[i]))

    # scaling factors for Poisson equation
    q = Constant((farad ** 2 * L ** 2) / (eps_0 * R_gas * temp))

    # scaled volume of solvated ions
    scale_vol = {'H':0.0, 'OH':0.0, 'HCO3':0.0, 'CO32':0.0, 'CO2':0.0, 'CO':0.0, 'H2':0.0, cat_str:0.0}

    for i in species:
        scale_vol[i] = Constant(solv_size[i] ** 3 * bulk_conc[i] * N_A) 

    # flux prefactors
    J_prefactor = {'H':0.0, 'OH':0.0, 'HCO3':0.0, 'CO32':0.0, 'CO2':0.0, 'CO':0.0, 'H2':0.0, cat_str:0.0}

    for i in species:
        J_prefactor[i] = L / (diff_coeff_eff[i] * bulk_conc[i]) # flux dimensions assumed as NT-1L-2

    # Reynolds number for laminar flow
    # the electrolyte_flow_goem_multiplier is used to vary (vel_e / A_cross_e)
    # the electrolyte velocity is assumed to be the maximum velocity at the center of the channel
    Re = (density_e * (vel_e / A_cross_e) * L_electrode * electrolyte_flow_geom_multiplier) / viscosity_e 
    # base case Re=28

    # Schmidt number dictionary initiate
    Sc = {'H':0.0, 'OH':0.0, 'HCO3':0.0, 'CO32':0.0, 'CO2':0.0, 'CO':0.0, 'H2':0.0, cat_str:0.0}

    # Sherwood number dictionary initiate
    Sh = {'H':0.0, 'OH':0.0, 'HCO3':0.0, 'CO32':0.0, 'CO2':0.0, 'CO':0.0, 'H2':0.0, cat_str:0.0}

    # mass transfer coeff at S3 dictionary initiate
    k_elec = {'H':0.0, 'OH':0.0, 'HCO3':0.0, 'CO32':0.0, 'CO2':0.0, 'CO':0.0, 'H2':0.0, cat_str:0.0}

    for i in species:
        Sc[i] = viscosity_e / (density_e * diff_coeff[i]) #Schmidt number
        Sh[i] = 1.017 * ((L_electrode * 2 / L_cross_e) * Re * Sc[i]) ** (1.0/3) #Sherwood number (eq 22.2-4 Bird)
        k_elec[i] = (diff_coeff[i] / L_electrode) * Sh[i] #Sherwood equation

    voltage_scaled = Constant(voltage_multiplier) #this voltage is at the OHP and not at the electrode surface
    
    ### From here starts the FEniCS part of the code ###
    
    # Read mesh from file
    mesh = Mesh(basepath_utilities+'L_'+str(int(L*1e+9))+'_R_'+str(int(R*1e+9))+'.xml') #locally
    #mesh = Mesh('L_'+str(int(L*1e+9))+'_R_'+str(int(R*1e+9))+'_sparse_vol.xml') # on cluster

    # defining boundary at the pore exit face
    class boundary_pore_exit(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1.0e-12
            return near(x[2], 1, tol)

    # defining boundary at the pore entry
    class boundary_pore_entry(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1.0e-12
            return near(x[2], 0, tol)

    # defining pore wall boundary
    class boundary_wall(SubDomain):
        def inside(self, x, on_boundary):
            if (R == 5.0e-9 or R == 50.0e-9) and L == 10.0e-9:
                tol = 5.0e-3
            else:
                tol = 1.0e-3
            return near((x[0] ** 2 + x[1] ** 2), (aspect_pore ** 2), tol) 

    # Mark boundaries
    # The dimension given to the MeshFunction is 0 for a vertex
    boundary_markers = MeshFunction('size_t', mesh, 2)  
    boundary_markers.set_all(9999)

    # invoking boundary entities
    b_entry = boundary_pore_entry()
    b_exit = boundary_pore_exit()
    b_wall = boundary_wall()

    # boundaries markers analogous to formulation doc
    b_entry.mark(boundary_markers, 1)
    b_exit.mark(boundary_markers, 3)
    b_wall.mark(boundary_markers, 2)

    # Redefine boundary integration measure
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    identifier = 'v_'+str(voltage_multiplier)\
    +'_L_'+str(int(L*1e+9))+'_R_'+str(int(R*1e+9))+'_P_g_'+str(press_gas)+'_D_eff_'+str(pore_geom_multiplier)\
    +'_Re_'+str(electrolyte_flow_geom_multiplier)+'_rough_'+str(roughness_factor)

    #basepath = '/Users/divyabohra/Documents/Project2/source/exp/MPNP_pore/out/' # use for local machine
    basepath = '/media/disk2/scratch/divya/MPNP_pore/out/' # use for local machine
    #basepath = '/'+model+'/' #use for cluster

    newpath = basepath+stamp+'_experiment/'+identifier+'/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Define function space for system of concentrations
    degree = 1
    P3 = FiniteElement('Lagrange', tetrahedron, degree)
    element = MixedElement([P3, P3, P3, P3, P3, P3, P3, P3, P3])  # 8 concentrations and 1 potential
    V = FunctionSpace(mesh, element)
    W = VectorFunctionSpace(mesh, 'Lagrange', degree)

    # Define test functions
    v_H, v_OH, v_HCO3, v_CO32, v_CO2, v_CO, v_H2, v_cat, v_p = TestFunctions(V)

    # Define functions for the concentrations and potential
    u = Function(V)  # at t_n+1
    # initialization of all variables
    u_0 = Expression(('1.0','1.0','1.0','1.0','1.0','1.0','1.0','1.0','0.0'), degree=1)  
    # initializing concentration as bulk and voltage as 0 V                                                           
    u_n = interpolate(u_0, V) # interpolate used instead of project to solve UMFPack error

    # Split system functions to access components
    u_H, u_OH, u_HCO3, u_CO32, u_CO2, u_CO, u_H2, u_cat, u_p = split(u)
    u_nH, u_nOH, u_nHCO3, u_nCO32, u_nCO2, u_nCO, u_nH2, u_ncat, u_np = split(u_n)

    bulk = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0]
    # Dirichlet condition for voltage at the boundaries and conc of gases at S1
    bc1 = DirichletBC(V.sub(8), Constant(0.0), boundary_markers, 1)
    bc2 = DirichletBC(V.sub(8), Constant(0.0), boundary_markers, 3)
    bc3 = DirichletBC(V.sub(8), voltage_scaled, boundary_markers, 2) 
    bc4 = DirichletBC(V.sub(4), eq_conc_CO2_scaled, boundary_markers, 1) 
    bc5 = DirichletBC(V.sub(5), eq_conc_CO_scaled, boundary_markers, 1)
    bc6 = DirichletBC(V.sub(6), eq_conc_H2_scaled, boundary_markers, 1)

    bcs = [bc1,bc2,bc3,bc4,bc5,bc6]
    # Neumann condition will be used for all solution species at all boundaries

    # storing coordinates of the mesh
    coor_array = mesh.coordinates()
    num_vertices = mesh.num_vertices()

    # R_i are the rates of production of species i (scaled)
    # cat, CO and H2 are not being consumed or formed in any homogeneous reaction
    R_H = - scale_R['H'] * (kw2 * (u_H * bulk_conc['H']) * (u_OH * bulk_conc['OH']) - kw1)

    R_OH = - scale_R['OH'] * (kw2 * (u_H * bulk_conc['H']) * (u_OH * bulk_conc['OH']) \
    + ka1 * (u_OH * bulk_conc['OH']) * (u_HCO3 * bulk_conc['HCO3']) \
    + kb1 * (u_CO2 * bulk_conc['CO2']) * (u_OH * bulk_conc['OH']) - kw1 \
    - ka2 * (u_CO32 * bulk_conc['CO32']) - kb2 * (u_HCO3 * bulk_conc['HCO3']))

    R_HCO3 = - scale_R['HCO3'] * (ka1 * (u_OH * bulk_conc['OH']) * (u_HCO3 * bulk_conc['HCO3']) \
    + kb2 * (u_HCO3 * bulk_conc['HCO3']) - ka2 * (u_CO32 * bulk_conc['CO32']) \
    - kb1 * (u_CO2 * bulk_conc['CO2']) * (u_OH * bulk_conc['OH']))

    R_CO32 = - scale_R['CO32'] * (ka2 * (u_CO32 * bulk_conc['CO32']) \
    - ka1 * (u_OH * bulk_conc['OH'] * (u_HCO3 * bulk_conc['HCO3'])))

    R_CO2 = - scale_R['CO2'] * (kb1 * (u_CO2 * bulk_conc['CO2']) * (u_OH * bulk_conc['OH']) \
    - kb2 * (u_HCO3 * bulk_conc['HCO3'])) 

    CO_FE = 1 - H2_FE
    
    current_planar = current_rough / roughness_factor

    J_CO2_wall = Constant((J_prefactor['CO2'] / farad) * current_planar * 0.5 * (CO_FE))  #at OHP
    J_CO_wall = Constant((J_prefactor['CO'] / farad) * current_planar * 0.5 * (CO_FE) * (-1.0))  #at OHP
    J_H2_wall = Constant((J_prefactor['H2'] / farad) * current_planar * 0.5 * (H2_FE) * (-1.0))  #at OHP
    J_OH_wall = Constant((J_prefactor['OH'] / farad) * current_planar * (-1.0))  #at OHP
    
    J_pore_exit_CO2 = Constant(J_prefactor['CO2'] * k_elec['CO2'] * bulk_conc['CO2']) * (u_CO2 - 1)
    J_pore_exit_CO = Constant(J_prefactor['CO'] * k_elec['CO'] * bulk_conc['CO']) * (u_CO - 1)
    J_pore_exit_H2 = Constant(J_prefactor['H2'] * k_elec['H2'] * bulk_conc['H2']) * (u_H2 - 1)
    J_pore_exit_OH = Constant(J_prefactor['OH'] * k_elec['OH'] * bulk_conc['OH']) * (u_OH - 1)
    J_pore_exit_H = Constant(J_prefactor['H'] * k_elec['H'] * bulk_conc['H']) * (u_H - 1)
    J_pore_exit_HCO3 = Constant(J_prefactor['HCO3'] * k_elec['HCO3'] * bulk_conc['HCO3']) * (u_HCO3 - 1)
    J_pore_exit_CO32 = Constant(J_prefactor['CO32'] * k_elec['CO32'] * bulk_conc['CO32']) * (u_CO32 - 1)
    J_pore_exit_cat = Constant(J_prefactor[cat_str] * k_elec[cat_str] * bulk_conc[cat_str]) * (u_cat - 1)

    #'''
    ## without staging
    time_step = 1.0e-3
    total_sim_time = 1.0

    T = total_sim_time / time_constant  # final time
    dt = time_step / time_constant    # step size
    num_steps = total_sim_time / time_step  # number of steps
    del_t = Constant(dt)
    tot_num_steps = int(num_steps)
    ## without staging 
    '''

    ## staging time step size
    time_step_1 = 1.0e-3
    time_step_2 = 1.0e-2 
    total_sim_time_1 = 1.0 #in sec
    total_sim_time_2 = 11.0 #in sec

    T_1 = total_sim_time_1 / time_constant  # final time
    dt_1 = time_step_1 / time_constant    # step size
    num_steps_1 = int(total_sim_time_1 / time_step_1)  # number of steps
    del_t_1 = Constant(dt_1)

    T_2 = total_sim_time_2 / time_constant  # final time
    dt_2 = time_step_2 / time_constant    # step size
    num_steps_2 = int((total_sim_time_2 - total_sim_time_1) / time_step_2)  # number of steps
    del_t_2 = Constant(dt_2)

    del_t = del_t_1
    tot_num_steps = num_steps_1 + num_steps_2
    ## staging time step size
    
    '''

    F_H = ((u_H - u_nH) / (del_t)) * v_H * dx + dot(grad(u_H), grad(v_H)) * dx \
    + z['H'] * u_H * dot(grad(u_p), grad(v_H)) * dx - R_H * v_H * dx \
    + (u_H / (1 - (scale_vol['H'] * u_H + scale_vol['OH'] * u_OH + scale_vol['HCO3'] * u_HCO3 \
    + scale_vol['CO32'] * u_CO32 + scale_vol['CO2'] * u_CO2 + scale_vol['CO'] * u_CO + scale_vol['H2'] * u_H2 \
    + scale_vol[cat_str] * u_cat))) \
    * dot((scale_vol['H'] * grad(u_H) + scale_vol['OH'] * grad(u_OH) + scale_vol['HCO3'] * grad(u_HCO3) \
    + scale_vol['CO32'] * grad(u_CO32) + scale_vol['CO2'] * grad(u_CO2) + scale_vol['CO'] * grad(u_CO) \
    + scale_vol['H2'] * grad(u_H2) + scale_vol[cat_str] * grad(u_cat)), grad(v_H)) * dx \
    + J_pore_exit_H * v_H * ds(3)

    F_OH = ((u_OH - u_nOH) / (del_t)) * v_OH * dx + dot(grad(u_OH), grad(v_OH)) * dx \
    + z['OH'] * u_OH * dot(grad(u_p), grad(v_OH)) * dx - R_OH * v_OH * dx \
    + (u_OH / (1 - (scale_vol['H'] * u_H + scale_vol['OH'] * u_OH + scale_vol['HCO3'] * u_HCO3 \
    + scale_vol['CO32'] * u_CO32 + scale_vol['CO2'] * u_CO2 + scale_vol['CO'] * u_CO + scale_vol['H2'] * u_H2 \
    + scale_vol[cat_str] * u_cat))) \
    * dot((scale_vol['H'] * grad(u_H) + scale_vol['OH'] * grad(u_OH) + scale_vol['HCO3'] * grad(u_HCO3) \
    + scale_vol['CO32'] * grad(u_CO32) + scale_vol['CO2'] * grad(u_CO2) + scale_vol['CO'] * grad(u_CO) \
    + scale_vol['H2'] * grad(u_H2) + scale_vol[cat_str] * grad(u_cat)), grad(v_OH)) * dx \
    + J_pore_exit_OH * v_OH * ds(3) + J_OH_wall * v_OH * ds(2)

    F_HCO3 = ((u_HCO3 - u_nHCO3) / (del_t)) * v_HCO3 * dx \
    + dot(grad(u_HCO3), grad(v_HCO3)) * dx - R_HCO3 * v_HCO3 * dx \
    + z['HCO3'] * u_HCO3 * dot(grad(u_p), grad(v_HCO3)) * dx \
    + (u_HCO3 / (1 - (scale_vol['H'] * u_H + scale_vol['OH'] * u_OH + scale_vol['HCO3'] * u_HCO3 \
    + scale_vol['CO32'] * u_CO32 + scale_vol['CO2'] * u_CO2 + scale_vol['CO'] * u_CO + scale_vol['H2'] * u_H2 \
    + scale_vol[cat_str] * u_cat))) \
    * dot((scale_vol['H'] * grad(u_H) + scale_vol['OH'] * grad(u_OH) + scale_vol['HCO3'] * grad(u_HCO3) \
    + scale_vol['CO32'] * grad(u_CO32) + scale_vol['CO2'] * grad(u_CO2) + scale_vol['CO'] * grad(u_CO) \
    + scale_vol['H2'] * grad(u_H2) + scale_vol[cat_str] * grad(u_cat)), grad(v_HCO3)) * dx \
    + J_pore_exit_HCO3 * v_HCO3 * ds(3)

    F_CO32 = ((u_CO32 - u_nCO32) / (del_t)) * v_CO32 * dx \
    + dot(grad(u_CO32), grad(v_CO32)) * dx - R_CO32 * v_CO32 * dx \
    + z['CO32'] * u_CO32 * dot(grad(u_p), grad(v_CO32)) * dx \
    + (u_CO32 / (1 - (scale_vol['H'] * u_H + scale_vol['OH'] * u_OH + scale_vol['HCO3'] * u_HCO3 \
    + scale_vol['CO32'] * u_CO32 + scale_vol['CO2'] * u_CO2 + scale_vol['CO'] * u_CO + scale_vol['H2'] * u_H2 \
    + scale_vol[cat_str] * u_cat))) \
    * dot((scale_vol['H'] * grad(u_H) + scale_vol['OH'] * grad(u_OH) + scale_vol['HCO3'] * grad(u_HCO3) \
    + scale_vol['CO32'] * grad(u_CO32) + scale_vol['CO2'] * grad(u_CO2) + scale_vol['CO'] * grad(u_CO) \
    + scale_vol['H2'] * grad(u_H2) + scale_vol[cat_str] * grad(u_cat)), grad(v_CO32)) * dx \
    + J_pore_exit_CO32 * v_CO32 * ds(3)

    F_CO2 = ((u_CO2 - u_nCO2) / (del_t)) * v_CO2 * dx \
    + dot(grad(u_CO2), grad(v_CO2)) * dx - R_CO2 * v_CO2 * dx \
    + (u_CO2 / (1 - (scale_vol['H'] * u_H + scale_vol['OH'] * u_OH + scale_vol['HCO3'] * u_HCO3 \
    + scale_vol['CO32'] * u_CO32 + scale_vol['CO2'] * u_CO2 + scale_vol['CO'] * u_CO + scale_vol['H2'] * u_H2 \
    + scale_vol[cat_str] * u_cat))) \
    * dot((scale_vol['H'] * grad(u_H) + scale_vol['OH'] * grad(u_OH) + scale_vol['HCO3'] * grad(u_HCO3) \
    + scale_vol['CO32'] * grad(u_CO32) + scale_vol['CO2'] * grad(u_CO2) + scale_vol['CO'] * grad(u_CO) \
    + scale_vol['H2'] * grad(u_H2) + scale_vol[cat_str] * grad(u_cat)), grad(v_CO2)) * dx \
    + J_pore_exit_CO2 * v_CO2 * ds(3) + J_CO2_wall * v_CO2 * ds(2)

    F_cat = ((u_cat - u_ncat) / (del_t)) * v_cat * dx + dot(grad(u_cat), grad(v_cat)) * dx \
    + z[cat_str] * u_cat * dot(grad(u_p), grad(v_cat)) * dx \
    + (u_cat / (1 - (scale_vol['H'] * u_H + scale_vol['OH'] * u_OH + scale_vol['HCO3'] * u_HCO3 \
    + scale_vol['CO32'] * u_CO32 + scale_vol['CO2'] * u_CO2 + scale_vol['CO'] * u_CO + scale_vol['H2'] * u_H2 \
    + scale_vol[cat_str] * u_cat))) \
    * dot((scale_vol['H'] * grad(u_H) + scale_vol['OH'] * grad(u_OH) + scale_vol['HCO3'] * grad(u_HCO3) \
    + scale_vol['CO32'] * grad(u_CO32) + scale_vol['CO2'] * grad(u_CO2) + scale_vol['CO'] * grad(u_CO) \
    + scale_vol['H2'] * grad(u_H2) + scale_vol[cat_str] * grad(u_cat)), grad(v_cat)) * dx \
    + J_pore_exit_cat * v_cat * ds(3)

    F_CO = ((u_CO - u_nCO) / (del_t)) * v_CO * dx \
    + dot(grad(u_CO), grad(v_CO)) * dx \
    + (u_CO / (1 - (scale_vol['H'] * u_H + scale_vol['OH'] * u_OH + scale_vol['HCO3'] * u_HCO3 \
    + scale_vol['CO32'] * u_CO32 + scale_vol['CO2'] * u_CO2 + scale_vol['CO'] * u_CO + scale_vol['H2'] * u_H2 \
    + scale_vol[cat_str] * u_cat))) \
    * dot((scale_vol['H'] * grad(u_H) + scale_vol['OH'] * grad(u_OH) + scale_vol['HCO3'] * grad(u_HCO3) \
    + scale_vol['CO32'] * grad(u_CO32) + scale_vol['CO2'] * grad(u_CO2) + scale_vol['CO'] * grad(u_CO) \
    + scale_vol['H2'] * grad(u_H2) + scale_vol[cat_str] * grad(u_cat)), grad(v_CO)) * dx \
    + J_pore_exit_CO * v_CO * ds(3) + J_CO_wall * v_CO * ds(2)

    F_H2 = ((u_H2 - u_nH2) / (del_t)) * v_H2 * dx \
    + dot(grad(u_H2), grad(v_H2)) * dx \
    + (u_H2 / (1 - (scale_vol['H'] * u_H + scale_vol['OH'] * u_OH + scale_vol['HCO3'] * u_HCO3 \
    + scale_vol['CO32'] * u_CO32 + scale_vol['CO2'] * u_CO2 + scale_vol['CO'] * u_CO + scale_vol['H2'] * u_H2 \
    + scale_vol[cat_str] * u_cat))) \
    * dot((scale_vol['H'] * grad(u_H) + scale_vol['OH'] * grad(u_OH) + scale_vol['HCO3'] * grad(u_HCO3) \
    + scale_vol['CO32'] * grad(u_CO32) + scale_vol['CO2'] * grad(u_CO2) + scale_vol['CO'] * grad(u_CO) \
    + scale_vol['H2'] * grad(u_H2) + scale_vol[cat_str] * grad(u_cat)), grad(v_H2)) * dx \
    + J_pore_exit_H2 * v_H2 * ds(3) + J_H2_wall * v_H2 * ds(2)

    F_p = - (eps_rel * ((55 - (n_water[cat_str] * u_cat * bulk_conc[cat_str] + n_water['H'] * u_H * bulk_conc['H']) * 1.0e-3) / 55) \
    + 6 * (((n_water[cat_str] * u_cat * bulk_conc[cat_str] + n_water['H'] * u_H * bulk_conc['H']) * 1.0e-3) / 55)) \
    * dot(grad(u_p), grad(v_p)) * dx + (z['H'] * u_H * bulk_conc['H'] + z['OH'] * u_OH * bulk_conc['OH'] \
    + z['HCO3'] * u_HCO3 * bulk_conc['HCO3'] + z['CO32'] * u_CO32 * bulk_conc['CO32'] \
    + z[cat_str] * u_cat * bulk_conc[cat_str]) * q * v_p * dx
    
    F = F_H + F_OH + F_HCO3 + F_CO32 + F_CO2 + F_cat + F_CO + F_H2 + F_p
        
    J = derivative(F, u, u) # Gateaux derivative in dir. of u

    H = np.ones(num_vertices)
    OH = np.ones(num_vertices)
    HCO3 = np.ones(num_vertices)
    CO32 = np.ones(num_vertices)
    CO2 = np.ones(num_vertices)
    CO = np.ones(num_vertices)
    H2 = np.ones(num_vertices)
    cat = np.ones(num_vertices)
    p = np.zeros(num_vertices)

    # Time-stepping
    t = 0
    for n in range(tot_num_steps):
        
        '''
        ## with staging
        # Update current time
        if t < T_1:
            dt = dt_1
            del_t = del_t_1
            print(int(t/dt))
        else: #t >= T_1 and t < T2:
            dt = dt_2
            del_t = del_t_2
            print(int(num_steps_1 + (t - T_1)/dt))
        ## with staging
        '''

        t += dt

        # mumps linear solver works faster than gmres according to local tests
        # using the default linear solver gives UMFPack error
        solve(F == 0, u, bcs, solver_parameters={
                'nonlinear_solver': 'newton',
                'newton_solver':{
                    'linear_solver':'mumps',
                    'maximum_iterations':50,
                    'relative_tolerance': 1.0e-4,
                    'absolute_tolerance': 1.0e-4,
                    'relaxation_parameter':0.9}})  

        # Save solution to file (VTK)
        _u_H, _u_OH, _u_HCO3, _u_CO32, _u_CO2, _u_CO, _u_H2, _u_cat, _u_p = u.split()

        _u_H_nodal_values_array = _u_H.compute_vertex_values()
        _u_OH_nodal_values_array = _u_OH.compute_vertex_values()
        _u_HCO3_nodal_values_array = _u_HCO3.compute_vertex_values()
        _u_CO32_nodal_values_array = _u_CO32.compute_vertex_values()
        _u_CO2_nodal_values_array = _u_CO2.compute_vertex_values()
        _u_CO_nodal_values_array = _u_CO.compute_vertex_values()
        _u_H2_nodal_values_array = _u_H2.compute_vertex_values()
        _u_cat_nodal_values_array = _u_cat.compute_vertex_values()
        _u_p_nodal_values_array = _u_p.compute_vertex_values()

        # storing median (scaled) values of the ion concentrations for calculating
        # CO2 concentration using Sechenov equation
        med_OH = np.median(_u_OH_nodal_values_array)
        med_HCO3 = np.median(_u_HCO3_nodal_values_array)
        med_CO32 = np.median(_u_CO32_nodal_values_array)
        med_cat = np.median(_u_cat_nodal_values_array)

        # median ion concentration for the time step
        conc_ions['OH'] = med_OH * bulk_conc['OH'] # in mol/m3
        conc_ions['HCO3'] = med_HCO3 * bulk_conc['HCO3'] # in mol/m3
        conc_ions['CO32'] = med_CO32 * bulk_conc['CO32'] # in mol/m3
        conc_ions[cat_str] = med_cat * bulk_conc[cat_str] # in mol/m3

        # updating concentration of CO2 at S1 based on Sechenov eq.
        eq_conc_CO2 = CO2_conc(temp=temp,fugacity_CO2=fugacity_CO2,conc_ions=conc_ions,h_sechenov=h_sechenov)
        eq_conc_CO2_scaled = Constant(eq_conc_CO2 / bulk_conc['CO2'])
        bc4 = DirichletBC(V.sub(4), eq_conc_CO2_scaled, boundary_markers, 1)
        bcs = [bc1,bc2,bc3,bc4,bc5,bc6]

        # creating a numpy array of concentration values at every time step in the whole domain
        H = np.vstack((H,_u_H_nodal_values_array))
        OH = np.vstack((OH,_u_OH_nodal_values_array))
        HCO3 = np.vstack((HCO3,_u_HCO3_nodal_values_array))
        CO32 = np.vstack((CO32,_u_CO32_nodal_values_array))
        CO2 = np.vstack((CO2,_u_CO2_nodal_values_array))
        CO = np.vstack((CO,_u_CO_nodal_values_array))
        H2 = np.vstack((H2,_u_H2_nodal_values_array))
        cat = np.vstack((cat,_u_cat_nodal_values_array))
        p = np.vstack((p,_u_p_nodal_values_array))

        CO2_min = np.amin(_u_CO2_nodal_values_array)
        print(CO2_min)

        # Update previous solution
        u_n.assign(u)
        print(datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
        print(int(t / dt)) # comment out when staging

    end_time = datetime.now().strftime('%y-%m-%d-%H-%M-%S')

    # Save solution to file in VTK format
    vtkfile_CO = File(newpath+'/solution_CO.pvd')
    vtkfile_CO << _u_CO
    vtkfile_K = File(newpath+'/solution_K.pvd')
    vtkfile_K << _u_cat
    vtkfile_H2 = File(newpath+'/solution_H2.pvd')
    vtkfile_H2 << _u_H2
    vtkfile_CO2 = File(newpath+'/solution_CO2.pvd')
    vtkfile_CO2 << _u_CO2
    vtkfile_OH = File(newpath+'/solution_OH.pvd')
    vtkfile_OH << _u_OH
    vtkfile_H = File(newpath+'/solution_H.pvd')
    vtkfile_H << _u_H
    vtkfile_HCO3 = File(newpath+'/solution_HCO3.pvd')
    vtkfile_HCO3 << _u_HCO3
    vtkfile_CO32 = File(newpath+'/solution_CO32.pvd')
    vtkfile_CO32 << _u_CO32
    vtkfile_p = File(newpath+'/solution_p.pvd')
    vtkfile_p << _u_p
    
    # estimating the electric field value as a function of x for the last computed value of potential profile
    field = project(-grad(u_np), W)
    field_values = field.compute_vertex_values()

    grad_H = project(grad(u_nH), W)
    grad_H_array = grad_H.compute_vertex_values()

    grad_OH = project(grad(u_nOH), W)
    grad_OH_array = grad_OH.compute_vertex_values()

    grad_HCO3 = project(grad(u_nHCO3), W)
    grad_HCO3_array = grad_HCO3.compute_vertex_values()

    grad_CO32 = project(grad(u_nCO32), W)
    grad_CO32_array = grad_CO32.compute_vertex_values()

    grad_CO2 = project(grad(u_nCO2), W)
    grad_CO2_array = grad_CO2.compute_vertex_values()

    grad_CO = project(grad(u_nCO), W)
    grad_CO_array = grad_CO.compute_vertex_values()

    grad_H2 = project(grad(u_nH2), W)
    grad_H2_array = grad_H2.compute_vertex_values()

    grad_cat = project(grad(u_ncat), W)
    grad_cat_array = grad_cat.compute_vertex_values()

    # time points as array without staging
    tau_array = np.linspace(0,T,tot_num_steps)
    
    '''
    ## with staging
    tau_array_1 = np.linspace(0, T_1, num_steps_1)
    tau_array_2 = np.linspace(T_1 + dt_2, T_2, num_steps_2)
    tau_array = np.concatenate((tau_array_1,tau_array_2))
    ## with staging
    '''

    # saving unscaled values of concentrations, coordinates, time and electric field in an npz
    np.savez(
        newpath+'arrays_unscaled.npz',
        H=H,
        OH=OH,
        HCO3=HCO3,
        CO32=CO32,
        CO2=CO2,
        CO=CO,
        H2=H2,
        cat=cat,
        p=p,
        coor=coor_array,
        tau=tau_array,
        field_values=field_values,
        H_grad=grad_H_array,
        OH_grad=grad_OH_array,
        HCO3_grad=grad_HCO3_array,
        CO32_grad=grad_CO32_array,
        CO2_grad=grad_CO2_array,
        CO_grad=grad_CO_array,
        H2_grad=grad_H2_array,
        cat_grad=grad_cat_array)
    # on cluster
    #np.savez('arrays_unscaled.npz', \
    
    # collecting scaled concentrations  and times in SI units (numpy arrays)
    c_H, t_H, grad_H_array_scaled = scale_conc_time(
        species='H',
        C=H,
        grad_c=grad_H_array,
        bulk_conc=bulk_conc,
        tau=tau_array,
        diff_coeff_eff=diff_coeff_eff,
        L=L)

    c_OH, t_OH, grad_OH_array_scaled = scale_conc_time(
        species='OH',
        C=OH,
        grad_c=grad_OH_array,
        bulk_conc=bulk_conc,
        tau=tau_array,
        diff_coeff_eff=diff_coeff_eff,
        L=L)

    c_HCO3, t_HCO3, grad_HCO3_array_scaled = scale_conc_time(
        species='HCO3',
        C=HCO3,
        grad_c=grad_HCO3_array,
        bulk_conc=bulk_conc,
        tau=tau_array,
        diff_coeff_eff=diff_coeff_eff,
        L=L)

    c_CO32, t_CO32, grad_CO32_array_scaled = scale_conc_time(
        species='CO32',
        C=CO32,
        grad_c=grad_CO32_array,
        bulk_conc=bulk_conc,
        tau=tau_array,
        diff_coeff_eff=diff_coeff_eff,
        L=L)

    c_CO2, t_CO2, grad_CO2_array_scaled = scale_conc_time(
        species='CO2',
        C=CO2,
        grad_c=grad_CO2_array,
        bulk_conc=bulk_conc,
        tau=tau_array,
        diff_coeff_eff=diff_coeff_eff,
        L=L)

    c_CO, t_CO, grad_CO_array_scaled = scale_conc_time(
        species='CO',
        C=CO,
        grad_c=grad_CO_array,
        bulk_conc=bulk_conc,
        tau=tau_array,
        diff_coeff_eff=diff_coeff_eff,
        L=L)
        
    c_H2, t_H2, grad_H2_array_scaled = scale_conc_time(
        species='H2',
        C=H2,
        grad_c=grad_H2_array,
        bulk_conc=bulk_conc,
        tau=tau_array,
        diff_coeff_eff=diff_coeff_eff,
        L=L)

    c_cat, t_cat, grad_cat_array_scaled = scale_conc_time(
        species=cat_str,
        C=cat,
        grad_c=grad_cat_array,
        bulk_conc=bulk_conc,
        tau=tau_array,
        diff_coeff_eff=diff_coeff_eff,
        L=L)

    coor_scaled = coor_array * L

    psi = p * thermal_voltage

    field_values_scaled = field_values * thermal_voltage / L 

    eps_rel_conc_ss = eps_rel * ((55 - (n_water[cat_str] * c_cat + n_water['H'] * c_H) * 1.0e-3) / 55) \
    + 6 * (((n_water[cat_str] * c_cat + n_water['H'] * c_H) * 1.0e-3) / 55)
    #eps_rel_OHP = eps_rel_conc_ss[-1][0]

    charge_density = c_cat[-1] - c_HCO3[-1] - 2 * c_CO32[-1] - c_OH[-1] + c_H[-1] # at t=final as a function of distance

    # on cluster
    #np.savez('arrays_scaled.npz', \
    np.savez(
        newpath+'arrays_scaled.npz',
        coor_scaled=coor_scaled,
        psi=psi,
        t_H=t_H,
        c_H=c_H,
        t_OH=t_OH,
        c_OH=c_OH,
        t_HCO3=t_HCO3,
        c_HCO3=c_HCO3,
        t_CO32=t_CO32,
        c_CO32=c_CO32,
        t_CO2=t_CO2,
        c_CO2=c_CO2,
        t_CO=t_CO,
        c_CO=c_CO,
        t_H2=t_H2,
        c_H2=c_H2,
        t_cat=t_cat,
        c_cat=c_cat,
        eps_rel=eps_rel_conc_ss,
        field_values=field_values_scaled,
        charge_density=charge_density,
        H_grad=grad_H_array_scaled,
        OH_grad=grad_OH_array_scaled,
        HCO3_grad=grad_HCO3_array_scaled,
        CO32_grad=grad_CO32_array_scaled,
        CO2_grad=grad_CO2_array_scaled,
        CO_grad=grad_CO_array,
        H2_grad=grad_H2_array_scaled,
        cat_grad=grad_cat_array_scaled)
    
    #time_step = time_step_1 # storing the smallest time step used in the simulation in metadata
    #total_sim_time = tot_sim_time_2 # if time is staged

    # create and open metadata file
    f_meta = open(newpath+'metadata.json', 'w') # locally
    #f_meta = open('metadata.json', 'w') # on cluster

    metadata_dict = {
        'concentration_elec':concentration_elec,
        'cation':cation,
        'voltage_multiplier':voltage_multiplier,
        'H2_FE':H2_FE,
        'L':L,
        'R':R,
        'time_step':time_step,
        'total_sim_time':total_sim_time,
        'porosity': porosity_eff,
        'tortuosity': tortuosity_eff,
        'constrictivity': constrictivity_eff,
        'y_CO2': y_CO2,
        'press_gas': press_gas,
        'pore_geom_multiplier': pore_geom_multiplier,
        'electrolyte_flow_geom_multiplier': electrolyte_flow_geom_multiplier,
        'end_time':end_time,
        'eq_conc_CO': eq_conc_CO,
        'eq_conc_H2': eq_conc_H2,
        'current_planar': current_planar,
        'CO2_min':CO2_min}

    r = json.dumps(metadata_dict, indent=0)
    f_meta.write(r)
    f_meta.close()


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='experiment parameters')

    parser.add_argument('--concentration_elec', metavar='electrolyte_concentration', required=False, \
    help='float val, 1.0 M', default=1.0, type=float)

    parser.add_argument('--voltage_multiplier', metavar='thermal_voltage_multiplier', \
    required=False, help='float val, -1.0', default=-1.0, type=float)

    parser.add_argument('--H2_FE', metavar='faradaic efficiency for hydrogen in fraction', \
    required=False, help='float val, 0.05', default=0.05, type=float)

    parser.add_argument('--current_rough', metavar='steady state current in A/m2', \
    required=False, help='float val, 3000.0 (300 mA/cm2)', default=3000.0, type=float)

    parser.add_argument('--L', metavar='cylinder length', \
    required=False, help='float val, 100.0e-9', default=100e-9, type=float)

    parser.add_argument('--R', metavar='cylinder radius', \
    required=False, help='float val, 5.0e-9', default=5e-9, type=float)

    parser.add_argument('--cation', metavar='monovalent cation in solution', \
    required=False, help='str, K/Cs/Li', default='K', type=str)

    parser.add_argument('--porosity_eff', metavar='effective porosity', \
    required=False, help='float val, <1', default=0.5, type=float)

    parser.add_argument('--tortuosity_eff', metavar='effective tortuosity', \
    required=False, help='float val, >1', default=1.5, type=float)

    parser.add_argument('--constrictivity_eff', metavar='effective constrictivity', \
    required=False, help='float val, <1', default=0.9, type=float)

    parser.add_argument('--press_gas', metavar='total gas pressure at S1', \
    required=False, help='float val', default=1.0, type=float)

    parser.add_argument('--pore_geom_multiplier', metavar='control variation in pore geometry', \
    required=False, help='float val', default=1.0, type=float)

    parser.add_argument('--electrolyte_flow_geom_multiplier', metavar='control variation in electrolyte \
    flow perpendicular to catalyst', required=False, help='float val', default=1.0, type=float)

    parser.add_argument('--params_file', metavar='yaml file with parameter values', \
    required=False, help='str, parameters/parameters_pore', default='parameters_pore', type=str)

    parser.add_argument('--y_CO2', metavar='CO2 fraction at S1', \
    required=False, help='float val, <1', default=0.95, type=float)

    parser.add_argument('--roughness_factor', metavar='overall current density is divided by the roughness factor', \
    required=False, help='float value, 100.0-300.0', default=150.0, type=float)

    args = parser.parse_args()

    solveEDL(
        concentration_elec=args.concentration_elec,
        voltage_multiplier=args.voltage_multiplier,
        H2_FE=args.H2_FE,
        current_rough=args.current_rough,
        L=args.L,
        R=args.R,
        press_gas=args.press_gas,
        cation=args.cation,
        porosity_eff=args.porosity_eff,
        tortuosity_eff=args.tortuosity_eff,
        constrictivity_eff=args.constrictivity_eff,
        params_file=args.params_file,
        y_CO2=args.y_CO2,
        pore_geom_multiplier=args.pore_geom_multiplier,
        electrolyte_flow_geom_multiplier=args.electrolyte_flow_geom_multiplier,
        roughness_factor=args.roughness_factor)
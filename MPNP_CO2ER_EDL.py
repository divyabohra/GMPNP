'''
Created on Mon Oct 22 12:05:00 2018

@author: divyabohra

This script solves the PNP/GMPNP system for CO2ER for steady state \
concentration of solution species as well as value of potential and \
electric field as a function of space and time.

Dirichlet conditions are used at both the left and right boundaries \
for potential and for concentration of species in the bulk.\
Flux boundary conditions are used at the OHP for all species

The geometry and the mesh are generated using a separate script.

4 heterogeneous reactions are considered:
    CO2 + H2O + 2e- -> CO + 2OH-
    2H2O + 2e- -> H2 + 2OH-
    2(H+ + e-) -> H2
    CO2 + 2(H+ + e-) -> CO + H2O

The rates of the above reactions are input to the simulation in the form of \
partial current density data.

3 homogeneous reactions are considered:
    H2O <=> H+ + OH- (k_w1, k_w2)
    HCO3- + OH- <=> CO32- + H2O (k_a1, k_a2)
    CO2 + OH- <=> HCO3- (k_b1, k_b2)
The values of the forward and backward rate constants are taken from \
literature and can be found in parameters.yaml file

Species solved for (i): H+, OH-, HCO3-, CO32-, CO2, cat+, Cl-

'''

from __future__ import print_function
from fenics import *  # pylint: disable=unused-wildcard-import
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
# tau is scaled time, C is scaled concentration, chi is scaled distance, \
# phi is scaled potential


def scale(
        species='H',
        tau=None,
        C=None,
        initial_conc={'H': 0.0},
        diff_coeff={'H': 0.0},
        L_n=0.0,
        L_debye=0.0):

    t = (tau * L_debye * L_n) / diff_coeff[species]
    c = C * initial_conc[species]

    return t, c


def solve_EDL(
        concentration_elec=0.1,
        model='MPNP',
        voltage_multiplier=-1.0,
        H2_FE=0.2,
        mesh_structure='variable',
        current_OHP_ss=10.0,
        L_n=50.0e-6,
        stabilization='N',
        H_OHP=None,
        cation='K',
        params_file='parameters',
        dry_run=True
):

    tol = 1.0e-14  # tolerance for coordinate comparisons
    stamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')

    # change below path for accessing utilities
    basepath_utilities = 'utilities/'

    # read rate constants of homogeneous reactions, diffusion coefficients \
    # and diffusion length from yaml file storing default parameters
    f_params = open(basepath_utilities+params_file+'.yaml')
    data = yaml.safe_load(f_params)

    rate_constants = data['rate_constants']

    # see code notes at top for reactions corresponding to rate constants
    kw1 = rate_constants['kw1']
    kw2 = rate_constants['kw2']
    ka1 = rate_constants['ka1']
    ka2 = rate_constants['ka2']
    kb1 = rate_constants['kb1']
    kb2 = rate_constants['kb2']

    # storing the cation string in cat_str
    cat_str = cation

    # hydration numbers of cations
    n_water = {'H': 10.0, cat_str: 0.0}

    if cat_str == 'K':
        n_water[cat_str] = 4
    elif cat_str == 'Li':
        n_water[cat_str] = 5
    elif cat_str == 'Cs':
        n_water[cat_str] = 3
    elif cat_str == 'Na':
        n_water[cat_str] = 5

    species = ['H', 'OH', 'HCO3', 'CO32', 'CO2', cat_str]

    diff_coeff = {'H': 0.0, 'OH': 0.0, 'HCO3': 0.0, 'CO32': 0.0, 'CO2': 0.0,
                  cat_str: 0.0}

    # saving diffusion coefficient of solution species
    for i in species:
        diff_coeff[i] = data['diff_coef']['D_'+i]

    # storing solvated sizes of solution species
    solv_size = {'H': 0.0, 'OH': 0.0, 'HCO3': 0.0, 'CO32': 0.0, 'CO2': 0.0,
                 cat_str: 0.0}

    for i in species:
        solv_size[i] = data['solv_size']['a_'+i]

    # all parameter values used are in SI units
    farad = data['nat_const']['F']  # Faradays constant
    temp = data['nat_const']['T']  # temperature
    k_B = data['nat_const']['k_B']  # Boltzmann constant
    e_0 = data['nat_const']['e_0']  # elementary electron charge
    eps_0 = data['nat_const']['eps_0']  # permittivity of vacuum
    eps_rel = data['nat_const']['eps_rel']  # relative permittivity of water
    R = data['nat_const']['R']  # gas constant
    N_A = data['nat_const']['N_A']  # Avogadros number

    f_params.close()

    # read bulk electrolyte concentrations as calculated by bulk_soln.py
    f_conc = open(
        basepath_utilities+'bulk_soln_'+str(concentration_elec)+'KHCO3.yaml'
    )
    data = yaml.safe_load(f_conc)

    bulk_pH = data['bulk_conc_post_CO2']['final_pH']

    # storing bulk concentration of solution species
    initial_conc = {'H': 0.0, 'OH': 0.0, 'HCO3': 0.0, 'CO32': 0.0, 'CO2': 0.0,
                    cat_str: 0.0}

    # storing charge of solution species
    z = {'H': 1, 'OH': -1, 'HCO3': -1, 'CO32': -2, 'CO2': 0, cat_str: 1}

    for i in species:
        initial_conc[i] = data['bulk_conc_post_CO2']['concentrations']['C0_'+i]

    f_conc.close()

    # H_OHP is the current density due to proton consumption
    # assumed to be 0 by default
    if H_OHP is None:
        current_H_frac = 0.0
    else:
        current_H_frac = 0.001  # initializing at a low value

    # estimation of the Debye length from a Boltzmann distribution
    L_debye = sqrt(
        (eps_0 * eps_rel * k_B * temp) /
        (2 * e_0 ** 2 * concentration_elec * 1.0e+3 * N_A)
    )

    L_D = Constant(L_debye/L_n)  # scaled Debye length

    thermal_voltage = (k_B * temp) / e_0  # thermal voltage

    # using the smallest diffusion coeff of all species
    time_constant = L_debye * L_n / diff_coeff['CO32']

    # scaling factor for homogeneous reaction rate stiochiometry
    scale_R = {'H': 0.0, 'OH': 0.0, 'HCO3': 0.0, 'CO32': 0.0, 'CO2': 0.0,
               cat_str: 0.0}

    for i in species:
        scale_R[i] = Constant((L_n ** 2) / (diff_coeff[i] * initial_conc[i]))

    # scaling factors for Poisson equation
    q = Constant((farad ** 2 * L_n ** 2) / (eps_0 * R * temp))

    # scaled volume of solvated ions
    scale_vol = {'H': 0.0, 'OH': 0.0, 'HCO3': 0.0, 'CO32': 0.0, 'CO2': 0.0,
                 cat_str: 0.0}

    for i in species:
        scale_vol[i] = Constant(solv_size[i] ** 3 * initial_conc[i] * N_A)

    # scaling factors for flux boundary conditions for OH and CO2
    J_H_prefactor = L_n / (diff_coeff['H'] * initial_conc['H'] * farad)
    J_OH_prefactor = L_n / (diff_coeff['OH'] * initial_conc['OH'] * farad)
    J_CO2_prefactor = L_n / (diff_coeff['CO2'] * initial_conc['CO2'] * farad)

    # this voltage is at the OHP and not at the electrode surface
    voltage_scaled = Constant(voltage_multiplier)

    # defining folder name to store simulation results
    identifier = 'voltage_' + str(voltage_multiplier) + '_H2_FE_' + str(H2_FE)\
        + '_current_' + str(current_OHP_ss)\
        + '_H_OHP_' + str(H_OHP) + '_cation_' + cat_str

    # choosing the right mesh for the system size
    L_sys = int(L_n * 1.0e+6)
    if mesh_structure == 'variable':
        mesh_structure = mesh_structure + '_' + str(L_sys) + 'um'
        if L_sys == 1:
            mesh_number = 1090
        elif L_sys == 5:
            mesh_number = 1490
        elif L_sys == 10:
            mesh_number = 1990
        elif L_sys == 50:
            mesh_number = 5990
    elif mesh_structure == 'uniform':
        mesh_number = 1000

    # Read mesh from file
    mesh = Mesh(
        basepath_utilities + '1D_' + mesh_structure + '_mesh_'
        + str(mesh_number) + '.xml.gz'
    )

    # defining boundary where Dirichlet conditions apply
    def boundary_R(x, on_boundary):
        if on_boundary:
            if near(x[0], 1, tol):
                return True
            else:
                return False
        else:
            return False

    # defining boundary where van Neumann conditions apply
    def boundary_L(x, on_boundary):
        if on_boundary:
            if near(x[0], 0, tol):
                return True
            else:
                return False
        else:
            return False

    if dry_run:
        # without staging time for trouble shooting purposes
        # runs for 100 time steps
        # 1e-5 sufficient for 50 microns, 1e-3 sufficient for 1 micron or less.
        time_step = 1.0e-5
        total_sim_time = 1.0e-3

        T = total_sim_time / time_constant  # final time
        dt = time_step / time_constant    # step size
        num_steps = total_sim_time / time_step  # number of steps
        del_ts = [Constant(dt)]
        del_t = del_ts[0]
        tot_num_steps = int(num_steps)

    else:
        # we use 2 time step sizes serially over the
        # total simulation time to reach steady state
        time_step_1 = 1.0e-5
        time_step_2 = 1.0e-3
        total_sim_time_1 = 0.1  # in sec
        total_sim_time_2 = 10.1  # in sec

        T_1 = total_sim_time_1 / time_constant  # final time
        dt_1 = time_step_1 / time_constant    # step size
        num_steps_1 = int(total_sim_time_1 / time_step_1)

        T_2 = total_sim_time_2 / time_constant  # final time
        dt_2 = time_step_2 / time_constant    # step size
        num_steps_2 = int((total_sim_time_2 - total_sim_time_1) / time_step_2)

        del_ts = [Constant(dt_1), Constant(dt_2)]
        del_t = del_ts[0]  # start with smaller time step size
        dts = [dt_1, dt_2]
        dt = dts[0]
        tot_num_steps = num_steps_1 + num_steps_2

    basepath = 'out/'+model+'/'

    newpath = basepath+stamp+'_experiment/'+identifier
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Define function space for system of concentrations
    degree = 1
    P1 = FiniteElement('P', interval, degree)
    # 6 concentrations and 1 potential
    element = MixedElement([P1, P1, P1, P1, P1, P1, P1])
    V = FunctionSpace(mesh, element)
    W = VectorFunctionSpace(mesh, 'P', degree)
    Y = FunctionSpace(mesh, 'P', degree)

    # Define test functions
    (  # pylint: disable=unbalanced-tuple-unpacking
        v_H,
        v_OH,
        v_HCO3,
        v_CO32,
        v_CO2,
        v_cat,
        v_p
    ) = TestFunctions(V)

    # Define functions for the concentrations and potential
    u = Function(V)  # at t_n+1
    # initialization of all variables
    u_0 = Expression(
        ('1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '0.0'),
        degree=1)
    # initializing concentration as bulk and voltage as grounded
    u_n = project(u_0, V)

    # accessing components at t=t_n+1
    (  # pylint: disable=unbalanced-tuple-unpacking
        u_H,
        u_OH,
        u_HCO3,
        u_CO32,
        u_CO2,
        u_cat,
        u_p
    ) = split(u)

    # accessing components t=t_n
    (  # pylint: disable=unbalanced-tuple-unpacking
        u_nH,
        u_nOH,
        u_nHCO3,
        u_nCO32,
        u_nCO2,
        u_ncat,
        u_np
    ) = split(u_n)

    bulk = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    # Constant values for concentrations in bulk and grounded potential
    bc1 = DirichletBC(V, bulk, boundary_R)
    # Dirichlet condition for voltage at the OHP
    bc2 = DirichletBC(V.sub(6), voltage_scaled, boundary_L)
    bcs = [bc1, bc2]

    solver_parameters = {
        'nonlinear_solver': 'newton',
        'newton_solver': {
            'maximum_iterations': 50,
            'relative_tolerance': 1.0e-4,
            'absolute_tolerance': 1.0e-4
        }
    }

    # Neumann condition will be used for cation, HCO3, CO32, H, OH
    # and CO2 at the OHP (x=0) (no flux bc, currents)
    CO_FE = 1 - H2_FE  # Faradaic efficiency of CO production

    # at OHP
    J_CO2 = Constant(J_CO2_prefactor * current_OHP_ss * 0.5 * (CO_FE))
    J_OH = Constant(
        J_OH_prefactor * current_OHP_ss * (1 - current_H_frac) * (-1.0)
    )
    J_H = Constant(J_H_prefactor * current_OHP_ss * current_H_frac)

    # storing coordinates of the mesh and the number of vertices
    coor_array = mesh.coordinates()
    num_vertices = mesh.num_vertices()

    # R_i are the rates of production of species i (scaled)
    # cation is not being consumed or formed in any homogeneous reaction
    R_H = - scale_R['H'] * (
        kw2 * (u_H * initial_conc['H']) * (u_OH * initial_conc['OH']) - kw1
    )

    R_OH = - scale_R['OH'] * (
        kw2 * (u_H * initial_conc['H']) * (u_OH * initial_conc['OH'])
        + ka1 * (u_OH * initial_conc['OH']) * (u_HCO3 * initial_conc['HCO3'])
        + kb1 * (u_CO2 * initial_conc['CO2']) * (u_OH * initial_conc['OH'])
        - kw1 - ka2 * (u_CO32 * initial_conc['CO32']) - kb2
        * (u_HCO3 * initial_conc['HCO3'])
    )

    R_HCO3 = - scale_R['HCO3'] * (
        ka1 * (u_OH * initial_conc['OH']) * (u_HCO3 * initial_conc['HCO3'])
        + kb2 * (u_HCO3 * initial_conc['HCO3'])
        - ka2 * (u_CO32 * initial_conc['CO32'])
        - kb1 * (u_CO2 * initial_conc['CO2']) * (u_OH * initial_conc['OH'])
    )

    R_CO32 = - scale_R['CO32'] * (
        ka2 * (u_CO32 * initial_conc['CO32'])
        - ka1 * (u_OH * initial_conc['OH'] * (u_HCO3 * initial_conc['HCO3']))
    )

    R_CO2 = - scale_R['CO2'] * (
        kb1 * (u_CO2 * initial_conc['CO2']) * (u_OH * initial_conc['OH'])
        - kb2 * (u_HCO3 * initial_conc['HCO3'])
    )

    F_p = - (
        eps_rel * (
            (55 - (n_water[cat_str] * u_cat * initial_conc[cat_str]
                   + n_water['H'] * u_H * initial_conc['H']) * 1.0e-3) / 55
        )
        + 6 * (
            ((n_water[cat_str] * u_cat * initial_conc[cat_str]
                + n_water['H'] * u_H * initial_conc['H']) * 1.0e-3) / 55
        )
    ) * dot(grad(u_p), grad(v_p)) * dx + (
        z['H'] * u_H * initial_conc['H']
        + z['OH'] * u_OH * initial_conc['OH']
        + z['HCO3'] * u_HCO3 * initial_conc['HCO3']
        + z['CO32'] * u_CO32 * initial_conc['CO32']
        + z[cat_str] * u_cat * initial_conc[cat_str]
    ) * q * v_p * dx

    if model == 'PNP':
        F_H = ((u_H - u_nH) / (del_t * L_D)) * v_H * dx \
            + dot(grad(u_H), grad(v_H)) * dx \
            + z['H'] * u_H * dot(grad(u_p), grad(v_H)) * dx - R_H * v_H * dx

        F_OH = ((u_OH - u_nOH) / (del_t * L_D)) * v_OH * dx \
            + dot(grad(u_OH), grad(v_OH)) * dx \
            + z['OH'] * u_OH * dot(grad(u_p), grad(v_OH)) * dx \
            - R_OH * v_OH * dx

        F_HCO3 = ((u_HCO3 - u_nHCO3) / (del_t * L_D)) * v_HCO3 * dx \
            + dot(grad(u_HCO3), grad(v_HCO3)) * dx - R_HCO3 * v_HCO3 * dx \
            + z['HCO3'] * u_HCO3 * dot(grad(u_p), grad(v_HCO3)) * dx

        F_CO32 = ((u_CO32 - u_nCO32) / (del_t * L_D)) * v_CO32 * dx \
            + dot(grad(u_CO32), grad(v_CO32)) * dx - R_CO32 * v_CO32 * dx \
            + z['CO32'] * u_CO32 * dot(grad(u_p), grad(v_CO32)) * dx

        F_CO2 = ((u_CO2 - u_nCO2) / (del_t * L_D)) * v_CO2 * dx \
            + dot(grad(u_CO2), grad(v_CO2)) * dx - R_CO2 * v_CO2 * dx \
            + J_CO2 * v_CO2 * ds

        F_cat = ((u_cat - u_ncat) / (del_t * L_D)) * v_cat * dx \
            + dot(grad(u_cat), grad(v_cat)) * dx \
            + z[cat_str] * u_cat * dot(grad(u_p), grad(v_cat)) * dx

        F = F_H + F_OH + F_HCO3 + F_CO32 + F_CO2 + F_cat + F_p

    elif model == 'MPNP':
        F_H = ((u_H - u_nH) / (del_t * L_D)) * v_H * dx \
            + dot(grad(u_H), grad(v_H)) * dx \
            + z['H'] * u_H * dot(grad(u_p), grad(v_H)) * dx \
            - R_H * v_H * dx \
            + (
                u_H / (1 - (
                    scale_vol['H'] * u_H
                    + scale_vol['OH'] * u_OH
                    + scale_vol['HCO3'] * u_HCO3
                    + scale_vol['CO32'] * u_CO32
                    + scale_vol['CO2'] * u_CO2
                    + scale_vol[cat_str] * u_cat
                ))
            ) * dot(
                scale_vol['H'] * grad(u_H)
                + scale_vol['OH'] * grad(u_OH)
                + scale_vol['HCO3'] * grad(u_HCO3)
                + scale_vol['CO32'] * grad(u_CO32)
                + scale_vol['CO2'] * grad(u_CO2)
                + scale_vol[cat_str] * grad(u_cat),
                grad(v_H)
            ) * dx

        F_OH = ((u_OH - u_nOH) / (del_t * L_D)) * v_OH * dx \
            + dot(grad(u_OH), grad(v_OH)) * dx \
            + z['OH'] * u_OH * dot(grad(u_p), grad(v_OH)) * dx \
            - R_OH * v_OH * dx \
            + (
                u_OH / (1 - (
                    scale_vol['H'] * u_H
                    + scale_vol['OH'] * u_OH
                    + scale_vol['HCO3'] * u_HCO3
                    + scale_vol['CO32'] * u_CO32
                    + scale_vol['CO2'] * u_CO2
                    + scale_vol[cat_str] * u_cat
                ))
            ) * dot(
                scale_vol['H'] * grad(u_H)
                + scale_vol['OH'] * grad(u_OH)
                + scale_vol['HCO3'] * grad(u_HCO3)
                + scale_vol['CO32'] * grad(u_CO32)
                + scale_vol['CO2'] * grad(u_CO2)
                + scale_vol[cat_str] * grad(u_cat),
                grad(v_OH)
            ) * dx

        F_HCO3 = ((u_HCO3 - u_nHCO3) / (del_t * L_D)) * v_HCO3 * dx \
            + dot(grad(u_HCO3), grad(v_HCO3)) * dx \
            + z['HCO3'] * u_HCO3 * dot(grad(u_p), grad(v_HCO3)) * dx \
            - R_HCO3 * v_HCO3 * dx \
            + (
                u_HCO3 / (1 - (
                    scale_vol['H'] * u_H
                    + scale_vol['OH'] * u_OH
                    + scale_vol['HCO3'] * u_HCO3
                    + scale_vol['CO32'] * u_CO32
                    + scale_vol['CO2'] * u_CO2
                    + scale_vol[cat_str] * u_cat
                ))
            ) * dot(
                scale_vol['H'] * grad(u_H)
                + scale_vol['OH'] * grad(u_OH)
                + scale_vol['HCO3'] * grad(u_HCO3)
                + scale_vol['CO32'] * grad(u_CO32)
                + scale_vol['CO2'] * grad(u_CO2)
                + scale_vol[cat_str] * grad(u_cat),
                grad(v_HCO3)
            ) * dx

        F_CO32 = ((u_CO32 - u_nCO32) / (del_t * L_D)) * v_CO32 * dx \
            + dot(grad(u_CO32), grad(v_CO32)) * dx \
            + z['CO32'] * u_CO32 * dot(grad(u_p), grad(v_CO32)) * dx \
            - R_CO32 * v_CO32 * dx \
            + (
                u_CO32 / (1 - (
                    scale_vol['H'] * u_H
                    + scale_vol['OH'] * u_OH
                    + scale_vol['HCO3'] * u_HCO3
                    + scale_vol['CO32'] * u_CO32
                    + scale_vol['CO2'] * u_CO2
                    + scale_vol[cat_str] * u_cat
                ))
            ) * dot(
                scale_vol['H'] * grad(u_H)
                + scale_vol['OH'] * grad(u_OH)
                + scale_vol['HCO3'] * grad(u_HCO3)
                + scale_vol['CO32'] * grad(u_CO32)
                + scale_vol['CO2'] * grad(u_CO2)
                + scale_vol[cat_str] * grad(u_cat),
                grad(v_CO32)
            ) * dx

        F_CO2 = ((u_CO2 - u_nCO2) / (del_t * L_D)) * v_CO2 * dx \
            + dot(grad(u_CO2), grad(v_CO2)) * dx \
            - R_CO2 * v_CO2 * dx \
            + J_CO2 * v_CO2 * ds \
            + (
                u_CO2 / (1 - (
                    scale_vol['H'] * u_H
                    + scale_vol['OH'] * u_OH
                    + scale_vol['HCO3'] * u_HCO3
                    + scale_vol['CO32'] * u_CO32
                    + scale_vol['CO2'] * u_CO2
                    + scale_vol[cat_str] * u_cat
                ))
            ) * dot(
                scale_vol['H'] * grad(u_H)
                + scale_vol['OH'] * grad(u_OH)
                + scale_vol['HCO3'] * grad(u_HCO3)
                + scale_vol['CO32'] * grad(u_CO32)
                + scale_vol['CO2'] * grad(u_CO2)
                + scale_vol[cat_str] * grad(u_cat),
                grad(v_CO2)
            ) * dx

        F_cat = ((u_cat - u_ncat) / (del_t * L_D)) * v_cat * dx \
            + dot(grad(u_cat), grad(v_cat)) * dx \
            + z[cat_str] * u_cat * dot(grad(u_p), grad(v_cat)) * dx \
            + (
                u_cat / (1 - (
                    scale_vol['H'] * u_H
                    + scale_vol['OH'] * u_OH
                    + scale_vol['HCO3'] * u_HCO3
                    + scale_vol['CO32'] * u_CO32
                    + scale_vol['CO2'] * u_CO2
                    + scale_vol[cat_str] * u_cat
                ))
            ) * dot(
                scale_vol['H'] * grad(u_H)
                + scale_vol['OH'] * grad(u_OH)
                + scale_vol['HCO3'] * grad(u_HCO3)
                + scale_vol['CO32'] * grad(u_CO32)
                + scale_vol['CO2'] * grad(u_CO2)
                + scale_vol[cat_str] * grad(u_cat),
                grad(v_cat)
            ) * dx

        F = F_H + F_OH + F_HCO3 + F_CO32 + F_CO2 + F_cat + F_p

    if stabilization == 'Y':
        # normalised over the system length
        h = project(CellDiameter(mesh)).compute_vertex_values()

        rho = {
            'H': np.zeros(num_vertices), 'OH': np.zeros(num_vertices),
            'HCO3': np.zeros(num_vertices), 'CO32': np.zeros(num_vertices),
            'CO2': np.zeros(num_vertices), cat_str: np.zeros(num_vertices)
        }

        Pe = {
            'H': np.zeros(num_vertices), 'OH': np.zeros(num_vertices),
            'HCO3': np.zeros(num_vertices), 'CO32': np.zeros(num_vertices),
            'CO2': np.zeros(num_vertices), cat_str: np.zeros(num_vertices)
        }

        rho_large = {
            'H': np.zeros(num_vertices), 'OH': np.zeros(num_vertices),
            'HCO3': np.zeros(num_vertices), 'CO32': np.zeros(num_vertices),
            'CO2': np.zeros(num_vertices), cat_str: np.zeros(num_vertices)
        }

        fact = 1
        # rho_small = h
        rho_small = fact ** 2 * h ** 2 / 4  # value of rho if Pe is <=1

    H = np.ones(num_vertices)
    OH = np.ones(num_vertices)
    HCO3 = np.ones(num_vertices)
    CO32 = np.ones(num_vertices)
    CO2 = np.ones(num_vertices)
    cat = np.ones(num_vertices)
    p = np.zeros(num_vertices)

    # Time-stepping
    t = 0
    for n in range(tot_num_steps):

        if dry_run:
            print(int(t / dt))
        else:
            # Update current time
            if t >= T_1:
                # choose the larger time step beyond T_1
                dt = dts[1]
                del_t = del_ts[1]
                # adjust number of steps
                print(int(num_steps_1 + (t - T_1) / dt))
            else:
                print(int(t / dt))

        t += dt

        if stabilization == 'Y':
            # norm of the gradient of potential projected on mesh
            norm_grad_phi = project(
                sqrt(inner(grad(u_np), grad(u_np)))
            ).compute_vertex_values()

            for specie in species:
                if z[specie] != 0:
                    Pe[specie] = \
                        (fact * h * norm_grad_phi * abs(z[specie])) / 2

                    rho_large[specie] = \
                        fact * h / (2 * abs(z[specie]) * norm_grad_phi)
                    # rho_large[specie] = h

                    # check if Pe number of > or <= 1
                    for n in range(num_vertices):
                        if Pe[specie][n] > 1.0 + tol:
                            rho[specie][n] = rho_large[specie][n]
                        else:
                            rho[specie][n] = rho_small[n]
                else:
                    continue

            # defining functions over scalar function space Y
            rho_H = Function(Y)
            rho_OH = Function(Y)
            rho_HCO3 = Function(Y)
            rho_CO32 = Function(Y)
            rho_cat = Function(Y)

            rho_H.vector().set_local(np.flip(rho['H'], 0))
            rho_OH.vector().set_local(np.flip(rho['OH'], 0))
            rho_HCO3.vector().set_local(np.flip(rho['HCO3'], 0))
            rho_CO32.vector().set_local(np.flip(rho['CO32'], 0))
            rho_cat.vector().set_local(np.flip(rho[cat_str], 0))

            if model == 'PNP':

                F_stab_PNP = -1.0 * (
                    rho_H * z['H'] * (
                        (u_H - u_nH) / (del_t * L_D)
                        + z['H'] * dot(grad(u_H), grad(u_p))
                        + R_H
                    ) * dot(grad(u_p), grad(v_H)) * dx
                    + rho_OH * z['OH'] * (
                        (u_OH - u_nOH) / (del_t * L_D)
                        + z['OH'] * dot(grad(u_H), grad(u_p))
                        + R_OH
                    ) * dot(grad(u_p), grad(v_OH)) * dx
                    + rho_HCO3 * z['HCO3'] * (
                        (u_HCO3 - u_nHCO3) / (del_t * L_D)
                        + z['HCO3'] * dot(grad(u_HCO3), grad(u_p))
                        + R_HCO3
                    ) * dot(grad(u_p), grad(v_HCO3)) * dx
                    + rho_CO32 * z['CO32'] * (
                        (u_CO32 - u_nCO32) / (del_t * L_D)
                        + z['CO32'] * dot(grad(u_CO32), grad(u_p))
                        + R_CO32
                    ) * dot(grad(u_p), grad(v_CO32)) * dx
                    + rho_cat * z[cat_str] * (
                        (u_cat - u_ncat) / (del_t * L_D)
                        + z[cat_str] * dot(grad(u_cat), grad(u_p))
                    ) * dot(grad(u_p), grad(v_cat)) * dx
                )

                # Solve variational problem for time step
                solve(
                    F + F_stab_PNP + J_OH * v_OH * ds + J_H * v_H * ds == 0,
                    u,
                    bcs,
                    solver_parameters=solver_parameters
                )

            elif model == 'MPNP':
                # SUPG stabilization formulation not
                # well suited for MPNP due to volume term
                print("Warning:stabilization not implemented for MPNP!")

                solve(
                    F + J_OH * v_OH * ds + J_H * v_H * ds == 0,
                    u,
                    bcs,
                    solver_parameters=solver_parameters
                )

        else:
            solve(
                F + J_OH * v_OH * ds + J_H * v_H * ds == 0,
                u,
                bcs,
                solver_parameters=solver_parameters
            )

        # Save solution to file (VTK)
        _u_H, _u_OH, _u_HCO3, _u_CO32, _u_CO2, _u_cat, _u_p = u.split()

        _u_H_nodal_values_array = _u_H.compute_vertex_values()
        _u_OH_nodal_values_array = _u_OH.compute_vertex_values()
        _u_HCO3_nodal_values_array = _u_HCO3.compute_vertex_values()
        _u_CO32_nodal_values_array = _u_CO32.compute_vertex_values()
        _u_CO2_nodal_values_array = _u_CO2.compute_vertex_values()
        _u_cat_nodal_values_array = _u_cat.compute_vertex_values()
        _u_p_nodal_values_array = _u_p.compute_vertex_values()

        # creating a numpy array of concentration values
        # at every time step in the whole domain
        H = np.vstack((H, _u_H_nodal_values_array))
        OH = np.vstack((OH, _u_OH_nodal_values_array))
        HCO3 = np.vstack((HCO3, _u_HCO3_nodal_values_array))
        CO32 = np.vstack((CO32, _u_CO32_nodal_values_array))
        CO2 = np.vstack((CO2, _u_CO2_nodal_values_array))
        cat = np.vstack((cat, _u_cat_nodal_values_array))
        p = np.vstack((p, _u_p_nodal_values_array))

        # fraction (wrt bulk) of protons at the OHP
        H_OHP_frac = _u_H_nodal_values_array[0]

        # adjust proton consumption current iteratively
        # if H_OHP frac is higher than a specified limit
        if H_OHP is not None:
            if H_OHP_frac < 0:
                current_H_frac = current_H_frac / 1.1
            elif H_OHP_frac < (H_OHP - 0.05):
                current_H_frac = current_H_frac / 1.05
            elif H_OHP_frac < (H_OHP - 0.025):
                current_H_frac = current_H_frac / 1.01
            elif (H_OHP_frac > H_OHP and
                    H_OHP_frac <= (H_OHP + 0.4) and
                    current_H_frac <= 1.0):
                current_H_frac = current_H_frac * 1.04
            elif H_OHP_frac > (H_OHP + 0.4) and current_H_frac <= 1.0:
                current_H_frac = current_H_frac * 1.15

            print(H_OHP_frac)
            print(current_H_frac)

            # new flux of OH and H after adjustment of
            # the proton consumption current at OHP
            J_OH = Constant(
                -1.0 * J_OH_prefactor * current_OHP_ss * (1 - current_H_frac)
            )

            J_H = Constant(J_H_prefactor * current_OHP_ss * current_H_frac)

        # Update previous solution
        u_n.assign(u)

    end_time = datetime.now().strftime('%y-%m-%d-%H-%M-%S')

    # estimating the electric field value as a function
    # of x for the last computed value of potential profile
    field = project(-grad(u_np), W)  # pylint: disable=E1130
    field_values = field.compute_vertex_values()
    field_values_rescaled = field_values * thermal_voltage / L_n
    field_OHP = field_values_rescaled[0] * 1.0e-9  # in V/nm

    if dry_run:
        # time points as array without staging
        tau_array = np.linspace(0, T, tot_num_steps)
    else:
        # with staging
        tau_array_1 = np.linspace(0, T_1, num_steps_1)
        tau_array_2 = np.linspace(T_1 + dts[1], T_2, num_steps_2)
        # concatenate the two time step arrays
        tau_array = np.concatenate((tau_array_1, tau_array_2))

    if stabilization != 'Y':
        Pe = None
        rho = None

    np.savez(
        newpath+'/arrays_unscaled.npz',
        H=H,
        OH=OH,
        HCO3=HCO3,
        CO32=CO32,
        CO2=CO2,
        cat=cat,
        p=p,
        coor=coor_array,
        tau=tau_array,
        field_values=field_values)

    # rescaling the output. all outputs in SI unitsand numpy arrays
    t_H, c_H = scale(
        species='H',
        tau=tau_array,
        C=H,
        initial_conc=initial_conc,
        diff_coeff=diff_coeff,
        L_n=L_n,
        L_debye=L_debye)

    t_OH, c_OH = scale(
        species='OH',
        tau=tau_array,
        C=OH,
        initial_conc=initial_conc,
        diff_coeff=diff_coeff,
        L_n=L_n,
        L_debye=L_debye)

    t_HCO3, c_HCO3 = scale(
        species='HCO3',
        tau=tau_array,
        C=HCO3,
        initial_conc=initial_conc,
        diff_coeff=diff_coeff,
        L_n=L_n,
        L_debye=L_debye)

    t_CO32, c_CO32 = scale(
        species='CO32',
        tau=tau_array,
        C=CO32,
        initial_conc=initial_conc,
        diff_coeff=diff_coeff,
        L_n=L_n,
        L_debye=L_debye)

    t_CO2, c_CO2 = scale(
        species='CO2',
        tau=tau_array,
        C=CO2,
        initial_conc=initial_conc,
        diff_coeff=diff_coeff,
        L_n=L_n,
        L_debye=L_debye)

    t_cat, c_cat = scale(
        species=cat_str,
        tau=tau_array,
        C=cat,
        initial_conc=initial_conc,
        diff_coeff=diff_coeff,
        L_n=L_n,
        L_debye=L_debye)

    coor_scaled = coor_array * L_n

    psi = p * thermal_voltage

    pH_OHP = - math.log10(c_H[-1][0] / 1000)

    eps_rel_conc_ss = \
        eps_rel * ((55 - (n_water[cat_str] * c_cat
                    + n_water['H'] * c_H) * 1.0e-3) / 55) \
        + 6 * (((n_water[cat_str] * c_cat + n_water['H'] * c_H) * 1.0e-3) / 55)

    eps_rel_OHP = eps_rel_conc_ss[-1][0]

    # at steady state as a function of x
    charge_density = \
        c_cat[-1] - c_HCO3[-1] - 2 * c_CO32[-1] - c_OH[-1] + c_H[-1]

    np.savez(
        newpath+'/arrays_scaled.npz',
        x=coor_scaled,
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
        t_cat=t_cat,
        c_cat=c_cat,
        eps_rel=eps_rel_conc_ss,
        field_values=field_values_rescaled,
        charge_density=charge_density)

    # initiating empty lists for storing OHP concentration values
    H_surf = []
    OH_surf = []
    HCO3_surf = []
    CO32_surf = []
    CO2_surf = []
    cat_surf = []
    p_surf = []

    for i in range(0, len(t_cat)):
        H_surf += [c_H[i][0]]
        OH_surf += [c_OH[i][0]]
        HCO3_surf += [c_HCO3[i][0]]
        CO32_surf += [c_CO32[i][0]]
        CO2_surf += [c_CO2[i][0]]
        cat_surf += [c_cat[i][0]]
        p_surf += [psi[i][0]]

    potential_OHP = p_surf[-1]

    CO2_OHP_frac = CO2_surf[-1] / initial_conc['CO2']

    pH_overpotential = - 0.059 * (bulk_pH - pH_OHP) * 1.0e+3  # in mV

    # in mV
    CO2_overpotential = (0.059 / 2) * math.log10(1 / CO2_OHP_frac) * 1.0e+3

    # current density attributed to proton consumption
    current_H = current_H_frac * current_OHP_ss

    # time_step = time_step_1 # if staged
    # total_sim_time = tot_sim_time_2 # if time is staged

    # create and open metadata file
    f_meta = open(newpath+'/metadata.json', 'w')

    metadata_dict = {
        'concentration_elec': concentration_elec,
        'cation': cation,
        'model': model,
        'stabilization': stabilization,
        'voltage_multiplier': voltage_multiplier,
        'H2_FE': H2_FE,
        'L_n_EDL': L_n,
        'time_constant': time_constant,
        'time_step': time_step,
        'total_sim_time': total_sim_time,
        'mesh_number': mesh_number,
        'mesh_structure': mesh_structure,
        'eps_rel_OHP': eps_rel_OHP,
        'field_OHP': field_OHP,
        'current_OHP_ss': current_OHP_ss,
        'current_H': current_H,
        'H_OHP_vs_bulk': H_OHP,
        'potential_OHP': potential_OHP,
        'pH_OHP': pH_OHP,
        'CO2_OHP_frac': CO2_OHP_frac,
        'pH_overpotential': pH_overpotential,
        'CO2_overpotential': CO2_overpotential,
        'end_time': end_time}

    r = json.dumps(metadata_dict, indent=0)
    f_meta.write(r)
    f_meta.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment parameters')

    parser.add_argument(
        '--concentration_elec',
        metavar='electrolyte_concentration',
        required=False,
        help='float val, 0.1 M',
        default=0.1,
        type=float
    )

    parser.add_argument(
        '--model',
        metavar='model_type',
        required=False,
        help='str, PNP/MPNP',
        default='MPNP',
        type=str
    )

    parser.add_argument(
        '--voltage_multiplier',
        metavar='thermal_voltage_multiplier',
        required=False,
        help='float val, -1.0',
        default=-1.0,
        type=float
    )

    parser.add_argument(
        '--mesh_structure',
        metavar='bias in mesh structure',
        required=False,
        help='str, uniform/variable',
        default='variable',
        type=str
    )

    parser.add_argument(
        '--H2_FE',
        metavar='faradaic efficiency for hydrogen in fraction',
        required=False,
        help='float val, 0.2',
        default=0.2,
        type=float
    )

    parser.add_argument(
        '--current_OHP_ss',
        metavar='steady state current in A/m2',
        required=False,
        help='float val, 10.0',
        default=10.0,
        type=float
    )

    parser.add_argument(
        '--L_n',
        metavar='system size',
        required=False,
        help='float val, 50.0e-6',
        default=50.0e-6,
        type=float
    )

    parser.add_argument(
        '--stabilization',
        metavar='SUPG',
        required=False,
        help='str, Y/N',
        default='N',
        type=str
    )

    parser.add_argument(
        '--H_OHP',
        metavar='build up of protons at the OHP relative to the bulk',
        required=False,
        help='float val, None/1.1/2.0',
        default=None,
        type=float
    )

    parser.add_argument(
        '--cation',
        metavar='monovalent cation in solution',
        required=False,
        help='str, K/Cs/Li',
        default='K',
        type=str
    )

    parser.add_argument(
        '--params_file',
        metavar='yaml file with parameter values',
        required=False,
        help='str, parameters',
        default='parameters',
        type=str
    )

    parser.add_argument(
        '--dry_run',
        metavar='run 100 time steps as test',
        required=False,
        help='boolean value',
        default=True,
        type=bool
    )

    args = parser.parse_args()

    solve_EDL(
        concentration_elec=args.concentration_elec,
        model=args.model,
        voltage_multiplier=args.voltage_multiplier,
        mesh_structure=args.mesh_structure,
        H2_FE=args.H2_FE,
        current_OHP_ss=args.current_OHP_ss,
        L_n=args.L_n,
        stabilization=args.stabilization,
        H_OHP=args.H_OHP,
        cation=args.cation,
        dry_run=args.dry_run
    )

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Created on Mon Oct 22 12:05:00 2018

@author: divyabohra

This script solves the reaction-diffusion system for CO2ER for steady state \
concentration of solution species.

The geometry and the mesh are generated using a separate script.

3 heterogeneous reactions are considered:
    CO2 + H2O + 2e- -> CO + 2OH-
    CO2 + H2O + 2e- -> HCOO- + OH-
    2H2O + 2e- -> H2 + 2OH-
The rates of the above reactions are input to the simulation in the form of \
partial current density data from previous publications.

3 homogeneous reactions are considered:
    H2O <=> H+ + OH- (k_w1, k_w2)
    HCO3- + OH- <=> CO32- + H2O (k_a1, k_a2)
    CO2 + OH- <=> HCO3- (k_b1, k_b2)
The values of the forward and backward rate constants are taken from \
literature.

Species solved for (i): H+, OH-, HCO3-, CO32-, CO2, K+

'''

from __future__ import print_function
from fenics import *  # pylint: disable=unused-wildcard-import
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
import json


tol = 1.0e-14  # tolerance for coordinate comparisons
stamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')


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


# the function below writes a experiment config file for each run \
# and saves it in the experiment folder
def metadata(
    concentration_KHCO3=1.0,
    model='rxn_diffusion',
    L_n=None, bulk_pH=7.0,
    time_constant=None,
    total_sim_time=None,
    time_step=None,
    mesh_number=1000,
    mesh_structure='uniform',
    H2_FE=0.0,
    CO_FE=0.0,
    current_OHP_ss=0.0,
    pH_OHP=0.0,
    f=None,
    pH_overpotential=0.0,
    CO2_overpotential=0.0,
    CO2_OHP_frac=0.0
):

    metadata_dict = {
        'concentration_KHCO3': concentration_KHCO3,
        'model': model,
        'L_n': L_n,
        'bulk_pH': bulk_pH,
        'time_constant': time_constant,
        'total_sim_time': total_sim_time,
        'time_step': time_step,
        'mesh_number': mesh_number,
        'mesh_structure': mesh_structure,
        'H2_FE': H2_FE,
        'CO_FE': CO_FE,
        'current_OHP': current_OHP_ss,
        'pH_OHP': pH_OHP,
        'pH_overpotential': pH_overpotential,
        'CO2_overpotential': CO2_overpotential,
        'CO2_OHP_frac': CO2_OHP_frac
    }

    r = json.dumps(metadata_dict, indent=0)
    f.write(r)

    return


# function to read csv file to return lists
# if using experimental data
def readIVdata(filename):
    volt = []
    HCOO = []
    CO = []
    H2 = []

    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            volt.append(row[0])
            HCOO.append(row[1])
            CO.append(row[2])
            H2.append(row[3])

    return volt, HCOO, CO, H2


def main(
    concentration_KHCO3=0.1,
    H2_FE=0.2,
    L_n=50.0e-6,
    mesh_number=5000,
    mesh_structure='uniform',
    current_OHP_ss=10.0
):

    basepath_utilities = 'utilities/'

    # read rate constants of homogeneous reactions, diffusion coefficients \
    # and diffusion length from yaml file storing default parameters
    f = open(basepath_utilities+'parameters.yaml')
    data = yaml.load(f)

    rate_constants = data['rate_constants']

    # see code notes at top for reactions corresponding to rate constants
    kw1 = rate_constants['kw1']
    kw2 = rate_constants['kw2']
    ka1 = rate_constants['ka1']
    ka2 = rate_constants['ka2']
    kb1 = rate_constants['kb1']
    kb2 = rate_constants['kb2']

    species = ['H', 'OH', 'HCO3', 'CO32', 'CO2', 'K']

    diff_coeff = {'H': 0.0, 'OH': 0.0, 'HCO3': 0.0, 'CO32': 0.0, 'CO2': 0.0,
                  'K': 0.0}

    # saving diffusion coefficient of solution species
    for i in species:
        diff_coeff[i] = data['diff_coef']['D_'+i]

    farad = data['nat_const']['F']  # Faradays constant

    f.close()

    def solve_rxn_diff(
        concentration_KHCO3=concentration_KHCO3,
        H2_FE=H2_FE,
        L_n=L_n,
        mesh_number=mesh_number,
        mesh_structure=mesh_structure,
        current_OHP_ss=current_OHP_ss
    ):

        # using the smallest diffusion coeff of all species
        time_constant = L_n ** 2 / diff_coeff['CO32']

        # Below function is used to scale back the calculated variables \
        # from dimentionless form to SI units
        # tau is scaled time, C is scaled concentration, chi is scaled distance
        def scale(species='H', tau=None, C=None, chi=[0.0]):

            t = [(n * L_n ** 2) / diff_coeff[species] for n in tau]
            c = C * initial_conc[species]
            x = [n * L_n for n in chi]

            return t, c, x

        # read bulk electrolyte concentrations as calculated by bulk_soln.py
        f = open(
            basepath_utilities + 'bulk_soln_' + str(concentration_KHCO3)
            + 'KHCO3.yaml'
        )
        data = yaml.load(f)

        bulk_pH = data['bulk_conc_post_CO2']['final_pH']

        # storing bulk concentration of solution species
        initial_conc = {'H': 0.0, 'OH': 0.0, 'HCO3': 0.0, 'CO32': 0.0,
                        'CO2': 0.0, 'K': 0.0}

        for i in species:
            initial_conc[i] = \
                data['bulk_conc_post_CO2']['concentrations']['C0_'+i]

        f.close()

        # scaling factor for homogeneous reaction rate stiochiometry
        scale_R = {'H': 0.0, 'OH': 0.0, 'HCO3': 0.0, 'CO32': 0.0,
                   'CO2': 0.0, 'K': 0.0}

        for i in species:
            scale_R[i] = \
                Constant((L_n ** 2) / (diff_coeff[i] * initial_conc[i]))

        # scaling factors for flux boundary conditions \
        # for OH and CO2
        J_OH_prefactor = \
            L_n / (diff_coeff['OH'] * initial_conc['OH'] * farad)
        J_CO2_prefactor = \
            L_n / (diff_coeff['CO2'] * initial_conc['CO2'] * farad)

        CO_FE = 1 - H2_FE

        # at OHP
        J_CO2 = Constant(J_CO2_prefactor * current_OHP_ss * 0.5 * (CO_FE))
        J_OH = Constant(J_OH_prefactor * current_OHP_ss * (-1.0))

        identifier = 'H2_FE_' + str(H2_FE) + '_current_' + str(current_OHP_ss)\
            + '_L_n_' + str(L_n)
        basepath = 'out/'
        newpath = basepath + stamp + '_experiment/' + identifier
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        total_sim_time = 10  # in sec
        time_step = 2.0e-2  # in sec

        T = total_sim_time / time_constant  # final time
        dt = time_step / time_constant    # step size
        num_steps = T / dt  # number of steps
        del_t = Constant(dt)

        # Read mesh from file
        mesh = Mesh(
            basepath_utilities + '1D_' + mesh_structure + '_mesh_'
            + str(mesh_number) + '.xml.gz'
        )

        # Define function space for system of concentrations
        degree = 1
        P1 = FiniteElement('P', interval, degree)
        element = MixedElement([P1, P1, P1, P1, P1])  # 5 concentrations
        V = FunctionSpace(mesh, element)

        # Define test functions
        (  # pylint: disable=unbalanced-tuple-unpacking
            v_H,
            v_OH,
            v_HCO3,
            v_CO32,
            v_CO2
         ) = TestFunctions(V)

        # Define functions for the concentrations and potential
        u = Function(V)  # at t_n+1
        # initialization of all variables
        u_0 = Expression(('1.0', '1.0', '1.0', '1.0', '1.0'), degree=1)
        # initializing concentration as bulk
        u_n = project(u_0, V)

        # Split system functions to access components
        (  # pylint: disable=unbalanced-tuple-unpacking
            u_H,
            u_OH,
            u_HCO3,
            u_CO32,
            u_CO2
        ) = split(u)

        (  # pylint: disable=unbalanced-tuple-unpacking
            u_nH,
            u_nOH,
            u_nHCO3,
            u_nCO32,
            u_nCO2
        ) = split(u_n)

        bulk = [1.0, 1.0, 1.0, 1.0, 1.0]
        # Constant values for concentrations in bulk
        bcs = DirichletBC(V, bulk, boundary_R)

        # Neumann condition will be used for all species at the OHP (x=0)
        # R_i are the rates of production of species i (scaled)
        # cation is not being consumed or formed in any homogeneous reaction

        R_H = - scale_R['H'] * (
            kw2 * (u_H * initial_conc['H']) * (u_OH * initial_conc['OH']) - kw1
        )

        R_OH = - scale_R['OH'] * (
            kw2 * (u_H * initial_conc['H']) * (u_OH * initial_conc['OH'])
            + ka1 * (u_OH * initial_conc['OH'])
            * (u_HCO3 * initial_conc['HCO3'])
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
            - ka1 * (u_OH * initial_conc['OH']
                     * (u_HCO3 * initial_conc['HCO3']))
        )

        R_CO2 = - scale_R['CO2'] * (
            kb1 * (u_CO2 * initial_conc['CO2']) * (u_OH * initial_conc['OH'])
            - kb2 * (u_HCO3 * initial_conc['HCO3'])
        )

        F = ((u_H - u_nH) / del_t) * v_H * dx \
            + dot(grad(u_H), grad(v_H)) * dx \
            - R_H * v_H * dx \
            + ((u_OH - u_nOH) / del_t) * v_OH * dx \
            + dot(grad(u_OH), grad(v_OH)) * dx \
            - R_OH * v_OH * dx \
            + ((u_HCO3 - u_nHCO3) / del_t) * v_HCO3 * dx \
            + dot(grad(u_HCO3), grad(v_HCO3)) * dx \
            - R_HCO3 * v_HCO3 * dx \
            + ((u_CO32 - u_nCO32) / del_t) * v_CO32 * dx \
            + dot(grad(u_CO32), grad(v_CO32)) * dx \
            - R_CO32 * v_CO32 * dx \
            + ((u_CO2 - u_nCO2) / del_t) * v_CO2 * dx \
            + dot(grad(u_CO2), grad(v_CO2)) * dx \
            - R_CO2 * v_CO2 * dx \
            + J_OH * v_OH * ds + J_CO2 * v_CO2 * ds

        # storing coordinates of the nodes as a list
        coor_array = mesh.coordinates()
        coor = coor_array.tolist()
        coor_list = [item for sublist in coor for item in sublist]

        H = np.ones(len(coor_list))
        OH = np.ones(len(coor_list))
        HCO3 = np.ones(len(coor_list))
        CO32 = np.ones(len(coor_list))
        CO2 = np.ones(len(coor_list))

        # Time-stepping
        t = 0
        for n in range(int(num_steps)):

            # Update current time
            t += dt
            # Solve variational problem for time step
            solve(
                F == 0,
                u,
                bcs,
                solver_parameters={
                    'nonlinear_solver': 'newton',
                    'newton_solver': {
                        'maximum_iterations': 100,
                        'relative_tolerance': 1.0e-6,
                        'absolute_tolerance': 1.0e-6
                    }
                }
            )

            # Save solution to file (VTK)
            _u_H, _u_OH, _u_HCO3, _u_CO32, _u_CO2 = u.split()

            _u_H_nodal_values_array = _u_H.compute_vertex_values()
            _u_OH_nodal_values_array = _u_OH.compute_vertex_values()
            _u_HCO3_nodal_values_array = _u_HCO3.compute_vertex_values()
            _u_CO32_nodal_values_array = _u_CO32.compute_vertex_values()
            _u_CO2_nodal_values_array = _u_CO2.compute_vertex_values()

            # creating a numpy array of concentration values at \
            # every time step in the whole domain
            H = np.vstack((H, _u_H_nodal_values_array))
            OH = np.vstack((OH, _u_OH_nodal_values_array))
            HCO3 = np.vstack((HCO3, _u_HCO3_nodal_values_array))
            CO32 = np.vstack((CO32, _u_CO32_nodal_values_array))
            CO2 = np.vstack((CO2, _u_CO2_nodal_values_array))

            # Update previous solution
            u_n.assign(u)
            print(n)

        # storing time points as a list
        tau_array = np.linspace(0, T, num_steps)
        tau = tau_array.tolist()

        np.savez(
            newpath + '/arrays_unscaled.npz',
            H=H,
            OH=OH,
            HCO3=HCO3,
            CO32=CO32,
            CO2=CO2,
            coor_array=coor_array,
            tau_array=tau_array
        )

        # rescaling the output. all outputs in SI units.
        # c_is are numpy arrays, x and t_is are lists
        t_H, c_H, x = scale(
            species='H',
            tau=tau,
            C=H,
            chi=coor_list)

        t_OH, c_OH, x = scale(
            species='OH',
            tau=tau,
            C=OH,
            chi=coor_list)

        t_HCO3, c_HCO3, x = scale(
            species='HCO3',
            tau=tau,
            C=HCO3,
            chi=coor_list)

        t_CO32, c_CO32, x = scale(
            species='CO32',
            tau=tau,
            C=CO32,
            chi=coor_list)

        t_CO2, c_CO2, x = scale(
            species='CO2',
            tau=tau,
            C=CO2,
            chi=coor_list)

        # below experesion implies that electroneutrality is
        # maintained throughout the solution.
        c_K = c_HCO3 + 2 * c_CO32 + c_OH - c_H

        pH_OHP = - math.log10(c_H[-1][0] / 1000)

        np.savez(
            newpath + '/arrays_scaled.npz',
            x=np.array(x),
            t_H=np.array(t_H),
            c_H=c_H,
            t_OH=np.array(t_OH),
            c_OH=c_OH,
            t_HCO3=np.array(t_HCO3),
            c_HCO3=c_HCO3,
            t_CO32=np.array(t_CO32),
            c_CO32=c_CO32,
            t_CO2=np.array(t_CO2),
            c_CO2=c_CO2,
            c_K=c_K
        )

        # storing concentration values at OHP as a function of time
        # these values can be plotted to see the time evolution of
        # species at the OHP
        H_surf = []
        OH_surf = []
        HCO3_surf = []
        CO32_surf = []
        CO2_surf = []
        K_surf = []

        for i in range(0, len(t_H)):
            H_surf += [c_H[i][0]]
            OH_surf += [c_OH[i][0]]
            HCO3_surf += [c_HCO3[i][0]]
            CO32_surf += [c_CO32[i][0]]
            CO2_surf += [c_CO2[i][0]]
            K_surf += [c_K[i][0]]

        pH_overpotential = - 0.059 * (bulk_pH - pH_OHP) * 1.0e+3  # in mV

        # in mV
        CO2_overpotential = (0.059 / 2) * math.log10(initial_conc['CO2']
                                                     / CO2_surf[-1]) * 1.0e+3
        CO2_OHP_frac = CO2_surf[-1] / initial_conc['CO2']

        CO2_OHP_frac = CO2_surf[-1] / initial_conc['CO2']

        # create and open metadata file
        f = open(newpath+'/metadata.json', 'w')

        metadata(
            concentration_KHCO3=concentration_KHCO3,
            L_n=L_n,
            bulk_pH=bulk_pH,
            time_constant=time_constant,
            total_sim_time=total_sim_time,
            time_step=time_step,
            mesh_number=mesh_number,
            mesh_structure=mesh_structure,
            H2_FE=H2_FE,
            CO_FE=CO_FE,
            current_OHP_ss=current_OHP_ss,
            pH_OHP=pH_OHP,
            f=f,
            pH_overpotential=pH_overpotential,
            CO2_overpotential=CO2_overpotential,
            CO2_OHP_frac=CO2_OHP_frac
        )

        f.close()

    solve_rxn_diff(
        concentration_KHCO3=concentration_KHCO3,
        H2_FE=H2_FE,
        L_n=L_n,
        mesh_number=mesh_number,
        mesh_structure=mesh_structure,
        current_OHP_ss=current_OHP_ss
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment parameters')

    parser.add_argument(
        '--concentration_KHCO3',
        metavar='electrolyte_concentration',
        required=False,
        help='float val, 0.1 M',
        default=0.1,
        type=float
    )

    parser.add_argument(
        '--mesh_number',
        metavar='no. of mesh vertices',
        required=False,
        help='int, 1000/5000',
        default=5000,
        type=int
    )

    parser.add_argument(
        '--mesh_structure',
        metavar='bias in mesh structure',
        required=False,
        help='str, uniform/stretched/variable_50um',
        default='uniform',
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
        '--L_n',
        metavar='Nernst boundary layer thickness',
        required=False,
        help='float val, 50.0e-6',
        default=50.0e-6,
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

    args = parser.parse_args()

    main(
        concentration_KHCO3=args.concentration_KHCO3,
        H2_FE=args.H2_FE,
        L_n=args.L_n,
        mesh_number=args.mesh_number,
        mesh_structure=args.mesh_structure,
        current_OHP_ss=args.current_OHP_ss
    )

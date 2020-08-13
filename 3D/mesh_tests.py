'''
@author: divyabohra

This script tests different geometrical tolerances to check whether
all elements on the boundary wall are marked by the class boundary_wall
by comparing the integral of all facets marked with the expected
surface area of the cylindrical wall.
'''

from __future__ import print_function
from fenics import *  # pylint: disable=unused-wildcard-import
import yaml
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, date, time
import os
import argparse
import json

tol = 1e-12

R = 5.0
L = 80.0

aspect_pore = R/L

basepath_utilities = '/home/divya/Documents/src/pnp/utilities/'

# Read mesh from file
mesh = Mesh(basepath_utilities+'L_'+str(int(L))+'_R_'+str(int(R))+'.xml')


# defining boundary at the pore exit face
class boundary_pore_exit(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 1, tol)


# defining boundary at the pore entry
class boundary_pore_entry(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 0, tol)


# defining pore wall boundary
class boundary_wall(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[0] ** 2 + x[1] ** 2), (aspect_pore ** 2), 1e-3)


# Mark boundaries
# The dimension given to the MeshFunction is 0 for a vertex
boundary_markers = MeshFunction('size_t', mesh, 2)
boundary_markers.set_all(9999)  # pylint: disable=no-member

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

# Define function space for system of concentrations
degree = 1
P3 = FiniteElement('Lagrange', tetrahedron, degree)
V = FunctionSpace(mesh, P3)

# Define functions for the area
u_area = Function(V)
u_area = Constant(1.0)

A_1 = A_3 = pi * aspect_pore ** 2
A_2 = 2 * pi * aspect_pore

form4 = u_area*ds(2)  # integration on the whole facet marked with number 4
ans = assemble(form4)  # compute the length of the circle
print(ans, "vs.", A_2)  # compare

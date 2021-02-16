import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from dolfin import *
from fenics import *
from matplotlib import cm

T = 5           # final time
num_steps = 1000     # number of time steps
dt = 1 #T / num_steps # time step size

# Create mesh and define function space
nx = ny = 51
mesh = RectangleMesh(Point(-0.2, -0.2), Point(1.2, 1.2), nx, ny)
V = FunctionSpace(mesh, 'CG', 2)



w1 = -2.56  # Basal inhibition of GFR
w13 = -1.6 #-1.6  # GFR inhibition by E2ER
w11 = 6.8  # GFR self activation
w12 = 0.4  # GFR activation by ERM
w2 = -3.04  # Basal inhibition of EPM
w23 = -1.6 #-1.6  # EPM inhibition by E2ER
w22 = 6.64 #6.64  # ERM self activation
w21 = 0.4  # ERM activation by GFR
w3 = 5.0  # E2ER self activation
w3e = 0.5  # E2ER activation by E2
E2 = 2
x3 = 1/(1+np.exp(-(w3 + w3e * E2)))





velocity = Expression(('1/(1+exp(-(w1 + w13 * x3 + w11 * x[0] + w12 * x[1])))  - x[0]',
                               '1/(1+exp(-(w2 + w23 * x3 + w22 * x[1] + w21 * x[0])))  - x[1]'), 
                              x3=x3, 
                              w1 = -2.56,
                              w13 = -1.6,
                              w11 = 6.8,
                              w12 = 0.4, w2 = -3.04, w23 = -1.6, w22 = 6.64, w21 = 0.4, 
                              w3 = 5.0,  w3e = 0.5 , degree=3)
# Define variational problem
u = TrialFunction(V) 
v = TestFunction(V)

# Define initial value
#u0 = Expression('exp(-(pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2)) )', degree=5)
u0 = Expression('100.0',degree=0)

u0 = interpolate(u0, V)

F =  dot(grad(u), grad(v))*dx \
    - u*dot(velocity, grad(v))*dx \
    - (1.0/dt)*dot(u - u0, v)*dx

a, L = lhs(F), rhs(F)

# Create VTK file for saving solution
vtkfile = File('fokker-planck.pvd')

# Time-stepping

t = 0
u = Function(V)
for n in range(num_steps):
    if n%100 == 0:

        plot(u0)
        plt.show()
        print(n)

    t += dt

   # Compute solution
    solve(a == L, u) 

   # Update previous solution
    u0.assign(u)
    
vtkfile << (u, t)

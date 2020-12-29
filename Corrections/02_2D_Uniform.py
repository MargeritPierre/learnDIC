#%% Import useful libraries
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% GENERATE images *************************************************
# Image size
N1 = 30
N2 = 25
N = N1*N2
# translation
u1 = 1.2
u2 = -2.6
# Gaussian std
sigma = 5
# Image coordinates
x1 = np.arange(0,N1)
x2 = np.arange(0,N2)
x1,x2 = np.meshgrid(x1,x2)
# Images
I = np.exp(-((x1-N1/2)**2+(x2-N2/2)**2)/sigma**2)
i = np.exp(-((x1-u1-N1/2)**2+(x2-u2-N2/2)**2)/sigma**2)
# Display images
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x1,x2,I,color = 'blue',label='Reference')
ax.plot_wireframe(x1,x2,i,color = 'red',label='Current')
plt.legend()
plt.xlabel('X1'),plt.ylabel('X2')
plt.show()

#%% Interpolation function *******************************************
# Linear interpolation of an Image I at real-valued coordinates (x1,x2)
def interp(i,x):
    x1 = np.arange(0,i.shape[1])
    x2 = np.arange(0,i.shape[0])
    f = interpolate.RegularGridInterpolator((x1,x2),i.transpose(),'linear',False,0)
    return f(x)
# Display example
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x1,x2,I,color = 'blue',label='Reference')
x = np.random.rand(100,2)*(np.array([N1,N2])-1)
ax.plot3D(x[:,0],x[:,1],interp(I,x), '+r')
plt.legend()
plt.xlabel('X1'),plt.ylabel('X2')
plt.show()

#%% Gradient function ************************************************
# using finite difference schemes
def d_dx1(I):
    return np.c_[I[:,1]-I[:,0] , 0.5*(I[:,2:]-I[:,0:-2]) , I[:,-1]-I[:,-2]]
def d_dx2(I):
    na = np.newaxis
    return np.r_[I[1,na]-I[0,na] , 0.5*(I[2:,:]-I[0:-2,:]) , I[-1,na]-I[-2,na]]
# Display gradients
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x1,x2,I,color = 'blue',label='I')
ax.plot_wireframe(x1,x2,d_dx1(I),color = 'red',label='dI/dx')
ax.plot_wireframe(x1,x2,d_dx2(I),color = 'green',label='dI/dy')
plt.legend()
plt.xlabel('X1'),plt.ylabel('X2')
plt.show()

#%% Gauss-Newton algorithm
# Error function J = 1/2*sum(r**2)
# Define the procedure
display = True
eps = 1e-3 # convergence criterion
maxIt = 100 # prevent infinite loops
# Initialize
u = np.array([0,0])
X = np.array([x1.flatten(),x2.flatten()]).transpose()
du = 1e10*np.array([[1],[1]])
it = 0
while np.sqrt(np.sum(du**2))>eps and it<maxIt:
    it = it+1
    # Current configuration
    x = X + u
    # Compute the residual
    r = interp(i,x)-interp(I,X)
    # Interpolated image gradients
    di_dx1 = interp(d_dx1(i),x)
    di_dx2 = interp(d_dx2(i),x)
    # Jacobian j = dJ_du
    j = np.array([[sum(di_dx1*r)],[sum(di_dx2*r)]])
    # Hessian H = d²J_du²
    H = np.array([[sum(di_dx1*di_dx1),sum(di_dx1*di_dx2)],[sum(di_dx1*di_dx2),sum(di_dx2*di_dx2)]])
    # Displacement update
    du = np.linalg.solve(H,j)
    u = u - du.transpose()
    # Display
    if display:
        print([-du,u])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(x1,x2,interp(I,X).reshape([N2,N1]),color = 'blue',label='Reference')
        ax.plot_wireframe(x1,x2,interp(i,x).reshape([N2,N1]),color = 'red',label='Current')
        ax.plot_wireframe(x1,x2,r.reshape([N2,N1]),color = 'red',label='Current')
        plt.legend()
        plt.xlabel('X1'),plt.ylabel('X2')
        plt.show()















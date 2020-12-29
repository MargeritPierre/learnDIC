#%% Import useful libraries
import numpy as np
import matplotlib.pyplot as plt

#%% Interpolation function *******************************************
# Linear interpolation of a signal s at real-valued coordinates x
def interp(s,x):
    return np.interp(x,np.arange(0,len(s)),s)
# Display example
s = np.exp(-np.linspace(-1,1,1000)**2/0.3**2)
xi = np.linspace(0,len(s)-1,25)
si = interp(s,xi)
plt.plot(s,label='Original')
plt.plot(xi,interp(s,xi),'+',label='Interpolated')
plt.legend()
plt.show()

#%% GENERATE signal *************************************************
N = 100 # number of points
U = np.array([-0.05,0.03])*N # displacement at ends of the signal
sigma = 0.1*N # gaussian std
i = np.arange(0,N) # sample indices
s = np.exp(-(i-N/2)**2/sigma**2)
idef = i + np.linspace(U[0],U[1],N)
S = interp(s,idef)
# Display signal
plt.plot(S,label='Reference')
plt.plot(s,label='Current')
plt.legend()
plt.show()

#%% Gradient function ************************************************
# using finite difference schemes: 
# centered: ds(i) = (s(i+1) - s(i-1))/2
# left: ds(i) = s(i) - s(i-1)
# right: ds(i) = s(i+1) - s(i)
def d_dx(s):
    return np.hstack((s[1]-s[0] , s[2:]-s[0:-2] , s[-1]-s[-2]))
# Display example
plt.plot(s,label='Signal')
plt.plot(d_dx(s),label='Derivative')
plt.legend()
plt.show()

#%% Gauss-Newton algorithm
# Error function J = 1/2*sum(r**2)
# Define the procedure
def GN(s,S,X,display=True):
    eps = 1e-6 # convergence criterion
    maxIt = 100 # prevent infinite loops
    # Initialize
    ee = np.linspace(0,1,len(X)) # normalized coordinate
    u = np.array([[0],[0]])
    x = X + u[0]*(1-ee) + u[1]*ee
    du = 1e10*np.array([[1],[1]])
    it = 0
    while np.sqrt(np.sum(du**2))>eps and it<maxIt:
        it = it+1
        # Compute the residual
        r = interp(s,x)-interp(S,X)
        # Interpolated signal gradient
        ds_dx = interp(d_dx(s),x)
        # Derivatives with respect to parameters
        ds_du1 = ds_dx*(1-ee)
        ds_du2 = ds_dx*ee
        # Jacobian j = dJ_du
        j = np.array([[sum(ds_du1*r)],[sum(ds_du2*r)]])
        # Hessian H = d²J_du²
        H = np.array([[sum(ds_du1*ds_du1),sum(ds_du1*ds_du2)],[sum(ds_du2*ds_du1),sum(ds_du2*ds_du2)]])
        # Displacement update
        du = np.linalg.solve(H,j)
        u = u - du
        # Current configuration
        x = X + u[0]*(1-ee) + u[1]*ee
        # Display
        if display:
            print([-du,u])
            plt.plot(interp(S,X),label='Reference')
            plt.plot(interp(s,x),label='Current')
            plt.plot(r,label='Residual')
            plt.legend()
            plt.show()
    return u 
# Execute the procedure
X = np.arange(0,N) # domain of interest==reference configuration
u = GN(s,S,X)
print(u)















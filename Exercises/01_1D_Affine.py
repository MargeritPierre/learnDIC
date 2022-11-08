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
plt.plot(s,label='Original')
plt.plot(xi,interp(s,xi),'+',label='Interpolated')
plt.legend()
plt.show()

#%% GENERATE signal *************************************************
N = 100 # number of points
u1 = -0.05*N # displacement at ends of the signal
u2 = 0.03*N # displacement at ends of the signal
sigma = 0.1*N # gaussian std
i = np.arange(0,N) # sample indices
s = np.exp(-(i-N/2)**2/sigma**2)
idef = i + np.linspace(u1,u2,N)
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
    return np.hstack((s[1]-s[0] , (s[2:]-s[0:-2])/2 , s[-1]-s[-2]))
# Display example
plt.plot(s,label='Signal')
plt.plot(d_dx(s),label='Derivative')
plt.legend()
plt.show()

#%% Gauss-Newton algorithm
# Error function J = 1/2*sum(r**2)
# Define the procedure
X = np.arange(0,N) # domain of interest==reference configuration
ee = np.linspace(0,1,len(X)) # normalized coordinate X/L
display = True # display iterations
eps = 1e-6 # convergence criterion
maxIt = 100 # prevent infinite loops
# Initialize
u = np.array([[0],[0]]) # parameter initialization
du = 1e10*np.array([[1],[1]])
it = 0
# Updating loop
while np.sqrt(np.sum(du**2))>eps and it<maxIt:
    it = it+1
    # Current configuration
    x = 
    # Compute the residual
    r = interp(s,x)-interp(S,X)
    # Interpolated signal gradient
    ds_dx = interp(d_dx(s),x)
    # Derivatives with respect to parameters
    ds_du1 = 
    ds_du2 = 
    # Jacobian j = dJ_du
    j = 
    # Hessian H = d²J_du²
    H = 
    # Displacement update
    du = 
    u = u - du
    # Display
    if display:
        print([it,-du,u])
        plt.plot(interp(S,X),label='Reference')
        plt.plot(interp(s,x),label='Current')
        plt.plot(r,label='Residual')
        plt.legend()
        plt.show()
print(u)















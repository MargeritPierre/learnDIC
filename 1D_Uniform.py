#%% Import useful libraries
import numpy as np
import matplotlib.pyplot as plt


#%% Generate signals *************************************************
# Reference S: centered gaussian
# Current s: shifted gaussian 
N = 100 # number of points
sigma = 0.1*N # gaussian std
shift = 0.03*N # current gaussian shift
i = np.arange(0,N) # sample indices
S = np.exp(-(i-N/2)**2/sigma**2)
s = np.exp(-(i-shift-N/2)**2/sigma**2)
# Display signals
plt.plot(S,label='Reference')
plt.plot(s,label='Current')
plt.legend()
plt.show()


#%% Interpolation function *******************************************
# Linear interpolation of a signal s at real-valued coordinates x
def interp(s,x):
    return np.interp(x,i,s)
# Display example
ii = np.arange(0,N,2.7)
plt.plot(i,s,label='Original')
plt.plot(ii,interp(s,ii),'+',label='Interpolated')
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
def GN(s,S,X,display=True):
    eps = 1e-6 # convergence criterion
    maxIt = 100 # prevent infinite loops
    # Initialize
    u = 0
    du = 1e10
    it = 0
    while np.abs(du)>eps and it<maxIt:
        it = it+1
        # Current configuration
        x = X + u
        # Compute the residual
        r = interp(s,x)-interp(S,X)
        # Interpolated signal gradient
        ds_dx = interp(d_dx(s),x)
        # Jacobian j = dJ_du
        j = sum(ds_dx*r)
        # Hessian H = d²J_du²
        H = sum(ds_dx*ds_dx)
        # Displacement update
        du = -j/H
        u = u + du
        # Display
        if display:
            print([-du,u,u-shift])
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

#%% With additive noise
NoiseStd = 0.05
Sn = S + NoiseStd*np.random.randn(N)
sn = s + NoiseStd*np.random.randn(N)
u = GN(sn,Sn,X) # execute the procedure

#%% Displacement variance
NoiseStd = 0.001
# Brute sampling
nSampling = 2000
u = [] ;
for sampling in range(nSampling):
    Sn = S + NoiseStd*np.random.rand(N)
    sn = s + NoiseStd*np.random.rand(N)
    u.append(GN(sn,Sn,X,False)) 
print('mean u: ' + str(np.mean(u)))
print('std u: ' + str(np.std(u)))
# Perrturbation analysis
# Compute the residual
r = interp(s,X+shift)-interp(S,X)
# Interpolated signal gradient
dS_dX = interp(d_dx(S),X)
# Jacobian j = dJ_du
A = dS_dX/sum(dS_dX*dS_dX)
stdU = np.sqrt(np.sum(A**2))*NoiseStd
print('estimated std u: ' + str(stdU))
# Display the pdf
ubins = np.linspace(-1,1,40)*stdU*3 + shift
h = np.histogram(u,ubins)
pdf=h[0]/nSampling/np.diff(ubins)
plt.plot((h[1][0:-1]+h[1][1:])/2,pdf,label='Sampled PDF')
plt.plot(ubins,np.exp(-(ubins-shift)**2/stdU**2)/stdU/np.sqrt(np.pi),label='Estimated PDF')
plt.legend()
plt.show()















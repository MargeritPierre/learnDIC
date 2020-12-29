#%% Import useful libraries
import os
import numpy as np
from scipy import interpolate
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.tri as mptri

#%% PLOT FUNCTIONS
def plotImg(I):
    I = np.array(I)
    I = np.repeat(I[:,:,np.newaxis],3,2) # three colors
    plt.imshow(I)
def plotMesh(x,elems,colorData=np.array([]),edgeColor = 'black'):
    plt.fill(x[elems,0].transpose(),x[elems,1].transpose(),edgecolor=edgeColor,fill=False)
    #plt.scatter(Xg[:,0],Xg[:,1],c = "k")
    if colorData.size==0: return 
    tri = np.vstack((elems[:,[0,1,2]],elems[:,[2,3,0]]))
    tri = mptri.Triangulation(x[:,0], x[:,1], tri)
    plt.tricontourf(tri,colorData.flatten())

#%% LOAD IMAGES *************************************************
# Change this folder to fit your configuration
folder = 'D:/POSTDOC/04_TEACHING/ENPC/DMS_Identification/TP2/Images'
# Load images
files = os.listdir(folder)
IMG = [mpimg.imread(folder + '/' +  name) for name in files]
# Show images
for img in IMG:
    plotImg(img)
    plt.show()
# Image info
Np1 = IMG[0].shape[1]
Np2 = IMG[0].shape[0]
nPixels = Np1*Np2
# Pixel coordinates
Xp1 = np.arange(0,Np1)
Xp2 = np.arange(0,Np2)
Xp1,Xp2 = np.meshgrid(Xp1,Xp2)

#%% CREATE THE GRID MESH *************************************************
# mesh: a list of Nodes Xg and a list of quad elements 
# Grid parameters
xmin = 100 ; xmax = Np1-100 ; dx = 50
ymin = 220 ; ymax = 580 ; dy = dx
# Node coordinates
Xg1 = np.arange(xmin,xmax+1,dx)
Xg2 = np.arange(ymin,ymax+1,dy)
Xg1,Xg2 = np.meshgrid(Xg1,Xg2)
Ng1 = Xg1.shape[1]
Ng2 = Xg2.shape[0]
# flatten
Xg = np.array([Xg1.flatten(),Xg2.flatten()]).transpose()
nNodes = Xg.shape[0]
# Quads
p0 = np.arange(0,Ng1)[np.newaxis,:] + Ng1*np.arange(0,Ng2)[:,np.newaxis]
p0 = p0[0:-1,0:-1].flatten()
quads = p0[:,np.newaxis] + np.array([0,1,Ng1+1,Ng1])
nElems = quads.shape[0]
# plot
plotImg(IMG[0])
plotMesh(Xg,quads)
plt.show()

#%% GET PIXELS IN EACH QUAD *******************************************
# create a SPARSE matrix IN of shape (nPixels,nElems)
pp = [] # linear pixel index
ee = [] # element index
# Get linear pixel indices included in each quad
for q in range(nElems):
    xq = Xg[quads[q,:],:] # quad point coordinates
    box = np.array([np.min(xq,0),np.max(xq,0)]) # quad bounding box
    p1 = np.arange(np.ceil(box[0,0]),box[1,0]) # horizontal indices
    p2 = np.arange(np.ceil(box[0,1]),box[1,1]) # vertical indices
    pl = (p1[np.newaxis,:] + Np1*p2[:,np.newaxis]).flatten() # linear indices
    pp.append(pl)
    ee.append(np.tile(q,pl.shape))
# concatenate all indices
pp = np.hstack(pp) ;
ee = np.hstack(ee) ;
vv = np.tile(1.0,pp.shape)
IN = sparse.csr_matrix((vv,(pp,ee)),(nPixels,nElems))
# Region Of Interest
ROI = np.sum(IN,1) # shape (nPixels,1)
inROI = np.where(ROI)[0]
# restrict to the ROI
IN = IN[inROI,:]
# show the ROI
plotImg(ROI.reshape((Np2,Np1)))

#%% BUILD THE MATRIX OF SHAPE FUNCTIONS *******************************************
# create a SPARSE matrix N of shape (nPixels,nNodes) containing the FEM shape functions
# bilinear quad shape function, for local coordinates xi in [0,1]²
# first determine the local coordinates of each pixel in its corresponding element
p0 = Xg[quads[:,0],:] # upper-left corner point
p0 = IN.dot(p0) # project for each pixel of the ROI
(p2,p1) = divmod(inROI,Np1) ; # pixel coordinates
pp = np.array([p1,p2]).transpose() # pixel coordinates 
xi = (pp-p0)/[dx,dy] # LOCAL coordinates xi in [0,1]²
# create the shape functions
N0 =  # upper-left corner
N1 =  # upper-right corner
N2 =  # lower-right corner
N3 =  # lower-left corner
# Build the sparse matrix
ee = IN.dot(np.arange(0,nElems)).astype(int) # associated element
pp = np.tile(inROI,(1,4)).flatten() # pixel indices
nn = quads[ee,:].transpose().flatten() # node indices
vv = np.hstack((N0,N1,N2,N3)) # shape function values
N = sparse.csr_matrix((vv,(pp,nn)),(nPixels,nNodes))
# Display a function
fn = 0.5+0.5*np.sin(2*np.pi*Xg[:,0]/dx*0.1)*np.sin(2*np.pi*Xg[:,1]/dy*0.2) # function value on nodes
fp = N.dot(fn) # function values at pixels
plotImg(fp.reshape((Np2,Np1)))

#%% Interpolation function *******************************************
# Linear interpolation of an Image i at real-valued coordinates x = (x1,x2)
def interp(i,x):
    X1 = np.arange(0,i.shape[1])
    X2 = np.arange(0,i.shape[0])
    f = interpolate.RegularGridInterpolator((X1,X2),i.transpose(),'linear',False,0)
    return f(x)

#%% Gradient function ************************************************
# using finite difference schemes
def d_dx1(I):
    return np.c_[I[:,1]-I[:,0] , 0.5*(I[:,2:]-I[:,0:-2]) , I[:,-1]-I[:,-2]]
def d_dx2(I):
    na = np.newaxis
    return np.r_[I[1,na]-I[0,na] , 0.5*(I[2:,:]-I[0:-2,:]) , I[-1,na]-I[-2,na]]
# Display the norm of the gradient of an image
I = IMG[0]
g = np.sqrt(d_dx1(I)**2 + d_dx2(I)**2)
plotImg(g)

#%% Gauss-Newton algorithm ************************************************
# Error function J = 1/2*sum(r**2)
# Define the procedure
display = True
eps = 1e-4 # convergence criterion
maxIt = 100 # prevent infinite loops
I = IMG[0] # reference image
i = IMG[1] # current image
# Initialize
Nr = N[inROI,:] # shape functions restricted to the ROI
u = np.zeros((nNodes,2)) # nodal displacement to identify
X = Nr.dot(Xg) # current configuration
for i in IMG[1:]:
    du = 1e10*np.ones((nNodes,2))
    it = 0
    while np.sqrt(np.sum(du**2))>eps and it<maxIt:
        it = it+1
        # Current configuration
        x = 
        # Compute the residual
        r = 
        # Interpolated image gradients
        di_dx1 = 
        di_dx2 = 
        # residual derivatives
        dr_du1 = 
        dr_du2 = 
        dr_du = 
        # Jacobian j = dJ_du
        j =  
        # Hessian H = d²J_du²
        H = 
        # Displacement update
        du = 
        du = du.reshape((2,nNodes)).transpose()
        u = u - du
        # Display
        if display:
            print(np.sqrt(np.sum(du**2)))
            plotImg(i)
            plotMesh(Xg + u,quads,np.sqrt(np.sum(du**2,1)))
            plt.show()
        
#%% PLOT DATA ON THE MESH ************************************************
dataToPlot = u[:,1]
plotImg(i)
plotMesh(Xg + u,quads,dataToPlot,edgeColor='none')
plt.colorbar()

#%% FIRST GRADIENT & STRAINS ************************************************
# build SPARSE matrices G so that df/dxj = G[j]*fn
# df/dxj = d(N*fn)/dxj = (dN/dxj)*fn so G[j] = dN/dxj
# first we evaluate the derivatives at gauss points
xi = np.array([[1/4,1/4],[3/4,1/4],[3/4,3/4],[1/4,3/4]]) # gauss points local coordinates
#xi = np.array([[1/2,1/2]]) # gauss points local coordinates
nGP = xi.shape[0]
# shape functions at gauss points
N0gp =  # upper-left corner
N1gp =  # upper-right corner
N2gp =  # lower-right corner
N3gp =  # lower-left corner
Ngp = np.vstack((N0gp,N1gp,N2gp,N3gp)).transpose()
# derivative w.r.t xi1 at gauss points
dN0_dxi1 =  # upper-left corner
dN1_dxi1 =  # upper-right corner
dN2_dxi1 =  # lower-right corner
dN3_dxi1 =  # lower-left corner
# derivative w.r.t xi2 at gauss points
dN0_dxi2 = # upper-left corner
dN1_dxi2 = # upper-right corner
dN2_dxi2 = # lower-right corner
dN3_dxi2 = # lower-left corner
# derivatives w.r.t x1 & x2 , shape: (nGP,4)
dN_dx1 = np.vstack((dN0_dxi1,dN1_dxi1,dN2_dxi1,dN3_dxi1)).transpose()/dx
dN_dx2 = np.vstack((dN0_dxi2,dN1_dxi2,dN2_dxi2,dN3_dxi2)).transpose()/dy
# build the sparse matrices indices, shape: (nGP*nElems,4)
vv = np.tile(Ngp,(nElems,1))
vv1 = np.tile(dN_dx1,(nElems,1))
vv2 = np.tile(dN_dx2,(nElems,1))
pp = np.tile(np.arange(nGP*nElems)[:,np.newaxis],(1,4))
nn = np.repeat(quads,nGP,0)
# sparse matrices
matShape = (nGP*nElems,nNodes) ;
pp = pp.flatten()
nn = nn.flatten()
Ngp = sparse.csr_matrix((vv.flatten(),(pp,nn)),matShape)
dN_dx1 = sparse.csr_matrix((vv1.flatten(),(pp,nn)),matShape)
dN_dx2 = sparse.csr_matrix((vv2.flatten(),(pp,nn)),matShape)
# gradient matrices
iNgp = spsolve(Ngp.transpose()*Ngp,Ngp.transpose())
G = [iNgp*dN_dx1,iNgp*dN_dx2]
# Check with known functions
G[0].dot(Xg[:,0]) # dx1_dx1 = 1
G[0].dot(Xg[:,1]) # dx2_dx1 = 0
G[1].dot(Xg[:,0]) # dx1_dx2 = 0
G[1].dot(Xg[:,1]) # dx2_dx2 = 1
        
#%% STRAINS ************************************************
# EPSILON = [[E11],[E22],[2*E12]] = B*[[u1],[u2]]
O = sparse.csr_matrix((nNodes,nNodes))
B11 = 
B22 = 
B12 = 
B = sparse.vstack((B11,B22,B12))
# plot the strains
dataToPlot = B11.dot(u.transpose().flatten()[:,np.newaxis])
plotImg(i)
plotMesh(Xg + u,quads,dataToPlot,edgeColor='none')
plt.colorbar()
















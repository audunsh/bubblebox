import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import numpy as np
import numba as nb
import evince as ev
import time
from scipy.interpolate import interp1d

# Author: Audun Skau Hansen, January, 2021

class lines():
    def __init__(self, Nx, periodic = False):
        self.Nx = Nx
        self.periodic = periodic
        self.line = self.getline(Nx)
        
    
    def getcurve(self, Nx):
        return interp1d(np.linspace(0,1,self.Nx),self.line, 2)(np.linspace(0,1,Nx))

    def getinstance(self):
        Z = self.line
        t = np.linspace(0,1,self.Nx)
        for i in range(10,15):
            nn = i+3
            #zn = 
            Z += interp1d(np.linspace(0,1,nn), np.random.uniform(-1,1,nn), 2)(t)
        return Z
        
    def getline(self,Nx):
        Z = np.zeros(Nx, dtype = float)
        t = np.linspace(0,1,Nx)
        for i in range(10):
            nn = i+3
            Zn = np.random.uniform(-1,1,nn)
            if self.periodic:
                Zn[-1] = Zn[0]
            Z += interp1d(np.linspace(0,1,nn), Zn, 2)(t)
        return Z
    
class colorscheme():
    def __init__(self):
        self.CC = np.array([[.1,  .3, .98],
               [.1,  .1, .98],
               [.3,  .1, 1]])
        self.CC = np.array([[.6,  1, .1],
               [.1,  .9, .1],
               [.1,  .5, .1]])

        colorx = np.zeros((3,5))

        colorx[:,0] = np.array([0.44,0.7,.76])
        colorx[:,1] = np.array([0.1,0.1,.2])
        colorx[:,2] = np.array([0.97,0.9,.63])
        colorx[:,3] = np.array([0.37,0.02,.13])
        colorx[:,4] = np.array([0.41,0.87,.62])

        colorx = colorx**1.1
        #self.CC = colorx

        #self.CC = np.random.uniform(0,1,(3,3))
        self.c = interp1d(np.linspace(0,1,3), self.CC) #(np.linspace(0,1,800)).T
    def getcol(self,i):
        return self.c(i)
   
def gen_field(b, bins = 10, Z = None, alpha = .9):
    #generate aa field of velocities
    

    if Z is None:
        Z = np.zeros((bins,bins, 2))
    
    bi = np.array(.5*bins*b.pos/b.Lx + bins/2, dtype = int)
    Z[bi[0], bi[1]] = alpha*Z[bi[0], bi[1]] + (1-alpha)*b.vel_.T
    Zv = Z[bi[0], bi[1]]
   
    dv = Z[bi[0], bi[1]]
    
    return Z, dv

#matplotlib.rcParams['figure.figsize'] = (6.4,4.8)
#matplotlib.rcParams['figure.figsize'] = (4.8,4.8)

#@nb.jit()
def pos2heatmap(pos, masses, Lx, Ly, Nz = 11, col = None):
    """
    Create a 2D density array for a set of various bubbles
    Author: Audun Skau Hansen (a.s.hansen@kjemi.uio.no)

    Keyword arguments:
    
    pos              -- array (2,n_particles) containing positions
    masses           -- array (n_particles) containing masses
    Lx, Ly           -- box dimensions
    Nz               -- number of bins in each dimension

    Returns an Nz by Nz by 3 array containing colors according to 
    bubble density
    """
    #Z = np.zeros((Nz,Nz, 4), dtype = nb.float64)
    Z = np.zeros((Nz,Nz, 4), dtype = float)
    col = colorscheme()
    
    masses_norm = masses/masses.max()
    for i in range(pos.shape[1]):
        ix, iy = int( Nz*(pos[0,i] + Lx)/(2*Lx) ), int( Nz*(pos[1,i] + Ly)/(2*Ly) )
        # Using the mean for N+1 samples here:
        # mean_{N+1} = (N*mean_N + new_sample)/(N + 1)
        Z[ix,iy, :3] = (Z[ix,iy, :3]*Z[ix,iy, 3] + col.getcol(masses_norm[i]) ) /(Z[ix,iy, 3]+1)
        Z[ix,iy, 3] += 1

    return Z[:,:,:3]


@nb.jit
def distances_reduced(coords, L2x, L2y):
    Lx = np.abs(L2x/2.0)
    Ly = np.abs(L2y/2.0)
    d = np.zeros((coords.shape[1],coords.shape[1]-1), dtype = float)
    #c = 0
    for i in range(coords.shape[1]):
        ix = coords[0,i]
        iy = coords[1,i]

        for j in range(i+1, coords.shape[1]):
            dx = ix - coords[0,j]
            dy = iy - coords[1,j]



            if np.abs(dx)>Lx:
                if dx>0:
                    dx += L2x
                else:
                    dx -= L2x
                #dx += np.sign(dx)*L2

            if np.abs(dy)>Ly:
                if dy>0:
                    dy += L2y
                else:
                    dy -= L2y



            d_ = np.sqrt(dx**2 + dy**2)
            d[i,j-1] = d_#+ (coords[i,2] - coords[j,2])**2 
            d[j,i] = d_
            #c += 1
    return d

def fcc(L = 1, Nc = 3, vel = 0):
    """
    Generate a FCC setup

    Keyword arguments:
    
    L              -- lattice constant
    Nc             -- half the number of cells in the simulation box

    Returns an fcc simulation box containing 4*Nc**3 particles, 
    with a total volume of (L*Nc)**3.
    """
    Lc = L*Nc
    
    coords = []
    for i in range(-Nc, Nc):
        for j in range(-Nc, Nc):
            for k in range(-Nc, Nc):
                coords.append([i,j,k])

                coords.append([i+.5,j+.5,k])
                coords.append([i+.5,j,k+.5])
                coords.append([i,j+.5,k+.5])

    coords = np.array(coords)

    coords = Lc*(coords+.5)/Nc
    coords -= np.mean(coords, axis = 0)[None,:]

    
    
    return coords




@nb.jit()
def repel(coords, screen = 10, L2 = 1.0):
    # 2d force vector
    d = np.zeros((2, coords.shape[1]), dtype = float64)
    
    
    for i in range(coords.shape[1]):
        for j in range(i+1, coords.shape[1]):
            dx = coords[0,i] - coords[0,j]
            
            
            
            if np.abs(dx)<=screen:
                dy = coords[1,i] - coords[1,j]
                
                # PBC
                
                if np.abs(dx)>L2:
                    if dx>0:
                        dx -= 2*L2
                    else:
                        dx += 2*L2
                    #dx += np.sign(dx)*L2
                if np.abs(dy)>L2:
                    if dy>0:
                        dy -= 2*L2
                    else:
                        dy += 2*L2
                        
                        
                
                d_ = (dx**2 + dy**2)**-1 #**-.5
                d[0,i] -= dx*d_
                d[1,i] -= dy*d_
                
                d[0,j] += dx*d_
                d[1,j] += dy*d_
                
                
    return d

@nb.jit(nopython=True)
def distance_matrix(coords):
    d = np.zeros((coords.shape[0], coords.shape[0]), dtype = nb.float64)
    for i in range(coords.shape[0]):
        for j in range(i+1, coords.shape[0]):
            d_ = ((coords[i,0] - coords[j,0])**2 + (coords[i,1] - coords[j,1])**2)**.5  #+ (coords[i,2] - coords[j,2])**2 
            d[i,j] = d_
            d[j,i] = d_
    return d

@nb.jit(nopython=True)
def distances(coords):
    d = np.zeros(int((coords.shape[1]-1)*(coords.shape[1])/2), dtype = nb.float64)
    c = 0
    for i in range(coords.shape[1]):
        ix = coords[0,i]
        iy = coords[1,i]

        for j in range(i+1, coords.shape[1]):
            d[c] = (ix - coords[0,j])**2 + (iy - coords[1,j])**2  #+ (coords[i,2] - coords[j,2])**2 
            c += 1
    return d


@nb.jit(nopython=True)
def no_forces(coords, size2, interactions = None, r2_cut = 9.0, force = None, pair_list = None):
    return 0


# force mapping and calculations

@nb.jit(nopython=True)
def compute_no_force(eps, sig, rr):
    return 0

def no_force():
    return 0

@nb.jit(nopython=True)
def compute_lj_force(eps, sig, rr):
    return -24*eps*(2*sig**12*rr**-7 - sig**6*rr**-4)

def lj_force():
    return 1

@nb.jit(nopython=True)
def compute_coulomb_force(eps, sig, rr):
    """
    eps = Q1 (charge of particle 1)
    sig = Q2 (charge of particle 2)
    rr  = distance between particles
    """
    return -eps*sig*rr**-2

def coulomb_force():
    return 3

@nb.jit(nopython=True)
def compute_hook_force(eps, sig, rr):
    """
    eps = Q1 (charge of particle 1)
    sig = Q2 (charge of particle 2)
    rr  = distance between particles
    """
    return eps*(rr-sig)

def hook_force():
    return 2

@nb.jit(nopython=True)
def compute_custom_force(eps, sig, rr):
    return 0

def custom_force():
    return 4



@nb.jit(nopython=True)
def compute_force(force_id, eps, sig, rr):
    if force_id==0:
        return 0 #no force
    if force_id==1:
        return compute_lj_force(eps, sig, rr)
    if force_id==2:
        return compute_hook_force(eps, sig, rr)
    if force_id==3:
        return compute_coulomb_force(eps, sig, rr)
    if force_id==4:
        return compute_custom_force(eps, sig, rr)
    





@nb.jit(nopython=True)
def compute_pair_list(coords, r2_cut, size2):
    pair_list = np.zeros(( coords.shape[1], coords.shape[1]), dtype = nb.int32)
    #Lx, Ly = -L2x/2.0, -L2y/2.0
    size_a = np.abs(.5*size2)
    #size = .5*size2
    
    for i in range(coords.shape[1]):
        pair_list[i] = i
        c = 0
        ci = coords[:,i]
        for j in range(i+1, coords.shape[1]):
            
            cj= coords[:,j]
            #dx = cjx - cix #coords[0,i]
            #dy = cjy - ciy #coords[1,i]
            
            dij = cj - ci
            
            # PBC
            for k in range(size2.shape[0]):
                if size2[k]<0:
                    if np.abs(dij[k])>size_a[k]:
                        if dij[k]>0:
                            dij[k] += size2[k]
                        else:                                
                            dij[k] -= size2[k]
            
            
            if np.sum(dij**2) < r2_cut:
                pair_list[i, c] = j
                c += 1
        
    return pair_list

@nb.jit(nopython=True)
def forces(coords, size2, interactions, r2_cut = 9.0, pair_list = None):
    # 2d force vector
    d = np.zeros((size2.shape[0], coords.shape[1]), dtype = nb.float64)
    #Lx, Ly = -L2x/2.0, -L2y/2.0
    size = .5*size2
    size_a = np.abs(size)
    #u = 0 #pot energy
    
    #if interactions is None:
    #    interactions = np.ones((coords.shape[1], coords.shape[1], 3), dtype = nb.float64)
        

    for i in range(coords.shape[1]):
        #cix, ciy = coords[0,i],coords[1,i]
        ci = coords[:, i]

        #if pair_list is not None:
        for j in pair_list[i]:
            if j==i:
                break





            # distance-based interactions

            cj = coords[:, j]
            dij = cj - ci


            for k in range(size.shape[0]):
                if size[k]<0:
                    if np.abs(dij[k])>size_a[k]:
                        if dij[k]>0:
                            dij[k] += size2[k]
                        else:
                            dij[k] -= size2[k]


            # Compute distance squared
            rr = np.sum(dij**2)


            if rr<r2_cut: # Screen on distance squared

                force_i, eps, sig = interactions[i,j]
                #if eps>0:

                ljw = compute_force(force_i, eps, sig, rr) 

                #ljw   = -12*eps*(sig**12*rr**-7 - sig**6*rr**-4) # Lennard-Jones weight

                ljf = dij*ljw

                #ljf_x = ljw*dx # x-component
                #ljf_y = ljw*dy # y-component

                # Sum forces
                d[:,i] += ljf
                d[:,j] -= ljf
        
    return d

@nb.jit(nopython=True)
def lj_potential(coords, size2, interactions = None, r2_cut = 9.0, force = lj_force, pair_list = None):
    # Compute the Lennard-Jones potential energy
    size = .5*size2
    size_a = np.abs(size)
    #u = 0 #pot energy
    
    if interactions is None:
        interactions = np.ones((coords.shape[1], coords.shape[1], 2), dtype = nb.float64)
        
    u = 0

    for i in range(coords.shape[1]):
        #cix, ciy = coords[0,i],coords[1,i]
        ci = coords[:, i]

        if pair_list is not None:
            for j in pair_list[i]:
                if j==i:
                    break
                
                
                

                
                # distance-based interactions
                #dx, dy = coords[:,j] - ci

                #cjx, cjy = coords[0,j], coords[1,j]
                cj = coords[:, j]
                dij = cj - ci
                
                #dx = cjx - cix #coords[0,i]
                #dy = cjy - ciy #coords[1,i]
                for k in range(size.shape[0]):
                    if size[k]<0:
                        if np.abs(dij[k])>size_a[k]:
                            if dij[k]>0:
                                dij[k] += size2[k]
                            else:
                                dij[k] -= size2[k]
                
                
                # Compute distance squared
                rr = np.sum(dij**2)

                
                if rr<r2_cut: # Screen on distance squared

                    eps, sig = interactions[i,j]
                    u +=  4*eps*((sig/rr)**12 - (sig/rr)**6)
                    
        
    return u






@nb.jit()
def collisions(coords, vels, screen = 10.0, radius = -1.0, size2=0, masses = None,pair_list = None, sphere_collisions = False):
    """
    Hard-sphere collision and wall collisions
    Author: Audun Skau Hansen (a.s.hansen@kjemi.uio.no)

    Keyword arguments:
    
    coords           -- array (2,n_particles) containing positions
    vels             -- array (2,n_particles) containing velocities
    screen           -- screening distance (particles further away not considered)
    radius           -- for positive values, particles are considered hard spheres of equal radius
    Lx, Ly           -- box dimensions, for x:
                        ( x > 0 , _ ) = closed boundary at x and -x
                        ( x = 0 , _ ) = no boundary in x-direction
                        ( x < 0 , _ ) = periodic boundary at x and -x
    masses           -- array containing particle masses (default None)

    Returns
    velocities, coordinates

    Note
    ----
    This functions utilize numba's just-in-time compilation ( nb.jit() ) 
    for optimized performance and readability.
    """
    #v = vels*1
    r2 = 4*radius**2
    R2 = 2*radius
    c = 0
    
    size = .5*size2
    size_a = np.abs(size)
    v = vels*1
    
    
    for i in range(coords.shape[1]):
        #cix, ciy = coords[0,i],coords[1,i]
        ci = coords[:, i]

        if sphere_collisions:
            if pair_list is not None:
                for j in pair_list[i]:
                    if j==i:
                        break
                    
                    
                    

                    
                    # distance-based interactions
                    #dx, dy = coords[:,j] - ci

                    #cjx, cjy = coords[0,j], coords[1,j]
                    cj = coords[:, j]
                    dij = cj - ci
                    
                    #dx = cjx - cix #coords[0,i]
                    #dy = cjy - ciy #coords[1,i]
                    for k in range(size.shape[0]):
                        if size[k]<0:
                            if np.abs(dij[k])>size_a[k]:
                                if dij[k]>0:
                                    dij[k] += size2[k]
                                else:
                                    dij[k] -= size2[k]
                    
                    
                    # Compute distance squared
                    rr = np.sum(dij**2)
                    
                    if rr<r2:
                        #collision detected
                        c += 1
                        
                        # velocities
                        vij = vels[:, i] - vels[:, j]
                        
                        # weight
                        w_dij = np.dot(vij, dij)/rr*dij
                        
                        if masses is None:
                            v[:, i] -= w_dij
                            v[:, j] += w_dij
                        else:
                            mi = masses[i]
                            mj = masses[j]
                            m1_m2 = mi+mj
                            wi = 2*mj/m1_m2
                            wj = 2*mi/m1_m2

                            v[:,i] -=  w_dij*wi
                            v[:,j] +=  w_dij*wi
                                
        # collision with wall
        for k in range(size.shape[0]):
            if size[k]>0:
                if np.abs(coords[k,i])>size[k]:
                    v[k,i] *= -1
                    c += 1
                            
    return v, c
                    
                    
                    
                    
    





def arrange_in_grid(pos, Lx, Ly, n_bubbles):
    """
    Place n_bubbles in a grid on a Lx by Ly area

    Keyword arguments:
    pos        -- position vector
    Lx, Ly     -- dimensions of box
    n_bubbles  -- number of bubbles
    """
    
    a = Ly/Lx
    nx = int(np.ceil(np.sqrt(n_bubbles/a)))
    ny = int(np.ceil(a*nx))
    

    
    count = 0
    dn_x = np.linspace(0, 2*Lx, nx+1)[1]*.5
    dn_y = np.linspace(0, 2*Ly, ny+1)[1]*.5
    for i in np.linspace(-Lx, Lx, nx+1)[:-1]:
        for j in np.linspace(-Ly, Ly, ny+1)[:-1]:
            pos[:, count] = [i+dn_x,j+dn_y]
            count += 1
            if count>=pos.shape[1]:
                break
        if count>=pos.shape[1]:
                break

    return pos





class mdbox():
    """
    Simple 2D Lennard-Jones liquid simulation

    Keyword arguments:
    
    box              -- boundary conditions in (x,y) direction:
                        ( x > 0 , _ ) = closed boundary at x and -x
                        ( x = 0 , _ ) = no boundary in x-direction
                        ( x < 0 , _ ) = periodic boundary at x and -x
    n_bubbles        -- number of bubbles (int, default 100)
    masses           -- array containing particle masses (default 1)
    pos              -- array of shape (3, n_bubbles) containing positions
    vel              -- array of scalars with maximum random velocity
    radius           -- 
    relax            -- minimize potential energy on initialization using simulated annealing
    grid             -- initial spacing in a grid (default True)

    Example usage (in a notebook):

    import hylleraas.bubblebox as bb

    %matplotlib notebook

    system = mdbox(n_bubbles = 100, size = (10,10)) #initialize 10 by 10 closed box containing 100 bubbles

    system.run() #run simulation interactively 
    

    """

    def __init__(self, n_bubbles = 100, masses = None, vel = 0.0, size = (0,0), grid = True, pair_list = True, fields = False, sphere_collisions = False):
        # Initialize system
        self.sphere_collisions = sphere_collisions
        

        # Boundary conditions
        
        self.size = np.array(size)
        self.size2 = np.array(size)*2
        self.ndim = len(size)

        # obsolete parameter for hard-sphere collisions
        self.radius = .1
        self.n_bubbles = n_bubbles
        
        # list to hold force algorithms 
        self.force_stack = (no_force, lj_force)
        
        # array to keep track of forces and interaction parameters
        self.interactions = np.ones((self.n_bubbles, self.n_bubbles, 3), dtype = float)
        
        # array to keep track of the positions of the bubbles
        self.pos = np.zeros((self.ndim,self.n_bubbles), dtype = float)
        
        # arrange bubbles in grid
        n_bubbles_axis = int(np.ceil(n_bubbles**(1/self.ndim)))
        grid_axes = []
        for i in range(self.size.shape[0]):
            grid_axes.append(np.linspace(-self.size[i], self.size[i], n_bubbles_axis+1)[:-1])
            
        self.pos = np.array(np.meshgrid(*grid_axes)).reshape(self.ndim, int(n_bubbles_axis**self.ndim))[:,:self.n_bubbles]
        
        # move to center
        self.pos = self.pos - np.mean(self.pos, axis = 1)[:, None]

        
        

        if masses is None:
            self.masses = np.ones(self.n_bubbles, dtype = int)
            
            self.masses_inv = np.array(self.masses, dtype = float)**-1
            #self.n_bubbles = n_bubbles
            
        else:
            self.masses = masses
            self.n_bubbles = len(masses)
            self.set_interactions(self.masses)
            self.masses_inv = np.array(self.masses, dtype = float)**-1

        
        
        

        

        self.pos_old = self.pos*1 # retain previous timesteps to compute velocities
        
        # all bubbles active by default
        self.active = np.ones(self.pos.shape[1], dtype = bool)

        
        # Set velocities (and half-step velocity for v-verlet iterations)
        self.vel = np.random.multivariate_normal(np.zeros(self.ndim), vel*np.eye(self.ndim), self.n_bubbles).T
        self.vel[:] -= np.mean(self.vel, axis = 1)[:, None]
        self.vel_ = self.vel # verlet integrator velocity at previous timestep

        # Integrator 
        self.advance = self.advance_vverlet

        # Algorithm for force calculation
        self.forces = forces
        self.force = lj_force
        self.r2_cut = 9.0 #distance cutoff squared for force-calculation
        
        # Time and timestep
        self.t = 0
        self.dt = 0.001

        # Collision counter
        self.col = 0

        # Prime for special first iteration
        self.first_iteration = True
        self.iteration = 0

        # Pair list for efficient force-calculation
        self.pair_list = None
        if pair_list:
            self.pair_list_update_frequency = 10
            self.pair_list_buffer = 1.2
            self.pair_list = compute_pair_list(self.pos, self.r2_cut*self.pair_list_buffer, self.size2)

        # fields
        self.fields = fields
        if self.fields:
            self.nbins, self.Z = 20, np.zeros((20,20,2), dtype = float)
 


    def resize_box(self, size):
        # New boundary conditions
        #self.Lx = size[0]
        #self.Ly = size[1]

        #self.L2x = 2*size[0]
        #self.L2y = 2*size[1]
        
        self.size = np.array(size)
        self.size2 = np.array(size)*2


    

    
    

    def advance_vverlet(self):
        """
        Advance one step in time according to the Velocity-Verlet algorithm
        """
        if self.first_iteration:
            self.Fn = self.forces(self.pos, self.size2, self.interactions, r2_cut = self.r2_cut, pair_list = self.pair_list)
            self.first_iteration = False
        if self.pair_list is not None:
            if self.iteration % self.pair_list_update_frequency == 0:
                self.pair_list = compute_pair_list(self.pos, self.r2_cut*self.pair_list_buffer, self.size2)
        

        Fn = self.Fn

        # field
        if self.fields:
            self.Z, dv = gen_field(self, self.nbins, self.Z)
            #self.vel_ = .99*self.vel_ + .01*dv.T
            Fn += .1*dv.T
        
        self.d_pos = self.vel_*self.dt + .5*Fn*self.dt**2*self.masses_inv

        pos_new = self.pos + self.d_pos
        
        forces_new = self.forces(pos_new, self.size2, self.interactions, r2_cut = self.r2_cut, pair_list = self.pair_list)

        self.vel_ = self.vel_ + .5*(forces_new + Fn)*self.dt*self.masses_inv

        self.Fn = forces_new
        
        
        # impose PBC
        for i in range(self.ndim):
            if self.size[i]<0:
                pos_new[i, :]  = (pos_new[i,:] + self.size[i]) % (self.size2[i]) - self.size[i]
        

        # impose wall and collision boundary conditions
        self.vel_, self.col = collisions(pos_new, self.vel_, screen = 10.0, radius = self.radius, size2 = self.size2, masses = self.masses, pair_list = self.pair_list, sphere_collisions = self.sphere_collisions)

        #update arrays (in order to retain velocity)
        self.vel = (pos_new - self.pos_old)/(2*self.dt)
        self.pos_old[:] = self.pos
        self.pos[:, self.active] = pos_new[:, self.active]

        # Track time
        self.t += self.dt
        self.iteration += 1
        
        
    def advance_euler(self):
        """
        Advance one step in time according to the explicit Euler algorithm
        """
        self.vel += self.forces(self.pos, self.size2, self.interactions, self.r2_cut, self.force)*self.dt*self.masses_inv
        self.pos[self.active] += self.vel[self.active]*self.dt
        
        # impose PBC
        for i in range(self.ndim):
            if self.size[i]<0:
                pos_new[i, :]  = (pos_new[i,:] + self.size[i]) % (self.size2[i]) - self.size[i]

        # impose wall and collision bounary conditions
        self.vel_, self.col = collisions(self.pos, self.vel_, screen = 10.0, radius = self.radius, size2 = self.size2, masses = self.masses, pair_list = self.pair_list)
        
            
        # Track time
        self.t += self.dt
        
    def compute_energy(self):
        """
        Compute total energy of system
        """
        return self.compute_potential_energy() + self.compute_kinetic_energy()


    def compute_potential_energy(self):
        """
        Compute total potential energy in system
        """
        #return lj_potential(self.pos, self.interactions, L2x = self.L2x, L2y = self.L2y, r2_cut = self.r2_cut)
        return lj_potential(self.pos, self.size2, interactions = self.interactions, r2_cut = self.r2_cut, force = self.force, pair_list = self.pair_list)

    def compute_kinetic_energy(self):
        """
        Compute total kinetic energy in system
        """
        # Vektorisert funksjon med hensyn på ytelse
        return .5*np.sum(self.masses*np.sum(self.vel_**2, axis = 0))
    
    def kinetic_energies(self):
        """
        Compute kinetic energy of each bubble
        """
        # Vektorisert funksjon med hensyn på ytelse
        return .5*self.masses*np.sum(self.vel_**2, axis = 0)
        
    
    
    
    def evolve(self, t = 1.0):
        """
        Let system evolve in time for t seconds
        """
        t1 = self.t+t
        while self.t<t1:
            self.advance()
            
    """
    Setters
    """


    def set_charges(self, charges):
        """
        Set charges for Coulomb interactions
        
        Arguments
        ===
        - charges: a numpy.ndarray (or list) containing float or integer charges for all particles
        
        """
        assert(len(charges) == self.n_bubbles), "Number of charges must equal number of bubbles (%i)." %self.n_bubbles
        for i in range(self.n_bubbles):
            for j in range(self.n_bubbles):
                self.interactions[i,j] = np.array([charges[i],charges[j]])
           
    def set_masses(self, masses, bubbles = None):
        """
        Set masses 
        
        Arguments
        ===
        - masses: a numpy.ndarray (or list) containing float or integer charges for all particles
        
        """
        if bubbles is None:
            self.masses[:] = masses
        else:
            self.masses[bubbles] = masses
        
        self.masses_inv = np.array(self.masses, dtype = float)**-1
            
        
    def set_forces(self, force, force_params, bubbles_a = None, bubbles_b = None):
        """
        Set the force acting between bubbles_a and bubbles_b (or all)
        """
        
        #self.force_stack.append(force)
        
        if bubbles_a is None:
            bubbles_a, bubbles_b = np.arange(self.n_bubbles),np.arange(self.n_bubbles)

        nx, ny = np.meshgrid(bubbles_a,bubbles_b)
        
        # set pointer to the force function in the force stack
        self.interactions[nx,ny,:] = force()
        
        # set parameters to use for the specific pairs of particles
        self.interactions[nx,ny,1] = force_params[0]
        self.interactions[nx,ny,2] = force_params[1]
    def set_vel(self, vel):
        """
        set manually the velocities of the system
        """
        assert(np.all(vel.shape==self.vel.shape)), "Wrong shape of velocities"
        self.vel = vel
        self.vel_ = self.vel # verlet integrator velocity at previous timestep

        
        
        
        


    def set_interactions(self, masses):
        """
        Placeholder for proper parametrization of LJ-interactions
        """
        #self.interactions = np.ones((self.n_bubbles, self.n_bubbles, 3), dtype = float)

        
        epsilons = np.linspace(.5,10,100)
        sigmas   = np.linspace(1,np.sqrt(2), 100)

        

        for i in range(self.n_bubbles):
            mi = int(masses[i])
            for j in range(i+1, self.n_bubbles):
                
                mj = int(masses[j])

                eps = np.sqrt(epsilons[mi]*epsilons[mj])
                sig = sigmas[mi] + sigmas[mj]


                self.interactions[i,j] = [1, eps, sig]
            
            
            
    """
    Visualization tools (some obsolete to be deleted)
    """
    def visualize_state(self, axis = False, figsize = None):
        """
        Show an image of the current state with positions, velocities (as arrows) and boundaries of the box.
        """
        if figsize is None:
            figsize = (6,6)
            if self.L2x != 0 and self.L2y != 0:
                figsize = (4, 4*np.abs(self.L2y/self.L2x))
        
            plt.rcParams["figure.figsize"] = figsize

           

        col = colorscheme()
        
        plt.figure(figsize = figsize)
        plt.plot([-self.Lx, self.Lx, self.Lx, -self.Lx, -self.Lx],[-self.Ly, -self.Ly, self.Ly, self.Ly, -self.Ly], color = (0,0,0), linewidth = 2)
        plt.plot(self.pos[0], self.pos[1], 'o', alpha = .4, markersize = 8*1.8, color = col.getcol(.5))
        plt.plot(self.pos[0], self.pos[1], '.', alpha = 1, markersize = 10, color = (0,0,0))
        
        
        for i in range(len(self.vel[0])):
            plt.plot([self.pos[0,i], self.pos[0,i] + self.vel[0,i]],[self.pos[1,i], self.pos[1,i] + self.vel[1,i]], "-", color = (0,0,0))
            
            th = np.arctan2(self.vel[1,i],self.vel[0,i])
            plt.text(self.pos[0,i] + self.vel[0,i],self.pos[1,i] + self.vel[1,i], "▲", rotation = -90+360*th/(2*np.pi),ha = "center", va = "center") #, color = (0,0,0), fontsize = 20, rotation=0, ha = "center", va = "center")
        
        plt.xlim(-self.Lx-1, self.Lx+1)
        if self.Lx == 0:
            plt.xlim(-11, 11)
        plt.ylim(-self.Ly-1, self.Ly+1)
        if self.Ly == 0:
            plt.ylim(-11, 11)

        if not axis:
            plt.axis("off")
        plt.show()

    #def run(self, n_steps_per_vis = 5, interval = 1):
    #    run_system = animated_system(system = self, n_steps_per_vis=n_steps_per_vis, interval = interval)
    #    plt.show()

    def run(self, nsteps, n_iterations_per_step = 1):
        for i in range(nsteps):
            for j in range(n_iterations_per_step):
                self.advance()
            self.update_view()

    def view(self):
        self.mview = ev.MDView(self)
        return self.mview
    
    def update_view(self):
        self.mview.pos = self.pos.T.tolist()


    # "algebra"

    def __add__(self, other):
        ret_n_bubbles = self.n_bubbles + other.n_bubbles
        ret_pos = np.concatenate(self.pos, other.pos, axis = 1)
        ret_vel = np.concatenate(self.vel, other.vel, axis = 1)
        ret_interactions = 0


class box(mdbox):
    """
    Wrapper to mdbox, for backwards compatability.
    """
    def __init__(self, n_bubbles = 100, masses = None, pos = None, vel = 0.0, box = (0,0), relax = False, grid = True, pair_list = True):
        # Initialize gas
        self.__init__ = mdbox.__init__
        self.__init__(self, n_bubbles = n_bubbles, size = box, masses = masses, vel = vel, grid = True, pair_list = True )



class animated_system():
    def __init__(self, system = None, n_steps_per_vis = 5, interval = 1):
        self.n_steps_per_vis = n_steps_per_vis
        self.system = system
        figsize = (6,6)
        if self.system.size2[0] != 0 and self.system.size2[1] != 0:
            figsize = (4, 4*np.abs(self.system.size2[1]/self.system.size2[0]))
            #self.Lx, self.Ly = self.system.Lx, self.system.Ly
        else:
            figsize = (4,4)

    
        plt.rcParams["figure.figsize"] = figsize

        
        self.fig, self.ax = plt.subplots()
        #self.col = colorscheme()

        self.scatterplot = False
        self.unique_masses = np.unique(self.system.masses)
        if len(self.unique_masses)>1:
            self.scatterplot = True
        


        
        self.ani = FuncAnimation(self.fig, self.update, interval=interval, 
                                          init_func=self.setup_plot, blit=True,cache_frame_data=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        
        
        x,y = self.system.pos
        #s = 10 + 2*self.system.masses
        
        #c = (1,0,0)
        #c = np.random.uniform(0,1,(self.system.N_bubbles, 3))
        
        c = colorscheme()
        
        Lx = self.system.size[0]
        Ly = self.system.size[1]

        if Lx>0:
            
            wx1 = plt.plot([-Lx, -Lx], [-Ly, Ly], color = (0,0,0), linewidth = 2.0)
            wx2 = plt.plot([Lx, Lx], [-Ly, Ly], color = (0,0,0), linewidth = 2.0)

        if Ly>0:
            wx3 = plt.plot([-Lx,Lx], [-Ly, -Ly], color = (0,0,0), linewidth = 2.0)
            wx4 = plt.plot([-Lx,Lx], [ Ly,  Ly], color = (0,0,0), linewidth = 2.0)
        
        if self.scatterplot:
            s = np.sqrt(self.system.masses)*10 #/self.system.Lx

            c = c.getcol(self.system.masses/self.system.masses.max()).T
            self.bubbles = self.ax.scatter(x, y, c=c, s=s, edgecolor="k", marker = "8")
            #self.bubbles = self.ax.scatter(x, y, s= s, edgecolor="k", marker = "8")
        else:
            self.bubbles = self.ax.plot(x, y, "o", color = c.getcol(.4), markersize = 4)[0]


        self.ax.axis([-Lx-1, Lx+1, -Ly-1, Ly+1])
        L05x = 0.1
    
        L05x = max(1, Lx*0.05)
        L05y = max(1, Ly*0.05)
        if self.system.size2[0] ==0:
            L05x = max(L05x, 10*np.abs(self.system.pos[0,:]).max())
        if self.system.size2[1] == 0:
            L05y = max(L05y, 10*np.abs(self.system.pos[1,:]).max())

        plt.xlim(-Lx-L05x, Lx+L05x)
        plt.ylim(-Ly-L05y, Ly+L05y)
        
        return self.bubbles,


        
        
    #@nb.jit
    def update(self, i):

        for i in range(self.n_steps_per_vis):
            self.system.advance()

        if self.scatterplot:
            self.bubbles.set_offsets(self.system.pos.T)
        else:
            x,y = self.system.pos
            self.bubbles.set_data(x,y)
            
        return self.bubbles,
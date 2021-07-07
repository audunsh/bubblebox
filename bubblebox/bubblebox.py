import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import numpy as np
import numba as nb
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
        #self.CC = np.random.uniform(0,1,(3,3))
        self.c = interp1d(np.linspace(0,1,3), self.CC) #(np.linspace(0,1,800)).T
    def getcol(self,i):
        return self.c(i)
   

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
        for j in range(1, coords.shape[0]):
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


@nb.jit()
def no_forces(coords, interactions = None, L2x = 0.0, L2y = 0.0, r2_cut = 9.0):
    return 0

@nb.jit(nopython=True)
def forces(coords, interactions = None, L2x = 0.0, L2y = 0.0, r2_cut = 9.0):
    # 2d force vector
    d = np.zeros((2, coords.shape[1]), dtype = nb.float64)
    Lx, Ly = -L2x/2.0, -L2y/2.0
    
    #u = 0 #pot energy
    
    if interactions is None:
        interactions = np.ones((coords.shape[1], coords.shape[1], 2), dtype = nb.float64)
        

    for i in range(coords.shape[1]):
        cix, ciy = coords[0,i],coords[1,i]
        for j in range(i+1, coords.shape[1]):
            
            
            
            

            
            # distance-based interactions
            #dx, dy = coords[:,j] - ci

            cjx, cjy = coords[0,j], coords[1,j]
            dx = cjx - cix #coords[0,i]
            dy = cjy - ciy #coords[1,i]
            
            # PBC
            if L2x<0:
                if np.abs(dx)>Lx:
                    if dx>0:
                        dx += L2x
                    else:
                        dx -= L2x
                    #dx += np.sign(dx)*L2
            if L2y<0:
                if np.abs(dy)>Ly:
                    if dy>0:
                        dy += L2y
                    else:
                        dy -= L2y
            
            # Compute distance squared
            rr = dx**2 + dy**2 

            if rr<r2_cut: # Screen on distance squared

                eps, sig = interactions[i,j]
                if eps>0:
                        
                    ljw   = -12*eps*(sig**12*rr**-7 - sig**6*rr**-4) # Lennard-Jones weight

                    ljf_x = ljw*dx # x-component
                    ljf_y = ljw*dy # y-component

                    # Sum forces
                    d[0,i] += ljf_x
                    d[1,i] += ljf_y

                    d[0,j] -= ljf_x
                    d[1,j] -= ljf_y
    return d


@nb.jit()
def lj_pot(coords, species = None, L2 = 1.0, r_cut = 9.0, pbc_x = True, pbc_y = True):
    # 2d force vector
    #d = np.zeros((2, coords.shape[1]), dtype = float)
    
    u = 0 #pot energy
    
    if species is None:
        species = np.zeros((coords.shape[1], coords.shape[1]), dtype = int)
    
    for i in range(coords.shape[1]):
        for j in range(i+1, coords.shape[1]):
            
            sp = species[i,j]
            

            if sp == 0:
                # no interaction
                pass
            else:
                # distance-based interactions
                dx = coords[0,j] - coords[0,i]
                dy = coords[1,j] - coords[1,i]
                
                # PBC
                
                if pbc_x:
                    if np.abs(dx)>L2:
                        if dx>0:
                            dx -= 2*L2
                        else:
                            dx += 2*L2
                        #dx += np.sign(dx)*L2
                if pbc_y:
                    if np.abs(dy)>L2:
                        if dy>0:
                            dy -= 2*L2
                        else:
                            dy += 2*L2
                
                
                
                rr = np.sqrt(dx**2 + dy**2)
                if rr<r_cut:

                    if sp == 2:
                        
                        eps = 1.0
                        sig = np.sqrt(2)
                        
                        
                        
                        u +=  (sig/rr)**12 - 2*(sig/rr)**6

                    if sp == 4:

                        
                        
                        u +=  (1/rr)**12 - 2*(1/rr)**6

                        


                
                
    return u


#@nb.jit()
def l_j_pot(coords, species = None):
    # 2d force vector
    d = np.zeros((2, coords.shape[1]), dtype = float)
    
    if species is None:
        species = np.zeros((coords.shape[1], coords.shape[1]), dtype = int)
    
    
    U = 0
    
    for i in range(coords.shape[1]):
        for j in range(i+1, coords.shape[1]):
            
            sp = species[i,j]
            
            if sp == 0:
                # No contribution to potential energy
                pass
            
            else:
                dx = coords[0,i] - coords[0,j]
                dy = coords[1,i] - coords[1,j]
                distance = (dx**2 + dy**2)**-.5 
                
                if sp == 1:
                    # elastic collision
                    print(" Elastic collision unavailable")
                    
                    pass
                
                if sp == 2:
                    # -2r0/r**3 + r0/r**6
                    
                    # l-j repulsive force
                    U +=  4*distance**12
                    
                
                if sp == 3:
                    # l-j attractive force
                    U -= 4*distance**6
                    
                    
                    
                    
                if sp == 4:
                    # l-j force
                    
                    U += 4*(distance**12 - distance**6)
                             
    return U



@nb.jit()
def collisions(coords, vels, screen = 10.0, radius = -1.0, Lx = 0.0, Ly = 0.0, masses = None):
    """
    Hard-sphere collision and wall collisions
    Author: Audun Skau Hansen (a.s.hansen@kjemi.uio.no)

    Keyword arguments:
    
    coords           -- array (2,n_particles) containing positions
    vels             -- array (2,n_particles) containing velocities
    screen           -- screening distance (particles further away not considered)
    radius           -- for positive values, particles are considered hard spheres of equial radius
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
    v = vels*1
    r2 = 4*radius**2
    R2 = 2*radius
    c = 0
    
    if radius>0:
        pos_x = np.argsort(coords[0])
    #else:
    #    pos_x = np.arange(coords.shape[1], dtype = int)
    
    for ii in range(coords.shape[1]):
        #i = pos_x[ii]
        if radius>0:
            i = pos_x[ii]
            for jj in range(ii+1, coords.shape[1]):
                j = pos_x[jj]
                
                dx = coords[0,i] - coords[0,j]
                if np.abs(dx)<R2:
                
                    
                    #for i in range(coords.shape[1]):
                    #for j in range(i+1, coords.shape[1]):
                    dy = coords[1,i] - coords[1,j]
                
                    
                    #d_ = (dx**2 + dy**2)**-1 #**-.5
                    if dx**2 + dy**2 <r2:
                    
                        #d[0,i] += dx*d_
                        #d[1,i] += dy*d_

                        #d[0,j] -= dx*d_
                        #d[1,j] -= dy*d_
                        dij = np.array([dx, dy]) #coords[:, i] - coords[:, j]
                        #dji = -dij
                        
                        
                        vij = v[:, i] - v[:, j]
                        
                        #w = np.dot(dij, vij)/np.dot(vij, vij)
                        
                        
                        #vji = -vij
                        
                        #w = (vij[0]*dij[0] + vij[1]*dij[1])/(vij[0]*vij[0] + vij[1]*vij[1])
                        #w = (vij[0]*dij[0] + vij[1]*dij[1])/(dij[0]*dij[0] + dij[1]*dij[1])
                        
                        w_dij = (vij[0]*dij[0] + vij[1]*dij[1])/(dx**2 + dy**2)*dij
                        #print(w)
                        #wji = np.dot(-vij, -dij)
                        #d[:, i] -= 
                        
                        #df = np.sum(v[:, i]**2 + v[:, j]**2)
                        
                        #if (dx**2 + dy**2)<1e-7:
                        #    print(w_dij, dx**2 + dy**2)
                        if masses is None:
                            v[0, i] -= w_dij[0] #*0.01
                            v[1, i] -= w_dij[1]
                            v[0, j] += w_dij[0] #*0.01
                            v[1, j] += w_dij[1]
                        else:
                            mi = masses[i]
                            mj = masses[j]
                            m1_m2 = mi+mj
                            wi = 2*mj/m1_m2
                            wj = 2*mi/m1_m2
                            

                            v[0, i] -= w_dij[0]*wi #*0.01
                            v[1, i] -= w_dij[1]*wi

                            v[0, j] += w_dij[0]*wj #*0.01
                            v[1, j] += w_dij[1]*wj


                        
                        #df_ = np.sum(v[:, i]**2 + v[:, j]**2)
                        #if np.abs(df_-df)>1e-11:
                        #    print(i,j, df, df_)
                        
                        
                        
                        
                        #v[0, i] = 0
                        #v[1, i] = 0
                        #v[0, j] = 0
                        #v[1, j] = 0
                        
                        
                else:
                    break
                

        # collision with wall
        if Lx>0:
            if np.abs(coords[0,ii])>Lx:
                
                #coords[0,i] = wall*np.sign(coords[0,i])
                v[0,ii] *= -1
                c += 1
        if Ly>0:
            if np.abs(coords[1,ii])>Ly:
                #coords[1,i] = wall*np.sign(coords[1,i])
                v[1,ii] *= -1
                c += 1
            
    return v, c




def arrange_in_grid(pos, Lx, Ly, n_bubbles):
    
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





class box():
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

    system = bb.box(n_bubbles = 100, box = (10,10)) #initialize 10 by 10 closed box containing 100 bubbles

    system.run() #run simulation interactively 
    

    """

    def __init__(self, n_bubbles = 100, masses = None, pos = None, vel = 0.0, box = (0,0), radius = -1, relax = False, grid = True):
        # Initialize gas

        # Boundary conditions
        self.Lx = box[0]
        self.Ly = box[1]

        self.L2x = 2*box[0]
        self.L2y = 2*box[1]







        self.radius = radius
        self.n_bubbles = n_bubbles
        

        # Coordinates - position
        if pos is not None:
            self.pos = np.array(pos)
            self.n_bubbles = self.pos.shape[1]
        else:
            self.pos = np.random.uniform(-self.Lx,self.Lx,(2,self.n_bubbles)) #place 
            self.pos[1,:] = np.random.uniform(-self.Ly,self.Ly,(self.n_bubbles)) 
            if self.Lx == 0:
                self.pos = np.random.uniform(-10,10,(2,self.n_bubbles)) #place 
            if self.Ly == 0:
                self.pos[1,:] = np.random.uniform(-10,10,(self.n_bubbles)) 
            if grid:
                if np.abs(self.Lx) > 0 and np.abs(self.Ly) > 0:
                    self.pos = arrange_in_grid(self.pos, np.abs(self.Lx), np.abs(self.Ly), self.n_bubbles)
                else:
                    self.pos = arrange_in_grid(self.pos, 1, 1, self.n_bubbles)
            
        

        if masses is None:
            self.masses = np.ones(self.n_bubbles, dtype = int)
            self.interactions = np.ones((self.n_bubbles, self.n_bubbles, 2), dtype = float)
            self.masses_inv = np.array(self.masses, dtype = float)**-1
            #self.n_bubbles = n_bubbles
            
        else:
            self.masses = masses
            self.n_bubbles = len(masses)
            self.set_interactions(self.masses)
            self.masses_inv = np.array(self.masses, dtype = float)**-1

        
        
        

        

        self.pos_old = self.pos*1 # retain previous timesteps to compute velocities
        self.active = np.ones(self.pos.shape[1], dtype = np.bool)

        # Coordinates - velocity
        r, th = np.random.normal(0,vel, self.n_bubbles), np.random.uniform(0,2*np.pi,self.n_bubbles)
        

        self.vel = np.array([r*np.cos(th), r*np.sin(th)]) #np.random.uniform(-1,1,(2,self.n_bubbles))*vel
        self.vel[:] -= np.mean(self.vel, axis = 1)[:, None]
        self.vel_ = self.vel # verlet integrator velocity at previous timestep
        

        # Integrator 
        self.advance = self.advance_vverlet

        # Algorithm for force calculation
        self.forces = forces
        self.r2_cut = 9.0 #distance cutoff squared for force-calculation
        
        # Time and timestep
        self.t = 0
        self.dt = 0.001

        # Collision counter
        self.col = 0

        
        

        
        # Thermostat / relaxation
        if relax:
            self.relax_sa(20000)

    def set_interactions(self, masses):
        self.interactions = np.ones((self.n_bubbles, self.n_bubbles, 2), dtype = float)

        # Placeholder for proper parametrization of interactions
        epsilons = np.linspace(.5,10,100)
        sigmas   = np.linspace(1,np.sqrt(2), 100)

        

        for i in range(self.n_bubbles):
            mi = masses[i]
            for j in range(i+1, self.n_bubbles):
                
                mj = masses[j]

                eps = np.sqrt(epsilons[mi]*epsilons[mj])
                sig = sigmas[mi] + sigmas[mj]


                self.interactions[i,j] = [eps, sig]

        
    def relax_positions(self):
        """
        deterministic relaxation using forces
        """
        for i in np.arange(20):
            #self.vel -= .1*repel(self.pos)*dt
            self.pos -= .1*self.forces(self.pos, self.interactions, self.L2x, self.L2y)
            #outside_x = np.abs(self.pos[0,:])>self.L
            #outside_y = np.abs(self.pos[1,:])>self.L
            
            #PBC
            
            self.pos[0, :]  = (self.pos[0,:] + self.Lx) % (2*self.Lx) - self.Lx
            self.pos[1, :]  = (self.pos[1,:] + self.Ly) % (2*self.Ly) - self.Ly
            
            #self.pos[0, outside_x] = np.random.uniform(-self.L, self.L, np.sum(outside_x))
            #self.pos[1, outside_y] = np.random.uniform(-self.L, self.L, np.sum(outside_y))
        self.pos *= .99
        
    def relax_sa(self, Nt, stepsize = 0.01, pbc = True):
        """
        Simulated annealing relaxation by maximization of the norm of the distance matrix
        """
        # simulated annealing thermostat


        f0 = np.sum(self.forces(self.pos, self.interactions, self.L2x, self.L2y)**2)/self.n_bubbles
        
        #dist = distances(self.pos)
        #dd = distances_reduced(self.pos, self.L2x, self.L2y)
        dd = np.min(distances_reduced(self.pos, self.L2x, self.L2y), axis = 1)
        f0 = -np.sum(dd) + np.sum((dd - np.mean(dd))**2)


        #f0 = np.sum((dist - np.mean(dist))**2)


        #print(f0)


        temp = 2
        print("initial", f0)

        for i in range(Nt):
            pos_new = self.pos*1
            pos_new[:, self.active] += np.random.uniform(-1,1, self.pos[:, self.active].shape)*stepsize
            #f1 = np.sum(forces(pos_new, self.interactions, self.L2x, self.L2y)**2)/self.n_bubbles
            #f1 = np.sum((self.pos - np.mean(self.pos, axis = 1)[:, None])**2)
            #dist = distances(pos_new)
            
            dd = np.min(distances_reduced(pos_new, self.L2x, self.L2y), axis = 1)
            f1 = -np.sum(dd) + np.sum((dd - np.mean(dd))**2)
            #if np.any(np.abs(pos_new)>=self.L):
            #    pass
            if np.exp(-(f1-f0)/temp)>0.9:
                # accept
                
                

                # impose PBC for both walled and periodic conditions, to evenly distribute particles
                if self.Lx<0:
                    pos_new[0, :]  = (pos_new[0,:] + self.Lx) % (2*self.Lx) - self.Lx
                if self.Ly<0:
                    pos_new[1, :]  = (pos_new[1,:] + self.Ly) % (2*self.Ly) - self.Ly

                self.pos = pos_new*1
                f0 = f1*1




                # impose wall boundary conditions
                #if self.walls_x or self.walls_y:
                #    self.vel, self.col = wall_collisions(pos_new, self.vel, wall = self.L, walls_x = self.walls_x, walls_y = self.walls_y)

                
            temp *= 0.99
            #if f0<=1:
            #    break
        dd = np.min(distances_reduced(self.pos, self.L2x, self.L2y), axis = 1)
        f1 = -np.sum(dd) + np.sum((dd - np.mean(dd))**2)
        print("final", f0)
        if f0>1:
            print("Warning, system may be in a poor initial state.")
            print("Annealing algorithm reports Force norm per bubble to be", f0)
        self.pos_old = self.pos*1 # retain previous timesteps to compute velocities

    
    
    
    
    
    
        
    def advance_vverlet(self):
        """
        velocity-Verlet timestep
        """
        #Fn = self.forces()
        Fn = self.forces(self.pos, self.interactions, self.L2x, self.L2y, self.r2_cut)
        
        self.d_pos = self.vel_*self.dt + .5*Fn*self.dt**2*self.masses_inv
        
        


        pos_new = self.pos + self.d_pos
        
        self.vel_ = self.vel_ + .5*(self.forces(pos_new, self.interactions, self.L2x, self.L2y, self.r2_cut) + Fn)*self.dt*self.masses_inv
        

        #self.vel, self.col = wall_collisions(self.pos, self.vel, radius = 1.0, wall = self.L)
        #self.pos[0,np.abs(self.pos[0,:])>self.L] 
        
        # impose PBC
        if self.Lx<0:
            pos_new[0, :]  = (pos_new[0,:] + self.Lx) % (2*self.Lx) - self.Lx
        if self.Ly<0:
            pos_new[1, :]  = (pos_new[1,:] + self.Ly) % (2*self.Ly) - self.Ly

        # impose wall and collision boundary conditions
        self.vel_, self.col = collisions(pos_new, self.vel_, screen = 10.0, radius = self.radius, Lx = self.Lx, Ly = self.Ly, masses = self.masses)
        
        #self.vel_*=.999


        #update arrays (in order to retain velocity)
        self.vel = (pos_new - self.pos_old)/(2*self.dt)
        self.pos_old[:] = self.pos
        self.pos[:, self.active] = pos_new[:, self.active]

        # Track time
        self.t += self.dt
        
    def advance_euler(self):
        """
        Explicit Euler timestep
        """
        self.vel += self.forces(self.pos, self.interactions, self.L2x, self.L2y, self.r2_cut)*self.dt*self.masses_inv
        self.pos += self.vel*self.dt
        
        # impose PBC
        if self.Lx<0:
            self.pos[0, :]  = (self.pos[0,:] + self.Lx) % (2*self.Lx) - self.Lx
        if self.Ly<0:
            self.pos[1, :]  = (self.pos[1,:] + self.Ly) % (2*self.Ly) - self.Ly

        # impose wall and collision bounary conditions
        self.vel_, self.col = collisions(self.pos, self.vel_, screen = 10.0, radius = self.radius, Lx = self.Lx, Ly = self.Ly, masses = self.masses)
        
            
        # Track time
        self.t += self.dt
    def kinetic_energy(self):
        """
        Compute total kinetic energy in system
        """
        # Vektorisert funksjon med hensyn på ytelse
        return .5*np.sum(self.masses*np.sum(self.vel**2, axis = 0))
    
    def kinetic_energies(self):
        """
        Compute kinetic energy of each bubble
        """
        # Vektorisert funksjon med hensyn på ytelse
        return .5*self.masses*np.sum(self.vel**2, axis = 0)
        
    
    
    
    def evolve(self, t = 1.0):
        """
        Let system evolve in time for t seconds
        """
        t1 = self.t+t
        while self.t<t1:
            self.advance()
           

            
            
            
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
        
        #for i in range(len(self.pos[0])):
        #    for j in range(i+1, len(self.pos[0])):
        #        plt.plot([self.pos[0,i], self.pos[0,j]],[self.pos[1,i], self.pos[1,j]], "-", color = (.5,.5,.5), linewidth = .4, alpha = .1)
        
        
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

    def run(self, n_steps_per_vis = 5, interval = 1):
        run_system = animated_system(system = self, n_steps_per_vis=n_steps_per_vis, interval = interval)
        plt.show()


class animated_system():
    def __init__(self, system = None, n_steps_per_vis = 5, interval = 1):
        self.n_steps_per_vis = n_steps_per_vis
        self.system = system
        figsize = (6,6)
        if self.system.L2x != 0 and self.system.L2y != 0:
            figsize = (4, 4*np.abs(self.system.L2y/self.system.L2x))
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
        
        Lx = self.system.Lx
        Ly = self.system.Ly

        if self.system.Lx>0:
            
            wx1 = plt.plot([-Lx, -Lx], [-Ly, Ly], color = (0,0,0), linewidth = 2.0)
            wx2 = plt.plot([Lx, Lx], [-Ly, Ly], color = (0,0,0), linewidth = 2.0)

        if self.system.Ly>0:
            wx3 = plt.plot([-Lx,Lx], [-Ly, -Ly], color = (0,0,0), linewidth = 2.0)
            wx4 = plt.plot([-Lx,Lx], [ Ly,  Ly], color = (0,0,0), linewidth = 2.0)
        
        if self.scatterplot:
            s = self.system.masses*10 #/self.system.Lx

            c = c.getcol(self.system.masses/self.system.masses.max()).T
            self.bubbles = self.ax.scatter(x, y, c=c, s=s, edgecolor="k", marker = "8")
            #self.bubbles = self.ax.scatter(x, y, s= s, edgecolor="k", marker = "8")
        else:
            self.bubbles = self.ax.plot(x, y, "o", color = c.getcol(.4), markersize = 4)[0]


        self.ax.axis([-self.system.Lx-1, self.system.Lx+1, -self.system.Ly-1, self.system.Ly+1])

        L05x = self.system.Lx*0.05
        L05y = self.system.Ly*0.05
        if self.system.L2x ==0:
            L05x = 10*np.abs(self.system.pos[0,:]).max()
        if self.system.L2y == 0:
            L05y = 10*np.abs(self.system.pos[1,:]).max()

        plt.xlim(-self.system.Lx-L05x, self.system.Lx+L05x)
        plt.ylim(-self.system.Ly-L05y, self.system.Ly+L05y)
        
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
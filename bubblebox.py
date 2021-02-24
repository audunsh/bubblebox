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
        Z = np.zeros(Nx, dtype = np.float)
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
        self.c = interp1d(np.linspace(0,1,3), self.CC) #(np.linspace(0,1,800)).T
    def getcol(self,i):
        return self.c(i)
   

matplotlib.rcParams['figure.figsize'] = (6.4,4.8)
matplotlib.rcParams['figure.figsize'] = (4.8,4.8)

@nb.jit()
def repel(coords, screen = 10, L2 = 1.0):
    # 2d force vector
    d = np.zeros((2, coords.shape[1]), dtype = np.float64)
    
    
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

@nb.jit
def distance_matrix(coords):
    d = np.zeros((coords.shape[0], coords.shape[0]), dtype = np.float64)
    for i in range(coords.shape[0]):
        for j in range(1, coords.shape[0]):
            d_ = (coords[i,0] - coords[j,0])**2 + (coords[i,1] - coords[j,1])**2  #+ (coords[i,2] - coords[j,2])**2 
            d[i,j] = d_
            d[j,i] = d_
    return d

@nb.jit()
def no_forces(coords, interactions = None, L2 = 1.0, r2_cut = 9.0, pbc_x = True, pbc_y = True):
    return 0

@nb.jit()
def forces(coords, interactions = None, L2 = 1.0, r2_cut = 9.0, pbc_x = True, pbc_y = True):
    # 2d force vector
    d = np.zeros((2, coords.shape[1]), dtype = np.float_)
    
    #u = 0 #pot energy
    
    if interactions is None:
        interactions = np.ones((coords.shape[1], coords.shape[1], 2), dtype = np.float_)
        

    for i in range(coords.shape[1]):
        for j in range(i+1, coords.shape[1]):
            
            
            
            

            
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
def forces_(coords, species = None, L2 = 1.0, r2_cut = 9.0, pbc_x = True, pbc_y = True):
    # 2d force vector
    d = np.zeros((2, coords.shape[1]), dtype = np.float_)
    
    #u = 0 #pot energy
    
    if species is None:
        species = np.zeros((coords.shape[1], coords.shape[1]), dtype = np.int_)
        
    
    
    
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
                
                """
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
                """
                
                
                
                rr = dx**2 + dy**2
                if rr<r2_cut:

                    if sp == 2:
                        
                        sig = np.sqrt(2)
                        eps = 1.0
                        
                        
                        
                        ljw   = -12*eps*(sig**12*rr**-7 - sig**6*rr**-4)

                        ljf_x = ljw*dx
                        ljf_y = ljw*dy


                        d[0,i] += ljf_x
                        d[1,i] += ljf_y

                        d[0,j] -= ljf_x
                        d[1,j] -= ljf_y
                        
                        

                    if sp == 4:

                        ljw   = -12*(rr**-7 - rr**-4)
                        
                        

                        ljf_x = ljw*dx
                        ljf_y = ljw*dy


                        d[0,i] += ljf_x
                        d[1,i] += ljf_y

                        d[0,j] -= ljf_x
                        d[1,j] -= ljf_y


                
                
    return d

@nb.jit()
def lj_pot(coords, species = None, L2 = 1.0, r_cut = 9.0, pbc_x = True, pbc_y = True):
    # 2d force vector
    #d = np.zeros((2, coords.shape[1]), dtype = np.float_)
    
    u = 0 #pot energy
    
    if species is None:
        species = np.zeros((coords.shape[1], coords.shape[1]), dtype = np.int_)
    
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
    d = np.zeros((2, coords.shape[1]), dtype = np.float_)
    
    if species is None:
        species = np.zeros((coords.shape[1], coords.shape[1]), dtype = np.int_)
    
    
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
def collisions(coords, vels, screen = 10.0, radius = -1.0, wall = 1.0, walls_x = True, walls_y = True, masses = None):
    # 2d force vector
    #d = coords*1
    v = vels*1
    r2 = 4*radius**2
    R2 = 2*radius
    c = 0
    
    if radius>0:
        pos_x = np.argsort(coords[0])
    #else:
    #    pos_x = np.arange(coords.shape[1], dtype = np.int)
    
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
        if walls_x:
            if np.abs(coords[0,ii])>wall:
                
                #coords[0,i] = wall*np.sign(coords[0,i])
                v[0,ii] *= -1
                c += 1
        if walls_y:
            if np.abs(coords[1,ii])>wall:
                #coords[1,i] = wall*np.sign(coords[1,i])
                v[1,ii] *= -1
                c += 1
            
    return v, c

@nb.jit()
def wall_collisions(coords, vels, screen = 10.0, radius = 1.0, wall = 1.0, walls_x = True, walls_y = True):
    # 2d force vector
    #d = coords*1
    v = vels*1
    c = 0
    
    for i in range(coords.shape[1]):
        if walls_x:
            if np.abs(coords[0,i])>wall:
                
                #coords[0,i] = wall*np.sign(coords[0,i])
                v[0,i] *= -1
                c += 1
        if walls_y:
            if np.abs(coords[1,i])>wall:
                #coords[1,i] = wall*np.sign(coords[1,i])
                v[1,i] *= -1
                c += 1
            
    return v, c






class sample():
    """
    Simple 2D gas simulation
    
    - N particles with equal mass
    - 
    
    """
    def __init__(self, N_bubbles = 30, masses = None, v0 = 0.0, L = None, radius = -1, relax = True, pbc = False):
        # Initialize gas

        # Boundary conditions
        self.pbc_x = False
        self.pbc_y = False

        

        if L is None:
            self.L = 40
            self.walls_x = False
            self.walls_y = False

        else:
            self.L = L
            if not pbc:
                self.walls_x = True
                self.walls_y = True
            else:
                self.walls_x = False
                self.walls_y = False
                self.pbc_x = True
                self.pbc_y = True


        self.radius = radius

        if masses is None:
            self.masses = np.ones(N_bubbles, dtype = np.int_)
            self.interactions = np.ones((N_bubbles, N_bubbles, 2), dtype = np.float_)
            self.masses_inv = np.array(self.masses, dtype = np.float_)**-1
            self.N_bubbles = N_bubbles
            
        else:
            self.masses = masses
            self.N_bubbles = len(masses)
            self.set_interactions(self.masses)
            self.masses_inv = np.array(self.masses, dtype = np.float_)**-1


        # Coordinates - position
        self.pos = np.random.uniform(-L,L,(2,self.N_bubbles)) #place 
        self.pos_old = self.pos*1 # retain previous timesteps to compute velocities

        # Coordinates - velocity
        self.vel = np.random.uniform(-1,1,(2,self.N_bubbles))*v0
        self.vel_ = self.vel # verlet integrator velocity at previous timestep
        self.vel[:] -= np.mean(self.vel, axis = 1)[:, None]

        # Integrator 
        self.advance = self.advance_vverlet

        # Algorithm for force calculation
        self.forces = forces
        
        # Time and timestep
        self.t = 0
        self.dt = 0.001

        # Collision counter
        self.col = 0

        
        

        
        # Thermostat / relaxation
        if relax:
            self.relax_sa(20000)

    def set_interactions(self, masses):
        self.interactions = np.ones((self.N_bubbles, self.N_bubbles, 2), dtype = np.float_)

        # Placeholder for proper parametrization of interactions
        epsilons = np.linspace(.5,10,100)
        sigmas   = np.linspace(1,np.sqrt(2), 100)

        

        for i in range(self.N_bubbles):
            mi = masses[i]
            for j in range(i+1, self.N_bubbles):
                
                mj = masses[j]

                eps = np.sqrt(epsilons[mi]*epsilons[mj])
                sig = sigmas[mi] + sigmas[mj]


                self.interactions[i,j] = [eps, sig]

        
    def relax_positions(self):
        for i in np.arange(20):
            #self.vel -= .1*repel(self.pos)*dt
            self.pos -= .1*self.forces(self.pos, self.interactions, self.L, pbc_x = self.pbc_x, pbc_y = self.pbc_y)
            #outside_x = np.abs(self.pos[0,:])>self.L
            #outside_y = np.abs(self.pos[1,:])>self.L
            
            #PBC
            
            self.pos[0, :]  = (self.pos[0,:] + self.L) % (2*self.L) - self.L
            self.pos[1, :]  = (self.pos[1,:] + self.L) % (2*self.L) - self.L
            
            #self.pos[0, outside_x] = np.random.uniform(-self.L, self.L, np.sum(outside_x))
            #self.pos[1, outside_y] = np.random.uniform(-self.L, self.L, np.sum(outside_y))
        self.pos *= .99
        
    def relax_sa(self, Nt, stepsize = 0.01):
        # simulated annealing thermostat
        f0 = np.sum(self.forces(self.pos, self.interactions, self.L, pbc_x = self.pbc_x, pbc_y = self.pbc_y)**2)/self.N_bubbles
        temp = 2
        #print("iniial", f0)

        for i in range(Nt):
            pos_new = self.pos + np.random.uniform(-1,1, self.pos.shape)*stepsize
            f1 = np.sum(forces(pos_new, self.interactions, self.L, pbc_x = self.pbc_x, pbc_y = self.pbc_y)**2)/self.N_bubbles
            #if np.any(np.abs(pos_new)>=self.L):
            #    pass
            if np.exp(-(f1-f0)/temp)>0.9:
                # accept
                
                self.pos = pos_new*1
                f0 = f1*1
                # impose PBC

                if self.pbc_x:
                    self.pos[0, :]  = (self.pos[0,:] + self.L) % (2*self.L) - self.L
                if self.pbc_y:
                    self.pos[1, :]  = (self.pos[1,:] + self.L) % (2*self.L) - self.L

                # impose wall bounary conditions
                if self.walls_x or self.walls_y:
                    self.vel, self.col = wall_collisions(pos_new, self.vel, wall = self.L, walls_x = self.walls_x, walls_y = self.walls_y)
                
            temp *= 0.99
            if f0<=1:
                break
        #print("final", f0)
        if f0>1:
            print("Warning, system may be in a poor initial state.")
            print("Annealing algorithm reports Force norm per bubble to be", f0)
        self.pos_old = self.pos*1 # retain previous timesteps to compute velocities

    
    
    
    
    
    
        
    def advance_vverlet(self, dt = 0.1):
        """
        velocity-Verlet timestep
        """
        #Fn = self.forces()
        Fn = self.forces(self.pos, self.interactions, self.L, pbc_x = self.pbc_x, pbc_y = self.pbc_y)
        
        #self.pos = 
        #self.pos_old = #


        pos_new = self.pos + self.vel_*dt + .5*Fn*dt**2*self.masses_inv
        
        self.vel_ = self.vel_ + .5*(self.forces(pos_new, self.interactions, self.L, pbc_x = self.pbc_x, pbc_y = self.pbc_y) + Fn)*dt*self.masses_inv
        
        #self.vel, self.col = wall_collisions(self.pos, self.vel, radius = 1.0, wall = self.L)
        #self.pos[0,np.abs(self.pos[0,:])>self.L] 
        
        # impose PBC
        if self.pbc_x:
            pos_new[0, :]  = (pos_new[0,:] + self.L) % (2*self.L) - self.L
        if self.pbc_y:
            pos_new[1, :]  = (pos_new[1,:] + self.L) % (2*self.L) - self.L

        # impose wall bounary conditions
        if self.walls_x or self.walls_y:
            #self.vel_, self.col = wall_collisions(pos_new, self.vel_, wall = self.L, walls_x = self.walls_x, walls_y = self.walls_y)
            self.vel_, self.col = collisions(pos_new, self.vel_, screen = 10.0, radius = self.radius, wall = self.L, walls_x = self.walls_x, walls_y = self.walls_y,  masses = self.masses)

        
        
        #update arrays (in order to retain velocity)
        self.vel = (pos_new - self.pos_old)/(2*self.dt)
        self.pos_old[:] = self.pos
        self.pos[:] = pos_new

        # Track time
        self.t += dt
        
    def advance_euler(self, dt):
        """
        Explicit Euler timestep
        """
        self.vel += self.forces(self.pos, self.interactions, self.L, pbc_x = self.pbc_x, pbc_y = self.pbc_y)*dt*self.masses_inv
        self.pos += self.vel*dt
        
        # impose PBC
        if self.pbc_x:
            self.pos[0, :]  = (self.pos[0,:] + self.L) % (2*self.L) - self.L
        if self.pbc_y:
            self.pos[1, :]  = (self.pos[1,:] + self.L) % (2*self.L) - self.L

        # impose wall bounary conditions
        if self.walls_x or self.walls_y:
            self.vel, self.col = wall_collisions(self.pos, self.vel, wall = self.L, walls_x = self.walls_x, walls_y = self.walls_y)
        
            
        # Track time
        self.t += dt
        
    
    
    
    def evolve(self, t = 1.0, dt = 0.1):
        t1 = self.t+t
        while self.t<t1:
            self.advance(dt)
        self.t += t
            
            
            
    """
    Visualization tools (some obsolete to be deleted)
    """
    def visualize_state(self, axis = False, figsize = (4,4)):
        col = colorscheme()
        
        plt.figure(figsize = figsize)
        plt.plot([-self.L, self.L, self.L, -self.L, -self.L],[-self.L, -self.L, self.L, self.L, -self.L], color = (0,0,0), linewidth = 2)
        plt.plot(self.pos[0], self.pos[1], 'o', alpha = .4, markersize = 8*1.8, color = col.getcol(.5))
        plt.plot(self.pos[0], self.pos[1], '.', alpha = 1, markersize = 10, color = (0,0,0))
        
        #for i in range(len(self.pos[0])):
        #    for j in range(i+1, len(self.pos[0])):
        #        plt.plot([self.pos[0,i], self.pos[0,j]],[self.pos[1,i], self.pos[1,j]], "-", color = (.5,.5,.5), linewidth = .4, alpha = .1)
        
        
        for i in range(len(self.vel[0])):
            plt.plot([self.pos[0,i], self.pos[0,i] + self.vel[0,i]],[self.pos[1,i], self.pos[1,i] + self.vel[1,i]], "-", color = (0,0,0))
            
            th = np.arctan2(self.vel[1,i],self.vel[0,i])
            plt.text(self.pos[0,i] + self.vel[0,i],self.pos[1,i] + self.vel[1,i], "▲", rotation = -90+360*th/(2*np.pi),ha = "center", va = "center") #, color = (0,0,0), fontsize = 20, rotation=0, ha = "center", va = "center")
        
        plt.xlim(-self.L-1, self.L+1)
        plt.ylim(-self.L-1, self.L+1)
        if not axis:
            plt.axis("off")
        plt.show()

    def run(self):
        run_system = animated_system(system = self)
        plt.show()

    def run___(self, axis = False, logoscreen = False):
        #mu = np.unique(self.masses)
    
        self.fig, self.ax = plt.subplots()
        col = colorscheme()
        
        #cluster = []
        #for i in np.unique(self.masses): #unique masses
        #    cluster.append(np.arange(self.masses.shape[0])[self.masses==i])
        #
        #Ln_ = []
        #for i in cluster:
        #    Ln_.append(plt.plot([], [], 'o', alpha = .4, markersize = 4*np.sqrt(self.masses[i[0]]), color = col.getcol(self.masses[i[0]]/self.masses.max()))[0] )
            

        #n_c = len(cluster)
        x,y = self.pos
        ln = self.ax.scatter(x,y)
        #ln.set_animated(True)


        def init():
            sv = 1
            self.ax.set_xlim(-self.L-sv,self.L+sv)
            self.ax.set_ylim(-self.L-sv,self.L+sv)
            if self.walls_x:
                wx1 = plt.plot([-self.L, -self.L], [-self.L, self.L], color = (0,0,0), linewidth = 2.0)
                wx2 = plt.plot([self.L, self.L], [-self.L, self.L], color = (0,0,0), linewidth = 2.0)

            if self.walls_y:
                wx3 = plt.plot([-self.L, self.L], [-self.L, -self.L], color = (0,0,0), linewidth = 2.0)
                wx4 = plt.plot([-self.L, self.L], [ self.L, self.L], color = (0,0,0), linewidth = 2.0)

            if not axis:
                self.ax.axis("off")
            if logoscreen:
                plt.text(0,-self.L+1, "BubbleBox", fontsize = 35, ha = "center",fontweight="bold", fontname = "Verdana", alpha = 1, color = col.getcol(0.0))
            plt.xlim(-self.L-1, self.L+1)
            plt.ylim(-self.L-1, self.L+1)
            
            #return Ln_[0],
            #ln.set_offsets([])
            return ln,



        def update(frame):
            #xdata.append(frame)
            #ydata.append(np.sin(frame))
            #c = 0
            #global U
            #global nn

            for i in range(10):
                self.advance(dt = self.dt)
                c += self.col


            #x,y = self.pos
            #mn.set_data(x,y)
            #for i in range(n_c):
            #    ci = cluster[i]
            #    Ln_[i].set_data(x[ci],y[ci])
            ln.set_offsets(self.pos)
            #return ln,
        

        self.ani = FuncAnimation(self.fig, update, frames=None, 
                            init_func=init, blit=True, interval = 1)

        
        plt.show() 
        
    def run__(self, axis = False, logoscreen = False):
        mu = np.unique(self.masses)
        if len(mu)==1:
            self.run_(axis = axis)
        else:
            self.fig, self.ax = plt.subplots()
            col = colorscheme()
            
            cluster = []
            for i in np.unique(self.masses): #unique masses
                cluster.append(np.arange(self.masses.shape[0])[self.masses==i])
            
            Ln_ = []
            for i in cluster:
                Ln_.append(plt.plot([], [], 'o', alpha = .4, markersize = 4*np.sqrt(self.masses[i[0]]), color = col.getcol(self.masses[i[0]]/self.masses.max()))[0] )
                

            n_c = len(cluster)


            def init():
                sv = 1
                self.ax.set_xlim(-self.L-sv,self.L+sv)
                self.ax.set_ylim(-self.L-sv,self.L+sv)
                if self.walls_x:
                    wx1 = plt.plot([-self.L, -self.L], [-self.L, self.L], color = (0,0,0), linewidth = 2.0)
                    wx2 = plt.plot([self.L, self.L], [-self.L, self.L], color = (0,0,0), linewidth = 2.0)

                if self.walls_y:
                    wx3 = plt.plot([-self.L, self.L], [-self.L, -self.L], color = (0,0,0), linewidth = 2.0)
                    wx4 = plt.plot([-self.L, self.L], [ self.L, self.L], color = (0,0,0), linewidth = 2.0)

                if not axis:
                    self.ax.axis("off")
                if logoscreen:
                    plt.text(0,-self.L+1, "BubbleBox", fontsize = 35, ha = "center",fontweight="bold", fontname = "Verdana", alpha = 1, color = col.getcol(0.0))
                plt.xlim(-self.L-1, self.L+1)
                plt.ylim(-self.L-1, self.L+1)
                
                return Ln_[0],



            def update(frame):
                #xdata.append(frame)
                #ydata.append(np.sin(frame))
                c = 0
                global U
                global nn

                for i in range(10):
                    self.advance(dt = self.dt)
                    c += self.col


                x,y = self.pos
                #mn.set_data(x,y)
                for i in range(n_c):
                    ci = cluster[i]
                    Ln_[i].set_data(x[ci],y[ci])
                
                return Ln_[0],
            

            self.ani = FuncAnimation(self.fig, update, frames=np.linspace(0, 1, 2), 
                                init_func=init, blit=True, interval = 1)

            
            plt.show() 


    def run_(self, axis = False):
        self.fig, self.ax = plt.subplots()
        col = colorscheme()
        #xdata, ydata = [], []
        ln, = plt.plot([], [], 'o', alpha = .4, markersize = 2*1.8, color = col.getcol(.4))
        mn, = plt.plot([], [], 'o', alpha = .4, markersize = 4*1.8, color = np.random.uniform(0,1,3))
        tn, = plt.plot([], [], '-', color = (0,0,0), alpha = 1, linewidth = 1.0)
        un, = plt.plot([], [], '-', color = (0,0,0), alpha = 1, linewidth = 1.0)

        temp = []
        pot = []

        #tx = plt.text(-self.L,self.L,"$E_K$ = %.2f" % 1.0)

        def init():
            self.ax.set_xlim(-self.L-1,self.L+1)
            self.ax.set_ylim(-self.L-1,self.L+1)
            if self.walls_x:
                wx1 = plt.plot([-self.L, -self.L], [-self.L, self.L], color = (0,0,0), linewidth = 2.0)
                wx2 = plt.plot([self.L, self.L], [-self.L, self.L], color = (0,0,0), linewidth = 2.0)

            if self.walls_y:
                wx3 = plt.plot([-self.L, self.L], [-self.L, -self.L], color = (0,0,0), linewidth = 2.0)
                wx4 = plt.plot([-self.L, self.L], [ self.L, self.L], color = (0,0,0), linewidth = 2.0)


            #ax.axis("off")
            plt.xlim(-self.L-1, self.L+1)
            plt.ylim(-self.L-1, self.L+1)
            return ln,



        def update(frame):
            #xdata.append(frame)
            #ydata.append(np.sin(frame))
            c = 0
            global U
            global nn

            for i in range(10):
                self.advance(dt = self.dt)
                c += self.col


            x,y = self.pos
            #mn.set_data(x,y)
            ln.set_data(x,y)
            #mn.set_data(x[Nd:],y[Nd:])
            if not axis:
                self.ax.axis("off")

            #v = .5*np.sum(np.sum(self.vel**2, axis = 0)*self.masses)

            #u = lj_pot(self.pos, self.species, L2 = self.L)
            #u = 0


            #U = (U*(nn-1) + (u+v))/nn
            #nn += 1
            #tx.set_text("$E_K$ = %.2f       $U$ = %.2f       %.2f" % (v,u, U))


            #return ln,
        
        self.ani = FuncAnimation(self.fig, update, frames=np.linspace(0, 1, 1000), 
                            init_func=init, blit=True, interval = 1)
        plt.show() 


class animated_system():
    def __init__(self, system = None):
        self.system = system
        
        self.fig, self.ax = plt.subplots()
        #self.col = colorscheme()

        self.scatterplot = False
        self.unique_masses = np.unique(self.system.masses)
        if len(self.unique_masses)>1:
            self.scatterplot = True
        


        
        self.ani = FuncAnimation(self.fig, self.update, interval=1, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        
        
        x,y = self.system.pos
        #s = 10 + 2*self.system.masses
        
        #c = (1,0,0)
        #c = np.random.uniform(0,1,(self.system.N_bubbles, 3))
        
        c = colorscheme()
        
        L = self.system.L
        if self.system.walls_x:
            
            wx1 = plt.plot([-L, -L], [-L, L], color = (0,0,0), linewidth = 2.0)
            wx2 = plt.plot([L, L], [-L, L], color = (0,0,0), linewidth = 2.0)

        if self.system.walls_y:
            wx3 = plt.plot([-L,L], [-L, -L], color = (0,0,0), linewidth = 2.0)
            wx4 = plt.plot([-L, L], [ L, L], color = (0,0,0), linewidth = 2.0)
        
        if self.scatterplot:
            s = self.system.masses*150/self.system.L

            c = c.getcol(self.system.masses/self.system.masses.max()).T
            self.bubbles = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1,
                                        cmap="jet", edgecolor="k", marker = "o")
        else:
            self.bubbles = self.ax.plot(x, y, "o", color = c.getcol(.4))[0]



        self.ax.axis([-self.system.L-1, self.system.L+1, -self.system.L-1, self.system.L+1])
        
        return self.bubbles,


        
        
    #@nb.jit
    def update(self, i, nsteps_per_vis = 10):

        for i in range(nsteps_per_vis):
            self.system.advance(dt = self.system.dt)

        if self.scatterplot:
            self.bubbles.set_offsets(self.system.pos.T)
        else:
            x,y = self.system.pos
            self.bubbles.set_data(x,y)

        return self.bubbles,
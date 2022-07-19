# Author: Audun Skau Hansen (a.s.hansen@kjemi.uio.no), 2022

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
import time

import evince as ev

square_lattice_basis = np.array([[ 1, 0],
                                 [-1, 0],
                                 [ 0, 1],
                                 [ 0,-1]], dtype = int)

# representations

class polymer():
    def __init__(self, config, conformation = None, basis = square_lattice_basis):
        self.config = config
        self.conformation = conformation
        self._edict = {"H":1, "P":0}
        self.basis = basis
        self.conformations = []
        
    def determine_conformations(self):
        """
        Explore and determine possible conformations
        
        **Note** The generated conformations contain a lot of 
                 redundancy due to symmetries.
                 However, we simply want a set which is compact 
                 enough for a somewhat efficient exploration of
                 small polymers.
        """
        self.advance_conformation([np.array([1,0])], [np.array([0,0]), np.array([1,0])])
        
        # generate transformed conformations
        transformed_conformations = []
        for i in self.conformations:
            transformed_conformations.append(i)
            transformed_conformations.append([np.array([j[0], -j[1]]) for j in i])
            transformed_conformations.append([np.array([-j[0], j[1]]) for j in i])
            transformed_conformations.append([np.array([-j[0], -j[1]]) for j in i])
            
            transformed_conformations.append([np.array([j[1],  j[0]]) for j in i])
            transformed_conformations.append([np.array([-j[1],  j[0]]) for j in i])
            transformed_conformations.append([np.array([j[1], -j[0]]) for j in i])
            transformed_conformations.append([np.array([-j[1], -j[0]]) for j in i])
            
        self.conformations = transformed_conformations

            
        
        
    def advance_conformation(self, conformation, points):
        """
        Recurrent function for discovering 
        possible conformations.
        """
        if len(conformation) >= len(self.config)-1:
            self.conformations.append(conformation)
        else:
            for i in self.basis:
                #if not np.all(i == -1*conformation[-1]):
                if np.sum((i + conformation[-1])**2)>0:
                    new_point = points[-1] + i
                    if not np.any(np.sum(np.abs(new_point - points), axis = 1)==0): # and np.sum(new_point**2)>0:
                        new_points = points + [new_point]
                        new_conformation = conformation + [i]
                        self.advance_conformation(new_conformation, new_points)

        
        
    def set_conformation(self, conformation):
        self.conformation = conformation
        
    def energy(self):
        # compute self energy of polymer
        e = np.array([self._edict[i] for i in self.config])
        return np.sum(np.abs(e[1:]-e[:-1]))
       
        
        


class lattice():
    def __init__(self, Nx = 10, Ny = 10):
        
        self.lattice = -1*np.ones((Nx,Ny), dtype = int) # initially populated with water
        
        self._edict = {"W":-1, "H":0, "P" :1}
        
        self.conformations = []
    
    def validate_placement(self, i,j,conf):
        """
        Only sites with water are replaced
        """
        if self.lattice[i,j]!=-1:
            return False
        di,dj = i,j
        for c in conf:
            di, dj = di+c[0], dj+c[1]
            if self.lattice[di,dj]!=-1 or di<0 or dj<0:
                return False
        
        return True
            
            
        
        
        
    def find_all_possible_placements(self, polym, uniques = True):
        """
        Returns all (unique) possible conformations of polym
        that fit in the lattice
        """
        config = [self._edict[k] for k in polym.config]
        
        valid_conformations = [] 
        
        unique_lattices = []
        for i in range(self.lattice.shape[0]):
            for j in range(self.lattice.shape[1]):
                if self.lattice[i,j] == -1:
                    for ci in range(len(polym.conformations)):
                        c = polym.conformations[ci]
                        if self.validate_placement(i,j,c):
                            conf = [np.array([i,j])] + c
                            conf = np.cumsum(np.array(conf), axis = 0)
                            
                            
                            # determine if conformation is unique                            
                            conf_unique = True
                            for k in valid_conformations:
                                if np.sum((k-conf)**2)==0:
                                    conf_unique = False
                            
                            if conf_unique:
                                valid_conformations.append(conf) #, self.place_polymer_at(i,j,polym, c)])
                                #unique_lattices.append(lat)
        return valid_conformations
                    

    def place_polymer_at(self, i,j,polym, conformation = None):
        """
        Returns a lattice where polymer is placed
        at coordinates i,j in lattice
        If no conformation is specified the function
        will chose the first available conformation
        which fits.
        
        """
        assert(i<self.lattice.shape[0]), "invalid placement"
        assert(j<self.lattice.shape[1]), "invalid placement"
        
        config = [self._edict[k] for k in polym.config]
        
        lattice = self.lattice*1
        
        if conformation is None:
            placed = False
            for c in range(len(polym.conformations)):
                pc = polym.conformations[c]
                if self.validate_placement(i,j,pc):
                    placed = True
                    print(pc)
                    # place polymer
                    di,dj = i,j
                    lattice[di,dj] = self._edict[polym.config[0]]
                    for k in range(len(polym.conformations[c])):
                        di += polym.conformations[c][k][0]
                        dj += polym.conformations[c][k][1]
                        lattice[di,dj] = self._edict[polym.config[k+1]]
                        
            if not placed:
                print("Unable to fit polymer in lattice")
        else:
            if self.validate_placement(i,j,conformation):
                di,dj = i,j
                lattice[di,dj] = self._edict[polym.config[0]]
                for k in range(len(conformation)):
                    di += conformation[k][0]
                    dj += conformation[k][1]
                    lattice[di,dj] = self._edict[polym.config[k+1]]
                    
            else:
                print("Conformation does not fit in lattice")
        return lattice
        
    
    def energy(self, lattice = None):
        """
        Computes the energy of the lattice 
        (if no lattice provided, the energy of self.lattice is computed)
        """
        if lattice is None:
            lattice = self.lattice
        energy = 0
        #for i in range(lattice.shape[0]-1):
        #    for j in range(lattice.shape[1]-1):
        #        if np.abs(lattice[i,j] - lattice[i,j+1]) == 1:
        #            energy += 1
        #        if np.abs(lattice[i,j] - lattice[i+1,j]) == 1:
        #            energy += 1

        return np.sum(np.abs(lattice[1:,:]  - lattice[:-1,:])==1) + np.sum(np.abs(lattice[:, 1:]  - lattice[:, :-1])==1)
        #return energy


    
    



# Visualization


def show_lattice_placement(l, p, c):

    """
    Show how the lattice l looks when polymer p is placed 
    as defined by positions c
    """



    if type(c) is not list:
        c = [c]
        
    n = int(np.sqrt(len(c)) + 1)
    
    counter = 0
    
    plt.figure(figsize=(2*l.lattice.shape[0], 2*l.lattice.shape[1]))
    
    dx, dy = l.lattice.shape[1]+3, l.lattice.shape[0]+3
    
    unique_lattices = [] #list to hold unique lattice configurations
    
    
    
    
    
    
    for i in range(n):
        for j in range(n):
            if counter<len(c):

                #compute energy of polymer
                energy_polymer = p.energy()

                #compute energy of empty cavity
                energy_cavity = l.energy() 


                pts = np.array(np.meshgrid(np.arange(l.lattice.shape[1])+dx*i, np.arange(l.lattice.shape[0])+dy*j)).reshape(2,-1).T
                lat = l.lattice*1

                config = [l._edict[k] for k in p.config]

                lat[c[counter][:,0], c[counter][:, 1]] = config

                # compute energy of filled cavity
                energy_filled = l.energy(lat)

                lat = lat.ravel()

                lw = lat==-1
                lh = lat== 0
                lp = lat== 1

                plt.plot(pts[lw,1], pts[lw,0], "o", markersize = 5, color = (.3,.3,.8))
                plt.plot(pts[lh,1], pts[lh,0], "o", markersize = 5, color = (.9,.9,.3))
                plt.plot(pts[lp,1], pts[lp,0], "o", markersize = 5, color = (.8,.4,.3))
                plt.plot(pts[lw,1], pts[lw,0], "o", markersize = 7, color = (0,0,0), zorder = -1)
                plt.plot(pts[lh,1], pts[lh,0], "o", markersize = 7, color = (0,0,0), zorder = -1)
                plt.plot(pts[lp,1], pts[lp,0], "o", markersize = 7, color = (0,0,0), zorder = -1)
                plt.plot(c[counter][:,0]+dy*j, c[counter][:,1]+dx*i, "-", color = (0,0,0), zorder = -1, linewidth = 2)


                #print("Polymer self-energy    :", energy_polymer)
                #print("Energy of empty cavity :", energy_cavity)
                #print("Energy of filled cavity:", energy_filled)
                #print("Total energy.          :", energy_filled - energy_cavity - energy_polymer)
                plt.text(dy*(j+.25), dx*i-1, "$\epsilon$ = %i" % (energy_filled - energy_cavity - energy_polymer), ha = "center", va = "center", fontsize = 8)
                counter += 1

    plt.axis("off")
    plt.show()




def show_conformations(polym):
    """
    Show all possible conformations of polym(er)
    """

    n = int(np.sqrt(len(polym.conformations))+1)
    sep = 6.5
    plt.figure(figsize = (10,10))
    c = 0
    lp = len(polym.conformations)
    hi = np.array([i for i in polym.config])=="H"
    pi = np.array([i for i in polym.config])=="P"
    
    for i in range(n):
        for j in range(n):
            if c<lp:
                conf = [np.array([0,0])] + polym.conformations[c]
                conf = np.cumsum(np.array(conf), axis = 0)

                conf = conf - np.mean(conf, axis = 0)[None, :]

                plt.plot(conf[:,0] + i*sep, conf[:,1]-j*sep, "-", color = (0,0,0))

                plt.plot(conf[hi,0] + i*sep, conf[hi,1]-j*sep, "o", color = (.8,.3,0), markersize = 2)
                plt.plot(conf[pi,0] + i*sep, conf[pi,1]-j*sep, "o", color = (0 ,.3,.8), markersize = 2)
                c +=1

    plt.xlim(-sep, sep*n+1)
    plt.ylim(-sep*n, sep)
    plt.axis("off")
    plt.show()
    
    
def remove_rotational_redundance(conf, remove_reversed = False):
    """
    Remove rotational redundancies from the set of conformations conf
    """
    nonredundant_set = []  #nonredundant conformations to be added here
    
    for c in conf:
        nonredundant = True # if nonredundant, add to nonredundant_set
        cc = np.array(c)
        
        # check for redundancies
        for m in nonredundant_set:
            mc = np.array(m)
            for j in range(4):
                if np.sum((mc-cc)**2) < 1e-10:
                    nonredundant = False
                cc = cc.dot(np.array([[0,1],[-1,0]])) #rotate polymer 90 degrees
            
            if remove_reversed:
                cc = cc[::-1] # reverse polymer
                for j in range(4):
                    if np.sum((mc-cc)**2) < 1e-10:
                        nonredundant = False
                    cc = cc.dot(np.array([[0,1],[-1,0]]))

                
        if nonredundant:
            nonredundant_set.append(c)
    return nonredundant_set
    



class animated_system():
    def __init__(self, system = None, n_steps_per_vis = 5, interval = 1, phase_color = True):
        self.n_steps_per_vis = n_steps_per_vis
        self.system = system
        figsize = (6,6)
        

    
        plt.rcParams["figure.figsize"] = figsize

        self.phase_color = phase_color

        
        self.fig, self.ax = plt.subplots()    
        
        cc = np.random.uniform(0,1,(3, 6))
        cc[:,0] = np.array([0,0,0])
        
        cc = np.array([[0.        , 0.        , 0.        ],
                    [0.8834592 , 0.36962255, 0.21858202],
                    [0.64961546, 0.79727038, 0.55362479],
                    [0.22449319, 0.56457326, 0.60815318],
                    [0.75835695, 0.729311  , 0.54213821]]).T
        
        
        
        
        
        self.color = interp1d(np.linspace(0,len(self.system.n_bubbles), 5), cc)


        
        self.ani = FuncAnimation(self.fig, self.update, interval=interval, 
                                          init_func=self.setup_plot)#, blit=True,cache_frame_data=True)
        
        
    def update(self, j):
        for i in range(self.n_steps_per_vis):
            #self.system.lattice = np.random.randint(0,2,self.system.lattice.shape) #advance()
            self.system.advance()
        self.bubbles.set_color(self.color(self.system.lattice.ravel()).T)
        if self.phase_color:
            self.bubbles.set_color(self.color(self.system.identical_neighbors().ravel()).T)

        self.infotext.set_text("%.2f" % self.system.energy())
        
        return self.bubbles,





    def setup_plot(self):
        
        
        if len(self.system.lattice.shape)==2:
            x,y = np.array(np.meshgrid(np.arange(self.system.lattice.shape[0]), np.arange(self.system.lattice.shape[1]))).reshape(2, -1)

            s = 50000/self.system.lattice.shape[0]**2
            self.bubbles = self.ax.scatter(x, y, c=self.color(self.system.lattice.ravel()).T, s=s, marker = "8")
            self.ax.axis("off")
            self.infotext = self.ax.text(1,-2,"test", ha = "left", va = "center")
            plt.xlim(-1, self.system.lattice.shape[0]+1)
            plt.xlim(-1, self.system.lattice.shape[1]+1)
        if len(self.system.lattice.shape)==3:
            """
            3D Lattice
            """
            x,y, z = np.array(np.meshgrid(np.arange(self.system.lattice.shape[0]), np.arange(self.system.lattice.shape[1]),np.arange(self.system.lattice.shape[2])), dtype = float).reshape(3, -1)

            y += .1*z
            x += .2*z
            
            
            
            s = 20000/(self.system.lattice.shape[0]**2 + z**2)
            self.bubbles = self.ax.scatter(x, y, c=self.color(self.system.lattice.ravel()).T, s=s, marker = "8", alpha = .1)
            self.ax.axis("off")
            self.infotext = self.ax.text(1,-2,"test", ha = "left", va = "center")
            plt.xlim(-1, self.system.lattice.shape[0]+1)
            plt.xlim(-1, self.system.lattice.shape[1]+1)
        return self.bubbles,
    
class latticebox():
    """
    ## Latticebox

    The latticebox extension models a lattice of monomers where the only interactions occur between neighboring sites. The Hamiltonian of the system is

    \begin{equation}
    H = \frac{1}{2}\sum_{I \in \Omega} \Big{(} \sum_{J \in \Omega_I} w_{\sigma(I)\sigma(J)} \Big{)},
    \end{equation}

    where $I$ are sites on the lattice $\Omega$, $J$ are sites in the neighborhood of $I$ ( $\Omega_I$ ), $\sigma(I) \in \{A,B,...\}$ yields the species occupying site $I$ and the matrix $w$ contains the interaction parameters $w_{AA}, w_{AB}, w_{BB}$ and so on.

    """
    
    def __init__(self, n_bubbles, size, interaction, T = 1.0, randomize = True, n_swaps_per_advance = 1, neighbor_swap = False):
        """
        ## Initialize the lattice
        ---

        Keyword arguments:
    
        n_bubbles        -- array with number of bubbles of varying species (must sum to number of lattice points)
        size             -- size of lattice (x,y) direction
        interaction      -- interaction matrix of dimension len(n_bubbles) x  len(n_bubbles)

        ## Example usage (in a notebook):
        ---

        import bubblebox as bb

        %matplotlib notebook

        system = latticebox(n_bubbles = [500,500], size = (10,10), interactions = np.eye(2)) #initialize 10 by 10 closed box containing 500 bubbles of type A, 500 bubbles of type B

        system.run() #run simulation interactively 

        """
        
        self.size = size
        self.lattice = np.zeros(size, dtype = int)
        
        self.z = 2*len(self.lattice.shape) #number of nearest neighbors
        
        assert(np.sum(n_bubbles) == self.lattice.size), "Inconsistent number of species and lattice"
        self.n_bubbles = n_bubbles
        
        
        counter = 0
        for m in range(len(self.n_bubbles)):
            dcounter = counter + self.n_bubbles[m]
            self.lattice.flat[counter:dcounter] = m
            counter = dcounter
            
        # randomize lattice
        if randomize:
            self.lattice = self.lattice.ravel()
            np.random.shuffle(self.lattice.ravel())
            self.lattice = self.lattice.reshape(size)
            
        self.interaction = interaction
        self.kT = 1*T
        self.n_swaps_per_advance = n_swaps_per_advance
        self.iterations = 0
        self.sample_threshold = 1000 #start sampling after this number of iterations
        
        # for sampling
        self.order = 0
        self.samples = 0
        self.s_energy = 0
        self.neighbor_swap = neighbor_swap
        self.E = self.energy()
        
    def energy(self, lattice = None):
        """
        compute the energy of the entire lattice
        """
        if lattice is None:
            lattice = self.lattice
        energy = 0
        for i in range(len(lattice.shape)):
            energy += self.interaction[np.roll(lattice, 1, axis = i).ravel(), lattice.ravel()].sum()
        return energy


    def identical_neighbors(self):
        # compute absolute differences between neighboring sites
        lnn = np.zeros_like(self.lattice)
        for i in range(len(self.lattice.shape)):
            lnn+= 1 - np.abs(self.lattice - np.roll(self.lattice,  1, axis = i))
            lnn+= 1 - np.abs(self.lattice - np.roll(self.lattice, -1, axis = i))
        return lnn/(2*len(self.lattice.shape))

    
    
    
    
    def energy_at_site(self, i, li = None):
        """
        Compute energy at site
        """
        
        dl = np.zeros(len(self.lattice.shape), dtype = int)
        dl[0] = 1
        
        energy = np.zeros((2, self.n_swaps_per_advance))
        
        
        for m in range(len(self.lattice.shape)):
            #print(self.lattice[tuple(i)], i)
            if li is None:
                li = self.lattice[tuple(i)]
            #print("li:", li)
            #print("i :", i)
            #print(self.interaction[[self.lattice[tuple(i)], li], self.lattice[tuple((i+np.roll(dl, m))%self.lattice.shape)]])
            #print("e:", i, li)
            #energy += self.interaction[[self.lattice[tuple(i)], li], self.lattice[tuple((i+np.roll(dl, m))%self.lattice.shape)]]
            #energy += self.interaction[[self.lattice[tuple(i)], li], self.lattice[tuple((i-np.roll(dl, m))%self.lattice.shape)]]

            #print(self.interaction[[self.lattice[i], li], self.lattice[tuple((i+np.roll(dl, m))%self.lattice.shape)]])
            energy += self.interaction[[self.lattice[i], li], self.lattice[tuple((i+np.roll(dl, m))%self.lattice.shape)]]
            energy += self.interaction[[self.lattice[i], li], self.lattice[tuple((i-np.roll(dl, m))%self.lattice.shape)]]

            
        return energy
    
    def sample(self):
        """update measurements"""
        
        
        self.s_energy = (self.s_energy*self.samples + self.energy())/(self.samples + 1)
        
        self.samples += 1
        return self.energy
        
    
    
    
    def evolve(self, N):
        for i in range(N):
            self.advance()
            
    def advance(self):
        """
        One Metropolis-Hastings step
        """

        # pick some pairs at random
        i,j = np.random.choice(self.lattice.size, 2*self.n_swaps_per_advance, replace = False).reshape(2,-1)

        new_lattice = np.zeros_like(self.lattice)
        new_lattice[:] = self.lattice[:]

        #perform swap
        new_lattice.flat[i] = self.lattice.flat[j]
        new_lattice.flat[j] = self.lattice.flat[i]

        E = self.energy(lattice = new_lattice)

        dE = E-self.E
        
        if np.exp(-dE/self.kT)>np.random.uniform(0,1):
            # accept move
            self.lattice = new_lattice
            self.E = E

        #else, reject move (do nothing)





        
    def advance_detailed(self):
        """
        One Metropolis-Hastings step
        """
        
        
        # if only neighbors are allowed to swap
        if self.neighbor_swap:
            i = np.random.choice(self.lattice.size, self.n_swaps_per_advance, replace = False)
            ir = np.array( np.unravel_index(i, self.lattice.shape), dtype = int)
            jr = (ir + (np.array([[1,0], [0,1], [-1,0], [0,-1]])[np.random.randint(0,4,self.n_swaps_per_advance)]).T)%np.array(self.lattice.shape)[:,None]
            
            j = np.ravel_multi_index(jr, self.lattice.shape)
            
            
            
        else:
            # pick self.n_swaps_per_advance number of pairs
            i,j = np.random.choice(self.lattice.size, 2*self.n_swaps_per_advance, replace = False).reshape(2,-1)

        
        li = self.lattice.flat[i]
        lj = self.lattice.flat[j]
        
        # remove all cases where li == lj
        
        
        
        # local energies before and after swaps
        energies = np.zeros((2, self.n_swaps_per_advance))
        
        
        
        if np.any(li!=lj):
            ir = np.array( np.unravel_index(i, self.lattice.shape), dtype = int)
            jr = np.array( np.unravel_index(j, self.lattice.shape), dtype = int )
            
            ax = np.zeros(len(self.lattice.shape), dtype = int)
            ax[0] = 1
            
            for k in range(len(self.lattice.shape)):
                # current energy in i-points
                dr = np.roll(ax, k, axis = 0)[:, None]
                neighbors_plus = (ir + dr)%np.array(self.lattice.shape)[:, None]
                neighbors_min  = (ir - dr)%np.array(self.lattice.shape)[:, None]
                
                p_p = self.lattice[tuple(neighbors_plus)]
                m_p = self.lattice[tuple(neighbors_min)]
                
                energies[0]+=self.interaction[li, p_p ]
                energies[0]+=self.interaction[li, m_p ]
                
                #swapped energy in i-points
                energies[1]+=self.interaction[lj, p_p ]
                energies[1]+=self.interaction[lj, m_p ]
                
                # current energy in j-points
                neighbors_plus = (jr + dr)%np.array(self.lattice.shape)[:, None]
                neighbors_min  = (jr - dr)%np.array(self.lattice.shape)[:, None]
                p_p = self.lattice[tuple(neighbors_plus)]
                m_p = self.lattice[tuple(neighbors_min)]
                
                energies[0]+=self.interaction[lj, p_p ]
                energies[0]+=self.interaction[lj, m_p ]
                
                #swapped energy in j-points
                energies[1]+=self.interaction[li, p_p ]
                energies[1]+=self.interaction[li, m_p ]
            
            
            # allow swaps depending on energy difference
            
            
            dE = energies[1]-energies[0]
            
            
            # if energy change is negative, accept move
            #swaps_for_sure = dE<0
            #self.lattice.flat[i[swaps_for_sure]] = lj[swaps_for_sure]
            #self.lattice.flat[j[swaps_for_sure]] = li[swaps_for_sure]
            
            
            # if not, accept with a probability depending on the energy difference
            swaps = np.exp(-dE/self.kT)>np.random.uniform(0,1,self.n_swaps_per_advance)
            
            swaps[li == lj] = False
            
            #print(dE[li == lj])
            #print(dE[li != lj])
            #print(np.exp(-dE/self.kT))
            #swaps[swaps_for_sure] = False #these have already been swapped
            
            self.lattice.flat[i[swaps]] = lj[swaps]
            self.lattice.flat[j[swaps]] = li[swaps]
        self.iterations += 1
    
    def run(self, phase_color = False, n_steps_per_vis = 100):
        self.run_system = animated_system(system = self, n_steps_per_vis=n_steps_per_vis, interval = 1, phase_color = phase_color)
        plt.show()
            

    def view(self):
        self.pos = np.array(np.meshgrid(*[np.arange(i) for i in self.lattice.shape])).reshape(len(self.lattice.shape),-1)
        self.pos = self.pos - .5*np.array(self.lattice.shape, dtype = int)[:, None]
        self.pos += .5
        
        # set masses 
        self.masses = np.ones(len(self.pos))
        
        self.mview = ev.LatticeView(self)
        return self.mview
    
    def update_view(self):
        self.mview.state = self.lattice.ravel().tolist()
        

            

# equilibration routine

def equilibrate(l, n_thresh = 1e5, dn = 2000, thresh = 2e-5):
    """
    Equilibrate a lattice system
    
    Aaguments
    n_thresh = number of iterations before we start
               accumulating energy
    dn       = number of iterations between each sample
    thresh   = assume equilibrated when relative change in energy
               falls below this threshold
    """
    
    # first, move system towards equilibrium 
    # before we start sampling
    l.evolve(int(n_thresh))
    
    # then start sampling the energy
    n_samples = 0 #number of samples
    energy_ac = 0 #accumulated energy
    for i in range(200):
        l.evolve(dn)
        ei = l.energy()/l.lattice.size #compute the lattice energy per particle
        
        energy_ac_prev = energy_ac #retain previous energy
        energy_ac = (energy_ac*n_samples + ei)/(n_samples +1) #update accumulated mean energy
        n_samples += 1 #update number of samples
        
        # compute relative change
        relative_change_in_energy = (energy_ac-energy_ac_prev)/energy_ac 
        
        if np.abs(relative_change_in_energy)<thresh:
            # if relative change in energy is below thresh, 
            # return the equilibrated lattice
            #print("Equilibrated lattice with %i samples, final relative change in energy: %.10e" % (n_samples, relative_change_in_energy))
            return l
        
    print("Failed to equilibrate")
    return l


# determine order

def order_2d(l):
    """
    for each site on the lattice,
    measure the number of differing neighbors
    
    should be used after equilibration
    """
    
    # compute absolute differences between neighboring sites
    lnn = 1 - np.abs(l.lattice - np.roll(l.lattice,  1, axis = 0))
    lnn+= 1 - np.abs(l.lattice - np.roll(l.lattice, -1, axis = 0))
    lnn+= 1 - np.abs(l.lattice - np.roll(l.lattice,  1, axis = 1))
    lnn+= 1 - np.abs(l.lattice - np.roll(l.lattice, -1, axis = 1))
    
    # for species A
    lnn_A = lnn[l.lattice==0]
    lnn_A_bins = np.bincount(lnn_A)
    # should be 1 if A perfectly separated
    # should be 0.5 if perfectly mixed
    
    
    # for species B
    lnn_B = lnn[l.lattice==1]
    lnn_B_bins = np.bincount(lnn_B)
    
    """
    #summarize results
    for i in range(len(lnn_A_bins)):
        print("A fraction of %.3f of species A have %i neighbors different than itself " % (lnn_A_bins[i]/l.number_of_species[0], i))
    
    print(" ")
        
    for i in range(len(lnn_B_bins)):
        print("A fraction of %.3f of species B have %i neighbors different than itself " % (lnn_B_bins[i]/l.number_of_species[1], i))
    """
    
    # if we only count sites surrounded by at least three molecules 
    # identical to the one occupying the site as being in phase A or B, 
    # and the remaining ones in liquid phase, we can return the following estimates
    # notice, however, that this is a matter of interpretation
    n_in_phase_A = lnn_A_bins[:1].sum()
    n_in_phase_B = lnn_B_bins[:1].sum()
    n_in_mixed_phase = lnn_A_bins[1:].sum() + lnn_B_bins[1:].sum()
    
    return lnn_A.mean()/4.0


def order_nd(l):
    """
    for each site on the lattice,
    measure the number of differing neighbors
    
    should be used after equilibration
    """
    
    # compute absolute differences between neighboring sites
    lnn = np.zeros_like(l.lattice)
    for i in range(len(l.lattice.shape)):
        lnn+= 1 - np.abs(l.lattice - np.roll(l.lattice,  1, axis = i))
        lnn+= 1 - np.abs(l.lattice - np.roll(l.lattice, -1, axis = i))

    
    # for species A
    lnn_A = lnn #[l.lattice==0]
    lnn_A_bins = np.bincount(lnn_A.ravel())
    
    #print(lnn_A_bins/np.sum(l.lattice==0))
    # should be 1 if A perfectly separated
    # should be 0.5 if perfectly mixed
    
    #return lnn_A.mean()/(2*nd)
    return lnn_A_bins/np.sum(lnn_A_bins), np.mean(lnn)
    

def initialize_lattice(T = .2, w_aa = -.5, w_ab = -.1, w_bb = -.6, chi_b = .5, n=100, n_swaps_per_advance = 10):
    # initialize a system
    I = np.array([[w_aa, w_ab],[w_ab, w_bb]]) #interaction matrix
    
    # an array containing the number of different kinds of particles
    number_of_species = np.array([n**2 - int(chi_b*n**2), int(chi_b*n**2)])


    # set up a system
    l = latticebox(dim = (n,n), 
                   number_of_species = number_of_species, 
                   interaction = I, 
                   n_swaps_per_advance = n_swaps_per_advance,
                   T = T)
    return l
  
            
        
def visualize(l, T = 0):
    fig, ax = plt.subplots()
    
    cc = np.random.uniform(0,1,(3, 6))
    cc[:,0] = np.array([0,0,0])

    cc = np.array([[0.        , 0.        , 0.        ],
                [0.8834592 , 0.36962255, 0.21858202],
                [0.64961546, 0.79727038, 0.55362479],
                [0.22449319, 0.56457326, 0.60815318],
                [0.75835695, 0.729311  , 0.54213821]]).T





    color = interp1d(np.linspace(0,3, 5), cc)

    x,y = np.array(np.meshgrid(np.arange(l.lattice.shape[0]), np.arange(l.lattice.shape[1]))).reshape(2, -1)

    s = 50000/l.lattice.shape[0]**2
    bubbles = ax.scatter(x, y, c=color(l.lattice.ravel()).T, s=s, marker = "8")
    ax.axis("off")
    plt.xlim(-1, l.lattice.shape[0]+1)
    plt.xlim(-1, l.lattice.shape[1]+1)
    #plt.show()
    plt.savefig("lm_at_temp_%.2f.png" % T)
    plt.close()
        

def initialize_lattice_3d(T = .2, w_aa = -.5, w_ab = -.1, w_bb = -.6, chi_b = .5, n=20, n_swaps_per_advance = 10):
    # initialize a system
    I = np.array([[w_aa, w_ab],[w_ab, w_bb]]) #interaction matrix
    
    # an array containing the number of different kinds of particles
    number_of_species = np.array([n**3 - int(chi_b*n**3), int(chi_b*n**3)])


    # set up a 3D system
    l = latticebox(dim = (n,n,n), 
                   number_of_species = number_of_species, 
                   interaction = I, 
                   n_swaps_per_advance = n_swaps_per_advance,
                   T = T, 
                   randomize = False)
    l.advance = l.advance_detailed
    
    return l

def initialize_lattice(T = .2, w_aa = -.8, w_ab = -.1, w_bb = -.8, chi_b = .5, n=60, n_swaps_per_advance = 10):
    # initialize a system
    I = np.array([[w_aa, w_ab],[w_ab, w_bb]]) #interaction matrix
    
    # an array containing the number of different kinds of particles
    number_of_species = np.array([n**2 - int(chi_b*n**2), int(chi_b*n**2)])


    # set up a system
    l = latticebox(dim = (n,n), 
                   number_of_species = number_of_species, 
                   interaction = I, 
                   n_swaps_per_advance = n_swaps_per_advance,
                   T = T, randomize = False)
    
    l.advance = l.advance_detailed
    
    
    return l

    
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

class bindingbox():
    def __init__(self, number_of_polymers, n_polymer_sites, number_of_ligands, number_of_solvent_molecules, interaction_matrix, initialize_in_ground_state = False, kT = 0.4):
        """
        Initialize model
        
        - explain how to initialize energy
        """
        
        self.nl = number_of_ligands 
        self.np = number_of_polymers # number of polymers
        self.lp = n_polymer_sites # number of binding sites at each polymer
        self.interaction_matrix = interaction_matrix # interaction matrix for each polymer
        self.lattice_size = number_of_ligands+number_of_polymers*n_polymer_sites+number_of_solvent_molecules
        
        self.polymer_partition = self.np*self.lp
        self.particle_partition = self.polymer_partition + number_of_ligands 
        
        
        # Temperature
        self.kT = kT
        

        self.lattice = np.zeros(self.polymer_partition+self.nl+1, dtype = bool)
        self.lattice[self.polymer_partition:] = True # free ligands
        self.lattice[-1] = False # a spot reserved for the solvent
        
        if initialize_in_ground_state:
            self.lattice[:] = False
            self.lattice[:self.nl] = True
            n_bound_ligands = np.sum(self.lattice[:self.polymer_partition])
            
            self.lattice[self.polymer_partition:self.particle_partition-n_bound_ligands] = True
            self.lattice[self.particle_partition-n_bound_ligands:] = False
        
        self.energy = self.compute_energy()
        
    def compute_average_populations(self):
        """
        
        """
        lattice = self.lattice[:self.polymer_partition].reshape(self.np, self.lp)
        occupancy = np.sum(lattice, axis = 1)
        
        oc = np.bincount(occupancy)/self.np
        occu = np.zeros(3)
        occu[:len(oc)] = oc
        
        return occu
        
        
    def compute_occupation(self):
        """
        Compute mean nuumber of occupied sites
        per polymer
        """
        lattice = self.lattice[:self.polymer_partition].reshape(self.np, self.lp)
        return np.mean(np.sum(lattice, axis = 1))
    
    def compute_sitewise_occupation(self):
        """
        Compute mean nuumber of occupied sites
        per polymer
        """
        lattice = self.lattice[:self.polymer_partition].reshape(self.np, self.lp)
        return np.mean(lattice, axis = 0)
    
    
    def compute_energy(self):
        """
        Compute energy of lattice
        """
        e = 0
        lattice = self.lattice[:self.polymer_partition].reshape(self.np, self.lp)
        for i in range(self.interaction_matrix.shape[0]):
            ei = self.interaction_matrix[i]
            ni = ei.shape[0]
            #print(self.interaction_matrix[i][None, :-i+1])
            #print(np.sum((lattice*np.roll(lattice, i, axis = 0))[:-i+1]))
            #print(self.interaction_matrix[i][None,:-i])
                  
            
            e += np.sum((lattice*np.roll(lattice, i, axis = 1))[:,i:ni]*ei[None,i:ni])
        return e
    
    
    def compute_energy_(self):
        """
        Compute energy of lattice
        """
        e = 0
        lattice = self.lattice[:self.polymer_partition].reshape(self.np, self.lp)
        for i in range(self.interaction_matrix.shape[0]):
            
            e += np.sum(lattice*np.roll(lattice, i, axis = 0))*self.interaction_matrix[i]
        return e
        
    
    
    def advance(self):
        """
        Do one Monte Carlo step
        """
        
        # pick to random different sites 
        #P = np.random.choice(self.lattice_size, 2, replace = False)
        P = np.random.randint(0,self.lattice_size, 2)
        P[P>self.lattice.shape[0]-1] = self.lattice.shape[0] - 1
        if P[0]!=P[1]:
            #print(P)

            # all 
            

            #print(P, self.lattice)

            # check if sites are occupied
            P0_occupied = self.lattice[P[0]]
            P1_occupied = self.lattice[P[1]]

            if P0_occupied!=P1_occupied:
                # if swap involves state change, perform swap

                # set occupation following change
                self.lattice[P[0]] = P1_occupied
                self.lattice[P[1]] = P0_occupied
                
                


                # compute new energy, and energy difference
                new_energy = self.compute_energy()

                energy_change = new_energy - self.energy
                
                # Monte Carlo step 
                if np.exp(-energy_change/self.kT)>np.random.uniform(0,1):
                    # accept move
                    
                    # update energy
                    self.energy = new_energy*1

                    # resolve number of bound ligands
                    n_bound_ligands = np.sum(self.lattice[:self.polymer_partition])
                    
                    self.lattice[self.polymer_partition:self.particle_partition-n_bound_ligands] = True
                    self.lattice[self.particle_partition-n_bound_ligands:] = False




                else:
                    # reject step and revert changes
                    self.lattice[P[0]] = P0_occupied
                    self.lattice[P[1]] = P1_occupied
                
    def evolve(self, Nt):
        """
        Perform Nt Monte Carlo steps
        """
        for i in range(Nt):
            self.advance()


def fit_langmuir_model(concentration, occupancy, A = 2):
    """
    fit to a Langmuir adsorption model
    
    \begin{equation}
        \theta(X) = A * \frac{KX}{1 + KX},
    \end{equation}
    
    where X is consentration and K is the equilibrium constant
    
    function returns K
    """
    
    R = lambda k, c=concentration, o = occupancy, A = A : (A*k*c/(1+k*c) - o)**2 #residual function
    
    #extract and return K
    return least_squares(R, np.array([1]) ).x[0]
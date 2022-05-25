import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation

class isingbox():
    def __init__(self, N, kT, flipratio = 0.1):
        """
        A 2D (N by N) Ising model
        
        Arguments
        ---
        N          Lattice dimensions
        kT         Boltsmann factor times the temperature 
        flipratio  The ratio of spins to flip at each iteration
        
        Methods
        ---
        reset(kT)          reset the lattice to all spins aligned, and set kT
        energy_per_site()  compute the energy per site
        
        
        Example usage
        ---
        
        I = ising(10, kT = 1.0) # an Ising lattice with 10x10 spins, kT = 1.0
        
        I.advance() # do one Monte Carlo integration
        
        I.evolve(10) # do 10 Monte Carlo integrations
        
        energy = I.compute_energy() # compute energy
        
        mangetization = I.compute_magnetization() # compute magnetization
        
        energy, magnetization = I.sample(100, 1000) # collect 100 samples, separated by 1000 MC-iterations
        
        
        
        
        """
        
        self.kT = kT
        self.N = int(N)
        self.N2 = int(N**2)
        self.Z = np.ones(self.N**2, dtype = int)
        
        #perhaps not start out in this state?
        #self.Z[np.random.choice(self.N2, int(self.N2/2), replace = False)] = -1

        self.Z_ind = np.arange(self.N2)
        self.n = int(flipratio*self.N2)
        self.flipratio = flipratio
        
        self.env = self.energy_env_torus() #_torus(arange(self.N2))
        
        self.n_logdist = 1000000
        self.logdist = self.kT*np.log(np.random.uniform(0,1,self.n_logdist))/2.0
        
        
        self.t = np.zeros(10, dtype = float)
        
    def reset(self, kT):
        """
        Reset the system to a non-magnetized state ( |M| = 0 )
        
        and set kT to the specified value
        
        """
        self.kT = kT
        self.Z = np.ones(int(self.N**2), dtype = int)
        #self.Z[np.random.choice(int(self.N2), int(self.N2/2), replace = False)] = -1
        self.Z_ind = np.arange(self.N2)
        self.n = int(self.flipratio*self.N2)
        self.env = self.energy_env_torus()
        
    def energy_per_site(self,Z):
        """Compute sitewise energy (returns a lattice)"""
        return (Z[2:,1:-1]+Z[:-2,1:-1] +Z[1:-1,2:]+Z[1:-1,:-2]) #[1:-1,1:-1]
    
    
    
    def log_uniform(self, n):
        """A logarithmic random distribution"""
        return self.logdist[np.random.randint(0,self.n_logdist,n)]
    
    def energy_env_torus(self):
        """Compute energy per site for torus PBC"""
        z = self.Z.reshape((self.N,self.N))
        return (np.roll(z, 1,0) + np.roll(z,-1,0) + np.roll(z,1,1) + np.roll(z,-1,1)).ravel()
        
    def update_env(self, signs, z_ind):
        """update energy environment with a change in Z[z_ind]"""
        
        self.env[z_ind-1] += signs
        self.env[(z_ind+1)%self.N2] += signs
        self.env[z_ind-self.N] += signs
        self.env[(z_ind+self.N)%self.N2] += signs
        
     
        
    def advance(self):
        """Do one Monte Carlo step"""
        
        sel = np.random.choice(self.N2, self.n, replace = False) #bottleneck
        
        
        dz = self.Z[sel]
        denv = self.env[sel]
        
        dE = dz*denv
        
        flips = -dE>self.log_uniform(self.n)
        dz[flips] *= -1            
        self.Z[sel] = dz
        
        self.update_env(2*dz[flips], sel[flips]) #bottleneck
        
   
        
    def evolve(self, n_steps):
        """Do n_steps Monte Carlo steps"""
        
        for i in range(n_steps):
            self.advance()
    
    def compute_magnetization(self):
        """Compute magnetization"""
        return np.sum(self.Z)
        
    def compute_energy(self):
        """Compute total energy""" 
        return -np.sum(self.Z*self.env)/2.0
        
        
    def sample(self, n_samples, n_separations):
        """
        Collect n_samples separated by n_separations Monte Carlo steps

        Returns 
        energies_per_site, specific heat, absolute_magnetization_per_site
        """

        energy        = np.zeros(n_samples)
        magnetization = np.zeros(n_samples)

        energy[0] = self.compute_energy()
        magnetization[0] = self.compute_magnetization()
        for i in range(1, n_samples):
            self.evolve(n_separations)

            energy[i] = self.compute_energy()
            magnetization[i] = self.compute_magnetization()

        return np.mean(energy)/self.N2, np.var(energy)/(self.N2*self.kT**2), np.mean(magnetization)/self.N2
    
    def run(self, phase_color = False, n_steps_per_vis = 100):
        """Run simulation interactively"""
        self.run_system = animated_system(system = self, n_steps_per_vis=n_steps_per_vis, interval = 1, phase_color = phase_color)
        plt.show()

class animated_system():
    """
    Support class for animations
    """
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
        
        
        
        
        
        self.color = interp1d(np.linspace(-1,1, 5), cc)


        
        self.ani = FuncAnimation(self.fig, self.update, interval=interval, 
                                          init_func=self.setup_plot)#, blit=True,cache_frame_data=True)
        
        
    def update(self, j):
        for i in range(self.n_steps_per_vis):
            #self.system.lattice = np.random.randint(0,2,self.system.lattice.shape) #advance()
            self.system.advance()
        self.bubbles.set_color(self.color(self.system.Z).T)
        if self.phase_color:
            self.bubbles.set_color(self.color(self.system.identical_neighbors().ravel()).T)

        #self.infotext.set_text("%.2f" % self.system.energy())
        
        return self.bubbles,





    def setup_plot(self):
        
        
        #if len(self.system.lattice.shape)==2:
        x,y = np.array(np.meshgrid(np.arange(self.system.N), np.arange(self.system.N))).reshape(2, -1)

        s = 50000/self.system.N2
        self.bubbles = self.ax.scatter(x, y, c=self.color(self.system.Z.ravel()).T, s=s, marker = "8")
        self.ax.axis("off")
        self.infotext = self.ax.text(1,-2,"test", ha = "left", va = "center")
        plt.xlim(-1, self.system.N+1)
        plt.xlim(-1, self.system.N+1)
        
        return self.bubbles,

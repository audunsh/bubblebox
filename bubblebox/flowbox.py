import numpy as np

class flowbox():
    """
    prototype (unfinished) lattice-boltzmann solver
    author: Audun Skau Hansen
    """
    def __init__(self, size = [100,500]):
        self.size = np.array(size)
        self.tau = 0.6
        self.rho0 = 100.0
        # create some interesting boundary masking

        Nx = 100
        Ny = Nx*size[1]/size[0]
        X = np.linspace(-Ny,Ny,size[1])
        Y = np.linspace(-Nx,Nx,size[0])

        X,Y=np.meshgrid(X,Y)
        #print(X.shape, lattice.shape)


        #self.bounds = Y*np.exp(-.01*(.1*(X+130.0)**2 + Y**2) ) > 0.0001
        self.bounds = (.2*(X+30)**2 + (Y-60)**2)**.5  < 10
        #self.bounds[((X+x0)**2 + (Y+y0)**2)**.5  < 10] = True
        for i in range(10):
            x0, y0 = np.random.uniform(-100,100,2)
            self.bounds[((X+x0)**2 + (Y+y0)**2)**.5  < 10] = True
        
        


        self.nd = len(size) # number of dimensions

        shp = list(size) + [9]

        self.lattice = np.ones(list(size) + [9], dtype = float)  + .5*np.random.uniform(-1,1,shp)

        self.lattice[:,:,7] +=  (1+0.2*np.cos(.05*np.pi*X/Nx*4.0))
        rho = np.sum(self.lattice,-1)
        for i in range(9):
            self.lattice[:,:,i] *= self.rho0 / rho

        ni = np.arange(-1,2)

        self.c = np.array(np.meshgrid(*[ni for i in range(self.nd)])).reshape(self.nd, -1).T

        self.cT = self.c.T

        #c2 = (np.arange(len(c))[:, None]*np.ones((len(c), len(c)))).T[np.all(c[:,None]==C[None,:], axis = 2)]
        #c2 = np.array(c2, dtype = int)


        c_i = (np.arange(len(self.c))[:, None]*np.ones((len(self.c), len(self.c)))).T[np.all(self.c[:,None]==-1*self.c[None,:], axis = 2)]
        self.c_i = np.array(c_i, dtype = int)


        weight_selection = np.array([4/9.0, 1/9.0, 1/36.0])

        self.weights = weight_selection[np.sum(np.abs(self.c), axis = 1)]

        self.weights *= np.sum(self.weights)**-1


        self.screen = []
        self.indxes = []
        for i in range(len(self.c)):
            self.screen.append(np.array(self.c[i] != 0, dtype = bool))
            self.indxes.append(np.arange(self.nd)[self.c[i] != 0])
    def advance(self, compute_vorticity = False):
        # collision step
        for i in range(len(self.c)):
            self.lattice[:,:,i]=np.roll(self.lattice[:,:,i], shift = self.c[i][self.screen[i]], axis = self.indxes[i])


        boundary_values = self.lattice[self.bounds, :]
        boundary_values = boundary_values[:, self.c_i]

        # compute density 
        rho = np.sum(self.lattice, axis = -1)
        u = self.lattice.dot(self.c)/rho[:, :, None]

        uc = np.dot(u,self.cT)

        lattice_eq = rho[:,:, None]*self.weights[None, None,:]*(3*uc + 9*uc**2/2.0 - 3*np.sum(u**2, axis = -1)[:, :, None]/2.0 + 1)

        self.lattice +=  (lattice_eq-self.lattice)/self.tau

        self.lattice[self.bounds,:] = boundary_values
        
        if compute_vorticity:
            self.vorticity = self.get_vorticity(rho, u)
            
    def evolve(self, nt):
        for i in range(nt):
            self.advance()
        
    def get_vorticity(self, rho = None, u = None):
        """
        Computes a (generalized) curl
        """
        if rho is None:
            rho = np.sum(self.lattice, axis = -1)
            
        if u is None:
            u = np.sum(self.lattice[:, :, :, None]*self.c[None, None, :], axis = 2)/rho[:, :, None]
        u[self.bounds,:] = 0

        #self.vorticity = np.zeros(size, dtype = float)

        
        vorticity = (np.roll(u[:,:,0], -1, axis=0) - np.roll(u[:,:,0], 1, axis=0)) - (np.roll(u[:,:,1], -1, axis=1) - np.roll(u[:,:,1], 1, axis=1))

        vorticity[self.bounds] = vorticity.max()
        
        return vorticity
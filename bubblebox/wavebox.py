import numpy as np


class wavebox():
    """
    Prototype (unfinished) wave-equation solver
    author: Audun Skau Hansen, 2022
    """
    def __init__(self, size = (1,1), resolution = (100,100), c = 0.8, dt = 0.005, boundaries = None):
        self.u = np.zeros(resolution, dtype = float)
        self.up = self.u*1
        self.upp = self.u*1
        
        self.nd = len(size)
        self.size = size
        self.resolution = resolution
        self.c = c
        self.t = 0
        self.dt = dt
        
        self.x, self.dx = [], []
        for i in range(self.nd):
            self.x.append(np.linspace(-1, 1, resolution[i])*self.size[i])
            self.dx.append(self.x[-1][1]-self.x[-1][0])
            
        
            
        # initialize solver
        
        self.u = self.initial_step()
        self.up = self.u*0.0
        
        self.boundaries = boundaries
        if boundaries is not None:
            self.boundaries, self.boundary_map = self.map_boundaries(boundaries)
            
    def source(self, t):
        return 0
        
    def compute_right_hand_side(self):
        rhs = 0
        for i in range(self.nd):
            rhs += (np.roll(self.u, shift = 1, axis = i) + np.roll(self.u, shift = -1, axis = i) - 2*self.u)/self.dx[i]**2
        
        return self.c**2*rhs + self.source(self.t)
    
    def initial_step(self):
        return .5*self.compute_right_hand_side()*self.dt**2 + self.u 
    
    def advance(self, dt = None):
        if dt is None:
            dt = self.dt
            
        self.upp = self.up*1
        self.up = self.u*1
        
        self.u = self.compute_right_hand_side()*dt**2 + 2*self.up - self.upp
        
        self.t += dt
        
        #self.impose_neumann_boundaries()
        #self.impose_dirichlet_boundaries()
        
        self.impose_boundaries()
    
    def evolve(self, nt):
        for i in range(nt):
            self.advance()
            
    def map_boundaries(self, boundaries):
        bm = np.zeros(boundaries.shape, dtype = int)-1

        for i in range(len(boundaries.shape)):
            bi = (np.roll(np.array(boundaries, dtype = int), -1, axis = i)-np.array(boundaries, dtype = int))<0

            bm[bi] = np.roll(np.arange(boundaries.size).reshape(boundaries.shape), -1, axis = i)[bi]

            bi = (np.roll(np.array(boundaries, dtype = int), 1, axis = i)-np.array(boundaries, dtype = int))<0

            bm[bi] = np.roll(np.arange(boundaries.size).reshape(boundaries.shape), 1, axis = i)[bi]


        
        return np.arange(bm.size)[bm.ravel()>-1], bm[bm>-1].ravel()
            
    def impose_dirichlet_boundaries(self, const = 0):
        self.u.flat[self.boundaries] = const
        
    def impose_neumann_boundaries(self):
        self.u.flat[self.boundaries] = self.u.flat[self.boundary_map]
        
    def impose_boundaries(self):
        """
        placeholder to be set by user
        """
        pass
    
    def set_dirichlet_boundary_conditions(self, boundaries):
        self.boundaries, self.boundary_map = self.map_boundaries(boundaries)
        self.impose_boundaries = self.impose_dirichlet_boundaries
        
    def set_neumann_boundary_conditions(self, boundaries):
        self.boundaries, self.boundary_map = self.map_boundaries(boundaries)
        self.impose_boundaries = self.impose_neumann_boundaries

    def representation(self):
        ret = self.u*1.0
        ret.flat[self.boundaries] = 1.1*self.u.max()
        return ret
        
        
        
def boundaries_edge(resolution):
    bo = np.zeros(resolution, dtype = bool)
    bo[:, 0] = True
    bo[:, -1] = True
    bo[0,:] = True
    bo[-1,:] = True
    return bo


def boundaries_single_slit(resolution):
    bo = np.zeros(resolution, dtype = bool)
    bo[:, 0] = True
    bo[:, -1] = True
    bo[0,:] = True
    bo[-1,:] = True
    return bo
    
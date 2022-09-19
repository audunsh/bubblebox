# Bubblebox Showcases
# Author: Audun Skau Hansen 2022

import bubblebox.mdbox
import bubblebox.binding_models
import bubblebox.ising
import bubblebox.lattice_models

from bubblebox.mdbox import mdbox, box, no_force, hook_force, coulomb_force, lj_force
from bubblebox.binding_models import bindingbox
from bubblebox.ising import isingbox
from bubblebox.lattice_models import latticebox

import numpy as np


from bubblebox.mdbox import no_forces, hook_force, coulomb_force

def classical_hydrogen(n_bubbles = 125, size = (-10,-10,-10), 
                       vel_proton =   np.array([0.0, 0.0, 0.0]), 
                       vel_electron = np.array([0.0, 1.0, 0.0]), 
                       pos_proton =   np.array([0.0, 0.0, 0.0]), 
                       pos_electron = np.array([1.0, 0.0, 0.0]), 
                       charge = 1.0, proton_fixed = False):
    """
    # classical_hydrogen

    Generate a classical Hydrogen system (proton and electron)
    
    ## Keyword arguments:

    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | n_bubbles      |  Number of bubbles       | 125 |
    | size   | size of simulation box        | (-10,-10,-10)  |
    | vel_proton   | velocity of proton        |  (0,0,0) |
    | pos_proton   | position of proton        |  (0,0,0) |
    | vel_proton   | velocity of electron        |  (0,1,0) |
    | vel_proton   | position of electron        |  (1,0,0) |
    | charge   | charge of particles        |  1 (-1) |
    | proton_fixed   | clamped nucleus        |  False |
    """
    
    # create a default system
    b = mdbox(n_bubbles = 2, size = size) 

    # set forces to repulsive Coulomb
    b.set_forces(coulomb_force, force_params = np.array([charge, -charge]))

    # set masses to relative size
    b.set_masses(np.array([1836.0, 1.0]))

    # set position of particles
    b.pos = np.array([pos_proton, pos_electron]).T
    
    # set velocity of particles
    b.set_vel(np.array([vel_proton, vel_electron]).T)

    # set proton to inactive (if fixed)    
    if proton_fixed:
        b.active = np.array([False, True])
    
    # return the generated system
    return b

def ideal_gas(vel = 1, n_bubbles = 216, size = (5,5,5), n_species = 3):
    """
    # ideal_gas

    Set up an ideal gas with no interactions

    ## Keyword arguments:

    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | n_bubbles      |  Number of bubbles       | 216 |
    | size   | size of simulation box        | (5,5,5)  |
    | n_species   | number of different bubbles/masses/colors        | 3  |
    | vel   | standard deviation in velocities        | 1.0 |
    """

    # set up a generic system
    b = mdbox(n_bubbles=n_bubbles, size = size , vel = vel)
    
    # set to no interaction 
    b.set_forces(no_force, force_params = np.array([1,1]))

    # set masses to n_species different kinds
    b.set_masses(np.random.randint(1,n_species+1, n_bubbles) )

    # return the generated system
    return b

def hard_sphere_gas(vel = 1, n_bubbles = 216, size = (5,5,5), n_species = 3):
    """
    # hard_sphere_gas 

    Set up an hard sphere gas [1].

    This is one of the first systems studied using Monte Carlo integration.

    [1] Alder, B. J., Frankel, S. P., & Lewinson, V. A. (1955). Radial Distribution Function Calculated by the Monte‐Carlo Method for a Hard Sphere Fluid. The Journal of Chemical Physics, 23(3), 417-419.

    ## Keyword arguments:

    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | n_bubbles      |  Number of bubbles       | 216 |
    | size   | size of simulation box        | (5,5,5)  |
    | n_species   | number of different bubbles/masses/colors        | 3  |
    | vel   | standard deviation in velocities        | 1.0 |
    """

    # set up a generic system
    b = mdbox(n_bubbles=n_bubbles, size = size , vel = vel, sphere_collisions=True)
    
    # set to no interaction 
    b.set_forces(no_force, force_params = np.array([1,1]))

    # set masses to n_species different kinds
    b.set_masses(np.random.randint(1,n_species+1, n_bubbles) )

    # return the generated system
    return b


def fcc_system(size=3, lattice_parameter = 2.0, vel = 0.0):
    """

    # fcc_system(...)
    Set up a face-centered cubic system
    ( see https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres )
    for relaxed geometry at the onset of the simulation
    
    ## Keyword arguments:

    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | lattice_parameter      |   the length of each cell       |  2.0 |
    | size   | number of cells         | 3  (total number of bubbles is 4*size**3)  |
    | n_species   | number of different bubbles/masses/colors        | 3  |
    | vel   | standard deviation in velocities        | 0.0 |
    """

    def fcc(Nc = 3):
        """
        ### Generate a FCC setup

        Returns an fcc simulation box containing 4*Nc**3 particles, 
        with a total volume of (L*Nc)**3.

        (keeping this function limited in scope to keep students unconfused)
        """
        #Lc = L*Nc
        
        coords = []
        for i in range(-Nc, Nc):
            for j in range(-Nc, Nc):
                for k in range(-Nc, Nc):
                    coords.append([i,j,k])

                    coords.append([i+.5,j+.5,k])
                    coords.append([i+.5,j,k+.5])
                    coords.append([i,j+.5,k+.5])

        coords = np.array(coords)

        coords = (coords+.5)/Nc
        coords -= np.mean(coords, axis = 0)[None,:]

        return coords

    # Get positions for the lattice
    pos = fcc(np.abs(size)).T

    # set up the mdbox system
    b = mdbox(n_bubbles = pos.shape[1], size = (size*lattice_parameter, size*lattice_parameter, size*lattice_parameter), vel = vel)

    # scale positions to fille the box
    b.pos = pos*size*lattice_parameter
    
    # return the system
    return b


def repulsive_gas(n_bubbles = 125, n_species = 5, size = (5,5,5), vel = 1.0, charge = 10.0):
    """
    Generate a gas of repulsive bubbles
    
    ## Keyword arguments:

    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | n_bubbles      |   number of bubbles in box       |  125 |
    | size   | size of simulation box          | (-6,-6,-6)  |
    | n_species   | number of different bubbles/masses/colors        | 5  |
    | vel   | standard deviation in velocities        | 0.0 |
    | charge   | scaling of repulsion        | 10.0 |
    """

    # create a generic (lj) system
    b = mdbox(n_bubbles = n_bubbles, size = size, vel = vel)

    # set the masses to randomly distributed
    b.set_masses(  np.random.randint(1,n_species+1, n_bubbles) )

    # set the forces to repulsive
    b.set_forces(coulomb_force, force_params = np.array([charge, charge]))
    
    # allow for relatively long range interactions
    b.r2_cut = 15

    return b

def attractive_gas(n_bubbles = 125, n_species = 5, size = (5,5,5), vel = 1.0, charge = 1.0):
    """
    Generate a gas of attractive bubbles

    ## Keyword arguments
    
    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | n_bubbles      |   number of bubbles in box       |  125 |
    | size   | size of simulation box          | (-6,-6,-6)  |
    | n_species   | number of different bubbles/masses/colors        | 5  |
    | vel   | standard deviation in velocities        | 0.0 |
    | charge   | scaling of repulsion        | 1.0 |
    """

    # create a generic (lj) system
    b = mdbox(n_bubbles = n_bubbles, size = size, vel = vel)

    # set the masses to randomly distributed
    b.set_masses(  np.random.randint(1,n_species+1, n_bubbles) )

    # set the forces to attractive
    b.set_forces(coulomb_force, force_params = np.array([charge, -1*charge]))

    # allow for relatively long range interactions
    b.r2_cut = 15

    return b

def spring_system(n_bubbles= 3, n_species = 1, force_params = np.array([1.0, 1.0]), vel = 0):
    """
    A system of particles interacting with spring interactions
    
    ## Keyword arguments

    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | n_bubbles      |   number of bubbles in box       |  3 |
    | size   | size of simulation box          | (-6,-6,-6)  |
    | n_species   | number of different bubbles/masses/colors        | 1  |
    | vel   | standard deviation in velocities        | 0.0 |
    | force_params   | np.array([a,b]) where potential is -a*(x1-x2+b)**-1        | np.array([1.0, 1.0])  |
    """

    # create generic system
    b = mdbox(n_bubbles = n_bubbles, size = (5,5,5), vel = vel)

    # set masses
    b.set_masses( np.random.randint(1,n_species+1, n_bubbles) )

    # set the forces to harmonic spring interactions
    b.set_forces(hook_force, force_params = force_params)

    # allow for relatively long range interaction (5 units)
    b.r2_cut = 25

    # return system
    return b
    
def flow_system():
    """
    A pipe with bubbles flowing through
    """
    
    
    # a generic system, periodic in one dimension
    b = mdbox(125, size = (2,2,-10))
    
    # set masses to more heavy in the middle
    b.set_masses( 1 + np.exp(-np.sqrt(np.sum(b.pos[:2]**2, axis = 0))))

    # set flow in one dimension
    b.pos[2] += np.random.uniform(-1,1, 125)
    b.vel[2] = np.random.normal(10,.4,b.n_bubbles)
    return b


def throttle_2d():
    """
    Bubbles moving through a throttle, prototype
    """
    nx = 50
    ny = 10

    sx = 70
    sy = 20
    b =mdbox(n_bubbles = nx*ny, size = (-sx,sy))
    b.pos = np.array(np.meshgrid(np.linspace(-sx,sx, nx+1)[:-1], np.linspace(-sy,sy, ny+1)[:-1])).reshape(2,-1)
    b.pos = b.pos - np.mean(b.pos, axis = 1)[:, None]

    s,dr = 1.6, 1030


    alpha = .001
    opening = 0.8
    lower_edge = b.pos[1] <  sy*(-1 + opening*np.exp(-alpha*b.pos[0]**2))
    upper_edge = b.pos[1] >  sy*( 1 - opening*np.exp(-alpha*b.pos[0]**2))

    t = np.linspace(-1,1,np.sum(lower_edge))*50
    b.pos[:,lower_edge] = np.array([t, -1 + sy*(-1 + opening*np.exp(-alpha*t**2))])
    b.pos[:,upper_edge] = np.array([t, 1 + sy*( 1 - opening*np.exp(-alpha*t**2))])

    b.interactions[:,:,1] = 10

    b.interactions[lower_edge, :, 1] = .1
    b.interactions[lower_edge, :, 2] = np.sqrt(2)
    b.interactions[lower_edge, :, 0] = lj_force()

    b.interactions[:, upper_edge, 1] = .1
    b.interactions[:, lower_edge, 2] = np.sqrt(2)
    b.interactions[:, lower_edge, 0] = lj_force()


    #upper_edge = np.sum((b.pos-np.array([0,s*sy]).reshape(2,-1))**2, axis = 0)<dr
    #lower_edge = np.sum((b.pos+np.array([0,s*sy]).reshape(2,-1))**2, axis = 0)<dr
    #
    b.masses[upper_edge] = 3
    b.masses_inv[upper_edge] = 3**-1
    b.masses[lower_edge] = 3
    b.masses_inv[lower_edge] = 3**-1
    #theta = np.linspace(-1,1,np.sum(upper_edge))*1.4
    #b.pos[:,upper_edge] = np.array([.8*dr**.5*np.sin(theta), -.8*dr**.5*np.cos(theta)+s*sy])

    #theta = np.linspace(-1,1,np.sum(lower_edge))*1.4
    #b.pos[:,lower_edge] = np.array([.8*dr**.5*np.sin(theta), .8*dr**.5*np.cos(theta)-s*sy])

    b.active[upper_edge] = False
    b.active[lower_edge] = False



    b.vel[0] = np.random.normal(6,.5,b.n_bubbles)
    
    return b


def fcc_custom(size=(3,3,3), lattice_parameter = 2.0):
    """
    # fcc_custom

    Set up a face-centered cubic system of varying size for each dimension
    ( see https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres )
    for relaxed geometry at the onset of the simulation
    

    ## Keyword arguments
    
    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | lattice_parameter      |   the length of each cell       |  2.0 |
    | size   | number of cells         | 3  (total number of bubbles is 4*size**3)  |
    """

    def fcc(Nx, Ny, Nz):
        """
        ### Generate a FCC setup

        Returns an fcc simulation box containing 4*Nc**3 particles, 
        with a total volume of (L*Nc)**3.

        (keeping this function limited in scope to keep students unconfused)
        """
        #Lc = L*Nc
        
        coords = []
        for i in range(-Nx, Nx):
            for j in range(-Ny, Ny):
                for k in range(-Nz, Nz):
                    coords.append([i,j,k])

                    coords.append([i+.5,j+.5,k])
                    coords.append([i+.5,j,k+.5])
                    coords.append([i,j+.5,k+.5])

        coords = np.array(coords)

        coords = (coords+.5)
        coords = coords*(np.array(size, dtype = float)**-1)[None,:]
        coords -= np.mean(coords, axis = 0)[None,:]

        return coords

    pos = fcc(*np.abs(np.array(size))).T

    b = mdbox(n_bubbles = pos.shape[1], size = np.array(size)*lattice_parameter, vel = 0)
    b.pos = lattice_parameter*pos*np.array(size, dtype = float)[:, None]
    
    return b

def throttle_3d_membrane(membrane_width = 1.0, lattice_parameter = 4.0, inflow_velocity = 6.0):
    """
    # throttle_3d_membrane
    """



    class custom_advance_box(mdbox):
        def custom_advance(self):
            super().advance_vverlet()
            
            # custom logic follows here
            self.vel_[:, self.pos[0]<self.inflow_region] = np.random.multivariate_normal([self.inflow_velocity, 0,0],np.eye(len(self.size))*0.1, np.sum(self.pos[0]<self.inflow_region)).T
            
            
    

    b = fcc_custom(size = (-5,1,1), lattice_parameter = lattice_parameter)


 

    bc = custom_advance_box(n_bubbles = b.n_bubbles, size = b.size)
    bc.inflow_region = .9*bc.pos[0].min()
    bc.inflow_velocity = inflow_velocity
    

    bc.pos = b.pos
    membrane_index = np.abs(bc.pos[0]+.01)<membrane_width
    bc.active[membrane_index] = False
    bc.masses[membrane_index] = 3
    bc.masses_inv[membrane_index] = 3**-1
    
    bc.vel[0] = np.random.normal(inflow_velocity,.5,bc.n_bubbles)

    # make the custom advance the new standard
    bc.advance = bc.custom_advance

    return bc

def throttle_expansion(membrane_width = 1.0, lattice_parameter = 3.0, inflow_velocity = 6.0, wall_1_velocity = 0.001, wall_2_velocity = 0.0011):

    """
    # throttle_3d_membrane

    This system will simulate throttling through a membrane,
    with walls pushing the bubbles through the membrane

    ## Keyword arguments
    
    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | membrane_width      |  the width of the membrane in the middle       |  1.0 |
    | lattice_parameter   | adjusts the distance between the particles         | 3.0   |
    | wall_1_velocity   | the velocity of the left wall pushing         | 0.001  |
    | wall_2_velocity   | the velocity of the right wall expanding         | 0.0011  |
    """

    # set up a custom advance routine allowing for 
    # special movemen of walls
    class custom_advance_box(mdbox):
        def custom_advance(self):
            super().advance_vverlet()
            
            # custom logic follows here

            # fixed velocity of inflowing particles
            #self.vel_[:, self.pos[0]<self.inflow_region] = np.random.multivariate_normal([self.inflow_velocity, 0,0],np.eye(len(self.size))*0.1, np.sum(self.pos[0]<self.inflow_region)).T
            
            # move walls
            self.pos[0, self.wall1] += wall_1_velocity
            self.pos[0, self.wall2] += wall_2_velocity

            
    
    # create a custom fcc system
    b = fcc_custom(size = (-5,1,1), lattice_parameter = lattice_parameter)

    # square root of the number of bubbles per wall 
    nw = 10
    n_walls = nw**2
    bc = custom_advance_box(n_bubbles = b.n_bubbles + 2*n_walls, size = b.size)

    # parameters used for custom logic: bubbles enter system with fixed velocity
    bc.inflow_region = .9*bc.pos[0].min()
    bc.inflow_velocity = inflow_velocity
    

    bc.pos[:, :b.n_bubbles] = b.pos
    bc.active[b.n_bubbles:] = False
    bc.masses[b.n_bubbles:] = 2
    bc.masses_inv[b.n_bubbles:] = 2**-1
    
    
    # create index arrays for each wall
    # so that pos[ wallN ] references the relevant wall
    bc.wall1 = np.zeros(bc.n_bubbles, dtype = bool)
    bc.wall1[b.n_bubbles:b.n_bubbles+n_walls] = True
    bc.wall2 = np.zeros(bc.n_bubbles, dtype = bool)
    bc.wall2[b.n_bubbles+n_walls:] = True
    
    # set wall position
    wx,wy = np.array(np.meshgrid(np.linspace(-bc.size[1], bc.size[1], nw+1)[:-1], np.linspace(-bc.size[2], bc.size[2], nw+1)[:-1])).reshape(2,-1)
    
    bc.pos[0, b.n_bubbles:b.n_bubbles+n_walls] = -np.abs(bc.size[0])
    bc.pos[1, b.n_bubbles:b.n_bubbles+n_walls] = wx
    bc.pos[2, b.n_bubbles:b.n_bubbles+n_walls] = wy
    
    bc.pos[0, b.n_bubbles+n_walls:] = 0
    bc.pos[1, b.n_bubbles+n_walls:] = wx
    bc.pos[2, b.n_bubbles+n_walls:] = wy
    
    # create inactive membrane
    membrane_index = np.abs(bc.pos[0]+.01)<membrane_width
    bc.active[membrane_index] = False
    bc.masses[membrane_index] = 3
    bc.masses_inv[membrane_index] = 3**-1
    
    # set default advance logic to custom advance
    bc.advance = bc.custom_advance

    # return system to user
    return bc



## lattice systems

def mixing_lattice():
    """
    short explanation of model here
    """
    interaction = -2*np.eye(3)
    interaction[1,0] = 2
    interaction[0,1] = 2

    interaction[1,2] = 5
    interaction[2,1] = 5

    interaction[0,2] = -10
    interaction[2,0] = -10
            
    nx = 15
    ny, nz = int(.1*nx**3), int(.05*nx**3)

    lb = latticebox(n_bubbles = np.array([nx**3-ny-nz,ny,nz]), size = np.array([nx,nx, nx]), interaction = interaction)
    lb.n_swaps_per_advance = 100
    lb.kT = 0.00001
    return lb


def isolated_chambers(temperature_factor_left = 1.0, temperature_factor_right = 8.0, lattice_parameter = 2.0):
    """
    
    # isolated_chambers
    
    Two isolated chambers with gases/liquids of varying temperature
    Auhors: Audun Skau Hansen and Hanan Gharayba, 2022
    
    ## Keyword arguuments
    
    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | lattice_parameter   | adjusts the distance between the particles         | 3.0   |
    | temperature_factor_right   | scale temperatures in right chamber         | 8.0  |
    | temperature_factor_left   | scale temperatures in left chamber         | 2.0  |
    
    """

    b = fcc_custom(lattice_parameter = 2.0, size = (4,2,2))

    # create a membrane in the middle
    membrane_width = 1
    left_index = b.pos[0]<-membrane_width
    right_index = b.pos[0]>membrane_width
    membrane_index = np.abs(b.pos[0])<=membrane_width
    
    # distinct masses (for varying colors)
    masses = np.zeros_like(b.masses)
    masses[left_index] = 1.0
    masses[right_index] = 2.0
    masses[membrane_index] = 3.0
    b.set_masses(masses)

    # define forces
    b.set_forces(no_force, left_index, right_index) #no interactions between particles
    b.set_forces(coulomb_force, left_index, membrane_index, force_params = np.ones(2)*1.0) # set membrane to repulsive
    b.set_forces(coulomb_force, right_index, membrane_index, force_params = np.ones(2)*1.0) # set membrane to repulsive

    # set velocities according to temperatures
    velocities = np.zeros_like(b.vel)

    velocities[:, left_index] = np.random.multivariate_normal(np.zeros(3), np.eye(3)*temperature_factor_left, np.sum(left_index)).T
    velocities[:, right_index] = np.random.multivariate_normal(np.zeros(3), np.eye(3)*temperature_factor_right, np.sum(right_index)).T
    b.set_vel(velocities)
    
    # make membrane compact and rigid (inactive)
    b.pos[0, membrane_index] = 0.0

    b.active[membrane_index] = False
    
    # return system to user
    return b


def harmonic_chain(n_bubbles = 120, size = (10,10,10), charge = 20,  force_params = np.array([4.0,1.0])):
    """
    
    # Harmonic Chain
    
    Generates a structure of n_bubbles connected to their neighbors with 
    harmonic potentials, and repelling all others with a coulomb potential
    Auhor: Audun Skau Hansen 2022
    
    ## Keyword arguuments
    
    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | n_bubbles   | number of bubbles         | 120   |
    | size   | size of simulation box         | (10,10,10)  |
    | charge  | repulsive charge         | 20.0  |
    
    """
    b = repulsive_gas(n_bubbles, size = size, charge = charge)
    
    for i in range(n_bubbles+1):
        bubbles_a = np.zeros(n_bubbles, dtype = bool)
        bubbles_b = np.zeros(n_bubbles, dtype = bool)
        bubbles_a[i%n_bubbles] = True
        bubbles_b[(i+1)%n_bubbles] = True
        
        b.set_forces(hook_force, bubbles_a = bubbles_a, bubbles_b = bubbles_b, force_params = force_params)
        
    t = np.linspace(0,2*np.pi, n_bubbles+1)[:-1]
    b.pos[0] = .8*size[0]*np.cos(t)
    b.pos[1] = .8*size[1]*np.sin(t)
    b.pos[2] *= 0
    b.r2_cut = 1000
    
        
    return b

def double_harmonic_chain(n_bubbles = 200, size = (10,10,10), charge = 20.0, force_params = np.array([4.0,1.0])):
    """
    
    # Harmonic Chain
    
    Generates a double-layered structure of n_bubbles connected to their neighbors with 
    harmonic potentials, and repelling all others with a coulomb potential
    Auhor: Audun Skau Hansen 2022
    
    ## Keyword arguuments
    
    | Argument      | Description | Default |
    | ----------- | ----------- | --- |
    | n_bubbles   | number of bubbles         | 120   |
    | size   | size of simulation box         | (10,10,10)  |
    | charge  | repulsive charge         | 20.0  |
    
    """

    b = repulsive_gas(n_bubbles, size = size, charge = charge)
    
    for i in range(n_bubbles+1):
        bubbles_a = np.zeros(n_bubbles, dtype = bool)
        bubbles_b = np.zeros(n_bubbles, dtype = bool)
        bubbles_a[i%n_bubbles] = True
        bubbles_b[(i+1)%n_bubbles] = True
        bubbles_b[(i+int(n_bubbles/2))%n_bubbles] = True
        b.set_forces(hook_force, bubbles_a = bubbles_a, bubbles_b = bubbles_b, force_params = force_params)
        
    t = np.linspace(0,2*np.pi, n_bubbles+1)[:-1]
    b.pos[0] = .8*size[0]*np.cos(t)
    b.pos[1] = .8*size[1]*np.sin(t)
    b.pos[2] *= 0
    b.r2_cut = 1000
    
        
    return b
import numpy as np
import scipy as scipy
import math
from scipy import spatial
from scipy.spatial import  voronoi_plot_2d

import matplotlib.pyplot as plt

def get_angle(vector1, vector2):
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def get_angle2(e_par_j, e_rho_j):
    e_par_j1 = np.zeros(3)
    e_rho_j1 = np.zeros(3)

    e_par_j1[0:2] = e_par_j
    e_rho_j1[0:2] = e_rho_j
    prod = np.cross(e_par_j1, e_rho_j1)
    e_par_j1 /= np.linalg.norm(e_par_j1)
    e_rho_j1 /= np.linalg.norm(e_rho_j1)
    dot_product = np.dot(e_par_j1, e_rho_j1)
    angle = np.arccos(dot_product)
    #angle *=  (180 / np.pi)

    if prod[2] < 0:
        angle = 2 * np.pi - angle
        #angle = 360 - angle
        #print('angle > 180; angle = ', angle)
    #else:
        #print('angle < 180; angle = ', angle)
    return(angle)
   # for i in range(3):


def neighbors_id(vor):
    #----------------------------------
    # sostavlenie spiska blishaishix sosedei (neigh_id)  neigh_id[i] = [id_1, id_2, ... , id_n];   i - index of particle, id_1,2...n - индексы ссоседних частиц
    #----------------------------------
    npoints=len(vor.points)
    neigh_id=[[] for i in range(npoints)]
    for rid in vor.ridge_points:
        neigh_id[rid[0]].append(rid[1])
        neigh_id[rid[1]].append(rid[0])
    return neigh_id
def get_perp_vector(vector):
    '''
    ffunction to calculate the perpendicular vector in 2D.
    INPUT: vector : numpy array
    '''
    e_perp = np.array([1, - vector[0] / vector[1] ])
    e_perp /= np.linalg.norm(e_perp)
    return e_perp
def get_angle_phi_ij(e_par_i, e_par_j, e_rho_j):
    '''
    fucnt to cal angle phi_ij. We have 2 situation:
    1) two vectors in one half-plane (above the x-axis)
    2) two vectors in different half-plane (first vec above the x-axis, second under)
    '''
    koord_angle = get_angle(e_rho_j, np.array([1, 0 ]))
    new_e_par_i = get_rotation(e_par_i, koord_angle)
    new_e_par_j = get_rotation(e_par_j, koord_angle)

    if new_e_par_i[1] * new_e_par_j[1] >= 0:
        phi_ij = get_angle(e_par_i, e_par_j)
    else:
        phi_ij = np.pi - get_angle(e_par_i, e_par_j)
    return phi_ij


def get_rotation(vector, angle):
    '''
    ffunction to calculate the perpendicular vector in 2D (by rotation in angle rad).
    INPUT: vector : numpy array
    '''
    return np.array([vector[0] * np.cos(angle) - vector[1] * np.sin(angle),  vector[0] * np.sin(angle) + vector[1] * np.cos(angle)])


def get_rot_vector(vector):
    '''
    ffunction to calculate the perpendicular vector in 2D (by rotation in pi/2 rad).
    INPUT: vector : numpy array
    '''
    return np.array([vector[0] * np.cos(np.pi / 2) - vector[1] * np.sin(np.pi / 2),  vector[0] * np.sin(np.pi / 2) + vector[1] * np.cos(np.pi / 2)])


class Set_fish:
    def __init__(self, position, direction = None, ID = None, get_info = False):
        """
        Initiate a Fish SINGLE (one unit) object

        Arguments
        ----------
        position : numpy.ndarray
            2-dimensional vector
        direction : numpy.ndarray, default : None
            2-dimensional unit vector giving the direction of the fish (velocity)
            If `None`, will be randomly sampled from the uniform sphere
        ID : any type, default : None
            A fish identifier
        get_info (verbose) : bool, default : False
            print data
        """


        self.position = position

        if direction is None:
            self.direction = np.random.randn(2)
            self.direction /= np.linalg.norm(self.direction)
        else:
            self.direction = direction / np.linalg.norm(direction)

        self.ID = ID
        self.get_info = get_info


    def update_orientation(self, theta_der_i, d_t):
        """
        change of position
        """
        #np.array([vector[0] * np.cos(np.pi / 2) - vector[1] * np.sin(np.pi / 2),  vector[0] * np.sin(np.pi / 2) + vector[1] * np.cos(np.pi / 2)])
        self.direction = np.array([self.direction[0] * np.cos(theta_der_i * d_t) - self.direction[1] * np.sin(theta_der_i * d_t),  self.direction[0] * np.sin(theta_der_i * d_t) + self.direction[1] * np.cos(theta_der_i * d_t)])

        self.direction /= np.linalg.norm(self.direction)
        #if theta_der_i >= 0:
            #self.direction = np.array([self.direction[0] * np.cos(theta_der_i * d_t) - self.direction[1] * np.sin(theta_der_i * d_t),  self.direction[0] * np.sin(theta_der_i * d_t) + self.direction[1] * np.cos(theta_der_i * d_t)])
        #if theta_der_i < 0:
            #theta_der_i =  abs(theta_der_i)
            #self.direction = np.array([self.direction[0] * np.cos(theta_der_i * d_t) + self.direction[1] * np.sin(theta_der_i * d_t),  -self.direction[0] * np.sin(theta_der_i * d_t) + self.direction[1] * np.cos(theta_der_i * d_t)])

    def update_position(self, der_position, d_t):
        self.position += der_position * d_t




class Set_MD:
    # MD performed by theory on article https://doi.org/10.1103/PhysRevLett.120.198101

    """A class for a setting fish parameters simulation.

    Arguments
    ----------
    number_type_fish : int, default : 1
                the number of type
    number_of_fish : int, default : 10
                The number of fish to be simulated
    repulsion_radius : float, default : 1.0
        Fish within this radius will repel each other
        (unit: length of a single fish).

    I_ii : float, default : 0.2
                dimensionless aligment intesity: tends to align with the same neighbors  with intensity k_v

    I_n : float, default : 0.2
                dimensionles nose intesity haractering the sponteniusly behavior

    I_f: float default : 0.01
                dimensionles far-field flow disturbance created by all other swimmers

    typical_length_r0 : float, default 5
                the swimmer surface radius (FIG(1, b) in ref article)

    d_t : float, default 0.01
                timestep in MD simulation

    box_lengths : numpy array (1 string and 2 column (2d system)), default [20, 20]
                the size of simulation box

    periodic_bond: list of bool, default: [False, False]
                argument for periodic boundary.

    speed : float, default : 1
                Speed of a fish.

    """
    def __init__(self,
                 number_type_fish = 1,
                 number_of_fish = 10,
                 I_ii = 0.2,
                 I_n = 0.2,
                 I_f = 0.1,
                 typical_length_r0 = 5,
                 speed = 1,
                 d_t = 0.01,
                 N_time_step = 1,
                 box_lengths = [20,20],
                 periodic_bond = [False, False],
                 get_info = False,
                 lmp_path_file = ""

                 ):

        self.number_type_fish = number_type_fish
        self.number_of_fish = int(number_of_fish)
        self.I_ii = float(I_ii)
        self.I_n = float(I_n)
        self.I_f = float(I_f)
        self.typical_length_r0 = typical_length_r0
        self.speed = speed
        self.d_t = d_t
        self.box_lengths = np.array(box_lengths,dtype=float)
        self.get_info = get_info
        self.fish = []
        self.N_time_step = int(N_time_step)
        self.lmp_path_file = lmp_path_file

    def calc_u_ij(self, theta_ji, e_theta_j, e_rho_j, rho_ij):
        '''
        func to calc of velocity induced by swimmer j at the position r_i
        '''
        u_ij = self.I_f * (e_theta_j * np.sin(theta_ji) + e_rho_j * np.cos(theta_ji)  ) / (np.pi * rho_ij * rho_ij)
        return u_ij

    def gen_normal_win_proc(self):
        """
        Generate motion by drawing from the Normal distribution

        Arguments:
            n_step: Number of steps

        Returns:
            A NumPy array with `n_steps` points
        """
        if self.N_time_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")

        w = np.zeros(self.N_time_step + 1)

        for i in range(1,self.N_time_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(self.N_time_step))

        return w


    def init_random(self):
        """
        Initialize the fish list
        """
        self.fish = [ Set_fish(position=self.box_lengths*np.random.random((2,)),
                           ID=i,
                           get_info=self.get_info
                           ) for i in range(self.number_of_fish) ]

    def get_pick_system(self, vor, f_directions, ts):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)

        #voronoi_plot_2d(vor, point_size=10, ax=ax)

        #fig = plt.figure(figsize=(20,20),frameon=False)
        #ax = plt.subplot(111)
        voronoi_plot_2d(vor, show_vertices=False, ax=ax)

        print("in get_pick, ts = ", ts)
        for jj in range(len(vor.points)):
            tvx     = 0.5 * f_directions[jj, 0]
            tvy     = 0.5 * f_directions[jj, 1]
            v       = (np.sqrt(tvx * tvx + tvy * tvy) + 0.0000001)
            #plt.plot(vor.points[jj][0], vor.points[jj][1],'.', color = 'black')
            plt.text(vor.points[jj][0], vor.points[jj][1] , str(jj))
            plt.arrow(vor.points[jj][0], vor.points[jj][1], 0.5*f_directions[jj, 0], 0.5 * f_directions[jj, 1], ec = 'black', fc = 'black', head_width = v * 0.4, head_length = v * 0.5, width = v * 0.25 * 0.0125, length_includes_head = True) # рисовка стрелки

        plt.savefig("fish_pick/Fish_sys_ts_" + str(ts)+'.png', transparent=True, bbox_inches='tight',pad_inches = 0)
        plt.close(fig)
    def calc_omega(self, u_ij, theta_i, f_directions):
        if f_directions[1] >= 0:
            theta = theta_i
        else:
            theta = -theta_i

        return u_ij[1] * np.cos(2 * theta) - u_ij[0] * np.sin(2 * theta)

    def get_vmd_file(self, position, direction, ts):
        '''
        function: outup lammpstrj file for vmd
        '''

        if ts == 0:
            file_save = open(self.lmp_path_file + 'Fish_md_In_' +str(self.I_n) + '_Iii_'+str(self.I_ii) + '_If_'+str(self.I_f)+'.lammpstrj', 'w')
        else:
            file_save = open(self.lmp_path_file + 'Fish_md_In_' +str(self.I_n) + '_Iii_'+str(self.I_ii) + '_If_'+str(self.I_f)+'.lammpstrj', 'a')
        file_save.write("ITEM: TIMESTEP" + '\n')
        file_save.write(str(ts) + '\n')
        file_save.write("ITEM: NUMBER OF ATOMS" + '\n')
        file_save.write(str(len(position)) + '\n')
        file_save.write("ITEM: BOX BOUNDS pp pp pp" + '\n')
        file_save.write(str(0.0) + " " + str(self.box_lengths[0]) + '\n')
        file_save.write(str(0.0) + " " + str(self.box_lengths[1]) + '\n')
        file_save.write(str(0.0) + " " + str(0.1) + '\n')
        file_save.write("ITEM: ATOMS id x y vx vy" + '\n')

        for i in range(len(position)):#range(self.number_of_fish):
            file_save.write(str(i) + ' ' + str(position[i, 0]) + ' ' + str(position[i, 1]) + ' ' + str(direction[i, 0]) + ' ' + str(direction[i, 0]) + '\n')
        file_save.close()

    def generate_wall_point(self, xmin, xmax, ymin, ymax):
        '''
        для более корректного учета соседей по ячейками вороного на крае косяка рыб
        на стенкам смоделированы "призрачные" рыбы (исключительно для коррекции ячеек вороного)
        в моделировании эти призрачные рыбы 1) статичны 2) не имеют влияние на рыб в косяке
        '''

        '''
        для более корректного учета соседей по ячейками вороного на крае косяка рыб
        на стенкам смоделированы "призрачные" рыбы (исключительно для коррекции ячеек вороного)
        в моделировании эти призрачные рыбы 1) статичны 2) не имеют влияние на рыб в косяке
        '''
        N = 15
        #d_r_wall =  1#self.d_r_wall
        x_box = (xmax - xmin)
        y_box = (ymax - ymin)
        y_up = ymax + 0.1 * y_box
        y_low = ymin - 0.1 * y_box
        x_l = xmin - 0.1 * x_box
        x_r = xmax + 0.1 * x_box
        x_box = x_r - x_l
        y_box = y_up - y_low
        d_r_wall_x = x_box / N
        d_r_wall_y = y_box / N
        N_x_gost = int( x_box / d_r_wall)
        N_y_gost = int( y_box / d_r_wall)

        N_r = [N, N]
        gost_fish = np.zeros((2 * (N + N), 2))
        for i in range(N):
            gost_fish[4 * j, 0] = x_l + d_r_wall_x * j
            gost_fish[4 * j, 1] = y_low
            gost_fish[ * j + 1, 0] = x_l + d_r_wall_x * j
            gost_fish[2 * j + 1, 1] = y_up

            gost_fish[2 * j - 1, 0] = x_l
            gost_fish[2 * j - 1, 1] = y_low + d_r_wall * j
            gost_fish[2 * j - 2, 0] = x_r
            gost_fish[2 * j - 2, 1] = y_low + d_r_wall * j
        return gost_fish




    def run_simulation(self):
        '''
        funt to run fush simulation
        arguments:

        N_time_step int, default : none
                the length of MD simulation
        '''
        # create arrays of initial position and orientation (velocity)
        flag = 0
        N_time_step = self.N_time_step
        print(type(self.number_of_fish))
        print(type(N_time_step))
        self.init_random()
        #positions = np.zeros((self.number_of_fish, N_time_step + 1, 2))
        #directions = np.zeros((self.number_of_fish, N_time_step + 1, 2))

        f_positions = np.zeros((self.number_of_fish, 2))
        f_directions = np.zeros((self.number_of_fish, 2))
        i_print = 1
        print("i = ", i_print)
        i_print += 1

        #for i in range(self.number_of_fish):
            ##f_positions[i,:] = self.fish[i].position
            ##f_directions[i,:] = self.fish[i].direction
            #positions[i,0,:] = self.fish[i].position
            #directions[i,0,:] = self.fish[i].direction



        w_p = self.gen_normal_win_proc()

        for ts in range(N_time_step + 1):
            print('N_time_step = ', N_time_step, ';  ts = ', ts)
            f_positions = np.zeros((self.number_of_fish, 2))
            f_directions = np.zeros((self.number_of_fish, 2))

            for i in range(self.number_of_fish):
                f_positions[i,:] = self.fish[i].position
                f_directions[i,:] = self.fish[i].direction

            gost_fish = self.generate_wall_point(min(f_positions[:,0]),max(f_positions[:,0]), min(f_positions[:,1]), max(f_positions[:,1]))

            f_positions = np.vstack((f_positions, gost_fish))
            self.get_vmd_file(f_positions, f_positions, ts)

            vor = scipy.spatial.Voronoi(f_positions[:, :])
            neigh_id = neighbors_id(vor)
            #self.get_pick_system(vor, f_directions, ts)
            count = 0
            for i in  range(self.number_of_fish):
                U_i = np.array([0.0, 0.0])
                Omega_i = 0
                theta_der_i = 0
                sum_cos_1 = 0
                fish_i = self.fish[i]
                r_i = f_positions[i] #fish_i.position
                v_i = f_directions[i] #fish_i.direction
                e_par_i = v_i
                e_perp_i = get_rot_vector(e_par_i)
                theta_i = get_angle(e_par_i, np.array([1, 0 ]))
                if e_par_i[1] < 0:
                    theta_i *= -1
                for j in neigh_id[i]:# range(len(vor.points)):
                    if j < self.number_of_fish:
                        fish_j = self.fish[j]
                        r_j = f_positions[j] #fish_j.position
                        e_par_j = f_directions[j] #fish_j.direction
                        rho_ij = r_i - r_j  # np vec        ====== CHEK ===========
                        #print('rho_ij = ', rho_ij, '; len = ', (rho_ij[0] ** 2 + rho_ij[1] ** 2) ** 0.5)
                        e_rho_j = rho_ij / np.linalg.norm(rho_ij) # np vec by unit scale [08, 0.6] = |1| ====== CHEK ===========
                        #print('e_rho_j = ', e_rho_j, '; len = ', (e_rho_j[0] ** 2 + e_rho_j[1] ** 2) ** 0.5)

                        rho_ij = np.linalg.norm(rho_ij) # module == length ====== CHEK ===========
                        #print('rho_ij = ', rho_ij)
                        e_theta_j = get_rot_vector(e_rho_j)


                        theta_ij = get_angle2(e_par_i, -e_rho_j)
                        theta_ji = get_angle2(e_par_j, e_rho_j)
                        phi_ij = get_angle_phi_ij(e_par_i, e_par_j, e_rho_j)

                        u_ij = self.calc_u_ij(theta_ji, e_theta_j, e_rho_j, rho_ij)     # fomule ====== CHEK ===========
                        U_i += u_ij

                        Omega_i += self.calc_omega(u_ij, theta_i, v_i) # u_ij[1] * np.cos(2 * theta_i) - u_ij[0] * np.sin(2 * theta_i)

                        theta_der_i += ((rho_ij * np.sin(theta_ij) + self.I_ii * np.sin(phi_ij)) * (1.0 + np.cos(theta_ij)))   # fomule ====== CHEK ===========
                        sum_cos_1 += (1 + np.cos(theta_ij))

                theta_der_i /= sum_cos_1
                #
                if flag == 0:
                    U_i = 0
                    Omega_i = 0

                theta_der_i += (self.I_n * w_p[ts]+ Omega_i)
                #if theta_der_i < 0:
                    #print(count, '; minus_1')
                    #count += 1
                #theta_der_i = theta_der_i * np.pi / 180
                #print('theta_der_i = ', theta_der_i, '; theta_der_i_degree = ', theta_der_i * 180 / np.pi)

                der_position = e_par_i + U_i
                #print('der_position = ', der_position, '; pos = ', r_i)
                self.fish[i].update_position(der_position, self.d_t)
                self.fish[i].update_orientation(theta_der_i, self.d_t)
                #del f_positions, f_directions





sim = Set_MD(number_type_fish = 1,
                 number_of_fish = 100,
                 I_ii = 0.3,
                 I_n = 1.5,
                 I_f = 0.01,
                 typical_length_r0 = 1,
                 speed = 1,
                 d_t = 0.1,
                 N_time_step = 10000,
                 box_lengths = [20,20],
                 periodic_bond = [False, False],
                 get_info = False
)
sim.run_simulation()









































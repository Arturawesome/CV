import numpy as np
import DataSetWorks1
from DataSetWorks1 import LammpsCoords

import StructuralAnalysis
from StructuralAnalysis import SystemStructur
import scipy as scipy
from scipy import spatial

import matplotlib.pyplot as plt
import pygame
import math
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.colors import LightSource, Normalize
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import scipy.io

from itertools import chain
from statistics import pvariance
import logging
import psutil
import time
import os
import decimal



class Snake(SystemStructur):
    def __init__(self,
                 heights: list,
                 mu_values: list,
                 Temperature: float,
                 percent: float,
                 r_pcf: float,
                 r_snake: float,

                 density: list,
                 list_psi: list,

                 istart: int,
                 all_cadr: int,
                 num_cadr: int,

                 txt_path:str,
                 img_path: str,
                 past_name: str,

                 full_md_name: str
                 ):

        # all path and txt variables
        self.img_path = img_path
        self.txt_path = txt_path
        self.past_name = past_name
        self.full_md_name = full_md_name

        # list cariables
        self.list_psi = list_psi

        # all nympy variables
        self.heights = np.array(heights, dtype=float)
        self.mu_values = np.array(mu_values,  dtype=float)
        self.density = np.array(density, dtype = float)
        self.z_distance = []
        self.z_coord = []
        self.st = None
        self.dat_xyz = None     # np.array after Add_Points() dat_xyz[:, 0] - MD id particle, dat_xyz[:, 1:4] - xyz position
        self.dat_xyz_init = None # np.array dat_xyz[:, 0] - MD id particle, dat_xyz[:, 1:4] - xyz position
        self.dat_xyz_previusly = None
        self.sigma2t = None
        #self.dat_id_part_inside_init = None
        self.dx_dy_ij = None
        self.r_ij = None
        self.triple_point_list = None
        self.r_pcf = r_pcf
        self.r_vor = None


        # all float variables
        self.Temperature = Temperature
        self.percent = percent  #
        self.x_box = None
        self.y_box = None
        self.dx = None
        self.dy = None
        self.xmin_init = None
        self.xmax_init = None
        self.ymin_init = None
        self.ymax_init = None

        # all integer variables
        self.num_h = len(self.heights)
        self.num_mu  = len(self.mu_values)
        self.istart = istart
        self.all_cadr = all_cadr
        self.num_cadr = num_cadr
        self.len_cadrs =  None
        self.nparts_init = None


        # VORONOI VARIABLES
        self.vor = None
        self.region_mask = None
        self.neigh_id = None

        # All snake's variables
        self.snakes_id = []
        self.angles_in_snakes = []
        self.len_snakes = []
        self.agnles_stat = None

        self.r_snake = r_snake #0.5#1.0#0.8

        # Structure variables
        self.psi_array_gist = None
        self.psi = None


        self.dy_diff = None
        self.dx_diff = None
        self.dr_diff = None

        self.min_rdf = None
        self.max_rdf = None
        self.R_field_D = None
        self.S_voro = None
        self.log_rdf_setka = None
        self.d_rdf = None
        self.counter_rdf = 0

    def __del__(self):
        del self.img_path
        del self.txt_path
        del self.full_md_name

        # list cariables
        del self.list_psi

        # all nympy variables
        del self.heights
        del self.mu_values
        del self.density
        del self.z_distance
        del self.z_coord
        del self.st
        del self.dat_xyz
        del self.dat_xyz_init
        del self.dat_xyz_previusly
        del self.sigma2t
        #self.dat_id_part_inside_init = None
        del self.dx_dy_ij
        del self.r_ij
        del self.triple_point_list
        del self.r_pcf

        # all float variables
        del self.Temperature
        del self.percent
        del self.x_box
        del self.dx
        del self.dy
        del self.xmin_init
        del self.xmax_init
        del self.ymin_init
        del self.ymax_init

        # all integer variables
        del self.num_h
        del self.num_mu
        del self.istart
        del self.all_cadr
        del self.len_cadrs
        del self.nparts_init


        # VORONOI VARIABLES
        del self.vor
        del self.region_mask
        del self.neigh_id

        # All snake's variables
        del self.snakes_id
        del self.angles_in_snakes
        del self.len_snakes
        del self.agnles_stat

        del self.r_snake

        # Structure variables
        del self.psi_array_gist
        del self.psi

        del self.dy_diff
        del self.dx_diff
        del self.dr_diff
        print("==============================")
        print("Snake Destucture: clear memory")
        print("==============================")
        class_name = self.__class__.__name__
        print('{} delete'.format(class_name))


    def Get_Voronoi_Neighbors(self):
        # function for
        self.vor = scipy.spatial.Voronoi(self.dat_xyz[:, 1:3])
        self.region_mask = np.full(len(self.vor.regions), False)
        npoints = len(self.vor.points)
        ver_index = np.full(len(self.vor.vertices),False)
        self.neigh_id = [[] for i in range(self.nparts_init)]
        self.dx_dy_ij = [[] for _ in range(self.nparts_init)]
        self.r_ij = [[] for _ in range(self.nparts_init)]
        self.triple_point_list = np.zeros(self.nparts_init)
        #self.dat_id_part_inside_init = np.zeros(self.nparts_init)

        #----------------------------------
        # otsechenie pogranichnux regionov i poisk min max ploshadei
        #----------------------------------
        for i in range(npoints):
            ind = self.vor.point_region[i]
            certies = self.vor.regions[ind]
            if (self.vor.points[i][0] > self.xmin_init  - 0.5 * self.dx  and self.vor.points[i][0] < self.xmax_init + 0.5*self.dx and self.vor.points[i][1] > self.ymin_init - 0.5*self.dy  and self.vor.points[i][1] < self.ymax_init + 0.5*self.dy):
                ver_index[certies] = True


        #----------------------------------
        # otsechenie pogranichnux regionov i poisk min max ploshadei
        #----------------------------------
        for jj in range(len(self.vor.regions)):
            region = self.vor.regions[jj]
            if not -1 in region and not False in ver_index[region]:
                self.region_mask[jj] = True
            else:
                self.region_mask[jj] = False

        #----------------------------------
        # sostavlenie spiska blishaishix sosedei (self.neigh_id)  self.neigh_id[i] = [id_1, id_2, ... , id_n];   i - index of particle, id_1,2...n - индексы ссоседних частиц
        #----------------------------------
        for rid in self.vor.ridge_points:   # get adjasent particles - > [id1, id2] -> id1 and id2 - index of particle in array
                                            # but trully id of particle from MD consist in dat_xyz[:, 0] ()
            rid0 = int(self.dat_xyz[rid[0], 0]) # index of particle
            rid1 = int(self.dat_xyz[rid[1], 0]) # index of adjasent particle

            # at least one particle in the region two adjacent particles: one inside the other outside the region
            # region_id = self.vor.point_region[rid[0]]
            if self.region_mask[self.vor.point_region[rid[0]]] + self.region_mask[self.vor.point_region[rid[1]]] == True:
                if rid1 not in self.neigh_id[rid0]:  # condition to avoid repetition

                    self.neigh_id[rid0].append(rid1)

                    # calc the distance the betwen central parrticle and its adjacent particle
                    x0 = self.dat_xyz_init[rid0, 1].copy()
                    y0 = self.dat_xyz_init[rid0, 2].copy()
                    z0 = self.dat_xyz_init[rid0, 3].copy()
                    x1 = self.dat_xyz_init[rid1, 1].copy()
                    y1 = self.dat_xyz_init[rid1, 2].copy()
                    z1 = self.dat_xyz_init[rid1, 3].copy()

                    x1, y1 = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1)
                    self.dx_dy_ij[rid0].append([(x1 - x0), (y1 - y0), abs(z1 - z0)])

                    self.z_distance.append(z1-z0)

                    rij = ((x1 - x0) **2 + (y1 - y0) ** 2) ** 0.5
                    self.r_ij[rid0].append(rij)

                    self.r_vor.append(rij)
                    if rij < self.r_snake:
                        self.triple_point_list[rid0] += 1

                if rid0 not in self.neigh_id[rid1]:
                    self.neigh_id[rid1].append(rid0)

                    # calc the distance the betwen central parrticle and its adjacent particle
                    x0 = self.dat_xyz_init[rid1, 1].copy()
                    y0 = self.dat_xyz_init[rid1, 2].copy()
                    z0 = self.dat_xyz_init[rid1, 3].copy()
                    x1 = self.dat_xyz_init[rid0, 1].copy()
                    y1 = self.dat_xyz_init[rid0, 2].copy()
                    z1 = self.dat_xyz_init[rid0, 3].copy()

                    x1, y1 = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1)
                    self.dx_dy_ij[rid1].append([(x1 - x0), (y1 - y0), abs(z1 - z0)])
                    self.z_distance.append(z1-z0)

                    rij = ((x1 - x0) **2 + (y1 - y0) ** 2) ** 0.5


                    self.r_ij[rid1].append(rij)
                    self.r_vor.append(rij)
                    if rij < self.r_snake:
                        self.triple_point_list[rid1] += 1

        del ver_index
        del rij
        del x0
        del x1
        del y0
        del y1
        del z0
        del z1
        del rid0
        del rid1

        return self.neigh_id, self.vor, self.region_mask

    def Consideration_Periodic_Boundary_Distance(self, x0, y0, x1, y1):
        if(abs(x1 - x0) > self.xmax_init/2.2 and x1 < x0):
            xi = x1 + self.xmax_init
        elif(abs(x1 - x0) > self.xmax_init/2.2 and x1 > x0):
            xi = x1 - self.xmax_init
        else:
            xi = x1


        if(abs(y1 - y0) > self.ymax_init/2.2 and y1 < y0):
            yi = y1 + self.ymax_init
        elif(abs(y1 - y0) > self.ymax_init/2.2 and y1 > y0):
            yi = y1 - self.ymax_init
        else:
            yi = y1
        return xi, yi

    def Add_Points(self):
        # ========================== INFORMATION ===============================
        # function of adding additional points to the array
        # of particles taking into account the periodic boundary

        self.dat_xyz = np.array(self.st[-1]).copy()
        id_part    = self.dat_xyz[:, 0].copy()
        x_part     = self.dat_xyz[:, 1].copy()
        y_part     = self.dat_xyz[:, 2].copy()
        z_part     = self.dat_xyz[:, 3].copy()


        # ======== define box size and area of cut line ===================
        box_data = self.st[1]
        x_left      = box_data[0][0].copy()
        x_right     = box_data[0][1].copy()
        y_up        = box_data[1][1].copy()
        y_bottom    = box_data[1][0].copy()

        self.x_box       = x_right - x_left
        self.y_box       = y_up - y_bottom
        self.dx = self.percent * self.x_box
        self.dy = self.percent * self.y_box

        for i in range(self.st[0]):

            if x_part[i] < x_left + self.dx:

                self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] + self.x_box, y_part[i], z_part[i]]], axis = 0).copy()

                if  y_part[i] < y_bottom + self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] + self.x_box , y_part[i] + self.y_box, z_part[i] ]], axis = 0).copy()
                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] + self.y_box, z_part[i] ]], axis = 0).copy()

                elif  y_part[i] > y_up - self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] + self.x_box, y_part[i] - self.y_box, z_part[i] ]], axis = 0).copy()
                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] - self.y_box, z_part[i] ]], axis = 0).copy()

            elif x_part[i] > x_right - self.dx:

                self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] - self.x_box, y_part[i], z_part[i] ]], axis = 0).copy()

                if y_part[i] < y_bottom + self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] - self.x_box, y_part[i] + self.y_box, z_part[i] ]], axis = 0).copy()
                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] + self.y_box, z_part[i] ]], axis = 0).copy()

                elif y_part[i] > y_up - self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] - self.x_box, y_part[i] - self.y_box, z_part[i] ]], axis = 0).copy()
                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] - self.y_box, z_part[i] ]], axis = 0).copy()

            elif y_part[i] > y_up - self.dy:

                self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] - self.y_box, z_part[i] ]], axis = 0).copy()

            elif y_part[i] < y_bottom + self.dy:

                self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] + self.y_box, z_part[i] ]], axis = 0).copy()

        del id_part
        del x_part
        del y_part
        del z_part
        del x_left
        del x_right
        del y_up
        del y_bottom

        #return self.dat_xyz, x_left, x_right, y_bottom, y_up, self.dx, self.dy

    def Plot_Gist(self, data, n_elem_gist, plot_type):
        ff=plt.figure(figsize=(15.0,15.0 * 0.8))
        plt.hist(data, n_elem_gist)
        plt.xlabel(plot_type)
        #plt.xlim([1, ])
        plt.savefig(self.img_path + 'Gist' + plot_type + '_.pdf',  bbox_inches='tight',pad_inches = 0.05)
        plt.close(ff)
        ff.clear()
        del ff

    def Plot_Voro_System(self, xy, plot_type, frame):
        ff=plt.figure(figsize=(30, 30 * (max(xy[:, 1]) - min(xy[:, 1])) / (max(xy[:, 0]) - min(xy[:, 0]))))
        ax = plt.subplot(111)
        ax.set_xlim([min(xy[:, 0]), max(xy[:, 0])])
        ax.set_ylim([min(xy[:, 1]), max(xy[:, 1])])

        ax.set_aspect('equal', 'datalim')
        ax.grid(color='r', linestyle='-', linewidth=2)

        for jj in range(len(self.vor.points)):
            region_id = self.vor.point_region[jj]
            if self.region_mask[region_id]:
                polygon = [self.vor.vertices[i] for i in self.vor.regions[region_id]]
                ax.fill(*zip(*polygon), facecolor = "white", edgecolor = 'black', linewidth = 0.7)
                ax.text(self.vor.points[jj][0], self.vor.points[jj][1], str(int(self.dat_xyz[jj, 0])))
                if jj<self.nparts_init:
                    ax.text(self.vor.points[jj][0], self.vor.points[jj][1], str(int(self.dat_xyz_init[jj, 0])))
                    ax.text(self.vor.points[jj][0], self.vor.points[jj][1], str(jj))

        rect = patches.Rectangle((self.xmin_init, self.ymin_init), self.xmax_init - self.xmin_init, self.ymax_init - self.ymin_init, linewidth = 5, edgecolor='Black', facecolor='none')
        ax.add_patch(rect)

        plt.savefig(self.img_path + plot_type +  '_' + str(frame) + '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
        ff.clear()
        plt.close(ff)
        del ff

    def Init_Main_Variable(self):
        return 0

    def Plot_Sigma_2t(self, plot_type):
        ff = plt.figure(figsize=(22,22 ))
        ax = plt.subplot(111)
        ax.plot(np.arange(0, len(self.sigma2t)), self.sigma2t, 'o-')    #mh1
        plt.savefig(self.img_path + plot_type +  '_.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
        ff.clear()
        plt.close(ff)

    def Plot_System(self, xy, plot_type, frame):
        ff = plt.figure(figsize=(30,30 * (max(xy[:, 1]) - min(xy[:, 1])) / (max(xy[:, 0]) - min(xy[:, 0]))))
        ax = plt.subplot(111)
        ax.set_xlim([min(xy[:, 0]), max(xy[:, 0])])
        ax.set_ylim([min(xy[:, 1]), max(xy[:, 1])])

        for id_part in range(self.nparts_init):
            ax.text(xy[id_part, 0], xy[id_part, 1], str(id_part), fontsize = 6)

        ax.set_aspect('equal', 'datalim')
        ax.plot(xy[:, 0], xy[:, 1], 'o')    #mh1

        plt.savefig(self.img_path + plot_type +  '_' + str(frame) + '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
        ff.clear()
        plt.close(ff)

    def Plot_Snakes(self, xy, plot_type, name_file, frame):

        #plot identified snakes
        full_name = self.txt_path + name_file + ".txt"
        f = open(full_name, "w")

        jj_using = np.full(self.nparts_init, False)

        ff=plt.figure(figsize=(30,30 * (max(xy[:, 1]) - min(xy[:, 1])) / (max(xy[:, 0]) - min(xy[:, 0]))))
        ax = plt.subplot(111)
        ax.set_xlim([min(xy[:, 0]), max(xy[:, 0])])
        ax.set_ylim([min(xy[:, 1]), max(xy[:, 1])])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_aspect('equal', 'datalim')


        ax.plot(xy[:, 0], xy[:, 1], '.', color = "black")    #mh1
        for list_el in self.snakes_id:    #mh1
            ax.plot(xy[list_el, 0], xy[list_el, 1], 'o')    #mh1

        for id_snake in range(len(self.snakes_id)):
            list_el = self.snakes_id[id_snake]

            for el in list_el:
                ax.text(xy[el, 0], xy[el, 1], str(id_snake), fontsize = 5)
                jj_using[el] = True
                f.write(str(xy[el, 0]) + " " + str(xy[el, 1]) + " " + str(id_snake) + "\n")

        for el in range(self.nparts_init):
            if not jj_using[el]:
                f.write(str(xy[el, 0]) + " " + str(xy[el, 1]) + " " + str(-1) + "\n")
        f.close()


        plt.savefig(self.img_path + plot_type +  '_' + str(frame) + '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
        plt.close(ff)
        ax.cla()
        ff.clear()

        del f
        del ff
        del full_name
        del jj_using

    def IdentShake(self):
        self.snakes_id = []
        self.angles_in_snakes = []

        jj_using_glob = np.zeros(self.nparts_init)
        jj_triple = np.zeros(self.nparts_init)
        #jj_triple = np.full(self.nparts_init, False)
        for jj in range(self.nparts_init ):
            if jj_using_glob[jj] > 0: # skip the particle if it's in a snake
                continue

            jj_using = [] #array of particles used in snake
            jj_using.append(jj) #jj - first particle in snake
            jj_using_glob[jj] = 1 #jj particle is in snake
            length = 0

            jj_el_snake = jj

            while True: #search end of chain
                r = []

                # distances between central partilce and her neighbors
                for i in range(len(self.neigh_id[jj_el_snake])):
                    if self.r_ij[jj_el_snake][i] < self.r_snake:
                        r.append([i, self.r_ij[jj_el_snake][i]])

                # condition for snake existing

                len_r = len(r)
                if len_r >2:
                    break
                if len_r > 1:
                    if len_r == 2:
                        if self.neigh_id[jj_el_snake][r[0][0]] in jj_using and self.neigh_id[jj_el_snake][r[1][0]] in jj_using:
                            break

                    for item in r:
                        if self.neigh_id[jj_el_snake][item[0]] not in jj_using:
                            length += item[1] #count number of particles in snake
                            jj_using.insert(0, self.neigh_id[jj_el_snake][item[0]]) #add particle to start of array of particles used in snake
                            jj_using_glob[self.neigh_id[jj_el_snake][item[0]]] = 1 #jj particle is in snake
                            jj_el_snake = self.neigh_id[jj_el_snake][item[0]]
                            break

                else:
                    break #find the end


            # ===============================================
            # ===============================================
            # Particle sorting in snakes??????????
            # ===============================================
            # ===============================================
            del(jj_el_snake)
            for jj_el_snake in jj_using: #loop by particles in current snake
                if jj_el_snake != jj_using[-1]: #skip considered particles, search for end of array
                    continue

                r = [] # for devination of snake r must be 3 or more EXAMPLE:  (---+----) [r1 r2 r3]     (---------) [r1 r2]
                        #                                                          |

                for i in range(len(self.neigh_id[jj_el_snake])):
                    if self.r_ij[jj_el_snake][i] < self.r_snake:
                        neigh_id = self.neigh_id[jj_el_snake][i]

                        r.append([i, self.r_ij[jj_el_snake][i]])

                len_r = len(r)

                if r:
                    if jj_triple[jj_el_snake] > 0:
                        jj_using.append(jj_el_snake)

                        if len_r == jj_triple[jj_el_snake]:
                            jj_using_glob[jj_el_snake] = 1

                        break

                    for item in r: #(index of neighbor, r)
                        neigh_part = self.neigh_id[jj_el_snake][item[0]]

                        if self.neigh_id[jj_el_snake][item[0]] not in jj_using and jj_triple[neigh_part] < 1:
                            length += item[1] #count number of particles in snake
                            jj_using.append(self.neigh_id[jj_el_snake][item[0]]) #add particle to array of particles used in snake

                            if self.triple_point_list[self.neigh_id[jj_el_snake][item[0]]] > 2:
                                jj_triple[self.neigh_id[jj_el_snake][item[0]]] += 1

                                if jj_triple[self.neigh_id[jj_el_snake][item[0]]] == self.triple_point_list[self.neigh_id[jj_el_snake][item[0]]]:
                                    jj_using_glob[self.neigh_id[jj_el_snake][item[0]]] = 1 #jj particle is in snake
                                continue

                            else:
                                jj_using_glob[self.neigh_id[jj_el_snake][item[0]]] = 1 #jj particle is in snake
                del r

            if len(jj_using) > 2:
                self.snakes_id.append(jj_using) #add current snake to array of snakes' ID
                self.angles_in_snakes.append([])
                self.len_snakes.append(length) #add length of current snake to array of snakes' lengths
            del jj_using


        for id_snake in range(len(self.snakes_id)):
            angles = []
            len_snake = len(self.snakes_id[id_snake])

            for index in range(len_snake - 2):
                if index + 2 < len_snake:

                    id_init = self.snakes_id[id_snake][index]
                    id_center = self.snakes_id[id_snake][index+1]
                    id_end = self.snakes_id[id_snake][index+2]

                    x0, y0, z0 = self.dat_xyz_init[id_init, 1], self.dat_xyz_init[id_init, 2], self.dat_xyz_init[id_init, 3]
                    x1, y1, z1 = self.dat_xyz_init[id_center, 1], self.dat_xyz_init[id_center, 2], self.dat_xyz_init[id_center, 3]
                    x2, y2, z2 = self.dat_xyz_init[id_end, 1], self.dat_xyz_init[id_end, 2], self.dat_xyz_init[id_end, 3]

                    x1t, y1t = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1)
                    x2t, y2t = self.Consideration_Periodic_Boundary_Distance(x1, y1, x2, y2)

                    x_init_center = x1t - x0
                    y_init_center = y1t - y0

                    x_center_end = x2t - x1
                    y_center_end = y2t - y1

                    radians = self.Angle_Of_Vectors([x_center_end, y_center_end], [x_init_center, y_init_center])
                    if math.isnan(radians):
                        continue
                    self.agnles_stat.append(radians)
                    angles.append(radians)


            self.angles_in_snakes[id_snake] = angles
            del angles

        del jj_using_glob
        del jj_triple

    def Init_Parameter(self):

        self.len_cadrs = self.num_cadr#len(range(istart, self.all_cadr, math.ceil((self.all_cadr ) / self.num_cadr)))
        #self.neigh_id,   self.dx_dy_ij init in Get_Voronoi_Neighbors
        #self.snakes_id, , self.len_snakes  init in Snake Ident
        self.len_snakes = []
        #self.angles_in_snakes
        self.agnles_stat = []
        self.z_distance = []    # in IdentShake
        self.z_coord = []
        self.r_vor = []

        self.dx_diff = np.zeros((self.nparts_init, self.num_cadr+1)) # Particle_Dispacment_1_Timestep and Get_Sigma2t
        self.dy_diff = np.zeros((self.nparts_init, self.num_cadr+1))  # Particle_Dispacment_1_Timestep and Get_Sigma2t
        self.dr_diff = [] # Particle_Dispacment_1_Timestep
        self.sigma2t = np.zeros(self.len_cadrs) #Get_Sigma2t

        self.psi = np.zeros((self.nparts_init, len(self.list_psi))) # in Calc_Parameter
        self.psi_array_gist = [[] for _ in range(len(self.list_psi))] # in Calc_Parameter

        self.R_field_D = np.zeros((self.nparts_init, self.num_cadr))
        self.min_rdf = 100000
        self.max_rdf = -1000000



    def Calc_Parameter(self, ts):
         #----------------------------------
        R_field = np.zeros(self.nparts_init)

        for jj in range(self.nparts_init):
            tt = np.array(self.r_ij[jj]).var()
            R_field[jj] = tt

            for psi_i in range(len(self.list_psi)):
                psi_k = self.list_psi[psi_i]
                self.psi_array_gist[psi_i].append(self.Get_Psik_Single(np.array(self.dx_dy_ij[jj]), psi_k)) # StructureAnalysis
                self.psi[jj, psi_i] += self.Get_Psik_Single(np.array(self.dx_dy_ij[jj]), psi_k)

            self.Particle_Dispacment_1_Timestep(jj, ts)
            if ts == self.len_cadrs-1:
                self.Get_Sigma2t(jj)


        for jj in range(self.nparts_init):
            tt = np.array(R_field[self.neigh_id[jj] + [jj]])
            tt = tt[tt > -0.5]
            tt2 = tt.var()
            self.R_field_D[jj, ts] = tt2

            if ts == 1:
                if np.log(tt2) > self.max_rdf:
                    self.max_rdf = np.log(tt2)
                if np.log(tt2) < self.min_rdf:
                    self.min_rdf = np.log(tt2)

            if ts > 1:
                if jj == 0:
                    self.counter_rdf += 1
                region_id = self.vor.point_region[jj]
                region_xy =  np.array([list(self.vor.vertices[i]) for i in self.vor.regions[region_id]])
                voronoi_area = self.Poly_Area(region_xy[:, 0], region_xy[:, 1]).copy()
                ind_rdf = int( (np.log(tt2) +  abs(self.min_rdf)) / self.d_rdf )

                if ind_rdf < 0 or ind_rdf >= self.len_rdf_setka:
                    continue
                else:
                    self.S_voro[ind_rdf] += voronoi_area

        if ts == 1:
            self.log_rdf_setka = np.linspace(self.min_rdf, self.max_rdf, 200)
            self.len_rdf_setka = len(self.log_rdf_setka)
            self.d_rdf = self.log_rdf_setka[1] - self.log_rdf_setka[0]
            self.S_voro = np.zeros(self.len_rdf_setka)


    def Particle_Dispacment_1_Timestep(self, id_part, ts):
        # func to calc gist for diffusion
        # INPUT:
        # solid_part - массив твердых частиц (в этом коде он не соответвует индексам частиц)
        # это массив номера строчек частиц

        # массивы координат частиц общиме массивы без разделения на фазы
        #          mas_part_dif0,              mas_part_dif_ts_1,                 mas_part_dif_ts
        # начальный (нулевой) таймстеп        предыдущий таймстеп                текущий таймстеп
        # x_left, x_right, y_up, y_bottom - размеры области моделирования

        #for i in range(self.nparts_init):
        x0 = self.dat_xyz_previusly[id_part, 1].copy()
        y0 = self.dat_xyz_previusly[id_part, 2].copy()

        x1 = self.dat_xyz_init[id_part, 1].copy()
        y1 = self.dat_xyz_init[id_part, 2].copy()

        x1, y1 = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1)

        self.dx_diff[id_part, ts] = x1 - x0
        self.dy_diff[id_part, ts] = y1 - y0
        self.dr_diff.append(((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0))**0.5)
        del x0, x1
        del y0, y1

    def Get_Sigma2t(self, id_part):
        # calc the sigma ^2 values:
        for ts_ in range(self.len_cadrs-1):
            dx = sum(self.dx_diff[id_part, 0: ts_+1])
            dy = sum(self.dy_diff[id_part, 0: ts_+1])
            self.sigma2t[ts_] += (dx*dx + dy*dy)  / self.nparts_init

    def Save_Txt_File(self, data, data_name, dimension):
        full_name = self.txt_path + data_name + ".txt"
        f = open(full_name, "w")
        if dimension == 1:
            a = len(data)
            for i in range(a):
                f.write(str(data[i]) + "\n")

        elif dimension == 2:
            row, col = data.shape
            for i in range(row):
                for j in range(col):
                    f.write(str(data[i, j]) + " ")
                f.write("\n")
        f.close()
        del f
    def Save_All(self):
        # System's parameters:
        bins = 50
        z_coord_hist, z_coord_bins = np.histogram(self.z_coord, bins=bins)
        self.Save_Txt_File(np.column_stack([z_coord_bins[0:bins], z_coord_hist]), "Z_coord" + self.past_name, 2)
        self.Save_Txt_File([sum(self.z_coord)/len(self.z_coord)], "Av_Zcoord" + self.past_name, 1)
        del z_coord_hist
        del z_coord_bins


        r_vor_hist, r_vor_bins = np.histogram(self.r_vor, bins= bins)
        self.Save_Txt_File(np.column_stack([r_vor_bins[0:bins], r_vor_hist]), "R_vor" + self.past_name, 2)
        self.Save_Txt_File([sum(self.r_vor)/len(self.r_vor)], "Av_r_vor" + self.past_name, 1)
        del r_vor_hist
        del r_vor_bins


        z_distance_hist, z_distance_bins = np.histogram(self.z_distance, bins=bins)
        self.Save_Txt_File(np.column_stack([z_distance_bins[0:bins], z_distance_hist]), "_Zdistance" + self.past_name, 2)
        self.Save_Txt_File([sum(self.z_distance)/len(self.z_distance)], "Av_Zdistance" + self.past_name, 1)
        del z_distance_hist
        del z_distance_bins


        dr_diff_hist, dr_diff_bins = np.histogram(self.dr_diff, bins=bins)
        self.Save_Txt_File(np.column_stack([dr_diff_bins[0:bins], dr_diff_hist]), "_dr_diff" + self.past_name, 2)
        self.Save_Txt_File([sum(self.dr_diff)/len(self.dr_diff)], "Av_dr_diff" + self.past_name, 1)
        del dr_diff_hist
        del dr_diff_bins

        self.Save_Txt_File(np.column_stack([np.exp(self.log_rdf_setka), self.S_voro/self.counter_rdf]), "S_vor(RDF)_" + self.past_name, 2)
        self.Save_Txt_File([np.sum(self.R_field_D[:, 1:self.num_cadr]) / self.num_cadr / self.nparts_init], "Av_RDF" + self.past_name, 1)


        #self.Plot_Sigma_2t("Gist_Sigma2t" + self.past_name)

        # Snakes's parameters:
        #self.Plot_Gist(self.len_snakes, bins, "_Length" + self.past_name)
        #self.Plot_Gist(self.agnles_stat, bins,"_Agnles" + self.past_name)


        # Triplets
        all_triplet = self.triple_point_list[self.triple_point_list>2]
        if len(all_triplet) > 0:
            self.Save_Txt_File(self.triple_point_list[self.triple_point_list>2], "Triplets" + self.past_name, 1)
            self.Save_Txt_File([sum(self.triple_point_list[self.triple_point_list>2])/len(self.triple_point_list[self.triple_point_list>2])], "AvTripletspoint" + self.past_name, 1)
        else:
            self.Save_Txt_File([0], "Triplets" + self.past_name, 1)
            self.Save_Txt_File([0], "AvTripletspoint" + self.past_name, 1)

        for id_psi in range(len(self.list_psi)):
            psik = self.list_psi[id_psi]
            psi_hist, psi_bins = np.histogram(self.psi_array_gist[id_psi], bins=bins)
            self.Save_Txt_File(np.column_stack([psi_bins[0:bins], psi_hist]), "GistPsi"+ str(psik) + self.past_name, 2)
            self.Save_Txt_File([sum(self.psi_array_gist[id_psi])/len(self.psi_array_gist[id_psi])], "AvPsi"+ str(psik) + self.past_name, 1)
            del psi_hist
            del psi_bins


        # System's parameters:
        self.Save_Txt_File(self.sigma2t, "Sigma2t" + self.past_name, 1)

        # Snakes's parameters:
        if len(self.snakes_id) > 3:
            len_snakes_hist, len_snakes_bins = np.histogram(self.len_snakes, bins=bins)
            self.Save_Txt_File(np.column_stack([len_snakes_bins[0:bins], len_snakes_hist]), "Length" + self.past_name, 2)
            self.Save_Txt_File([sum(self.len_snakes)/len(self.len_snakes)], "Av_Length" + self.past_name, 1)
            del len_snakes_hist
            del len_snakes_bins

            agnles_stat_hist, agnles_stat_bins = np.histogram(self.agnles_stat, bins=bins)
            self.Save_Txt_File(np.column_stack([agnles_stat_bins[0:bins], agnles_stat_hist]), "Angle" + self.past_name, 2)
            self.Save_Txt_File([sum(self.agnles_stat)/len(self.agnles_stat)], "Av_Angle" + self.past_name, 1)
            del agnles_stat_hist
            del agnles_stat_bins
        else:
            self.Save_Txt_File(np.array([[0, 0]]), "Length" + self.past_name, 2)
            self.Save_Txt_File(np.array([0]), "Av_Length" + self.past_name, 1)

            self.Save_Txt_File(np.array([[0, 0]]), "Angle" + self.past_name, 2)
            self.Save_Txt_File(np.array([0]), "Av_Angle" + self.past_name, 1)


    def Calc_Loop2(self):
        # open dump -> calc looprelative the pindividual particles -> del dump -> get ris and txt data

        dump = LammpsCoords(string_content = 'id x y z ',
                            required_data = 'id x y z ',
                            path = self.full_md_name)

        dump.OpenTrjGetInfo(1)

        self.st = dump.ReadState().copy()
        self.nparts_init = self.st[0]
        self.Init_Parameter()

        ts_index = 0
        dr_pcf = 0.1

        #radii_pcf =  np.arange(0., self.r_pcf + 1.1 * dr_pcf, dr_pcf)
        #gr = np.zeros(len(radii_pcf) - 1)
        count_cadrs = 0

        for index in range(self.num_cadr):

            index__1 = index
            count_cadrs += 1

            self.st = dump.ReadState().copy()
            self.dat_xyz_init = np.array(self.st[-1]).copy()

            self.xmin_init, self.xmax_init = min(self.dat_xyz_init[:, 1]), max(self.dat_xyz_init[:, 1])
            self.ymin_init, self.ymax_init = min(self.dat_xyz_init[:, 2]), max(self.dat_xyz_init[:, 2])
            self.z_coord.append(self.dat_xyz_init[:, 3])

            self.Add_Points()

            if min(self.dat_xyz[:, 0]) == 1:
                self.dat_xyz[:, 0] -= 1
            if min(self.dat_xyz_init[:, 0]) == 1:
                self.dat_xyz_init[:, 0] -= 1

            self.Get_Voronoi_Neighbors()
            self.IdentShake()

            # ============== PCF ==============================
            #pcf, radii_pcf = self.GetPCF(self.dat_xyz_init[:, 1:4], 2, r_pcf, dr_pcf)
            #gr += pcf
            # ============== PCF ==============================

            ts_index += 1
            print('ts_index = ', ts_index)
            if ts_index == 1:
                print("no_PLOT")
                #self.Plot_Snakes(self.dat_xyz_init[:, 1:3], "system/Ident" + self.past_name, "snakes_id" + self.past_name, index__1)

            if ts_index >= 2:
                self.Calc_Parameter(ts_index-1)

            self.dat_xyz_previusly = self.dat_xyz_init.copy()
            print(self.counter_rdf)

        dump.CloseFile()
        del dump

        self.Save_All()



#path_md = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/MD/MIPS_prove_rdf/"
path_md = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/MD/Nikita/Data"
path = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/MD/Nikita/Data"

txt_path = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/result/txt_0.6/"
img_path = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/result/ris_v3/"
prefix = "dump_100_100_"
density = [0.6]

num_h = 19
num_mu = 50#bins

heights = np.arange(1, 1 + 0.025 * num_h, 0.025)


mu_values = np.arange(5.0, 5. + 0.2 * num_mu, 0.2)

Temperature = 1.0
istart = 2
all_cadr = 97
num_cadr = 89
list_triplets = [3, 4, 5, 6]

mips_names = ["T_0.1_rho_1.0_Dr_0.05_gamma_2.0_Act_35_ts_0.000770715_eps_0.lammpstrj",
              "T_0.1_rho_1.0_Dr_0.05_gamma_2.0_Act_16_ts_0.000770715_eps_0.lammpstrj",
              "T_0.1_rho_1.0_Dr_0.05_gamma_2.0_Act_10_ts_0.000770715_eps_0.lammpstrj"]
past_names = ["Act_35",
              "Act_16",
              "Act_10"]
for id_rho in range(len(density)):
    rho = density[id_rho]

    for id_h in range(num_h):
        h = 1.0 + round(0.025, 3)*int(id_h)



        if id_h % 4 == 0:
            path_md = path + str(rho) + "_"
            path_md += (str(h) + "/")
            print(path_md)

        for id_mu in range(num_mu):
            mu = round(mu_values[id_mu], 3)

            if h < 1.025:
                if mu < 11:
                    continue

            str_h = '%.8f' % h
            str_mu = '%.8f' % mu
            str_rho = '%.8f' % rho
            str_temp = '%.8f' % Temperature

            new_prefix = prefix + str_rho + "_" + str_h + "_" + str_temp + "_" +str_mu
            past_name =  "_" + str_rho + "_" + str_h + "_" + str_temp + "_" + str_mu
            name = path_md + new_prefix + "_0.lammpstrj"

            #print("rho = %s;   h = %s;    mu = %s"%(rho, h, mu))
            #print(name, end="\n\n")


            KingCobra = Snake(heights = heights,
                   mu_values = mu_values,
                   Temperature = 1.,
                   percent = 0.14,
                   r_pcf = 5.,
                   r_snake = 0.9,
                   density = density,
                   list_psi = [3, 4, 5, 6],

                   istart = istart,
                   all_cadr = all_cadr,
                   num_cadr = num_cadr,

                   txt_path = txt_path,
                   img_path = img_path,
                   past_name = past_name,

                   full_md_name = name
                   )
            KingCobra.Calc_Loop2()
            del KingCobra
            #time.sleep(30)
















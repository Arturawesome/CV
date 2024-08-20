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


import sys

import os


import datetime



class ActivePhaseDiagram():
	def __init__(self,
			   save_path_txt: str,
			   save_path_img: str,
			   path_to_md: str,
			   name_md: str,
			   last_name_md: str,
			   num_cadr: int,
			   flag_R_d_filed: bool
			   ):
		self.save_path_txt = save_path_txt
		self.save_path_img = save_path_img
		self.path_to_md = path_to_md
		self.name_md = name_md
		self.last_name_md = last_name_md

		self.full_md_name = None
		self.st = None
		self.nparts_init = None
		self.num_cadr = num_cadr
		self.flag_R_d_filed = flag_R_d_filed

		self.dat_xyz_init = None
		self.dat_xyz = None
		self.dat_xyz_prev = None
		self.dt = None

		self.xmin_init = None
		self.xmax_init = None
		self.ymin_init = None
		self.ymax_init = None
		self.x_box = None
		self.y_box = None
		self.neigh_id = None
		self.vor = None
		self.region_mask = None
		self.max_rdf = -100000000000
		self.min_rdf = 100000000000
		self.rho_gas = 0
		self.rho_cond = 0

		self.solid_phase = None
		self.liquid_phase  = None
		self.gas_phase = None
		self.sl_border = None
		self.lg_border = None
		self.log_rdf_setka = None
		self.S_voro = None


		self.n_part_phase = {"g":None, "s":None, "l":None, "c":None, "slb":None, "lgb":None}
		self.power = {"g":None, "s":None, "l":None, "c":None}
		self.av_v = {"g":None, "s":None, "l":None, "c":None}
		self.v_gist = {"g":None, "s":None, "l":None, "c":None}
		self.diff = {"all":None, "s":None, "l":None, "g":None, "c":None}
		self.vv_corr = {"g":None, "s":None, "l":None, "c":None}
		self.dencity = {"g":None, "s":None, "l":None, "c":None}
		self.A = None






	def __del__(self):
		print("==============================")
		print("ActivePhaseDiagram Destucture: clear memory")
		print("==============================")
		class_name = self.__class__.__name__
		print('{} delete'.format(class_name))

	@staticmethod
	def Add_Points(st, xmin_init, xmax_init, ymin_init, ymax_init, percent):
		# ========================== INFORMATION ===============================
		# function of adding additional points to the array
		# of particles taking into account the periodic boundary
		dat_xyz = np.array(st[-1]).copy()
		id_part    = dat_xyz[:, 0].copy()
		x_part     = dat_xyz[:, 1].copy()
		y_part     = dat_xyz[:, 2].copy()
		z_part     = dat_xyz[:, 3].copy()

		# ======== define box size and area of cut line ===================
		box_data = st[1]

		x_box       = xmax_init - xmin_init
		y_box       = ymax_init - ymin_init
		dx = percent * x_box
		dy = percent * y_box

		for i in range(st[0]):
			if x_part[i] < xmin_init + dx:
				dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i] + x_box, y_part[i], z_part[i]]], axis = 0).copy()

				if  y_part[i] < ymin_init + dy:
					dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i] + x_box , y_part[i] + y_box, z_part[i] ]], axis = 0).copy()
					dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i], y_part[i] + y_box, z_part[i] ]], axis = 0).copy()

				elif  y_part[i] > ymax_init - dy:
					dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i] + x_box, y_part[i] - y_box, z_part[i] ]], axis = 0).copy()
					dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i], y_part[i] - y_box, z_part[i] ]], axis = 0).copy()

			elif x_part[i] > xmax_init - dx:
				dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i] - x_box, y_part[i], z_part[i] ]], axis = 0).copy()

				if y_part[i] < ymin_init + dy:
					dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i] - x_box, y_part[i] + y_box, z_part[i] ]], axis = 0).copy()
					dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i], y_part[i] + y_box, z_part[i] ]], axis = 0).copy()

				elif y_part[i] > ymax_init - dy:
					dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i] - x_box, y_part[i] - y_box, z_part[i] ]], axis = 0).copy()
					dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i], y_part[i] - y_box, z_part[i] ]], axis = 0).copy()

			elif y_part[i] > ymax_init - dy:
				dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i], y_part[i] - y_box, z_part[i] ]], axis = 0).copy()

			elif y_part[i] < ymin_init + dy:
				dat_xyz = np.append(dat_xyz, [[id_part[i], x_part[i], y_part[i] + y_box, z_part[i] ]], axis = 0).copy()

		del id_part, x_part, y_part, z_part

		return dat_xyz, dx, dy
		# ========================================================================================================
		# ========================================================================================================
		# ========================================================================================================

	@staticmethod
	def Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1, x_box, y_box):

		if(abs(x1 - x0) > x_box/2.2 and x1 < x0):
			xi = x1 + x_box
		elif(abs(x1 - x0) > x_box/2.2 and x1 > x0):
			xi = x1 - x_box
		else:
			xi = x1

		if(abs(y1 - y0) > y_box/2.2 and y1 < y0):
			yi = y1 + y_box
		elif(abs(y1 - y0) > y_box/2.2 and y1 > y0):
			yi = y1 - y_box
		else:
			yi = y1
		return xi, yi

	@classmethod
	def Get_Voronoi_Neighbors(self, nparts_init, dat_xyz, dat_xyz_init, xmin_init, xmax_init, ymin_init, ymax_init, x_box, y_box, dx, dy, flag_rvor):
		# function for
		vor = scipy.spatial.Voronoi(dat_xyz[:, 1:3])
		region_mask = np.full(len(vor.regions), False)
		npoints = len(vor.points)
		ver_index = np.full(len(vor.vertices),True)
		neigh_id = [[] for i in range(nparts_init)]

		r_ij = [[] for _ in range(nparts_init)]
		dx_l, dx_r, dy_b, dy_a = dx[0], dx[1], dy[0], dy[1]


		#----------------------------------
		# otsechenie pogranichnux regionov i poisk min max ploshadei
		#----------------------------------

		for i in range(len(vor.vertices)):
			ver=vor.vertices[i]
			if ver[0]<xmin_init + dx_l or ver[0]>xmax_init - dx_r or ver[1]<ymin_init + dy_b or ver[1]>ymax_init-dy_a:
				ver_index[i] = False

		#----------------------------------
		# otsechenie pogranichnux regionov i poisk min max ploshadei
		#----------------------------------
		for jj in range(len(vor.regions)):
			region = vor.regions[jj]
			if not -1 in region and not False in ver_index[region]:
				region_mask[jj] = True
			else:
				region_mask[jj] = False

		#----------------------------------
		# sostavlenie spiska blishaishix sosedei (self.neigh_id)  self.neigh_id[i] = [id_1, id_2, ... , id_n];   i - index of particle, id_1,2...n - индексы ссоседних частиц
		#----------------------------------
		for rid in vor.ridge_points:   # get adjasent particles - > [id1, id2] -> id1 and id2 - index of particle in array
											# but trully id of particle from MD consist in dat_xyz[:, 0] ()
			rid0 = int(dat_xyz[rid[0], 0]) # index of particle
			rid1 = int(dat_xyz[rid[1], 0]) # index of adjasent particle

			# at least one particle in the region two adjacent particles: one inside the other outside the region
			# region_id = self.vor.point_region[rid[0]]
			if region_mask[vor.point_region[rid[0]]] + region_mask[vor.point_region[rid[1]]] == True:
				if rid1 not in neigh_id[rid0]:  # condition to avoid repetition
					neigh_id[rid0].append(rid1)

					# calc the distance the betwen central parrticle and its adjacent particle
					if flag_rvor:
						x0 = dat_xyz_init[rid0, 1].copy()
						y0 = dat_xyz_init[rid0, 2].copy()
						x1 = dat_xyz_init[rid1, 1].copy()
						y1 = dat_xyz_init[rid1, 2].copy()
						x1, y1 = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1, x_box, y_box)

						rij = ((x1 - x0) **2 + (y1 - y0) ** 2) ** 0.5
						r_ij[rid0].append(rij)

				if rid0 not in neigh_id[rid1]:
					neigh_id[rid1].append(rid0)

					# calc the distance the betwen central parrticle and its adjacent particle
					if flag_rvor:
						x0 = dat_xyz_init[rid1, 1].copy()
						y0 = dat_xyz_init[rid1, 2].copy()
						x1 = dat_xyz_init[rid0, 1].copy()
						y1 = dat_xyz_init[rid0, 2].copy()
						x1, y1 = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1, x_box, y_box)

						rij = ((x1 - x0) **2 + (y1 - y0) ** 2) ** 0.5
						r_ij[rid1].append(rij)

		del ver_index, rid0, rid1,
		return neigh_id, vor, region_mask, r_ij
		# ========================================================================================================
		# ========================================================================================================
		# ========================================================================================================



	@staticmethod
	def Cacl_Phase_Parameter_Step1(nparts_init, neigh_id, partition, r_ij, vor, region_mask):
		#----------------------------------
		max_rdf = -100000000000
		min_rdf = 100000000000
		R_field = np.zeros(nparts_init)

		for jj in range(nparts_init):
			region_id = vor.point_region[jj]
			if region_mask[region_id]:
				R_field[jj] = np.array(r_ij[jj]).var()

		for jj in range(nparts_init):
			region_id = vor.point_region[jj]
			if region_mask[region_id]:
				tt = np.array(R_field[neigh_id[jj] + [jj]])
				tt = tt[tt > -0.5]
				tt2 = tt.var()

				if np.log(tt2) > max_rdf:
					max_rdf = np.log(tt2) + 0.2*np.log(tt2)
				if np.log(tt2) < min_rdf:
					min_rdf = np.log(tt2) - 0.2*np.log(tt2)
		min_rdf = -21#abs(0.8*min_rdf)
		max_rdf = 15#abs(0.1*max_rdf)

		log_rdf_setka = np.linspace(min_rdf, max_rdf, partition)
		len_rdf_setka = len(log_rdf_setka)
		d_rdf = log_rdf_setka[1] - log_rdf_setka[0]
		S_voro = np.zeros(len_rdf_setka)
		return log_rdf_setka, len_rdf_setka, d_rdf, S_voro, min_rdf, max_rdf


	@staticmethod
	def Cacl_Phase_Parameter_Step2(nparts_init, neigh_id,  r_ij, vor, region_mask, len_rdf_setka, S_voro, min_rdf, d_rdf):
		#----------------------------------
		R_field = np.zeros(nparts_init)

		for jj in range(nparts_init):
			region_id = vor.point_region[jj]
			if region_mask[region_id]:
				R_field[jj] = np.array(r_ij[jj]).var()


		for jj in range(nparts_init):
			region_id = vor.point_region[jj]
			if region_mask[region_id]:
				tt = np.array(R_field[neigh_id[jj] + [jj]])
				tt = tt[tt > -0.5]
				tt2 = tt.var()

				region_xy =  np.array([list(vor.vertices[i]) for i in vor.regions[region_id]])
				voronoi_area = SystemStructur.Poly_Area(region_xy[:, 0], region_xy[:, 1]).copy()
				ind_rdf = int((np.log(tt2) +  abs(min_rdf)) / d_rdf )

				if ind_rdf < 0 or ind_rdf >= len_rdf_setka:
					continue
				else:
					S_voro[ind_rdf] += voronoi_area
		return S_voro

	@staticmethod
	def Phase_Work(nparts_init, neigh_id, r_ij, vor, region_mask, gas_liq_param, liq_cryst_param, separate_type):
		if separate_type == "slg":
			solid_phase = np.full(len(vor.regions), 0)
			liquid_phase = np.full(len(vor.regions), 0)
			gas_phase = np.full(len(vor.regions), 0)

			R_field = np.zeros(nparts_init)

			for jj in range(nparts_init):
				tt = np.array(r_ij[jj]).var()
				R_field[jj] = tt

			for jj in range(nparts_init):
				region_id = vor.point_region[jj]
				if region_mask[region_id]:
					tt = np.array(R_field[neigh_id[jj] + [jj]])
					tt = tt[tt > -0.5]
					tt2 = np.log(tt.var())
					if tt2 > gas_liq_param:                      # gas
						gas_phase[region_id] = 1
					elif tt2 < liq_cryst_param:                       # solid
						solid_phase[region_id] = 1
					elif tt2 >= liq_cryst_param and tt2 <= gas_liq_param:     # liquid
						liquid_phase[region_id] = 1
			return solid_phase, liquid_phase, gas_phase


		if separate_type == "cg":
			gas_phase = np.full(len(vor.regions), 0)
			cond_phase = np.full(len(vor.regions), 0)
			R_field = np.zeros(nparts_init)

			for jj in range(nparts_init):
				tt = np.array(r_ij[jj]).var()
				R_field[jj] = tt

			for jj in range(nparts_init):
				region_id = vor.point_region[jj]
				if region_mask[region_id]:
					tt = np.array(R_field[neigh_id[jj] + [jj]])
					tt = tt[tt > -0.5]
					tt2 = np.log(tt.var())
					if tt2 > gas_liq_param:                      # gas
						gas_phase[region_id] = 1
					elif tt2 < gas_liq_param:                       # solid
						cond_phase[region_id] = 1
			return cond_phase, gas_phase

	@staticmethod
	def Phase_Work_Correction_slg(nparts_init, solid_phase, liquid_phase, gas_phase, vor, region_mask, neigh_id, num_corr):
		help_phase_solid = solid_phase.copy()
		help_phase_liquid = liquid_phase.copy()
		help_phase_gas = gas_phase.copy()

		solid_phase1 = solid_phase.copy()
		liquid_phase1 = liquid_phase.copy()
		gas_phase1 = gas_phase.copy()

		# ==========================================================
		# phase correction
		# ==========================================================
		for _ in range(num_corr):
			for jj in range(nparts_init):
				region_id = vor.point_region[jj]
				if region_mask[region_id]:
					neig = neigh_id[jj]
					neig_region_id = vor.point_region[neig]
					if (-sum(solid_phase1[neig_region_id] == 0) + len(solid_phase1[neig_region_id])) < 3:  # cut off the along crystal particle
						help_phase_solid[region_id] = 0
						help_phase_liquid[region_id] = 1

					if (-sum(gas_phase1[neig_region_id] == 0) + len(gas_phase1[neig_region_id])) < 3: # cut off the along gas particle
						help_phase_gas[region_id] = 0
						help_phase_liquid[region_id] = 1

					if (-sum(liquid_phase1[neig_region_id] == 0) + len(liquid_phase1[neig_region_id])) < 3:
						if sum(gas_phase1[neig_region_id] == 1) > 1 and sum(solid_phase1[neig_region_id] == 1) > 0:
							help_phase_liquid[region_id] = 1
							help_phase_gas[region_id] = 0
							continue

						if sum(gas_phase1[neig_region_id] == 1) > 1 :
							help_phase_liquid[region_id] = 0
							help_phase_gas[region_id] = 1

						if sum(solid_phase1[neig_region_id] == 1) > 1:
							help_phase_liquid[region_id] = 0
							help_phase_solid[region_id] = 1

			solid_phase1 = help_phase_solid.copy()
			liquid_phase1 = help_phase_liquid.copy()
			gas_phase1 = help_phase_gas.copy()


		solid_liquid_border = np.full(len(vor.regions), 0)
		liquid_gas_border = np.full(len(vor.regions), 0)




		for jj in range(nparts_init):
			region_id = vor.point_region[jj]
			if region_mask[region_id]:
				neig = neigh_id[jj]
				neig_region_id = vor.point_region[neig]
				if solid_phase1[region_id]:
					if sum(gas_phase1[neig_region_id]) > 0:
						for n_id in solid_phase1:
							help_phase_gas[n_id] = 0
							help_phase_liquid[n_id] = 1



		solid_phase1 = help_phase_solid.copy()
		liquid_phase1 = help_phase_liquid.copy()
		gas_phase1 = help_phase_gas.copy()


		for jj in range(nparts_init):
			region_id = vor.point_region[jj]
			if region_mask[region_id]:
				neig = neigh_id[jj]                         # индексы соседних частиц
				neig_region_id = vor.point_region[neig]     # индексы соседних ячеек вороного

				if  region_mask[region_id]:
					if (liquid_phase1[region_id] == 1) and (sum(solid_phase1[neig_region_id] == 0) < len(solid_phase1[neig_region_id])):
						liquid_phase1[region_id] = 0
						solid_liquid_border[region_id] = 1

					if (liquid_phase1[region_id] == 1) and (sum(help_phase_gas[neig_region_id] == 1) ) and  (sum(help_phase_liquid[neig_region_id] == 1)):
						liquid_phase1[region_id] = 0
						liquid_gas_border[region_id] = 1

		return solid_phase1, liquid_phase1,  gas_phase1, solid_liquid_border, liquid_gas_border

	@staticmethod
	def Phase_Work_Correction_cg(nparts_init, cond_phase, gas_phase, vor, region_mask, neigh_id, num_corr):

		help_phase_cond = cond_phase.copy()
		help_phase_gas = gas_phase.copy()

		cond_phase1 = cond_phase.copy()
		gas_phase1 = gas_phase.copy()

		# ==========================================================
		# phase correction
		# ==========================================================
		for _ in range(num_corr):
			for jj in range(nparts_init):
				region_id = vor.point_region[jj]
				if region_mask[region_id]:
					neig = neigh_id[jj]
					neig_region_id = vor.point_region[neig]
					if (-sum(cond_phase1[neig_region_id] == 0) + len(cond_phase1[neig_region_id])) < 3:  # cut off the along crystal particle
						help_phase_cond[region_id] = 0
						help_phase_gas[region_id] = 1

					if (-sum(gas_phase1[neig_region_id] == 0) + len(gas_phase1[neig_region_id])) < 3: # cut off the along gas particle
						help_phase_gas[region_id] = 0
						help_phase_cond[region_id] = 1



			cond_phase1 = help_phase_cond.copy()
			gas_phase1 = help_phase_gas.copy()

		cond_gas_border = np.full(len(vor.regions), 0)
		for jj in range(nparts_init):
			region_id = vor.point_region[jj]
			if region_mask[region_id]:
				neig = neigh_id[jj]                         # индексы соседних частиц
				neig_region_id = vor.point_region[neig]     # индексы соседних ячеек вороного

				if (gas_phase1[region_id] == 1) and (sum(help_phase_gas[neig_region_id] == 1) ) and  (sum(help_phase_cond[neig_region_id] == 1)):
					gas_phase1[region_id] = 0
					cond_gas_border[region_id] = 1

		return cond_phase1, gas_phase1, cond_gas_border




	def Plot_Voro_System(self, xy, plot_type, frame):
		ff=plt.figure(figsize=(30, 30 * (max(xy[:, 1]) - min(xy[:, 1])) / (max(xy[:, 0]) - min(xy[:, 0]))))
		ax = plt.subplot(111)
		ax.set_xlim([min(xy[:, 0]) - 10, max(xy[:, 0]) + 10])
		ax.set_ylim([min(xy[:, 1]) - 10, max(xy[:, 1]) + 10])

		ax.set_aspect('equal', 'datalim')


		for jj in range(len(self.vor.points)):
			region_id = self.vor.point_region[jj]
			if self.region_mask[region_id]:
				polygon = [self.vor.vertices[i] for i in self.vor.regions[region_id]]
				ax.fill(*zip(*polygon), facecolor = "white", edgecolor = 'black', linewidth = 0.7)

		rect = patches.Rectangle((self.xmin_init, self.ymin_init), self.xmax_init - self.xmin_init, self.ymax_init - self.ymin_init, linewidth = 5, edgecolor='Black', facecolor='none')
		ax.add_patch(rect)
		ax.plot(xy[:, 0], xy[:, 1], ".")

		plt.savefig(self.save_path_img + plot_type +  '_' + str(frame) + '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		ff.clear()
		plt.close(ff)
		del ff



	def PlotPhaseSystem(self, plot_type, ts):

		newcolors =[ '#00e600','#2997ff','#cc0000', '#cacece']  #slgb _ vver1
		face_color = ['#e6ffe6', '#e6f5ff','#ffe6e6', '#f2f2f2']  #s l g bord
		ff=plt.figure(figsize=(50,50), frameon=False)
		ax = plt.subplot(111)
		#ax.set_xlim([0, x_box])
		#ax.set_ylim([0, y_box])
		#ax.get_xaxis().set_ticks([])
		#ax.get_yaxis().set_ticks([])
		ax.set_aspect('equal', 'datalim')
		for jj in range(len(self.vor.points)):
			region_id = self.vor.point_region[jj]
			if self.region_mask[region_id]:

				polygon = [self.vor.vertices[i] for i in self.vor.regions[region_id]]

				if self.gas_phase[region_id] == 1:                      # gas
					plt.fill(*zip(*polygon), facecolor = newcolors[2], edgecolor = 'black', linewidth = 3.5)


				elif self.solid_phase[region_id] == 1:
					plt.fill(*zip(*polygon), facecolor = newcolors[0], edgecolor='black', linewidth=3.5)


				elif self.liquid_phase[region_id] == 1:     # liquid
					plt.fill(*zip(*polygon), facecolor = newcolors[1], edgecolor='black', linewidth=3.5)


				elif self.sl_border[region_id] == 1:
					plt.fill(*zip(*polygon),facecolor=newcolors[3],edgecolor='black', linewidth=3.5)

				elif self.lg_border[region_id] == 1:
					plt.fill(*zip(*polygon),facecolor=newcolors[3],edgecolor='black', linewidth=3.5)

		rect = patches.Rectangle((self.xmin_init, self.ymin_init), self.xmax_init - self.xmin_init, self.ymax_init - self.ymin_init, linewidth = 5, edgecolor=newcolors[3], facecolor='none')
		ax.add_patch(rect)
		#rect = patches.Rectangle((self.xmin_init + self.dx, self.ymin_init + self.dy), self.xmax_init - self.xmin_init - 2*self.dx , self.ymax_init - self.ymin_init - 2*self.dy, linewidth = 5, edgecolor='#ff8000ff', facecolor='none')
		rect = patches.Rectangle((self.xmin_init + self.dx_left, self.ymin_init + self.dy_below), self.xmax_init - self.xmin_init - self.dx_right -self.dx_left, self.ymax_init - self.ymin_init - self.dy_below -  self.dy_above, linewidth = 5, edgecolor='#ff8000ff', facecolor='none')

		ax.add_patch(rect)
		ax.plot(self.dat_xyz_init[:, 1], self.dat_xyz_init[:, 2], ".", color = "black")
		plt.savefig(self.save_path_img + plot_type +str(ts)+'.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
		plt.close(ff)

	def PlotPhaseSystem_cg(self, plot_type, ts):

		newcolors =[ '#00e600','#2997ff','#cc0000', '#cacece']  #slgb _ vver1
		face_color = ['#e6ffe6', '#e6f5ff','#ffe6e6', '#f2f2f2']  #s l g bord
		ff=plt.figure(figsize=(50,50), frameon=False)
		ax = plt.subplot(111)
		#ax.set_xlim([0, x_box])
		#ax.set_ylim([0, y_box])
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])
		ax.set_aspect('equal', 'datalim')
		for jj in range(len(self.vor.points)):
			region_id = self.vor.point_region[jj]
			if self.region_mask[region_id]:

				polygon = [self.vor.vertices[i] for i in self.vor.regions[region_id]]

				if self.gas_phase[region_id] == 1:                      # gas
					plt.fill(*zip(*polygon), facecolor = newcolors[2], edgecolor = 'black', linewidth = 3.5)

				elif self.cond_phase[region_id] == 1:     # cond
					plt.fill(*zip(*polygon), facecolor = newcolors[1], edgecolor='black', linewidth=3.5)

				elif self.cg_border[region_id] == 1:
					plt.fill(*zip(*polygon),facecolor=newcolors[3],edgecolor='black', linewidth=3.5)

		ax.plot(self.dat_xyz_init[:, 1], self.dat_xyz_init[:, 2], ".", color = "black")
		rect = patches.Rectangle((self.xmin_init, self.ymin_init), self.xmax_init - self.xmin_init, self.ymax_init - self.ymin_init, linewidth = 5, edgecolor='Black', facecolor='none')
		ax.add_patch(rect)
		rect = patches.Rectangle((self.xmin_init + self.dx, self.ymin_init + self.dy), self.xmax_init - self.xmin_init - 2*self.dx , self.ymax_init - self.ymin_init - 2*self.dy, linewidth = 5, edgecolor='red', facecolor='none')
		ax.add_patch(rect)

		plt.savefig(self.save_path_img + plot_type +str(ts)+'.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
		plt.close(ff)


	@staticmethod
	def PolyArea(x,y):
		return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


	def Calc_Density_For_Phase_Digram(self):
		s_gas = 0
		s_cond = 0

		cond_particles = sum(self.cond_phase)
		gas_particles = sum(self.gas_phase)

		for jj in range(len(self.vor.points)):
			region_id = self.vor.point_region[jj]
			if self.region_mask[region_id]:
				region_xy =  np.array([list(self.vor.vertices[i]) for i in self.vor.regions[region_id]])

				if self.gas_phase[region_id] == 1:                      # gas
					s_gas += self.PolyArea(region_xy[:, 0], region_xy[:, 1])

				elif self.cond_phase[region_id] == 1:     # cond
					s_cond += self.PolyArea(region_xy[:, 0], region_xy[:, 1])
		if s_cond >0:
			rho_cond = cond_particles / s_cond
		else:
			rho_cond = 0

		if s_gas > 0:
			rho_gas = gas_particles / s_gas
		else:
			rho_gas = 0

		if gas_particles < 0.008*self.nparts_init:
			rho_gas = 0
		if cond_particles < 0.008*self.nparts_init:
			rho_cond = 0
		return rho_gas, rho_cond

	def Plot_Phase_Parameter(self, x, y, plot_type):
		ff=plt.figure(figsize=(10, 10) )
		ax = plt.subplot(111)
		ax.plot(x, y, ".")

		plt.savefig(self.save_path_img + plot_type +  '.png', bbox_inches='tight')#,pad_inches = 0)    #mh1
		ff.clear()
		plt.close(ff)
		del ff
	def Save_Phase_Param(self, name):
		ff = open(self.save_path_txt + name + ".txt", "w")
		for i in range(300):
			ff.write(str(self.log_rdf_setka[i]) + " " + str(self.S_voro[i] / self.num_cadr) +"\n")
		ff.close()



	def Calc_APD_paramters(self, ts):

		if ts == 0:

			self.n_part_phase = {"g":np.zeros(self.num_cadr -1),
						"s":np.zeros(self.num_cadr -1),
						"l":np.zeros(self.num_cadr -1),
						"c":np.zeros(self.num_cadr -1),
						"slb":np.zeros(self.num_cadr -1),
						"lgb":np.zeros(self.num_cadr -1)}

			#self.power = {"g":np.zeros(self.num_cadr -1), "s":np.zeros(self.num_cadr -1), "l":np.zeros(self.num_cadr -1), "c":np.zeros(self.num_cadr -1)}
			#WE DO NOT HAVE THE PARTICLES ORIENTATION
			self.Kin = 0

			self.av_v = {"g":np.zeros(self.num_cadr -1), "s":np.zeros(self.num_cadr -1), "l":np.zeros(self.num_cadr -1), "c":np.zeros(self.num_cadr -1)}
			self.dencity = {"g":np.zeros(self.num_cadr -1), "s":np.zeros(self.num_cadr -1), "l":np.zeros(self.num_cadr -1), "c":np.zeros(self.num_cadr -1)}

			self.v_gist = {"g": -np.ones((self.num_cadr -1, self.nparts_init)), "s": -np.ones((self.num_cadr -1, self.nparts_init)), "l": -np.ones((self.num_cadr -1, self.nparts_init)), "c": -np.ones((self.num_cadr -1, self.nparts_init))}

			self.diff = {"all":None, "s":None, "l":None, "g":None, "c":None}
			self.vv_corr = {"g":None, "s":None, "l":None, "c":None}


		if ts >0:

			tss = ts - 1

			self.n_part_phase["g"][tss] = sum(self.gas_phase)
			self.n_part_phase["l"][tss] = sum(self.liquid_phase)
			self.n_part_phase["s"][tss] = sum(self.solid_phase)
			self.n_part_phase["c"][tss] = sum(self.solid_phase) + sum(self.liquid_phase) + sum(self.sl_border)
			self.n_part_phase["slb"][tss] = sum( self.sl_border)
			self.n_part_phase["lgb"][tss] = sum(self.lg_border)



			s_gas, s_liq, s_cond, s_solid = 0, 0, 0, 0



			for jj in range(len(self.vor.points)):
				region_id = self.vor.point_region[jj]
				if self.region_mask[region_id]:
					region_xy =  np.array([list(self.vor.vertices[i]) for i in self.vor.regions[region_id]])

					x1 = self.dat_xyz[jj, 1]
					y1 = self.dat_xyz[jj, 2]
					x0 = self.dat_xyz_prev[jj, 1]
					y0 = self.dat_xyz_prev[jj, 2]

					x1, y1 = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1, self.x_box, self.y_box)
					vx = (x1 - x0) / self.dt
					vy = (y1 - y0) / self.dt

					v = (vx * vx + vy * vy) ** 0.5

					self.Kin += (v * v / 2 /  self.num_cadr / self.nparts_init)

					if self.gas_phase[region_id] == 1:                      # gas
						s_gas += self.PolyArea(region_xy[:, 0], region_xy[:, 1])




						self.av_v["g"][tss] += v
						self.v_gist["g"][tss, jj] = v

					elif self.liquid_phase[region_id] == 1:     # cond
						s_liq += self.PolyArea(region_xy[:, 0], region_xy[:, 1])
						self.av_v["l"][tss] += v
						self.av_v["c"][tss] += v
						self.v_gist["l"][tss, jj] = v
						self.v_gist["c"][tss, jj] = v

					elif  self.solid_phase[region_id] == 1:     # cond
						s_solid += self.PolyArea(region_xy[:, 0], region_xy[:, 1])
						self.av_v["s"][tss] += v
						self.av_v["c"][tss] += v
						self.v_gist["s"][tss, jj] = v
						self.v_gist["c"][tss, jj] = v

					elif  self.sl_border[region_id] == 1:     # cond
						s_cond += self.PolyArea(region_xy[:, 0], region_xy[:, 1])
						self.av_v["c"][tss] += v
						self.v_gist["c"][tss, jj] = v




			s_cond += (s_liq + s_solid)

			n_cut = 20
			if s_solid > 0 and self.n_part_phase["s"][tss] > n_cut:
				self.dencity["s"][tss] = self.n_part_phase["s"][tss] / s_solid
				self.av_v["s"][tss] /= self.n_part_phase["s"][tss]
			else:
				self.dencity["s"][tss] = 0
				self.n_part_phase["s"][tss] = 0

			if s_liq > 0 and self.n_part_phase["l"][tss] > n_cut:
				self.dencity["l"][tss] = self.n_part_phase["l"][tss] / s_liq
				self.av_v["l"][tss] /= self.n_part_phase["l"][tss]
			else:
				self.dencity["l"][tss]
				self.n_part_phase["l"][tss] = 0

			if s_gas > 0 and self.n_part_phase["g"][tss] > n_cut:
				self.dencity["g"][tss] = self.n_part_phase["g"][tss] / s_gas
				self.av_v["g"][tss] /= self.n_part_phase["g"][tss]
			else:
				self.dencity["g"][tss] = 0
				self.n_part_phase["g"][tss] = 0

			if s_cond >0 and self.n_part_phase["c"][tss] > n_cut:
				self.dencity["c"][tss] = self.n_part_phase["c"][tss] / s_cond
				self.av_v["c"][tss] /=  (self.n_part_phase["l"][tss] + self.n_part_phase["s"][tss] + self.n_part_phase["slb"][tss])
			else:
				self.dencity["c"][tss] = 0


		return 0

	def Save_APD_paramters(self):
		ff = open(self.save_path_txt + self.name_md + "SRD.txt", "w")
		for i in range(300):
			ff.write(str(self.log_rdf_setka[i]) + " " + str(self.S_voro[i] / self.num_cadr) +"\n")
		ff.close()

		ff = open(self.save_path_txt +  self.name_md + "KinEn.txt", "w")
		ff.write(str(self.Kin))
		ff.close()

		ff = open(self.save_path_txt +  self.name_md + "AvVel.txt", "w")
		ff.write(str( sum(self.av_v["s"]) / (self.num_cadr - 1) ) + " " + str( sum(self.av_v["l"]) / (self.num_cadr - 1) ) + " " + str( sum(self.av_v["g"]) / (self.num_cadr - 1) )  + " " + str( sum(self.av_v["c"]) / (self.num_cadr - 1) )   )
		ff.close()

		ff = open(self.save_path_txt +  self.name_md + "PhDen.txt", "w")
		ff.write(str( sum(self.dencity["s"]) / (self.num_cadr - 1) ) + " "  + str(sum(self.dencity["l"]) / (self.num_cadr - 1) ) + " "  + str(sum(self.dencity["g"]) / (self.num_cadr - 1)) + " "  + str(sum(self.dencity["c"]) / (self.num_cadr - 1) ) )
		ff.close()

		ff = open(self.save_path_txt +  self.name_md + "NpartsInPhas.txt", "w")
		ff.write(str( sum(self.n_part_phase["s"]) / (self.num_cadr - 1) ) + " " + str(sum(self.n_part_phase["l"]) / (self.num_cadr - 1) )  + " " + str(sum(self.n_part_phase["g"]) / (self.num_cadr - 1) )   + " " + str(sum(self.n_part_phase["c"]) / (self.num_cadr - 1) )  + " " + str(sum(self.n_part_phase["slb"]) / (self.num_cadr - 1) )  + " " + str(sum(self.n_part_phase["lgb"]) / (self.num_cadr - 1) ))
		ff.close()






	def Define_dx_dy(self):
		x_sort = np.sort(self.dat_xyz_init[:, 1])
		y_sort = np.sort(self.dat_xyz_init[:, 2])
		dx_left =  x_sort[10] - self.xmin_init
		dx_right = self.xmax_init - x_sort[-11]

		dy_above = self.ymax_init - y_sort[-11]
		dy_below = y_sort[10] - self.ymin_init
		return dx_left, dx_right, dy_below, dy_above

	def Main_Loop(self):
		self.full_md_name = self.path_to_md + self.name_md + self.last_name_md
		print(self.full_md_name)
		dump = LammpsCoords(string_content = 'id x y z ',
                            required_data = 'id x y z ',
                            path = self.full_md_name)
		dump.OpenTrjGetInfo(1)
		self.st = dump.ReadState().copy()
		self.nparts_init = self.st[0]

		print(self.nparts_init)
		current_time = datetime.datetime.now()
		# Printing value of now.
		print("Time now at greenwich meridian is:", current_time)

		self.A = float(self.name_md.split("_")[3])
		kT = float(self.name_md.split("_")[-1])
		self.dt = 8000 * 0.0015 * (0.1/(self.A + kT)**0.5)**0.5


		for index in range(self.num_cadr):
			if index % 10 == 0:
				print(index, " / ", self.num_cadr)
				# using now() to get current time

			ts = index
			self.st = dump.ReadState().copy()
			self.dat_xyz_init = np.array(self.st[-1]).copy()

			self.xmin_init, self.xmax_init = min(self.dat_xyz_init[:, 1]), max(self.dat_xyz_init[:, 1])
			self.ymin_init, self.ymax_init = min(self.dat_xyz_init[:, 2]), max(self.dat_xyz_init[:, 2])
			self.x_box = self.xmax_init - self.xmin_init
			self.y_box = self.ymax_init - self.ymin_init
			#self.dat_xyz, self.dx, self.dy = self.Add_Points(self.st, self.xmin_init, self.xmax_init, self.ymin_init, self.ymax_init, 0.05)
			self.dat_xyz = self.dat_xyz_init.copy()

			self.dx_left, self.dx_right, self.dy_below, self.dy_above = self.Define_dx_dy()

			self.dx = [self.dx_left, self.dx_right]
			self.dy = [self.dy_below, self.dy_above]
			#self.dx = (self.xmax_init - self.xmin_init) * 0.04
			#self.dy = (self.ymax_init - self.ymin_init) * 0.04

			if min(self.dat_xyz[:, 0]) == 1:
				self.dat_xyz[:, 0] -= 1
			if min(self.dat_xyz_init[:, 0]) == 1:
				self.dat_xyz_init[:, 0] -= 1

			self.neigh_id, self.vor, self.region_mask, self.r_ij = self.Get_Voronoi_Neighbors(self.nparts_init, self.dat_xyz, self.dat_xyz_init, self.xmin_init, self.xmax_init,
				self.ymin_init, self.ymax_init, self.x_box, self.y_box, self.dx, self.dy, True)

			if self.flag_R_d_filed:
				if ts == 0:
					self.log_rdf_setka, self.len_rdf_setka, self.d_rdf, self.S_voro, self.min_rdf, _ = self.Cacl_Phase_Parameter_Step1(self.nparts_init, self.neigh_id, 550, self.r_ij,self.vor, self.region_mask)

				else:
					self.S_voro = self.Cacl_Phase_Parameter_Step2(self.nparts_init, self.neigh_id,  self.r_ij, self.vor, self.region_mask, self.len_rdf_setka, self.S_voro,self.min_rdf, self.d_rdf)

			self.solid_phase, self.liquid_phase, self.gas_phase = self.Phase_Work(self.nparts_init, self.neigh_id, self.r_ij, self.vor, self.region_mask, -2.5, -10, separate_type = "slg")
			self.solid_phase, self.liquid_phase, self.gas_phase, self.sl_border, self.lg_border = self.Phase_Work_Correction_slg(self.nparts_init, self.solid_phase, self.liquid_phase, self.gas_phase, self.vor, self.region_mask, self.neigh_id, 4)


			self.Calc_APD_paramters(ts)
			self.dat_xyz_prev = self.dat_xyz.copy()
			# self.cond_phase, self.gas_phase = self.Phase_Work(self.nparts_init, self.neigh_id, self.r_ij, self.vor, self.region_mask, -5, -10, separate_type = "cg")
			# self.cond_phase, self.gas_phase, self.cg_border = self.Phase_Work_Correction_cg(self.nparts_init, self.cond_phase, self.gas_phase, self.vor, self.region_mask, self.neigh_id, 5)
			if ts == 1:
				#self.PlotPhaseSystem_cg("Phases"+ self.name_md, index)
				self.PlotPhaseSystem("/States0/"+ self.name_md, index)

			# rho_gas, rho_cond = self.Calc_Density_For_Phase_Digram()
			# self.rho_gas += rho_gas/self.num_cadr
			# self.rho_cond += rho_cond/self.num_cadr


		if self.flag_R_d_filed:
		 	self.Plot_Phase_Parameter(self.log_rdf_setka, self.S_voro, "SRd" + self.name_md)
		# 	self.Save_Phase_Param(self.name_md)
		self.Save_APD_paramters()

		return self.log_rdf_setka, self.S_voro, self.rho_gas, self.rho_cond
















path_to_md = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/ActivePhaseDiagram/MD/MD/"
save_path_txt = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/ActivePhaseDiagram/Res/txtAlina/"
save_path_img = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/ActivePhaseDiagram/Res/risAlina/"

num_activ = 20
num_temp = 20
activity = [0.0]
act_start = 0.1
act_stop = 40

ln_act_start = np.log(act_start)
ln_act_stop = np.log(act_stop)
activity = np.exp(np.linspace(ln_act_start, ln_act_stop, 20))
activity[0] = 0
activity = np.sort(np.append(activity, np.array([12, 12.5, 13, 13.5, 14, 14.5, 15, 16, 16.5, 17])))


kt_start = 0.2
kt_stop =  kt_start * 20

ln_t_start = np.log(kt_start)
ln_t_stop = np.log(kt_stop)
kTlist = np.exp(np.linspace(ln_t_start, ln_t_stop, 20))
kTlist = np.sort(np.append(kTlist, np.linspace(0.3, 1.5, 23)))


log_rdf_setka_global = []
S_voro_glob = []
names_md_glob = []
rho_cond_glob = []
rho_gas_glob = []
flag_R_d_filed = True
for act_id in range(len(activity)):
	Act = activity[act_id]
	# if act_id<5:
	# 	continue
	for kt_id in range(len(kTlist)):#num_temp):
		if kt_id < 12:
			continue
		kT = kTlist[kt_id]

		name_md = "F1_APD_A_"+ str(round(Act, 4)) + "_T_" + str(round(kT, 4))
		names_md_glob.append(name_md)
		APD = ActivePhaseDiagram(
						save_path_txt = save_path_txt,
						save_path_img = save_path_img,
						path_to_md = path_to_md,
						name_md = name_md,
						last_name_md = ".lammpstrj",
						num_cadr = 2,
						flag_R_d_filed = flag_R_d_filed)
		log_rdf, S_voro, rho_gas, rho_cond = APD.Main_Loop()
		log_rdf_setka_global.append(log_rdf)
		S_voro_glob.append(S_voro)
		rho_gas_glob.append(rho_gas)
		rho_cond_glob.append(rho_cond)
		#sys.exit()

		if flag_R_d_filed:
			fs = open(save_path_txt + name_md + "RD_f.txt", "w")
			for i in range(len(S_voro)):
				fs.write(str(log_rdf[i]) + " " + str(S_voro[i]) + "\n ")
			fs.close()




















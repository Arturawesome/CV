
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
import os




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
		self.dat_xyz = None
		self.rho_gas = 0
		self.rho_cond = 0

		self.solid_phase = None
		self.liquid_phase  = None
		self.gas_phase = None
		self.sl_border = None
		self.lg_border = None
		self.log_rdf_setka = None
		self.S_voro = None



	def __del__(self):
		print("==============================")
		print("ActivePhaseDiagram Destucture: clear memory")
		print("==============================")
		class_name = self.__class__.__name__
		print('{} delete'.format(class_name))

	@staticmethod
	def Add_Points(dat_xyz, nparts_init, xmin_init, xmax_init, ymin_init, ymax_init, percent):
		# ========================== INFORMATION ===============================
		# function of adding additional points to the array
		# of particles taking into account the periodic boundary
		#dat_xyz = np.array(st[-1]).copy()
		id_part    = dat_xyz[:, 0].copy()
		x_part     = dat_xyz[:, 1].copy()
		y_part     = dat_xyz[:, 2].copy()
		z_part     = dat_xyz[:, 3].copy()

		# ======== define box size and area of cut line ===================

		x_box       = xmax_init - xmin_init
		y_box       = ymax_init - ymin_init
		dx = percent * x_box
		dy = percent * y_box

		for i in range(nparts_init):
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

		#----------------------------------
		# otsechenie pogranichnux regionov i poisk min max ploshadei
		#----------------------------------

		# cut by voronoi verices:
		# for i in range(len(vor.vertices)):
		# 	ver=vor.vertices[i]
		# 	if ver[0]<xmin_init + dx or ver[0]>xmax_init - dx or ver[1]<ymin_init + dy or ver[1]>ymax_init-dy:
		# 		ver_index[i] = False

		# cut by point position:
		for i in range(npoints):
			ind = vor.point_region[i]
			certies = vor.regions[ind]
			if vor.points[i][0] > xmin_init - dx  and vor.points[i][0] < xmax_init + dx and vor.points[i][1] > ymin_init - dy  and vor.points[i][1] < ymax_init +dy:
				ver_index[certies] = True


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


		# for jj in range(nparts_init):
		# 	region_id = vor.point_region[jj]
		# 	if region_mask[region_id]:
		# 		neig = neigh_id[jj]                         # индексы соседних частиц
		# 		neig_region_id = vor.point_region[neig]     # индексы соседних ячеек вороного
  #
		# 		if  region_mask[region_id]:
		# 			if (liquid_phase1[region_id] == 1) and (sum(solid_phase1[neig_region_id] == 0) < len(solid_phase1[neig_region_id])):
		# 				liquid_phase1[region_id] = 0
		# 				solid_liquid_border[region_id] = 1
  #
		# 			if (gas_phase1[region_id] == 1) and (sum(gas_phase1[neig_region_id] == 0) < len(gas_phase1[neig_region_id])):
		# 				gas_phase1[region_id] = 0
		# 				liquid_gas_border[region_id] = 1

		return solid_phase1, liquid_phase1,  gas_phase1, solid_liquid_border, liquid_gas_border

	@staticmethod
	def Phase_Work_Correction_cg(nparts_init, cond_phase, gas_phase, vor, region_mask, neigh_id, num_corr, flag_border):
		if flag_border == False:
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
			return cond_phase1, gas_phase1
		# else else else else else else else else else else else else
		else:
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

					if  region_mask[region_id]:
						if (gas_phase1[region_id] == 1) and (sum(cond_phase1[neig_region_id] == 0) < len(cond_phase1[neig_region_id])):
							gas_phase1[region_id] = 0
							cond_gas_border[region_id] = 1
			return cond_phase1, gas_phase1, cond_gas_border

	def Calc_Diff_In_Phase(self, xy):
		#/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/artur_data/MIPS/python_prog/hex_temp.py
		return 0

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
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])
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

		ax.plot(self.dat_xyz_init[:, 1], self.dat_xyz_init[:, 2], ".", color = "black")
		plt.savefig(self.save_path_img + plot_type +str(ts)+'.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
		plt.close(ff)

	def PlotPhaseSystem_cg(self, plot_type, ts, flag_border):
		if flag_border:

			newcolors =[ '#00e600','#2997ff','#cc0000', '#cacece']  #slgb _ vver1
			face_color = ['#e6ffe6', '#e6f5ff','#ffe6e6', '#f2f2f2']  #s l g bord
			ff=plt.figure(figsize=(50,50), frameon=False)
			ax = plt.subplot(111)
			ax.set_xlim([self.xmin_init - 0.01*(self.xmax_init - self.xmin_init), self.xmax_init+ 0.01*(self.xmax_init - self.xmin_init)])
			ax.set_ylim([self.ymin_init- 0.01*(self.ymax_init - self.ymin_init), self.ymax_init + 0.01*(self.ymax_init - self.ymin_init)])
			ax.get_xaxis().set_ticks([])
			ax.get_yaxis().set_ticks([])
			ax.set_aspect('equal', 'datalim')
			for jj in range(len(self.vor.points)):
				region_id = self.vor.point_region[jj]
				if self.region_mask[region_id]:

					polygon = [self.vor.vertices[i] for i in self.vor.regions[region_id]]

					if self.gas_phase[region_id] == 1:                      # gas
						plt.fill(*zip(*polygon), facecolor = newcolors[2], edgecolor = 'black', linewidth = 3.5)

					elif self.border_cg[region_id] == 1:
						plt.fill(*zip(*polygon), facecolor = newcolors[3], edgecolor = 'black', linewidth = 3.5)
					elif self.cond_phase[region_id] == 1:     # cond
						plt.fill(*zip(*polygon), facecolor = newcolors[1], edgecolor='black', linewidth=3.5)

					# elif self.cg_border[region_id] == 1:
					# 	plt.fill(*zip(*polygon),facecolor=newcolors[3],edgecolor='black', linewidth=3.5)

			ax.plot(self.dat_xyz_init[:, 1], self.dat_xyz_init[:, 2], ".", color = "black")
			rect = patches.Rectangle((self.xmin_init, self.ymin_init), self.xmax_init - self.xmin_init, self.ymax_init - self.ymin_init, linewidth = 5, edgecolor='Black', facecolor='none')
			ax.add_patch(rect)
			rect = patches.Rectangle((self.xmin_init + self.dx, self.ymin_init + self.dy), self.xmax_init - self.xmin_init - 2*self.dx , self.ymax_init - self.ymin_init - 2*self.dy, linewidth = 5, edgecolor='red', facecolor='none')
			ax.add_patch(rect)

			plt.savefig(self.save_path_img + plot_type +str(ts)+'.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
			plt.close(ff)

		else:
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

					# elif self.cg_border[region_id] == 1:
					# 	plt.fill(*zip(*polygon),facecolor=newcolors[3],edgecolor='black', linewidth=3.5)

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

	def Angle_of_vector_Ox(self, x, y):
		angle = pygame.math.Vector2(1, 0).angle_to((x, y))
		#if angle < 0:
			#angle += 360
		return math.radians(angle)

	def Ret_Psi_k(self, x0y0, xy_neighs, k):
		psik = 0
		for j in range(len(xy_neighs)):
			r_ij = xy_neighs[j, :] - x0y0
			theta_ij = self.Angle_of_vector_Ox(r_ij[0], r_ij[1])
			psik += np.exp(complex(0, k * theta_ij))
		psik/= len(xy_neighs)
		psik = abs(psik)
		return psik



	def Plot_Phase_Parameter(self, x, y, plot_type):
		ff=plt.figure(figsize=(10, 10) )
		ax = plt.subplot(111)
		ax.plot(x, y, "-")

		plt.savefig(self.save_path_img + plot_type +  '.png', bbox_inches='tight')#,pad_inches = 0)    #mh1
		ff.clear()
		plt.close(ff)
		del ff
	def Save_Phase_Param(self, name):
		ff = open(self.save_path_txt + name + ".txt", "w")
		for i in range(300):
			ff.write(str(self.log_rdf_setka[i]) + " " + str(self.S_voro[i] / self.num_cadr) +"\n")
		ff.close()

	def Save_OSC_Data(self, name, data):
		ff = open(self.save_path_txt + name + ".txt", "w")
		for i in range(len(data)):
			for j in range(len(data[i])):
				ff.write(str(data[i][j])+" ")
			ff.write("\n")
		ff.close()

	def Save_Density(self, name):
		ff = open(self.save_path_txt + name + ".txt", "w")
		for i in range(len(self.rho_gas_through_md)):
			ff.write(str(self.rho_gas_through_md[i]) +" ")
		ff.write("\n")
		for i in range(len(self.rho_cond_through_md)):
			ff.write(str(self.rho_cond_through_md[i]) +" ")
		ff.close()

	def Plot_All_SRd_for_md(self, x, y, plot_type):
		ff=plt.figure(figsize=(10, 10) )
		ax = plt.subplot(111)
		for i in range(len(y)):
			ax.plot(x, y[i], "-")
		plt.savefig(self.save_path_img + plot_type +  '.png', bbox_inches='tight')#,pad_inches = 0)    #mh1
		plt.close(ff)
	def PLot_Av_RDF(self, x, y, plot_type):
		index_crystal = list(np.where(x<-11))[0][-1]
		index_gas = list(np.where(x>0))[0][0]
		av_y = np.zeros(len(y))

		flag_crystal, flag_gas = 0, 0
		for i in range(len(y)):
			av_y[i] = sum(x * y[i]) / len(y[i]) / self.num_cadr
			b = np.array(y[i])
			if sum(b[0:index_crystal]) / self.num_cadr > 0.05*self.nparts_init:
				flag_crystal = 1
			if sum(b[index_gas:-1]) / self.num_cadr > 0.05*self.nparts_init:
				flag_gas = 1

		ff=plt.figure(figsize=(10, 10) )
		ax = plt.subplot(111)
		ax.plot(np.arange(1, len(y) + 1), av_y, "-")
		if flag_gas == 1:
			ax.text(1, av_y[1], "gas")
		if flag_crystal == 1:
			ax.text(2, av_y[2], "crystal")
		plt.savefig(self.save_path_img + plot_type +  '.png', bbox_inches='tight')#,pad_inches = 0)    #mh1

		ff.clear()
		plt.close(ff)
		return flag_crystal, flag_gas

#
	def Calc_OSC_paramters(self, ts):
		self.r_ij_stat.append(self.r_ij)
		for jj in range(self.nparts_init):
			region_id = self.vor.point_region[jj]
			if self.region_mask[region_id]:
				self.psi4[ts*self.nparts_init + jj] = self.Ret_Psi_k(self.vor.points[jj],  self.vor.points[self.neigh_id[jj]], 4)
				self.psi5[ts*self.nparts_init + jj] = self.Ret_Psi_k(self.vor.points[jj],  self.vor.points[self.neigh_id[jj]], 5)
				self.psi6[ts*self.nparts_init + jj] = self.Ret_Psi_k(self.vor.points[jj],  self.vor.points[self.neigh_id[jj]], 6)
				self.psi7[ts*self.nparts_init + jj] = self.Ret_Psi_k(self.vor.points[jj],  self.vor.points[self.neigh_id[jj]], 7)


	def Main_Loop(self):

		self.full_md_name = self.path_to_md + self.name_md + self.last_name_md
		print(self.full_md_name)
		dump = LammpsCoords(string_content = 'id x y ',
                            required_data = 'id x y',
                            path = self.full_md_name)
		dump.OpenTrjGetInfo(1)
		self.st = dump.ReadState().copy()
		self.nparts_init = self.st[0]
		print(self.nparts_init)
		self.S_voro_through_md = []
		self.rho_gas_through_md = []
		self.rho_cond_through_md = []

		self.psi4_through_md = []
		self.psi5_through_md = []
		self.psi6_through_md = []
		self.psi7_through_md = []

		self.r_ij_through_md = []

		for _ in range(49):
			self.st = dump.ReadState().copy()

		flag = True
		count = 0
		while flag:
			# relaxation
			for _ in range(200 - self.num_cadr):
				try:
					self.st = dump.ReadState().copy()
				except:
					flag = False
					break
			count += 1
			#
			self.rho_gas = 0
			self.rho_cond = 0
			counter_internal = 0
			self.psi6 = np.zeros(self.num_cadr * self.nparts_init)
			self.psi4 = np.zeros(self.num_cadr * self.nparts_init)
			self.psi5 = np.zeros(self.num_cadr * self.nparts_init)
			self.psi7 = np.zeros(self.num_cadr * self.nparts_init)
			self.r_ij_stat = []

			for index in range(self.num_cadr):
				ts = index
				try:
					self.st = dump.ReadState().copy()
				except:
					flag = False
					break
				self.dat_xyz_init = np.array(self.st[-1]).copy()
				self.dat_xyz_init = np.append(self.dat_xyz_init, np.zeros((self.nparts_init, 1)), axis=1).copy()
				counter_internal += 1

				self.xmin_init, self.xmax_init = min(self.dat_xyz_init[:, 1]), max(self.dat_xyz_init[:, 1])
				self.ymin_init, self.ymax_init = min(self.dat_xyz_init[:, 2]), max(self.dat_xyz_init[:, 2])
				self.x_box = self.xmax_init - self.xmin_init
				self.y_box = self.ymax_init - self.ymin_init


				self.dat_xyz, self.dx, self.dy = self.Add_Points(self.dat_xyz_init, self.nparts_init, self.xmin_init, self.xmax_init, self.ymin_init, self.ymax_init, 0.05)
				#self.dat_xyz = self.dat_xyz_init.copy()

				self.dx = 0#(self.xmax_init - self.xmin_init) * 0.04
				self.dy = 0#(self.ymax_init - self.ymin_init) * 0.04

				if min(self.dat_xyz[:, 0]) == 1:
					self.dat_xyz[:, 0] -= 1
				if min(self.dat_xyz_init[:, 0]) == 1:
					self.dat_xyz_init[:, 0] -= 1

				self.neigh_id, self.vor, self.region_mask, self.r_ij = self.Get_Voronoi_Neighbors(self.nparts_init, self.dat_xyz, self.dat_xyz_init, self.xmin_init, self.xmax_init, self.ymin_init, self.ymax_init, self.x_box, self.y_box, self.dx, self.dy, True)
#				psi_6[jj] = Ret_Psi_k(vor.points[jj], vor.points[neigh_id[jj]], 6)

				if self.flag_R_d_filed:
					if ts == 0:
						self.log_rdf_setka, self.len_rdf_setka, self.d_rdf, self.S_voro, self.min_rdf, _ = self.Cacl_Phase_Parameter_Step1(self.nparts_init, self.neigh_id, 550, self.r_ij,self.vor, self.region_mask)

					else:
						self.S_voro = self.Cacl_Phase_Parameter_Step2(self.nparts_init, self.neigh_id,  self.r_ij, self.vor, self.region_mask, self.len_rdf_setka, self.S_voro,self.min_rdf, self.d_rdf)

				#self.solid_phase, self.liquid_phase, self.gas_phase = self.Phase_Work(self.nparts_init, self.neigh_id, self.r_ij, self.vor, self.region_mask, -5, -10, separate_type = "slg")
				#self.solid_phase, self.liquid_phase, self.gas_phase, self.sl_border, self.lg_border = self.Phase_Work_Correction_slg(self.nparts_init, self.solid_phase, self.liquid_phase, self.gas_phase, self.vor, self.region_mask, self.neigh_id, 0)


				self.cond_phase, self.gas_phase = self.Phase_Work(self.nparts_init, self.neigh_id, self.r_ij, self.vor, self.region_mask, 0, -10, separate_type = "cg")

				self.cond_phase, self.gas_phase, self.border_cg = self.Phase_Work_Correction_cg(self.nparts_init, self.cond_phase, self.gas_phase, self.vor, self.region_mask, self.neigh_id, 0, True)
				if ts == -1:
				 	self.PlotPhaseSystem_cg("P_" + str(count)+ self.name_md, index, True)

				rho_gas, rho_cond = self.Calc_Density_For_Phase_Digram()
				self.rho_gas += rho_gas
				self.rho_cond += rho_cond
				self.Calc_OSC_paramters(ts)


			if counter_internal > 0:
				self.rho_gas_through_md.append(self.rho_gas/counter_internal)
				self.rho_cond_through_md.append(self.rho_cond/counter_internal)
				#print("self.psi4[np.where(self.psi4!=0)].shape():  ", len(self.psi4[np.where(self.psi4!=0)]))
				self.psi4_through_md.append(self.psi4[np.where(self.psi4!=0)])
				#print(len(self.psi4_through_md))
				self.psi5_through_md.append(self.psi5[np.where(self.psi5!=0)])
				self.psi6_through_md.append(self.psi6[np.where(self.psi6!=0)])
				self.psi7_through_md.append(self.psi7[np.where(self.psi7!=0)])
				self.S_voro_through_md.append(self.S_voro)

				r_ij_flat0 = [x for xs in self.r_ij_stat for x in xs]
				self.r_ij_flat = [x for xs in r_ij_flat0 for x in xs]
				self.r_ij_through_md.append(self.r_ij_flat)


			# if self.flag_R_d_filed:
			# 	self.Plot_Phase_Parameter(self.log_rdf_setka, self.S_voro, "SRd" + str(count) + self.name_md)
				#self.Save_Phase_Param(str(count) + "_" + self.name_md)

		if self.flag_R_d_filed:
			#flag_crystal, flag_gas = self.PLot_Av_RDF(self.log_rdf_setka, self.S_voro_through_md, "AV_SRd"  + self.name_md)
			self.Plot_All_SRd_for_md(self.log_rdf_setka, self.S_voro_through_md, "ALL_SRd"  + self.name_md)

			self.Save_OSC_Data("Psi4_" + self.name_md, self.psi4_through_md)
			self.Save_OSC_Data("Psi5_" + self.name_md, self.psi5_through_md)
			self.Save_OSC_Data("Psi6_" + self.name_md, self.psi6_through_md)
			self.Save_OSC_Data("Psi7_" + self.name_md, self.psi7_through_md)
			self.Save_OSC_Data("Srdf_" + self.name_md, self.S_voro_through_md)

			# self.Save_OSC_Data("Rij_" + self.name_md, self.r_ij_through_md)
			# self.Save_Density("RhoGasCond_"  + self.name_md)

			#print("len(self.r_ij_through_md) = ",len(self.r_ij_through_md))


		return self.log_rdf_setka, self.S_voro_through_md, self.rho_gas_through_md, self.rho_cond_through_md#, flag_crystal, flag_gas
















path_to_md = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/Osc/MD/"
save_path_txt = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/Osc/Res/txt/"
save_path_img = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/Osc/Res/ris/"

names = os.listdir(path_to_md)


#read_file = open(save_path_txt + "RaeadME2.txt", "w")
log_rdf_setka_global = []
S_voro_glob = []
names_md_glob = []
rho_cond_glob = []
rho_gas_glob = []
flag_R_d_filed = True
#names = ["dump_0.3535535_0.25_628.31900000_0.10000000_0.00500000_10.00000000_0.30000000_0_2.00000000.lammpstrj"]#s,
# 		 "dump_0.3535535_2.0_1256.63800000_0.10000000_0.00500000_10.00000000_0.30000000_0_2.00000000.lammpstrj",
# 		 "dump_2.121321_0.1_314.15950000_0.10000000_0.00500000_10.00000000_0.30000000_0_2.00000000.lammpstrj",
# 		 "dump_0.3535535_0.25_314.15950000_0.10000000_0.00500000_10.00000000_0.30000000_0_2.00000000.lammpstrj",
# 		 "dump_1.414214_0.25_1256.63800000_0.10000000_0.00500000_10.00000000_0.30000000_0_2.00000000.lammpstrj",
# 		 "dump_1.414214_2.0_1256.63800000_0.10000000_0.00500000_10.00000000_0.30000000_0_2.00000000.lammpstrj"]
#read_file.write("DUMP NAME; gas_phase; CrystalPhase; rho_gas; rho_cond\n\n")
for md_id in range(len(names)):
	name_md = names[md_id]
	file_stats = os.stat(path_to_md + name_md)
	if file_stats.st_size / (1024 * 1024) < 250:
		continue

	names_md_glob.append(name_md)
	APD = ActivePhaseDiagram(
					save_path_txt = save_path_txt,
					save_path_img = save_path_img,
					path_to_md = path_to_md,
					name_md = name_md[0:-10],
					last_name_md = ".lammpstrj",
					num_cadr = 10,
					flag_R_d_filed = flag_R_d_filed)
	#log_rdf, S_voro, rho_gas, rho_cond, flag_crystal, flag_gas = APD.Main_Loop()
	log_rdf, S_voro, rho_gas, rho_cond = APD.Main_Loop()
	log_rdf_setka_global.append(log_rdf)
	S_voro_glob.append(S_voro)
	print(rho_gas)
	print(rho_cond)


#	read_file.write(name_md + " " + str(flag_gas) + " " + str(flag_crystal) + " "+ str(rho_gas) + " " + str(rho_cond) + "\n")
	if flag_R_d_filed:
		fs = open(save_path_txt +  name_md[0:-10] + "RD_f.txt", "w")
		for i in range(len(S_voro)):
			fs.write(str(log_rdf[i]) + " " + str(S_voro[i]) + "\n ")
		fs.close()
#read_file.close()




















# USE GUIDE:
#
# FOLDERS AND LOCATION
# Place FIM_RDF_Suppl.py and FIM_Coefficients.py in the same folder (e.g. "FIM_folder")
# In FIM_folder create a folder “InputData”
# The RDF data should be in the InputData folder.
# See comments in FIM_Coefficients.py for RDF data format and Supplementary Materials of original article.
#
#
# FUNCTIONS:
#
# def Main_Loop(self) is the main function of the FluidInterpolatingMethod class,
# where the actions described in the original article are sequentially performed.
#
# All variables set in def __init__(self, etc) and def Init_Variables(self)
# you can set the additional variables in  def Init_Variables(self)
#
# All functions are named according to the original article.
# The example def Eq5_pi_s(self) describes equation 4 from the original article.
#


# ============================================================================
# Name:
# FIM_Coefficients.py

# Authors:
# 	N.P. Kryuchkov, A.D. Nasyrov (kruchkov_nkt@mail.ru, nasyrovartur151998@gmail.com)

# Copyright:
# 	Please cite the original work if you use this software package.

# Description:
# 	Python software package for calculating m, r, sigma2, phi values (see Eq. 4, 8) based on input files 
# 	RDF_LJ_Ts.txt RDF_LJ_Tm.txt RDF_LJ_Tf.txt

# Requirements:
# 	In the folder where FIM_RDF_Suppl.py and FIM_Coefficients are located, create a folder named InputData
# 	Numpy, matplotlib, scipy should be available

# !!!!!!!!!!!!!!!!!(EXAMPLE FORMAT IN SUPPLEMENTAL MATERIALS)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!(EXAMPLE FORMAT IN SUPPLEMENTAL MATERIALS)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# RDF INPUT FILES MUST HAVE THE FOLLOWING FORMAT:
# num_row num_col
# r_0 rdf_0-1 rdf_0-2 ... rdf_0-numcol
# r_1 rdf_1-1 rdf_1-2 ... rdf_1-numcol
# --------------------------------
# r_numrow rdf_numrow-1 ... rdf_numrow-numcol

# num_row - number of rows
# num_col - number of correlation peaks
# !!!!!!!!!!!!!!!!!(EXAMPLE FORMAT IN SUPPLEMENTAL MATERIALS)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!(EXAMPLE FORMAT IN SUPPLEMENTAL MATERIALS)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Inputs:
# 	path_to_input_files(str) — InputData folder name.
# 	name_rdf_Ts, name_rdf_Tm, name_rdf_Tf(str) - data names of rdf functions for Ts Tm Tf, respectively, which are located in the “InputData” folder
# 	!!Note that you do not specify a save file name, the save file name is generated internally by FIM_RDF_Supplement.py!!


# ============================================================================



import numpy as np
import scipy as scipy
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import scipy.optimize as opt

class FluidInterpolatingMethod():
	def __init__(self,
				path_to_input_files: str,
				name_rdf_Ts:str,
				name_rdf_Tm:str,
				name_rdf_Tf:str
				):
		self.path_input = path_to_input_files
		self.name_rdf_Ts = name_rdf_Ts
		self.name_rdf_Tm = name_rdf_Tm
		self.name_rdf_Tf = name_rdf_Tf


	def __del__(self):
		print("Destructor Called; Clear Memory")


	def Read_RDF_data(self, name):
		full_name_rdf = self.path_input + name
		f = open(full_name_rdf, "r")

		row_col_info = [float(item) for item in f.readline().strip().split(' ')]
		self.num_row, self.num_peaks = int(row_col_info[0]), int(row_col_info[1])
		data  = np.array([[float(item) for item in f.readline().strip().split(' ')] for i in range(int(self.num_row))])
		f.close()

		self.r = data[:, 0]
		self.dr = self.r[1] - self.r[0]

		return data[:, 1:self.num_peaks+1]


	def Save_txt(self, data, name):
		f = open(self.path_input + name, "w")
		for id_temp in range(self.num_t):
			for id_peak in range(self.num_peaks):
				f.write(str(data[id_temp, id_peak]) + " ")
			f.write("\n")
		f.close()


	def Calc_Dat_Peaks(self):
		print("In Calc_Dat_Peaks")
		self.dat_pike = np.zeros((self.num_t, self.num_peaks, 4)) # self.dat_pike[id_temp, id_peak, id_param]
																	#id_param : 0-Norm; 1- avr; 2-sigma2; 3-alpha
		for id_peak in range(self.num_peaks):
			for id_temp in range(self.num_t):
				norm = sum(self.rdf_smf[id_temp][:, id_peak] * self.dr)
				self.dat_pike[id_temp, id_peak, 0] = norm

				av_r = sum(self.r * self.rdf_smf[id_temp][:, id_peak] * self.dr) / norm
				self.dat_pike[id_temp, id_peak, 1] = av_r

				sigma2 = sum((self.r - av_r) * (self.r - av_r) * self.rdf_smf[id_temp][:, id_peak] * self.dr) / norm
				self.dat_pike[id_temp, id_peak, 2] = sigma2


	def Alpha_Minimization(self, alpha_Tm):
		return sum((alpha_Tm * self.p_iTs_alpha + (1 - alpha_Tm) * self.p_iTf_alpha - self.p_iTm_alpha) ** 2)


	def Find_Alpha_Depend(self, id_peak):
		self.p_iTs_alpha = self.pTs_pTm_pTf[0, id_peak, :]
		self.p_iTm_alpha = self.pTs_pTm_pTf[1, id_peak, :]
		self.p_iTf_alpha = self.pTs_pTm_pTf[2, id_peak, :]
		alpha_i_Tm = opt.minimize(self.Alpha_Minimization, x0=1, method='SLSQP', options={"maxiter":100000000})
		return alpha_i_Tm


	def Eq5_pi_s(self):
		self.s_norm_dist = np.zeros((3, self.num_peaks, len(self.r)))
		self.pTs_pTm_pTf = np.zeros((3, self.num_peaks, len(self.r)))

		for id_peak in range(self.num_peaks):
			self.s_norm_dist[0, id_peak, :] = (self.r - self.dat_pike[0, id_peak, 1]) / np.sqrt(self.dat_pike[0, id_peak, 2])
			self.s_norm_dist[1, id_peak, :] = (self.r - self.dat_pike[1, id_peak, 1]) / np.sqrt(self.dat_pike[1, id_peak, 2])
			self.s_norm_dist[2, id_peak, :] = (self.r - self.dat_pike[2, id_peak, 1]) / np.sqrt(self.dat_pike[2, id_peak, 2])
			x_new = self.s_norm_dist[0, id_peak, :]

			pTs = self.rdf_smf[0][:, id_peak] * np.sqrt(self.dat_pike[0, id_peak, 2]) / self.dat_pike[0, id_peak, 0]
			pTm = self.rdf_smf[1][:, id_peak] * np.sqrt(self.dat_pike[1, id_peak, 2]) / self.dat_pike[1, id_peak, 0]
			pTf = self.rdf_smf[2][:, id_peak] * np.sqrt(self.dat_pike[2, id_peak, 2]) / self.dat_pike[2, id_peak, 0]

			f1 = CubicSpline(self.s_norm_dist[0, id_peak, :], pTs)
			f2 = CubicSpline(self.s_norm_dist[1, id_peak, :], pTm)
			f3 = CubicSpline(self.s_norm_dist[2, id_peak, :], pTf)

			self.pTs_pTm_pTf[0, id_peak, :] = f1(x_new)
			self.pTs_pTm_pTf[1, id_peak, :] = f2(x_new)
			self.pTs_pTm_pTf[2, id_peak, :] = f3(x_new)

			self.s_norm_dist[0, id_peak, :] = x_new
			self.s_norm_dist[1, id_peak, :] = x_new
			self.s_norm_dist[2, id_peak, :] = x_new

			alpha_res = self.Find_Alpha_Depend(id_peak)
			self.dat_pike[:, id_peak, 3] = np.array([1, alpha_res.x[0], 0] )


	def Init_Variables(self):
		self.rdf_smf = []
		self.s_norm_dist = None
		self.pTs_pTm_pTf = None
		self.num_t = 3
		self.num_peaks = None
		self.dat_pike = None
		self.r = None
		self.dr = None
		self.name_norm = "FIM_Norm_smf.txt"
		self.name_avr = "FIM_avr_smf.txt"
		self.name_sigma2 = "FIM_sigma2_smf.txt"
		self.name_alpha = "FIM_alpha_smf.txt"
		self.save_path = ""


	def Main_Loop(self):
		self.Init_Variables()

		self.rdf_smf.append(self.Read_RDF_data(self.name_rdf_Ts))
		self.rdf_smf.append(self.Read_RDF_data(self.name_rdf_Tm))
		self.rdf_smf.append(self.Read_RDF_data(self.name_rdf_Tf))

		self.Calc_Dat_Peaks()
		print("Calc Norm, Avr, Sigma2 Done;  Calc Norm, Avr, Sigma2 Done;  Calc Norm, Avr, Sigma2 Done; \n")
		self.Eq5_pi_s()
		print("Calc Phi Done;  Calc Phi Done;  Calc Phi Done;  Calc Phi Done;  Calc Phi Done; \n")

		self.Save_txt(self.dat_pike[:, :, 0], self.name_norm)
		self.Save_txt(self.dat_pike[:, :, 1], self.name_avr)
		self.Save_txt(self.dat_pike[:, :, 2], self.name_sigma2)
		self.Save_txt(self.dat_pike[:, :, 3], self.name_alpha)
		print("Save Done; Save Done; Save Done; Save Done; Save Done; Save Done; Save Done; Save Done; \n")




FIM = FluidInterpolatingMethod(
		path_to_input_files = "InputData/",
		name_rdf_Ts = "RDF_LJ_Ts.txt",
		name_rdf_Tm = "RDF_LJ_Tm.txt",
		name_rdf_Tf = "RDF_LJ_Tf.txt",
		)
FIM.Main_Loop()











































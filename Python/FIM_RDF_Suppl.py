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
# The example def Eq4_pa_s(self) describes equation 4 from the original work.
#
# Plot_Fig3(self) - plots the third figure from the original article

# ============================================================================
# Name:
# FIM_RDF_Suppl.py

# Authors:
# 	N.P. Kryuchkov, A.D. Nasyrov (kruchkov_nkt@mail.ru, nasyrovartur151998@gmail.com)

# Copyright:
# 	Please cite the original work if you use this software package.

# Description:
# 	Python software package for interpolating RDF data over a range of temperatures.


# Requirements:
# 	In the folder where FIM_RDF_Suppl.py and FIM_Coefficients.py are located, create a folder named InputData
# 	Numpy, matplotlib, scipy should be available
# 	You should run FIM_Coefficients.py first and only then FIM_RDF_Suppl.py

# Inputs:
#
# path_to_input_files(str) — the name of the folder in which the rdf data is located.
# name_rdf_ps, name_rdf_pf (str) - data names of well-known rdf functions that are located in the path_to_input_files folder
# 	For an example of the file format, see FIM_Coefficients.py and the supplementary materials.
# Temp_ts, Temp_tm, Temp_tf (float) - temperatures of systems, where name_rdf_ps and name_rdf_ps are defined and additional point.
# !!Note that you do not specify save file names, save file names are generated internally by FIM_RDF_Suppl.py!!
# temperature_req (float) - temperature at which RDF should be calculated
# ============================================================================


import numpy as np
import scipy as scipy
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class FluidInterpolatingMethod():
	def __init__(self,
				path_to_input_files: str,
				name_rdf_ps:str,
				name_rdf_pf:str,

				Temp_ts:float,
				Temp_tm:float,
				Temp_tf:float,
				temperature_req:float,


				):
		self.path_input = path_to_input_files
		self.name_rdf_ps = name_rdf_ps
		self.name_rdf_pf = name_rdf_pf

		self.temp_ts = Temp_ts
		self.temp_tm = Temp_tm
		self.temp_tf = Temp_tf
		self.temp_req = temperature_req



	def __del__(self):
		print("Destructor called: Memory clean")


	def Read_RDF_data(self, name):
		full_name_rdf = self.path_input + name
		f = open(full_name_rdf, "r")

		row_col_info = [float(item) for item in f.readline().strip().split(' ')]
		self.num_row, self.num_peaks = int(row_col_info[0]), int(row_col_info[1])
		data  = np.array([[float(item) for item in f.readline().strip().split(' ')] for i in range(int(self.num_row))])
		f.close()
		self.r = data[:, 0]
		return data[:, 1:self.num_peaks+1]


	def Read_m_avr_sigma_alpha(self, name):
		full_name_mrsa = self.path_input + name
		f = open(full_name_mrsa, "r")
		data  = np.array([[float(item) for item in f.readline().strip().split(' ')] for i in range(int(self.num_t))])
		f.close()
		return data


	def Plot_Fig2_Article(self):
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (15, 5))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col
				axs[row, col].plot(self.s_norm_dist[0, id_peak, :], self.pTs_pTf[0, id_peak, :], "--", color = "black")
				axs[row, col].plot(self.s_norm_dist[0, id_peak, :], self.pTs_pTf[1, id_peak, :], "-", color = "grey")

				axs[row, col].plot(self.s_norm_dist[0, id_peak, :], self.pTreq[id_peak, :], "-", color = "orange")
				axs[row, col].set_xlim([-4, 4])

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path + 'FIM_Ris2.pdf', bbox_inches='tight')
		fig.clear()
		plt.close(fig)
		return 0


	def Save_Txt_data(self):
		f = open(self.save_path + "FIM_Pa(sTs).txt", "w")
		for row in range(self.num_row):
			for id_peak in range(self.num_peaks):
				f.write(str(self.pTs_pTf[0, id_peak, row]) + " ")
			f.write("\n")
		f.close()

		f = open(self.save_path + "FIM_Pa(sTf).txt", "w")
		for row in range(self.num_row):
			for id_peak in range(self.num_peaks):
				f.write(str(self.pTs_pTf[1, id_peak, row]) + " ")
			f.write("\n")
		f.close()

		f = open(self.save_path + "FIM_S_for_TsTfTreq.txt", "w")
		for row in range(self.num_row):
			for id_peak in range(self.num_peaks):
				f.write(str(self.s_norm_dist[0, id_peak, row]) + " ")
			f.write("\n")
		f.close()

		f = open(self.save_path + "FIM_Pa(sTreq).txt", "w")
		for row in range(self.num_row):
			for id_peak in range(self.num_peaks):
				f.write(str(self.pTreq[id_peak, row]) + " ")
			f.write("\n")
		f.close()

		f = open(self.save_path + "FIM_RDF_g(rTreq).txt", "w")
		for row in range(self.num_row):
			f.write( str(self.r[row]) + " "+ str(self.rdf_req[-1, row]) + "\n")
		f.close()


	def Plot_Suppl(self, plot_type):
		# Magnitude of peaks plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (11, 8))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col

				x_fit = self.temp_range_grid
				a_i, b_i, beta_i = self.a_b_beta_magnitude[id_peak, 0], self.a_b_beta_magnitude[id_peak, 1], self.a_b_beta_magnitude[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)
				f_mi_T = self.Eq6_f_T(self.temp_req, a_i, b_i, beta_i)

				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 0], "o", color = "red")
				axs[row, col].plot(self.temp_req, f_mi_T , "*", color = "red")

				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path +"FIM_" + plot_type[0] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)

		# Average distance plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (11, 8))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col
				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 1], "o", color = "red")

				x_fit = self.temp_range_grid
				a_i, b_i, beta_i = self.a_b_beta_avr[id_peak, 0], self.a_b_beta_avr[id_peak, 1], self.a_b_beta_avr[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)
				f_ri_T = self.Eq6_f_T(self.temp_req, a_i, b_i, beta_i)

				axs[row, col].plot(self.temp_req, f_ri_T , "*", color = "red")
				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path + "FIM_" + plot_type[1] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)

		# The sigma2 values plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (11, 8))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col
				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 2], "o", color = "red")

				x_fit = self.temp_range_grid
				a_i, b_i, beta_i = self.a_b_beta_sigma2[id_peak, 0], self.a_b_beta_sigma2[id_peak, 1], self.a_b_beta_sigma2[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)
				f_sigma2i_T = self.Eq6_f_T(self.temp_req, a_i, b_i, beta_i)

				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].plot(self.temp_req, f_sigma2i_T , "*", color = "red")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path + "FIM_" + plot_type[2] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)

		# The alpha values plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (13, 3.5))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col
				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 3], "o", color = "red")

				x_fit = self.temp_range_grid
				a_i, b_i, beta_i = self.a_b_beta_alpha[id_peak, 0], self.a_b_beta_alpha[id_peak, 1], self.a_b_beta_alpha[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)
				f_ai_T = self.Eq6_f_T(self.temp_req, a_i, b_i, beta_i)
				axs[row, col].plot(self.temp_req, f_ai_T , "*", color = "red")
				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path +"FIM_" + plot_type[3] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)


	def Plot_Fig3(self):

		fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 5))

		for id_peak in range(self.num_peaks):
			axs.plot(self.r, self.rdf_req[id_peak, :], "-", color = "orange")

		axs.plot(self.r, self.rdf_req[-1, :], "--", color = "red")
		axs.set_xlim([0, 10])
		plt.savefig(self.save_path + 'FIM_Ris3.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)


	def Eq6_f_T(self, x_temp, a_i, b_i, beta_i):
		return a_i + b_i * x_temp **(beta_i)

	def Eq6_f_T_1(self, id_peak_x, beta_i):
		id_peak = int(id_peak_x[0])
		x_temp = id_peak_x[1]
		fit_type = int(id_peak_x[2])

		dataTs = self.dat_pike[0, id_peak, fit_type]
		dataTf = self.dat_pike[2, id_peak, fit_type]

		Tf = self.temp_range[2]
		Ts = self.temp_range[0]
		return dataTs + (dataTf - dataTs)/(Tf ** beta_i - Ts ** beta_i) * (x_temp ** beta_i - Ts ** beta_i)


	def Eq6_f_T_2(self, id_peak_x, beta_i):
		id_peak = int(id_peak_x[0])
		x_temp = id_peak_x
		fit_type = int(id_peak_x[2])

		dataTs = self.dat_pike[0, id_peak, fit_type]
		dataTf = self.dat_pike[2, id_peak, fit_type]

		Tf = self.temp_range[2]
		Ts = self.temp_range[0]

		b = (dataTf - dataTs)/(Tf ** beta_i - Ts ** beta_i)
		a = dataTs - b * (Ts ** beta_i)
		return a, b


	def Get_Fit(self, fit_type:str):
		if fit_type == "magnitude":
			self.a_b_beta_magnitude = np.zeros((self.num_peaks, 3))
			for id_peak in range(self.num_peaks):
				beta_i_fisrt, _ = curve_fit(self.Eq6_f_T_1, (id_peak, self.temp_range[1], 0), [self.dat_pike[1, id_peak, 0]], maxfev = 10000000)
				a, b = self.Eq6_f_T_2((id_peak, self.temp_range[1], 0), beta_i_fisrt[0])
				self.a_b_beta_magnitude[id_peak, :] = [a, b, beta_i_fisrt[0]]

		elif fit_type == "avr":
			self.a_b_beta_avr = np.zeros((self.num_peaks, 3))
			for id_peak in range(self.num_peaks):
				beta_i_fisrt, _ = curve_fit(self.Eq6_f_T_1, (id_peak, self.temp_range[1], 1), [self.dat_pike[1, id_peak, 1]], maxfev = 10000000)
				a, b = self.Eq6_f_T_2((id_peak, self.temp_range[1], 1), beta_i_fisrt[0])
				self.a_b_beta_avr[id_peak, :] = [a, b, beta_i_fisrt[0]]

		elif fit_type == "sigma2":
			self.a_b_beta_sigma2 = np.zeros((self.num_peaks, 3))
			for id_peak in range(self.num_peaks):
				beta_i_fisrt, _ = curve_fit(self.Eq6_f_T_1, (id_peak, self.temp_range[1], 2), [self.dat_pike[1, id_peak, 2]], maxfev = 10000000)
				a, b = self.Eq6_f_T_2((id_peak, self.temp_range[1], 2), beta_i_fisrt[0])
				self.a_b_beta_sigma2[id_peak, :] = [a, b, beta_i_fisrt[0]]

		elif fit_type == "alpha":
			self.a_b_beta_alpha = np.zeros((self.num_peaks, 3))
			for id_peak in range(self.num_peaks):
				beta_i_fisrt, _ = curve_fit(self.Eq6_f_T_1, (id_peak, self.temp_range[1], 3), [self.dat_pike[1, id_peak, 3]], maxfev = 10000000)
				a, b = self.Eq6_f_T_2((id_peak, self.temp_range[1], 3), beta_i_fisrt[0])
				self.a_b_beta_alpha[id_peak, :] = [a, b, beta_i_fisrt[0]]


	def Eq4_pa_s(self):
		self.s_norm_dist = np.zeros((2, self.num_peaks, len(self.r)))
		self.pTs_pTf = np.zeros((2, self.num_peaks, len(self.r)))

		for id_peak in range(self.num_peaks):
			self.s_norm_dist[0, id_peak, :] = (self.r - self.dat_pike[0, id_peak, 1]) / np.sqrt(self.dat_pike[0, id_peak, 2])
			x_new = self.s_norm_dist[0, id_peak, :]

			self.s_norm_dist[1, id_peak, :] = (self.r - self.dat_pike[2, id_peak, 1]) / np.sqrt(self.dat_pike[2, id_peak, 2])

			pTs = self.rdf_ps_pf[0][:, id_peak] * np.sqrt(self.dat_pike[0, id_peak, 2]) / self.dat_pike[0, id_peak, 0]
			pTf = self.rdf_ps_pf[1][:, id_peak] * np.sqrt(self.dat_pike[2, id_peak, 2]) / self.dat_pike[2, id_peak, 0]

			f1 = CubicSpline(self.s_norm_dist[0, id_peak, :], pTs)
			f2 = CubicSpline(self.s_norm_dist[1, id_peak, :], pTf)

			self.pTs_pTf[0, id_peak, :] = f1(x_new)
			self.pTs_pTf[1, id_peak, :] = f2(x_new)


	def Eq7_pa_sTreq(self):
		self.pTreq = np.zeros((self.num_peaks, self.num_row))
		for id_peak in range(self.num_peaks):
			a_i, b_i, betta_i = self.a_b_beta_alpha[id_peak, 0], self.a_b_beta_alpha[id_peak, 1], self.a_b_beta_alpha[id_peak, 2]
			alpha_i_t = self.Eq6_f_T(self.temp_req, a_i, b_i, betta_i)
			self.pTreq[id_peak, :] = alpha_i_t * self.pTs_pTf[0, id_peak, :] + (1 - alpha_i_t) *  self.pTs_pTf[1, id_peak, :]


	def Eq8_grt_rdf(self):
		self.rdf_req = np.zeros((self.num_peaks + 1, len(self.pTreq[1, :])) )

		for id_peak in range(self.num_peaks):
			a_im, b_im, betta_im = self.a_b_beta_magnitude[id_peak, 0], self.a_b_beta_magnitude[id_peak, 1], self.a_b_beta_magnitude[id_peak, 2]
			a_ir, b_ir, betta_ir = self.a_b_beta_avr[id_peak, 0], self.a_b_beta_avr[id_peak, 1], self.a_b_beta_avr[id_peak, 2]
			a_is, b_is, betta_is = self.a_b_beta_sigma2[id_peak, 0], self.a_b_beta_sigma2[id_peak, 1], self.a_b_beta_sigma2[id_peak, 2]
			a_ia, b_ia, betta_ia = self.a_b_beta_alpha[id_peak, 0], self.a_b_beta_alpha[id_peak, 1], self.a_b_beta_alpha[id_peak, 2]

			f_mi_T = self.Eq6_f_T(self.temp_req, a_im, b_im, betta_im)
			f_ri_T = self.Eq6_f_T(self.temp_req, a_ir, b_ir, betta_ir)
			f_sigma2i_T = self.Eq6_f_T(self.temp_req, a_is, b_is, betta_is)

			rdf_for_interpolation = (f_mi_T/np.sqrt(f_sigma2i_T)) * self.pTreq[id_peak, :]
			r_new = self.s_norm_dist[0, id_peak, :] * np.sqrt(f_sigma2i_T) + f_ri_T

			f1 = CubicSpline(r_new, rdf_for_interpolation)
			self.rdf_req[id_peak, :] = f1(self.r)
			self.rdf_req[-1, :] += self.rdf_req[id_peak, :]


	def Init_Variables(self):
		self.rdf_ps_pf = []	# list of Ps and Pf peaks. Set in Main_Loop
		self.s_norm_dist = None	# shape ant type in def Eq4_pa_s
		self.pTs_pTf = None	# shape ant type in def Eq4_pa_s
		self.pTreq = None	# shape and type in def Eq7_pa_sTreq
		self.rdf_req = None # shape and type in def Eq8_grt_rdf
		self.temp_range = [self.temp_ts, self.temp_tm, self.temp_tf]

		self.num_t = 3	# len of self.temp_range
		self.num_peaks = None	# shape and type in def Read_RDF_data
		self.dat_pike = None # self.dat_pike[id_temperature, id_peak, id_param], id_param: 0-Norm, 1-Avr, 2-sigma2, 3-alpha; Set in def Main_Loop
		self.r = None # shape and type describe in def Read_r_grid; self.r set in Main_Loop


		self.a_b_beta_magnitude = None	#shape and type in def Get_Fit. (like np.zeros((self.num_peaks, 3)))
		self.a_b_beta_avr = None	#shape and type in def Get_Fit. (like np.zeros((self.num_peaks, 3)))
		self.a_b_beta_sigma2 = None	#shape and type in def Get_Fit. (like np.zeros((self.num_peaks, 3)))
		self.a_b_beta_alpha = None	#shape and type in def Get_Fit. (like np.zeros((self.num_peaks, 3)))

		self.name_m_smf = "FIM_Norm_smf.txt"
		self.name_avr_smf = "FIM_avr_smf.txt"
		self.name_sigma2_smf = "FIM_sigma2_smf.txt"
		self.name_alpha_smf = "FIM_alpha_smf.txt"
		self.temp_range_grid = np.arange(self.temp_ts, self.temp_tf + (self.temp_tf - self.temp_ts)/20,  (self.temp_tf - self.temp_ts)/20)
		self.save_path = ""



	def Main_Loop(self):

		self.Init_Variables()
		self.rdf_ps_pf.append(self.Read_RDF_data(self.name_rdf_ps))
		self.rdf_ps_pf.append(self.Read_RDF_data(self.name_rdf_pf))



		print("\n\nDownload RDF, Ps, Pf done; Download RDF, Ps, Pf done; Download RDF, Ps, Pf done; Download RDF, Ps, Pf done; Download RDF, Ps, Pf done; Download RDF, Ps, Pf done; \n\n")

		self.dat_pike = np.zeros((self.num_t, self.num_peaks, 4))	# self.dat_pike[id_temperature, id_peak, id_param], id_param: 0-Norm, 1-Avr, 2-sigma2, 3-alpha
		self.dat_pike[:,:, 0] = self.Read_m_avr_sigma_alpha(self.name_m_smf)
		self.dat_pike[:,:, 1] = self.Read_m_avr_sigma_alpha(self.name_avr_smf)
		self.dat_pike[:,:, 2] = self.Read_m_avr_sigma_alpha(self.name_sigma2_smf)
		self.dat_pike[:,:, 3] = self.Read_m_avr_sigma_alpha(self.name_alpha_smf)

		print("Download Norm, Avr, Sigma2, Alpha done; Download Norm, Avr, Sigma2, Alpha done; Download Norm, Avr, Sigma2, Alpha done \n")

		self.Get_Fit("magnitude")
		self.Get_Fit("avr")
		self.Get_Fit("sigma2")
		self.Get_Fit("alpha")

		print("Fit Norm, Avr, Sigma2, Alpha done;  Fit Norm, Avr, Sigma2, Alpha done;  Fit Norm, Avr, Sigma2, Alpha done; \n")

		self.Plot_Suppl(["norm", "avr", "sigma2", "alpha"])
		self.Eq4_pa_s()
		print("Eq.4:  Pa(s) done;  Eq.4:  Pa(s) done;  Eq.4:  Pa(s) done;  Eq.4:  Pa(s) done;  Eq.4:  Pa(s) done; \n")

		self.Eq7_pa_sTreq()
		print("Eq.7:  Pa(s, Treq) done;  Eq.7:  Pa(s, Treq) done;  Eq.7:  Pa(s, Treq) done;  Eq.7:  Pa(s, Treq) done; \n")

		self.Plot_Fig2_Article()
		self.Eq8_grt_rdf()
		print("Eq.8: g(r, Treq) done;  Eq.8: g(r, Treq) done;  Eq.8: g(r, Treq) done;  Eq.8: g(r, Treq) done; \n")

		self.Plot_Fig3()
		self.Save_Txt_data()


# =========================================================================================================
# =========================================================================================================
# =========================================================================================================
# =========================================================================================================
# =========================================================================================================




FIM = FluidInterpolatingMethod(
		path_to_input_files = "InputData/",
		name_rdf_ps = "RDF_LJ_Ts.txt",
		name_rdf_pf = "RDF_LJ_Tf.txt",
		Temp_ts = 6.0,
		Temp_tm = 23.886430,
		Temp_tf = 60,
		temperature_req = 10
		)
FIM.Main_Loop()































import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import pygame
import math
import matplotlib as mpl
from matplotlib import gridspec
#import matplotlib.cm as cm
from matplotlib import cm
from scipy import spatial
from scipy.optimize import curve_fit
from matplotlib.ticker import LinearLocator
from scipy import signal
from matplotlib.colors import LightSource, Normalize
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
plt.style.use('./article_style.mplstyle')

def plot_3d_system(X, Y, Z, Z2, name):
    newcolors =[ '#b5b5b5', '#0daf0d','#3e68e2','#eb2d2d','#ff8000','#840384','#85B4DC','#63CE9C','#CB63CE','#3A779F','#3A9F77','#BA8838', '#BA5438','#BA38A8', '#b5b5b5']
    ax = plt.figure().add_subplot(projection='3d')

    xlen = len(X)
    ylen = len(Y)

    X1, Y1 = np.meshgrid(X, Y)
    Z1= np.zeros((len(X), len(Y), len(Z)))

    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(len(Z)):
                Z1[i, j, k] = Z[k]

    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(len(Z)):
                ax.scatter(X[i], Y[j], Z1[i, j, k], marker='o', color = newcolors[int(Z2[i, j, k])])

    colortuple = ('y', 'b')
    colors = np.empty(X1.shape, dtype=str)

    for y in range(ylen):
        for x in range(xlen):
            colors[y, x] = colortuple[(x + y) % len(colortuple)]
    plt.savefig(name + '.pdf')


def plot_xy(x, y, name):
    ff=plt.figure(figsize=(10.0,10.0 * 0.6), frameon=False)
    plt.plot(x, y, '.', color = 'black')
    plt.savefig(name + '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
    plt.close(ff)

def plot_xy_2_2(x, y, name, graf_name):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (10,7))
    for i in range(2):
        for j in range(2):
            axs[i, j].plot(x[2*i + j], y[2*i + j], '.', color = 'black',)
            axs[i, j].set_title(name[2 * i + j])
    plt.tight_layout(pad = 0.4)
    plt.savefig(str(graf_name) + '.png',pad_inches = 0.05)
    plt.close(fig)


def read_er_m_p(f, N_relax, N_run, thermo):
    for i in range(int(N_relax / thermo)):
        f.readline()

    d=[[float(item) for item in f.readline().strip().split('\t')[2::]] for i in range(int(N_run/thermo))]
    return np.array(d)


def angle_of_vector(x0, y0, x1, y1):
    angle = pygame.math.Vector2(x0, y0).angle_to((x1, y1))
    if angle < 0:
        angle += 360
    if angle > 180:
        angle = 360 - angle
    return math.radians(angle)

def display_colorbar(x, y, z, name):

    cmap = plt.cm.copper
    ls = LightSource(315, 45)
    rgb = ls.shade(z, cmap)

    fig, ax = plt.subplots()
    ax = plt.subplot(111)

    # Use a proxy artist for the colorbar...
    im = ax.pcolormesh(x, y, z, cmap=cmap)
    fig.colorbar(im)

   #
    #fig.colorbar(im)

    ax.set_title(name, size='x-large')
    #plt.savefig(name + '.png',  bbox_inches='tight',pad_inches = 0.05)

def RegistPhaseChage(size, p_xy, ts_relax, ts_all):

    p_xy_medfilt = np.array(scipy.signal.medfilt(p_xy, 3))
    size_milling = np.mean(size[50 : ts_relax])
    change_flags = [0, 0, 0] #0- stay skooling, 1-change to milling, 2-skooling->milling->skoling,  3-change to turning (experimental)
    polar_gran_in_clear = 0.7
    flag_mill = 0
    flag_brak = 0
    # все хгначения меньше => либо мгновенный вылет, лыбо не хватает силы вылететь но skyling остается
    if sum(p_xy_medfilt[50 : ts_relax] < polar_gran_in_clear) >= len(p_xy_medfilt[50 : ts_relax])- 5:
        flag_mill = 1


    count_up_down = 0
    time_ts = np.arange(1, ts_all, 10)
    size_start = 0
    size_end = 0

    size_start = np.mean(size[ts_relax-25 : ts_relax])
    size_end = np.mean(size[ts_all-25 : ts_all])
    for i in range(len(time_ts) - 2):
        p_mean1 = np.mean(p_xy_medfilt[int(time_ts[i]) : int(time_ts[i + 1])] - 0.5)
        p_mean2 = np.mean(p_xy_medfilt[int(time_ts[i+1]) : int(time_ts[i + 2])] - 0.5)

        if p_mean1 * p_mean2 < 0:
            count_up_down += 1

    if count_up_down == 1:
        #1-change to milling
        change_flags[1] = 1

    elif count_up_down == 0:
        #0- stay skooling
        if size_end > 120:
            flag_mill = 1
            flag_brak = 1
        else:
            change_flags[0] = 1

    elif count_up_down > 1.5:
        #2-skooling->milling->skoling
        change_flags[2] = 1

    return change_flags, flag_mill, flag_brak




#path = '/run/media/artur/Новый том/FISH_MD_data/Fish/dump/'
path = '/run/media/artur/Новый том/FISH_MD_data/Fish/Fish/sp_fish/SC2Mil_dump3/'
dump_folders = ['dump']
print('/run/media/artur/Новый том/FISH_MD_data/Fish/dump/log_0_100_5_Ip_In_0.1_20000_100000_100_0_0.6_0.99_0.0.txt')
name = 'log_0_100_5_' + 'Ip'+'_' + 'In'+'_0.1_20000_100000_100_0_0.6_0.99_0.0.txt'

log_data = np.arange(10,20, 1) #0, 1, 2, 3, 4...



#In = np.arange(0.04, 0.6, 0.04)
#In = np.array([0.1, 0.2, 0.3, 0.35, 0.38, 0.4, 0.5])
In = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

#Ip = [3.0]
#num_osob = np.array([1, 2, 3, 4, 5, 10, 15, 20])
num_osob = np.array([1, 2, 3, 4, 5, 7, 10, 12, 15])
#F_k_osob = np.arange(0.0, 2, 0.1)
#F_k_osob = np.arange(0.1, 1, 0.1)
F_k_osob = 0.01*np.arange(4, 64, 4)#[0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]#[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,]#, 0.7, 0.8, 0.9]
#F_k_osob = np.array([0.9])
N_relax = 2e4
N_run = 1e5

name_log = []
name_log_not_path = []
print(F_k_osob)

for lgd in log_data:
    for i_n in In:
        for n_osoba in num_osob:
            for f_k_os in F_k_osob:
                name = path +'log_'+ str(lgd) + '_100_5_8.0_'+ str(round(i_n, 2)) + '_0.1_20000_100000_100_'+ str(n_osoba) + '_'+ str(f_k_os) + '_1.0_0.0.txt'
                name_log.append(name)
                name_log_not_path.append('log_'+ str(lgd) + '_100_5_8.0_'+ str(round(i_n, 2)) + '_0.1_20000_100000_100_'+ str(n_osoba) + '_'+ str(f_k_os) + '_1.0_0.0.txt')


def plot_3d_system(X, Y, Z, Z2, name):
    newcolors =[ '#b5b5b5', '#0daf0d','#3e68e2','#eb2d2d','#ff8000','#840384','#85B4DC','#63CE9C','#CB63CE','#3A779F','#3A9F77','#BA8838', '#BA5438','#BA38A8', '#b5b5b5']
    ax = plt.figure().add_subplot(projection='3d')

    # Make data.
    #X = np.arange(-5, 5, 0.25)
    xlen = len(X)
    #print(X)
    #Y = np.arange(-5, 5, 0.25)
    ylen = len(Y)
    X1, Y1 = np.meshgrid(X, Y)
    #R = np.sqrt(X**2 + Y**2)
    Z1= np.zeros((len(X), len(Y), len(Z)))
    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(len(Z)):
                Z1[i, j, k] = Z[k]
    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(len(Z)):
                ax.scatter(X[i], Y[j], Z1[i, j, k], marker='o', color = newcolors[int(Z2[i, j, k])])
    #for k in range(len(Z)):
        #ax.scatter(X1, Y1, Z1[:, :, k], marker='o')
    #Z = np.sin(R)
    #print(X)

    # Create an empty array of strings with the same shape as the meshgrid, and
    # populate it with two colors in a checkerboard pattern.
    colortuple = ('y', 'b')
    colors = np.empty(X1.shape, dtype=str)
    for y in range(ylen):
        for x in range(xlen):
            colors[y, x] = colortuple[(x + y) % len(colortuple)]

    ## Plot the surface with face colors taken from the array we made.
    #surf = ax.plot_surface(X1, Y1, Z, facecolors=colors, linewidth=0)

    ## Customize the z axis.
    #ax.set_zlim(-1, 1)
    #ax.zaxis.set_major_locator(LinearLocator(6))
    plt.savefig(name + '.pdf')
    #plt.show()







N_ts = int((N_relax + N_run) / 100)
ts_relax = 200
ts_all = 1200
time = np.array([range(ts_all)])
ts0_m = [20]#, 0, 201]
ts1_m = [1200, 200, 1200]
#prove_file = open('name_prove.txt', 'w')
#prove_file.write('file_name\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tf_p_d\n')
#fly_data = open('fly_Sc2Mil.txt', 'w')
no_influens_data = open('/home/artur/Documents/Fish/res/txt_data/no_influens_Sc2Mil.txt', 'w')
full_change_data = open('/home/artur/Documents/Fish/res/txt_data/full_change_Sc2Mil.txt', 'w')
sch_Mil_Schol_data = open('/home/artur/Documents/Fish/res/txt_data/sch_Mil_Schol_Sc2Mil.txt', 'w')
brak_data = open('/home/artur/Documents/Fish/res/txt_data/brak_Sc2Mil.txt', 'w')

size_gist = np.zeros(len(In) * len(F_k_osob) * len(num_osob) * len(log_data))

#fly_fish = np.zeros((len(In), len(F_k_osob), len(num_osob)))    # мгновенный вылет рыб, милинг остается
no_influens = np.zeros((len(In), len(F_k_osob), len(num_osob))) #stay scholing
full_change = np.zeros((len(In), len(F_k_osob), len(num_osob))) # change to milling
sch_Mil_Schol = np.zeros((len(In), len(F_k_osob), len(num_osob)))    # bifurccatiuon - > sc->mil->sc
brak_md = np.zeros((len(In), len(F_k_osob), len(num_osob)))    # bifurccatiuon - > sc->mil->sc
count_1 = 0
count_2 = 0
count_3 = 0
ts = 0.1
#alpha0 = np.zeros(len(range(ts_relax, ts_all)))
alphas = np.zeros(len(range(ts_all))-1)
radius = np.zeros(len(range(ts_all))-1)
for lgd in range(len(log_data)):
    for i_n in range(len(In)):
        for n_osoba in range(len(num_osob)):
            for f_k_os in range(len(F_k_osob)):
                k = (len(In) *len(num_osob) * len(F_k_osob)) * lgd + len(num_osob)*len(F_k_osob)*i_n + n_osoba * len(F_k_osob) + f_k_os
                #print(name_log[k])
                f = open(name_log[k], "r")

                st = read_er_m_p(f, 0, N_relax + N_run, 100)
                x_cm = st[:, 0]
                y_cm = st[:, 1]
                px = st[:, 2]
                py = st[:, 3]
                p_xy = np.sqrt(np.square(px) + np.square(py))
                erx = st[:, 4]
                ery = st[:, 5]
                er_xy = np.sqrt(np.square(erx) + np.square(ery))
                moment = abs(st[:, 6])
                size = st[:, 7]
                size_gist[k] = np.mean(size[ts_all - 20:ts_all])
                mean_v = np.mean(p_xy)
                c, flag_mill, flag_brak = RegistPhaseChage(size, p_xy, ts_relax, ts_all)
                f.close()
                #change_flags = [0, 0, 0] #0- stay skooling, 1-change to milling, 2-skooling->milling->skoling,  3-change to turning (experimental)
                if flag_mill == 0:
                    if c[0] > 0:
                        #no_influens[i_n, f_k_os, n_osoba] += c[0]
                        no_influens_data.write(name_log[k] + "\n")
                    if c[1] > 0:
                        #full_change[i_n, f_k_os, n_osoba] += c[1]
                        full_change_data.write(name_log[k] + "\n")
                    if c[2] > 0:
                        #sch_Mil_Schol[i_n, f_k_os, n_osoba] += c[2]
                        sch_Mil_Schol_data.write(name_log[k] + "\n")


                    no_influens[i_n, f_k_os, n_osoba] += c[0]
                    full_change[i_n, f_k_os, n_osoba] += c[1]
                    sch_Mil_Schol[i_n, f_k_os, n_osoba] += c[2]

                if flag_brak == 1:
                    brak_md[i_n, f_k_os, n_osoba] += 1
                    brak_data.write(name_log[k] + "\n")


no_influens_data.close()
full_change_data.close()
sch_Mil_Schol_data.close()
brak_data.close()


pesent_brak = 0
props = dict(boxstyle = 'square', facecolor = 'white', linewidth = 0.0, alpha = 0.8)
x = num_osob
y = F_k_osob
fig, axs = plt.subplots(nrows = len(In), ncols=3, figsize = (3.0, 4.3))

text_color = "Black"
for i_n in range(len(In)):
    cmap = plt.cm.YlOrRd
    cmap = plt.cm.hot_r

    cmap = cm.get_cmap('hot_r', 512)
    #                                   0.4 начало колорбара 1 - конец
    cmap = ListedColormap(cmap(np.linspace(0.0, 0.65, 512)))

    #norm=plt.Normalize(0, 1)
    #cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red","green"])

    num = no_influens[i_n, :, :] + full_change[i_n, :, :] + sch_Mil_Schol[i_n, :, :]

    if i_n != len(In) - 1:
        axs[i_n, 0].set_xticklabels([])
        axs[i_n, 1].set_xticklabels([])
        axs[i_n, 2].set_xticklabels([])
    axs[i_n, 1].set_yticklabels([])
    axs[i_n, 2].set_yticklabels([])

    if i_n == len(In) - 1:
        axs[i_n, 1].set_xlabel("Number of agents, n")

    if i_n == 0:
        axs[i_n, 0].set_title("No influence")
        axs[i_n, 1].set_title("Full change")
        axs[i_n, 2].set_title("Bifurcation")


    norm_max = 1#1.5
    z = no_influens[i_n, :, :] / num
    axs[i_n, 0].pcolormesh(x, y, z, cmap=cmap, vmin=0, vmax=norm_max)

    z = full_change[i_n, :, :] / num
    im = axs[i_n, 1].pcolormesh(x, y, z, cmap=cmap, vmin=0, vmax=norm_max)

    z = sch_Mil_Schol[i_n, :, :] / num
    axs[i_n, 2].pcolormesh(x, y, z, cmap=cmap, vmin=0, vmax=norm_max)

    axs_r = axs[i_n, 2].twinx()
    axs_r.get_yaxis().set_ticks([])
    axs_r.set_ylabel(f"$I_n = {In[i_n]}$", labelpad = 3)

cbar_ax = fig.add_axes([0.1, -0.025, 0.8, 0.02])
fig.colorbar(im, cax = cbar_ax, orientation = "horizontal")
cbar_ax.relim([0, 1])

fig.text(-0.02, 0.5, "Pull strength, k", va = 'center', rotation = 'vertical')
fig.text(0.4, -0.07, "Probabilty", va = 'center', rotation = 'horizontal')
plt.tight_layout(pad = 0.35)
plt.savefig("/home/artur/Documents/Fish/res/color_maps/0VerArticle_" +'SC2Mil.pdf', transparent=True, bbox_inches='tight', pad_inches = 0)








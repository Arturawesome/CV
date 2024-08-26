import numpy as np
import matplotlib.pyplot as plt

# Set step and time
h = 0.0001
t_end = 100
n_steps = int(t_end / h)

M = 1
L = 5
v0 = 1
# Beginnig conditions
x0 = 3.0
y0 = -4.0
vx0 = v0 * y0/L
vy0 = -x0/L

# Set arrays and dict for processing
t_values = np.arange(0, t_end + h, h)
dat_ten = np.zeros(len(t_values))
dat = {"t": np.arange(0, t_end + h, h),
       "x": np.zeros(len(t_values)),
       "y": np.zeros(len(t_values)),
       "vx": np.zeros(len(t_values)),
       "vy": np.zeros(len(t_values)),
       "T": np.zeros(len(t_values))
       }

dat["x"][0] = x0
dat["y"][0] = y0
dat["vx"][0] = vx0
dat["vy"][0] = vy0

# lambda function for g(t)
g = lambda t: 9.81 + 0.05* np.sin(2 * np.pi * t)

# function for tension
def tension(t, x, vx, vy):
    return g(t) * np.sqrt(1 - x**2/L**2) + (vx*vx + vy*vy)/L


# functions for RK methods
def f1(vx):
    return vx

def f2(t, x, vx, xy):
    T = tension(t, x, vx, vy)
    return -x * T / L

def f3(vy):
    return vy

def f4(t, x, y, vx, vy):
    T = tension(t, x, vx, vy)
    return -y * T / L - g(t)

# Main Loop
for i in range(n_steps):
    t = dat["t"][i]

    x = dat["x"][i]

    y = dat["y"][i]
    vx = dat["vx"][i]
    vy = dat["vy"][i]
    dat_ten[i] = tension(t, x, vx, vy)

    # Koefficients for x and vx
    k1_x = h * f1(vx)
    k1_vx = h * f2(t, x, vx, vy)

    k2_x = h * f1(vx + 0.5 * k1_vx)
    k2_vx = h * f2(t + 0.5 * h, x + 0.5 * k1_x, vx, vy)

    k3_x = h * f1(vx + 0.5 * k2_vx)
    k3_vx = h * f2(t + 0.5 * h, x + 0.5 * k2_x, vx, vy)

    k4_x = h * f1(vx + k3_vx)
    k4_vx = h * f2(t + h, x + k3_x, vx, vy)

    # Calculation the step and forward pass
    delta_x = (1/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    delta_vx = (1/6) * (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)

    dat["x"][i+1] = x + delta_x
    dat["vx"][i + 1] = vx + delta_vx


    k1_y = h * f3(vy)
    k1_vy = h * f4(t, x, y, vx, vy)

    k2_y = h * f3(vy + 0.5 * k1_vy)
    k2_vy = h * f4(t + 0.5 * h, x + 0.5 * k1_y, y + 0.5 * k1_y, vx, vy)

    k3_y = h * f3(vy + 0.5 * k2_vy)
    k3_vy = h * f4(t + 0.5 * h, x + 0.5 * k2_y, y + 0.5 * k2_y, vx, vy)

    k4_y = h * f3(vy + k3_vy)
    k4_vy = h * f4(t + h, x + k3_x, y + k3_y, vx, vy)

    delta_y = (1/6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    delta_vy = (1/6) * (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)

    #y_val_new = -np.sqrt(L * L - dat["x"][i + 1] * dat["x"][i + 1])
    dat["y"][i + 1] =  y + delta_y
    dat["vy"][i + 1] = vy + delta_vy


Ek0 = v0**2 / 2
Ep0 = 9.81 * (y0+5)
E0 = Ek0 + Ep0
h_max = E0 / 9.81
y_max = h_max - 5
x_max_theory = (L**2 - y_max**2)**0.5
print("max x Simulation = ", max(dat["x"]))
print("max x in Theory = ", x_max_theory)
print("Max vx in Simulation = ", max(dat["vx"]))
print("max vx in Theory = ", np.sqrt(2*(9.81 * (y0+5) + 0.5) ))
print("Period = ", 2*np.pi * (L/9.81)**0.5)


r_v = np.sqrt(dat["x"]**2 + dat["y"]**2)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.plot(t_values, r_v, label="r(t)")
ax.set_xlabel('Time (t)')
ax.set_ylabel('Tension')
ax.set_title('Графики зависимости r(t)')
plt.legend()
plt.grid(True)
plt.savefig("r(t).pdf")
plt.close(fig)


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.plot(t_values, dat["x"], label="x(t)")
ax.plot(t_values, dat["y"], label="y(t)")

ax.set_xlabel('Time (t)')
ax.set_ylabel('Position')
ax.set_title('Графики зависимости r(t)')
ax.set_yticks([3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7])
plt.legend()
plt.grid(True)
plt.savefig("xy(t).pdf")

plt.close(fig)


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.plot(dat["x"], dat["y"], label="xy")


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Xy')
plt.legend()
plt.grid(True)
plt.savefig("xy_t.pdf")
plt.close(fig)




fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.plot(t_values, dat["vx"], label="vx")
ax.plot(t_values, dat["vy"], label="vy")

ax.set_xlabel('Time (t)')
ax.set_ylabel('Velocity')
ax.set_title('Графики зависимости v(t)')
ax.set_yticks([3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7])
plt.legend()
plt.grid(True)
plt.savefig("v(t).pdf")
plt.close(fig)





with open("1datAll_t_" +str(t_end) + ".txt", "w") as f:
    for i in range(n_steps):
        t = dat["t"][i]
        x, y = dat["x"][i], dat["y"][i]
        vx, vy = dat["vx"][i], dat["vy"][i]
        f.write(str(t) + " " + str(x) + " " + str(y) + " " + str(vx) + " " + str(vy)+"\n" )


















fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (8, 8))

ax[0, 0].plot(t_values, tension(t_values, dat["x"], dat["vx"], dat["vy"]), label="r(t)")
ax[0, 0].set_xlabel('Time (t)')
ax[0, 0].set_ylabel('Tension(t), g(t)')
ax[0, 0].set_title('Tension')
ax[0, 0].legend()

ax[0, 1].plot(t_values, dat["x"], label="x(t)")
ax[0, 1].plot(t_values, dat["y"], label="y(t)")
ax[0, 1].set_xlabel('Time (t)')
ax[0, 1].set_ylabel('xt, yt')
ax[0, 1].set_title('Графики зависимости r(t)')
ax[0, 1].set_yticks([3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7])
ax[0, 1].legend()

ax[1, 0].plot(dat["x"], dat["y"], label="xy")
ax[1, 0].set_xlabel('X')
ax[1, 0].set_ylabel('Y')
ax[1, 0].set_title('Xy')

ax[1, 1].plot(t_values, dat["vx"], label="vx")
ax[1, 1].plot(t_values, dat["vy"], label="vy")
ax[1, 1].set_xlabel('Time (t)')
ax[1, 1].set_ylabel('Velocity')
ax[1, 1].set_title('Графики зависимости v(t)')

plt.savefig("All.pdf")
plt.close(fig)


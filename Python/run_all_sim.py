import os
import os.path
import subprocess
import random
import numpy as np
import time

name = "/home/artur/Documents/Molecular_Dinamic/HOOMD-blue/hoomd-v2.9.7/hoomd/md/TwoStepBDGPU.cu"
name_py = "/home/artur/Documents/Molecular_Dinamic/HOOMD-blue/in.py"
flow_amplitude = np.arange(0.0, 1.1, 0.1)
print(flow_amplitude)

for a_id in range(len(flow_amplitude)):
    f_a = flow_amplitude[a_id]
    print(round(f_a, 1))
    if a_id == 0:
        with open (name, 'r') as f:
            old_data = f.read()
            new_data = old_data.replace('F_x = 0.2 * 0.4194001362256668 * sin(0.5 * 8 * 3.14 * postype.x / L.x);', 'F_x = ' + str(round(f_a, 2)) + ' * 0.4194001362256668 * sin(0.5 * 8 * 3.14 * postype.x / L.x);')
            new_data = new_data.replace('F_y = -0.2 * (1 + cos(0.5 * 8 * 3.14 * postype.x / L.x));', 'F_y = -' + str(round(f_a, 2)) + ' * (1 + cos(0.5 * 8 * 3.14 * postype.x / L.x));')
        with open (name, 'w') as f:
            f.write(new_data)
            f.close()

        with open (name_py, 'r') as f:
            old_data = f.read()
            new_data = old_data.replace('f_a = 0.2', 'f_a = ' + str(round(f_a, 2)))
        with open (name_py, 'w') as f:
            f.write(new_data)
            f.close()
    else:
        f_a_1 = flow_amplitude[a_id-1]
        with open (name, 'r') as f:
            old_data = f.read()
            new_data = old_data.replace('F_x = ' + str(round(f_a_1, 2)) + ' * 0.4194001362256668 * sin(0.5 * 8 * 3.14 * postype.x / L.x);', 'F_x = ' + str(round(f_a, 2)) + ' * 0.4194001362256668 * sin(0.5 * 8 * 3.14 * postype.x / L.x);')
            new_data = new_data.replace('F_y = -' + str(round(f_a_1, 2)) + ' * (1 + cos(0.5 * 8 * 3.14 * postype.x / L.x));', 'F_y = -' + str(round(f_a, 2)) + ' * (1 + cos(0.5 * 8 * 3.14 * postype.x / L.x));')
        with open (name, 'w') as f:
            f.write(new_data)
            f.close()

        with open (name_py, 'r') as f:
            old_data = f.read()
            new_data = old_data.replace('f_a = ' + str(round(f_a_1, 2)), 'f_a = ' + str(round(f_a, 2)))
        with open (name_py, 'w') as f:
            f.write(new_data)
            f.close()

    time.sleep(30)

    bashCommand1 = "bash runSH.sh"
    subprocess.call(bashCommand1, shell=True)




























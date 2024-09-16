# optimisation of EPAC irradiation beam mode
# aim to optimise the AF5m beamline for high charge density in 2D (3D when FBPIC introduced)

import os
import numpy as mp
import time
from scipy import constants as cst
from scipy.optimize import minimize, differential_evolution, Bounds
from matplotlib import pyplot as plt

from sddsbeam import sddsbeam
import SDDSFile

# ONLY WORKS WITH PYTHON 3.8
# py -3.8 ... to run (uses pyenv)

def obj_func(K1_vals, Qbunch, Npart):
    #run elegant
    os.system('elegant IP_opt.ele -macro=Q1K1={0},Q2K1={1},Q3K1={2}'.format(K1_vals[0], K1_vals[1], K1_vals[2]))
    # take in the temporary beam file & sigma file
    time.sleep(0.7) # stops the optimisation from trying to read the file before it is written
    sig_IP = SDDSFile.read_sdds_file('IP_opt.sig')
    temp_beam = sddsbeam()
    temp_beam.read_SDDS_file('temp_IP.w2')
    Npart_temp = len(temp_beam.beam.x)
    print(Npart_temp)
    # find the Q/sigma^2 and return minus values (minimization)
    print(-(Qbunch*(Npart_temp/Npart))/((sig_IP.Sx[-1]/cst.milli)*(sig_IP.Sy[-1]/cst.milli)))
    return -(Qbunch*(Npart_temp/Npart))/((sig_IP.Sx[-1]/cst.milli)*(sig_IP.Sy[-1]/cst.milli))
    

if __name__ == "__main__":

    init_beam = sddsbeam()
    init_beam.read_SDDS_file('EPAC_1GeV_JS.sdds')

    Q = 233.896
    Npart_init = len(init_beam.beam.x)
    
    # eventually get these functionally from the lattice
    starting_K1 = [7.346730993433913, 2.29233339226338, -6.669619360805036]

    # K1 limits
    min_K1 = -10
    max_K1 = 10
    K1_limits = Bounds(lb=min_K1, ub=max_K1)

    # optimisation + results
    irrad_mode = minimize(obj_func, starting_K1, args=(Q, Npart_init), method='Nelder-Mead', bounds=K1_limits)

    print("Quad Settings: ",irrad_mode.x)
    print("Charge density at IP [pC/mm^2]: ",-irrad_mode.fun)
    
    # plotting out solution
    sdds_opt_sig = SDDSFile.read_sdds_file('IP_opt.sig')

    plt.figure(1)
    plt.plot(sdds_opt_sig.s, sdds_opt_sig.Sx/cst.milli)
    plt.plot(sdds_opt_sig.s, sdds_opt_sig.Sy/cst.milli)
    plt.title('Beam Irradiation Mode')
    plt.xlabel('s [m]')
    plt.ylabel(r'Beam Size [mm]')
    plt.draw()

    IP_beam = sddsbeam()
    IP_beam.read_SDDS_file('temp_IP.w2')
    
    plt.figure(2)
    plt.scatter(IP_beam.beam.x/cst.milli, IP_beam.beam.y/cst.milli, s=1)
    plt.title('Beam Irradiation Mode')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.draw()

    # show plots
    plt.show()

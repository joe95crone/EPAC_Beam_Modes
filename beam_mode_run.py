import h5py
import numpy as np
import scipy.constants as cst
from scipy import stats
from pathlib import Path
import os

# This code only runs with python 3.8

# JAMES TOOLS
from SimulationFramework.Modules.Beams import beam as SimBeam
from sddsbeam import sddsbeam
import SimulationFramework.Modules.SDDSFile as SDDSFile

class HDF5_read:
	def __init__(self):
		# Store an empty list for dataset names
		self.names = []

	def __call__(self, name, h5obj):
		# only h5py datasets have dtype attribute, so we can search on this
		if hasattr(h5obj,'dtype') and not name in self.names:
			self.names += [name]

class Converter:
    def __init__(self):
        self.args = args_class()
        self.file = h5py.File(self.args.input, 'r')
        self.HDF5_in = HDF5_read()
        self.file.visititems(self.HDF5_in)
        
    # funtion take in a list of the dataset names and writes back those that contain the label  
    def get_dataset_name(self, names,label):
        arr = []
        for i in range(len(names)):
            if label in names[i]:
                arr.append(names[i])
        return arr
	
    def get_data_arrays(self):
        # the label passed here can be file specific - Oli naming convention
        #mom_sets = self.get_dataset_name(self.HDF5_in.names,"Mixed Gas/momentum")
        #pos_sets = self.get_dataset_name(self.HDF5_in.names,"Mixed Gas/position")
        #weight_sets = self.get_dataset_name(self.HDF5_in.names,"Mixed Gas/weighting")

        # the label passed here can be file specific - Oli new naming convention
        mom_sets = self.get_dataset_name(self.HDF5_in.names,"Ionised N/momentum")
        pos_sets = self.get_dataset_name(self.HDF5_in.names,"Ionised N/position")
        weight_sets = self.get_dataset_name(self.HDF5_in.names,"Ionised N/weighting")

        # make array of momentum, position and weighting data
        mom_dat = np.array([np.array(self.file[mom_sets[i]]) for i in range(len(mom_sets))])
        pos_dat = np.array([np.array(self.file[pos_sets[i]]) for i in range(len(pos_sets))])
        weight_dat = np.array(self.file[weight_sets[0]])
        return (pos_dat, mom_dat, weight_dat)        

    def w_ave(self, a, weights=None):
    # Check if input contains data
        if not np.any(a):
            # If input is empty return NaN
            return np.nan
        else:
            # Calculate the weighted average
            average = np.average(a, weights=weights)
        return(average)

    def w_std(self, a, weights=None):
        # Check if input contains data
        if not np.any(a):
            # If input is empty return NaN
            return np.nan
        else:
            # Calculate the weighted standard deviation
            average = np.average(a, weights=weights)
            variance = np.average((a - average) ** 2, weights=weights)
        return(np.sqrt(variance))

    def get_vals_6d(self, beam, w):
        means = [self.w_ave(p, w) for p in beam]
        stds = [self.w_std(p, w) for p in beam]
        sliced_beam = [((p - self.w_ave(p, w)) / self.w_std(p, w)) for p in beam]
        return sliced_beam, w, means, stds

    def KDE_resample(self, pos_dat, mom_dat, weight_dat, frac):
        size = 17
        bandwidth = 0.001
        normbeam, weight, means, stds = self.get_vals_6d([pos_dat[0],pos_dat[1],pos_dat[2],mom_dat[0],mom_dat[1],mom_dat[2]], weight_dat)
        values = np.vstack(normbeam)
        kernel = stats.gaussian_kde(values, weights=(weight), bw_method=bandwidth)
        print('Re-sampling',int(frac*(2**size)),'particles from',len(pos_dat[0]),'particles')
        kdebeam = kernel.resample(int(frac*(2**size)))
        postbeam_uncut = np.transpose([(p*s)+m for p,m,s in zip(*[kdebeam, means, stds])])
        x,y,z,px,py,pz = np.transpose(postbeam_uncut)	
        postbeam = [x,y,z,px,py,pz]
        return postbeam

# using the option wonk = 1 means the beam will be de-wonked!
    def write_KDE_SDDS(self, KDE, weights, input_loc, output_loc, wonk=0):
        newbeam = SimBeam()
        newbeam.x = KDE[0]
        newbeam.y = KDE[1]
        newbeam.z = KDE[2]
        if wonk == 0:
            newbeam.px = KDE[3]
            newbeam.py = KDE[4] 
        elif wonk == 1:
            newbeam.px = KDE[5]*(KDE[3]/KDE[5] - np.mean(KDE[3]/KDE[5]))
            newbeam.py = KDE[5]*(KDE[4]/KDE[5] - np.mean(KDE[4]/KDE[5]))	
        newbeam.pz = KDE[5]
        newbeam.t = newbeam.z / (-1 * newbeam.Bz * cst.speed_of_light)
        newbeam.total_charge = np.sum(weights)*cst.elementary_charge # this is now charge
        single_charge = newbeam.total_charge / (len(newbeam.x))
        newbeam.charge = np.full(len(newbeam.x), single_charge)
        newbeam.nmacro = np.full(len(newbeam.x), 1)
        newbeam.code = 'KDE full'
        newbeam['longitudinal_reference'] = 'z'
        # write to file
        outputsdds = Path(input_loc).with_name(output_loc).with_suffix('.sdds')
        newbeam.write_SDDS_beam_file(str(outputsdds))

    def write_full_beam_SDDS(self, frac, wonk):
        pos_dat_full, mom_dat_full, weight_dat_full = self.get_data_arrays()
        KDE_full = self.KDE_resample(pos_dat_full, mom_dat_full, weight_dat_full, frac)
        self.write_KDE_SDDS(KDE_full, weight_dat_full, self.args.input, self.args.output, wonk)

    def write_core_beam_SDDS(self, frac, wonk):
        pos_dat_full, mom_dat_full, weight_dat_full = self.get_data_arrays()
        core = CoreCut(pos_dat_full, mom_dat_full, weight_dat_full)
        pos_dat_core, mom_dat_core, weight_dat_core = core.p_FWQM_cut()
        KDE_core = self.KDE_resample(pos_dat_core, mom_dat_core, weight_dat_core, frac*(len(weight_dat_core)/len(weight_dat_full)))
        self.write_KDE_SDDS(KDE_core, weight_dat_core, self.args.input, self.args.output+"_CORE", wonk)    

class CoreCut:
    nbins = 500
    kgmps2MeVpc = (cst.elementary_charge*cst.mega)/cst.speed_of_light

    def __init__(self, pos_dat, mom_dat, weight_dat):
        self.pos_dat = pos_dat
        self.mom_dat = mom_dat
        self.weight_dat = weight_dat
        self.n_macro = self.mom_dat.shape[1]+1
        self.n_part = np.sum(self.weight_dat)
        self.bunch_charge = (self.n_part*cst.elementary_charge)/cst.pico

    def find_nearest_index(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def t_cut(self):
        # histogram the z momentum data
        pz_bincount, pz_vals = np.histogram(self.mom_dat[2]/self.kgmps2MeVpc,bins=self.nbins,weights=self.weight_dat)

        # find maximum position in histogrammed data
        max_pos_pz = int(np.where(pz_bincount == np.amax(pz_bincount))[0])	
        # value in momentum data array that is closest to the histogram maximum
        max_pz_index = self.find_nearest_index(self.mom_dat[2]/self.kgmps2MeVpc,float(pz_vals[max_pos_pz]))

        # histogram of the t position data
        dt_dat = ((self.pos_dat[2]-np.mean(self.pos_dat[2]))/(cst.speed_of_light*cst.femto))
        t_bincount, t_vals = np.histogram(dt_dat,bins=self.nbins,weights=self.weight_dat)
	    # find t bin in histogrammed data closest to where pz is maximum in t 
        max_pos_t = self.find_nearest_index(t_vals,dt_dat[max_pz_index])
        # histograms zero positions in time (we use weighting, so can sometimes end up with non-real fractional particles in a bin!)
        t_zero_pos = np.array(np.where(t_bincount < 1)).flatten()
        print(min(t_bincount))
        print(t_zero_pos)
        # this condition finds the positions to cut the time data (when there is a gap in the time structure of the bunch)
        # if there is not (binning not granular enough or otherwise) we take all of bunch
        if len(t_zero_pos) == 0:
            higher_t_cut_bin = len(t_vals)-1
            lower_t_cut_bin = 0
        else:
            # higher value closest to maximum value in zero_pos list
            # conditional is applied if there is no zero count time bins after the central energy time index 
            if t_zero_pos.max() > max_pos_t:
                higher_t_cut_bin = t_zero_pos[t_zero_pos > max_pos_t].min()
            else:
                higher_t_cut_bin = len(t_vals)-1
            # lower value closest to maximum value in zero pos list
	        # conditional is applied if there is no zero count time bins before the central energy time index
            if t_zero_pos.min() < max_pos_t:
                lower_t_cut_bin = t_zero_pos[t_zero_pos < max_pos_t].max()
            else:
                lower_t_cut_bin = 0
	
        t_cut_index = []
        # get list of indexes for all of the macroparticles within the core bins
        for i in range(self.n_macro-1):
            if dt_dat[i] > t_vals[lower_t_cut_bin] and dt_dat[i] < t_vals[higher_t_cut_bin]:
                t_cut_index.append(i)

        # creation of time cut arrays
        mom_dat_t_cut = self.mom_dat[:,t_cut_index]
        pos_dat_t_cut = self.pos_dat[:,t_cut_index]
        weight_dat_t_cut = self.weight_dat[t_cut_index]
        return (pos_dat_t_cut, mom_dat_t_cut, weight_dat_t_cut, t_cut_index)
    
    def p_FWQM_cut(self):
        pos_dat_t_cut, mom_dat_t_cut, weight_dat_t_cut, t_cut_index = self.t_cut()
        n_macro_t_cut = len(t_cut_index)+1

        # Histogramming of data
        pz_bincount_t_cut, pz_vals_t_cut = np.histogram(mom_dat_t_cut[2]/self.kgmps2MeVpc,bins=self.nbins,weights=weight_dat_t_cut)
        # cutting all beam above QM
        pz_FWQM_index = np.where(mom_dat_t_cut[2]/self.kgmps2MeVpc > max(pz_vals_t_cut)/4)[0]
        
        # creation of momentum cut arrays
        mom_dat_FWQM = mom_dat_t_cut[:,pz_FWQM_index]
        pos_dat_FWQM = pos_dat_t_cut[:,pz_FWQM_index]
        weight_dat_FWQM = weight_dat_t_cut[pz_FWQM_index]
        return (pos_dat_FWQM, mom_dat_FWQM, weight_dat_FWQM)

def objective_function(K1_vals, beam_file, full_core):
    if full_core == "FULL":
        init_beam = sddsbeam()
        init_beam.read_SDDS_file(beam_file+'.sdds')
        os.system('elegant beam_mode.ele -macro=Q1K1={0},Q2K1={1},Q3K1={2},qbunch={3},beam={4}'.format(K1_vals[0], K1_vals[1], K1_vals[2], init_beam.beam.Charge[0]/cst.pico, beam_file+".sdds"))
    elif full_core == "CORE":
        init_beam = sddsbeam()
        init_beam.read_SDDS_file(beam_file+'_CORE.sdds')
        os.system('elegant beam_mode.ele -macro=Q1K1={0},Q2K1={1},Q3K1={2},qbunch={3},beam={4}'.format(K1_vals[0], K1_vals[1], K1_vals[2], init_beam.beam.Charge[0]/cst.pico, beam_file+"_CORE.sdds"))
    sig_IP = SDDSFile.read_sdds_file('beam_mode.sig')
    temp_beam = sddsbeam()
    temp_beam.read_SDDS_file('temp_IP.sdds')
    return (temp_beam.beam.Charge[0]/cst.pico)/((sig_IP.Sx[-1]/cst.milli)*(sig_IP.Sy[-1]/cst.milli)*(sig_IP.Ss[-1]/cst.milli))

class args_class():
    input = 'OF4_1GeV_20mm_Realistic_1.h5'
    output = 'OF4_1GeV_20mm_Realistic_1'

if __name__ == "__main__":
    # options are frac and wonk
    trial = Converter()
    trial.write_full_beam_SDDS(1, 1)
    trial.write_core_beam_SDDS(1, 1)

    args = args_class()

    #start_K1 = [7.346730993433913, 2.29233339226338, -6.669619360805036]
	
    # example is for the full beam 
    # needs to pull out the bunch charge and have a switch case for the core/full bunch
    #print("3D Irradiation [pC/mm^3]: ", objective_function(start_K1, args.output, "FULL"))
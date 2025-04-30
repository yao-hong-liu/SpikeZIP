# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:09:36 2024

@author: russo43
"""


%reset -f
%clear
    
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
from joblib import Parallel, delayed

plt.close('all')
  
### CHOOSE THE DATASET ###
# SPECS: Probe NPX v1, Sampling freq. 30kHz, N. channels supported 128chs, lsb = 2.37uV (range/resolution)
input_dataset_path =  '~~~~~~~~~~ Insert here the datapath ~~~~~~~~~~'

### CUT THE DATASET ###
time_interval_cut = 10     #[s], how much second of data to import
time_offset_cut = 50       #[s], where to start to cut the data


# %% TUNING PARAMETERS
# =============================================================================
### INPUT DATASET SPECS (NPX v1) ###
f_NPX = 30e3                                        # Hz, sampling frequency NPX data (reference)
n_chs = 128                                         # Total number of channels to process (max 385)
n_bit = 10                                          # Bit resolution NPX v1
NPX_uv_per_bit = 1.2/494/2**n_bit                   # V, lsb NPX

### RCS SPECS ###
f_ADC = 30e3                                        # Hz, sampling frequency LC-ADC
n_bit_ADC = 7                                       # n. bits, Resolution LC-ADC
range_ADC = 1.2                                     # V, input range LC-ADC
lsb_ADC = range_ADC/2**n_bit_ADC                    # V, step size LC-ADC

### UPSAMPLING SPECS ###
itp_cont_wave = 20                                  # Interpolation factor to make the reference data continous
f_NPX_cont = itp_cont_wave*f_NPX                    # Hz, frequency of continous-wave NPX data

### AMPLIFY DATA ###
gain_NPX2RCS = 750                                  # Gain (from NPX to RCS)


# %% SYSTEM PARAMETERS (DO NOT CHANGE)
# =============================================================================
### Cutting index ###
n_samples_cut = int(time_interval_cut*f_NPX)            # Number of samples to consider to cut to time_interval_cut seconds
start_sample_cut = int(time_offset_cut*f_NPX)           # Sample at which start to cut the data

### COMPRESSION RATIO PARAMETERS ###
nb_preamble = 5         # Number of bit for preamble
nb_header = 10          # Number of bit for ADC header
nb_ADC_addr = 5         # Number of bit for ADC address
nb_event = 5            # Number of bit for representing one event
n_ADCs = 8              # Number of ADCs per RCS


# %% IMPORT INPUT DATA
# =============================================================================
time_range_input = np.arange(n_samples_cut) + start_sample_cut
data = np.fromfile( input_dataset_path, count = n_samples_cut*385, dtype='int16', offset = start_sample_cut*n_chs*2 ).reshape(n_samples_cut, 385)[ time_range_input, -128:]     # Read NPXv1 dataset and time-cut it

NPX_datarate = 163.8e6/3                                                                # Expected input datarate NPXv1

data_NPX_raw = data_NPX*NPX_uv_per_bit                                        # Scale NPX reference data to uV scale
t_NPX = np.arange(0,np.size(data_NPX_raw,1))/f_NPX                            # Timestamp array NPX reference data  

NPX_data_volume = n_samples_cut*n_bit*n_chs                                   # Size on input data

NPX_swing = np.max(data_NPX_raw,1)-np.min(data_NPX_raw,1)                     # Swing of NPX reference data
NPX_max_swing = np.max(NPX_swing)                                             # MAX Swing of NPX reference data
ch_with_max_swing = np.squeeze(np.argwhere(NPX_swing==NPX_max_swing))
print("NPX max swing:", np.round(NPX_max_swing*1e6,2), 'uV', "(Ch", ch_with_max_swing, ")")
ch_plot = ch_with_max_swing

### PLOT
plt.figure()
plt.plot(t_NPX, 1e6*data_NPX_raw[ch_plot,:])
plt.title(f"NEUROPIXEL (REFERENCE) DATA, CH: {ch_plot}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude ($\mu$V)")
plt.grid(which='Both')
plt.legend()    

    
# %% PRE-FILTER INPUT DATA
# =============================================================================  
filter_type = 'bandpass'                # lowpass, highpass, bandpass
f_ord = 4                               # Filter order
f_passband_ripple = 0.1                 # Passband ripple
f_freq = np.array([300, 6000])          # Hz, cutoff freq
f_fs = f_NPX                            # Sampling rate

sos_cheb = signal.cheby1(f_ord, f_passband_ripple, f_freq, btype= filter_type, fs=f_fs, output='sos')
data_NPX_filtered = signal.sosfiltfilt(sos_cheb, data_NPX_raw.astype('float64'), axis=1, padtype='constant', padlen = int(f_NPX)*3).astype('float32')           # Filter reconstructed data
data_NPX = data_NPX_filtered


### PLOT
fig = plt.figure()
plt.title(f"Pre-filtered data \n(Red: original, Blue: filtered)")
for l in np.arange(0,16):
    plt.plot(t_NPX, ((1-l%16)*3e2 +(data_NPX_raw[l,:])*1e6 ), 'r')
    plt.plot(t_NPX, ((1-l%16)*3e2 +(data_NPX[l,:])*1e6 ), 'b')
plt.legend(ncol=4)    
plt.xlabel('Time (s)')
plt.ylabel('Ampl ($\mu$V)')
plt.grid()
  

# %% INTERPOLATION FOR CONTINOUS REFERENCE DATA
# =============================================================================
t_NPX_cont = np.arange(0, data_NPX.shape[1]*itp_cont_wave)/f_NPX_cont                           # Timestamp array continous NPX reference data  
    
interp_func_NPX_cont = interp1d(t_NPX, data_NPX, 'linear', fill_value="extrapolate")             
data_NPX_cont = interp_func_NPX_cont(t_NPX_cont)                                                # Continous (interpolated) NPX reference data
   
t_NPX = t_NPX_cont
data_NPX = data_NPX_cont
    
del t_NPX_cont, data_NPX_cont


# %% CLC FUNCTION DEFINITION
# =============================================================================
def CLC_comp_RCSv1( delta_modulated_vector):
    CLC_data = delta_modulated_vector.copy()
    # -- check whether input data are all 0
    if CLC_data.sum() == 0:
        return CLC_data
 
    # # -- initiate process
    data_value = np.abs( CLC_data)
    data_dir = np.ones( len(CLC_data), dtype='bool') # collect all data dir for error checking
    data_dir_1st = np.argwhere(CLC_data!=0)[0][0]
    if CLC_data[ data_dir_1st] > 0:
        dir_current = True
    else:
        dir_current = False
    dir_pre = dir_current
    data_dir[ :data_dir_1st+1] = dir_pre
    #-- sequential process
    for ind in range( data_dir_1st+1, len(CLC_data)):
        if data_value[ind] == 0:
            data_dir[ind] = dir_current
            continue
        else:
            if CLC_data[ind] > 0:
                dir_current = True
            else:
                dir_current = False
            data_dir[ind] = dir_current
            if dir_current ^ dir_pre: # XOR operater
                # orientation change
                dir_pre = dir_current
                if data_value[ind-1] == 0: # only do compemsation when the previous value is 0
                    if dir_current:
                        CLC_data[ind-1] = 1
                    else:
                        CLC_data[ind-1] = -1
                else:
                    print(ind)
    return CLC_data


# %% RESAMPLE DATA (RCS)
# =============================================================================
t_ADC = np.arange(0,time_interval_cut*f_ADC)/f_ADC                                  # Timestamp array LC-ADC 
data_NPX_scaled = data_NPX*gain_NPX2RCS
data_NPX_scaled = data_NPX_scaled - data_NPX_scaled.mean(axis=1, keepdims=True)     # Remove mean from each row

interp_func_ADC = interp1d(t_NPX, data_NPX_scaled, 'linear', fill_value="extrapolate")             
data_ADC = interp_func_ADC(t_ADC)                                                   # Data sampled at f_ADC

ADC_data_max_swing = np.max(np.max(data_ADC,1)-np.min(data_ADC,1))                  # Max swing of the resampled data
print("\n***************** ADC info *****************")
print("ADC data full swing:", np.round(ADC_data_max_swing*1e3,2), 'mV')
print("Gain NPX to ADC:", gain_NPX2RCS)
    

# %% LEVEL CROSSING ADC
# =============================================================================
data_ADC_quantized = data_ADC/lsb_ADC                                           # Quantized (absolute) number of steps
delta_ADC = np.diff(np.floor(data_ADC_quantized))                               # Delta value
delta_ADC = np.c_[np.zeros(len(delta_ADC)).T,delta_ADC]                         # Append an initial 0 to restore the original dimension    

##################### CLC FEATURE #####################
   
delta_ADC_orig = delta_ADC.copy()

temp_ind = np.zeros(delta_ADC.shape[0])
for jj in range( delta_ADC.shape[0]):                                       # Fill the first element with the first non zero value (in case first element is 0)
    non_zero_idx = np.squeeze(np.array(np.nonzero( delta_ADC[jj,:])))
    if len(non_zero_idx) == 0:
        delta_ADC[jj,0] = 1
    else:
        delta_ADC[jj,0] = delta_ADC[jj, non_zero_idx[0]]
    

delta_ADC_clc = np.copy(delta_ADC)                              # Create copy of delta_ADC for clc feature   
delta_sign = np.sign(delta_ADC)                                 # Sign of the matrix

row_indices = np.arange(delta_sign.shape[1])                        
non_zero_indices = np.where(delta_sign != 0, row_indices, 0)                # Find where the sign is non zero and replace with increasing integer values
last_non_zero_indices = np.maximum.accumulate(non_zero_indices, axis=1)     # Find the indices of the last non-zero value for each row 

result = delta_sign[np.arange(delta_ADC.shape[0])[:, None], last_non_zero_indices]  # Replace zeros with the sign of the last non-zero element for each row

signchange = ((np.roll(result, 1,axis=1) - result) != 0).astype(int)            # Check when polarity changes
delta_ADC_clc[signchange==1] = 0                                                # When polarity changes, put a 0
delta_ADC_clc[:,0] = delta_ADC_orig[:,0]
delta_ADC = np.copy(delta_ADC_clc)
    
ADC_quantized_max_swing = np.max(np.max(data_ADC_quantized,1)-np.min(data_ADC_quantized,1))     # Max swing of the quantized signal
        
        
# %% CALCULATE FRAME SIZE AND DATA VOLUME
# =============================================================================
Nevents_tot = np.sum(np.abs(delta_ADC))                             # Total number of events
Max_nevents_per_frame = np.max(np.sum(abs(delta_ADC),0))            # Max frame size (events)

N_events_per_ADC = np.zeros((n_ADCs,np.size(delta_ADC,1)))          # Total number of events per ADC in each packet
for z in range(n_ADCs):
    N_events_per_ADC[z,:] = np.sum(abs(delta_ADC[z*16:(z+1)*16,:]),0)
    
n_active_ADCs_per_frame = np.count_nonzero(N_events_per_ADC!=0,0)                              # Number of active ADCs per packet
n_events_per_frame  =np.sum(N_events_per_ADC,0)                                                # Total number of events per packet

data_no_protocol_size = n_active_ADCs_per_frame*(nb_ADC_addr) + n_events_per_frame*nb_event                                 # Total size of data (with channel and ADC address)
data_protocol_size = nb_preamble + n_active_ADCs_per_frame*(nb_header+nb_ADC_addr) + n_events_per_frame*nb_event            # Total size of a packet (data + protocol)
data_protocol_size[data_protocol_size<=5] = 0                                                                               # Remove preamble for packets with 0 events

max_frame_size_no_protocol = np.max(data_no_protocol_size)                      # Max packet size without protocol (only data and ADC/Ch addresses) (bits)
max_frame_size_w_protocol = np.max(data_protocol_size)                          # Max packet size with protocol (bits)
nb_no_protocol = np.sum(data_no_protocol_size)                                  # Total number of bits without protocol
nb_w_protocol = np.sum(data_protocol_size)                                      # Total number of bits with protocol
avg_frame_size_w_protocol =  np.mean(data_protocol_size)                        # Average packet size with protocol (bits)

RCS_data_volume = Nevents_tot*nb_event                                          # Size of ADC sampled data (only events + ch)


print("LC quantized data full swing:", int(ADC_quantized_max_swing), 'steps')
print("\n***************** ADC data volume *****************")
print("Total number of events:", int(Nevents_tot), "events, in", time_interval_cut, "seconds")
print("Event rate:", np.round(1e-6*Nevents_tot/time_interval_cut,2), "Meps" )
print("Max frame size (only events):", int(Max_nevents_per_frame), "events" )
print("Max frame size (without protocol):", int(max_frame_size_no_protocol), "bits (Ch and ADC addr. included)" )
print("Max frame size (with protocol):", int(max_frame_size_w_protocol), "bits" )
print("Total number of bit (without protocol):", np.round(1e-6*nb_no_protocol,2), "Mbits" )
print("Total number of bit (with protocol):", np.round(1e-6*nb_w_protocol,2), "Mbits" )
print("Compression ratio (without protocol):", np.round(NPX_data_volume/RCS_data_volume,2), "(NPX 10b and ADC 5b events)")
print("Compression ratio (with protocol):", np.round(NPX_datarate*time_interval_cut/nb_w_protocol,2), "(based on datarate)")
print("ADC estimated datarate", np.round(1e-6*nb_w_protocol/time_interval_cut,2), "Mbps")

    
# %%=============================================================================
# FRAME SIZE ANALYSIS
# =============================================================================
n_packs_rcs = len(data_protocol_size)                                                               # Total number of pack generated by the rcs
sorted_pack_size = np.sort(data_protocol_size)                                                      # Packet size sorted

bin_size = 100
edges = np.arange( 0, int(2*data_protocol_size.max()//bin_size+1)*bin_size+1, bin_size)

### PLOT
plt.figure(figsize=(7,5))
plt.grid(which='Both', linestyle='--', alpha=0.5)
hist, bin_edges = np.histogram(2*data_protocol_size, bins=edges)
hist = hist/ np.sum(hist)
plt.bar(bin_edges[:-1], hist, width=bin_size, align='edge', color = "cyan", edgecolor='black', linewidth=1.5, bottom = 3e-7)

plt.yscale('log')
plt.title(f"Distribution of frame size (Manchester coded) \n Compression ratio: {np.round(NPX_datarate*time_interval_cut/nb_w_protocol,2)}")
plt.xlabel("Frame size (bits)", fontsize = 14)
plt.ylabel("Normalized number of frames", fontsize = 14)   
plt.legend(loc = 'upper center')
plt.xticks(np.arange(0,7000,1000))
plt.xlim(-300,7000)


# %% RECONSTRUCTION
# =============================================================================
data_ADC_rec = np.cumsum(delta_ADC,1)*lsb_ADC                                   # Reconstructed signal

### COMPENSATE CLC ###
ADCs_matrix_clc = Parallel( n_jobs=10, verbose=10)( delayed( CLC_comp_RCSv1)(delta_ADC[tt, :]) for tt in range(n_chs))
ADCs_matrix_clc = np.array(ADCs_matrix_clc)
Rec_matrix_clc = ADCs_matrix_clc.astype('int16').cumsum( axis=1)
    
### PLOT
fig = plt.figure()
plt.title(f"Waveform reconstruction (after clc), Ch: {ch_plot}")
plt.plot(t_NPX, 1e3*data_NPX_scaled[ch_plot,:], label="Reference data continous: "+str(f_NPX_cont*1e-3)+"kHz", linewidth = '5')
plt.plot(t_ADC, 1e3*Rec_matrix_clc[ch_plot,:]*lsb_ADC, label = "Reconstructed after CLC")
plt.legend()    
plt.xlabel('Time (s)')
plt.ylabel('Ampl(mV)')
plt.grid()    


# %% POST-FILTER AFTER CLC FOR RECONSTRUCTION 
# =============================================================================  
''' Reuse the same filter parameters of the prefilter:
        1) HPF for CLC compensation effect - corner below 500Hz to not cut the action potential (in pre-filter, corner above 250Hz to remove the Local Field Potential) --> Any value between 250 to 500 is ok for both
        2) LPF for RCS high frequency noise, but here just simulation, so no noise will be added --> We can reuse the same filter of pre-filtering
    
    We refilter also the input because otherwise the reconstructed data will have a higher filter order'''


sos_cheb = signal.cheby1(f_ord, f_passband_ripple, f_freq, btype= filter_type, fs=f_fs, output='sos')
Rec_matrix_clc_filtered = signal.sosfiltfilt(sos_cheb, Rec_matrix_clc.astype('float64'), axis=1).astype('float32')           # Filter reconstructed data
data_NPX_post_filtered = signal.sosfiltfilt(sos_cheb, data_NPX_filtered.astype('float64'), axis=1).astype('float32')         # Filter input data

### PLOT
fig = plt.figure()
plt.title(f"Original (red) vs reconstructed bandpass-filtered (blue)")
for l in np.arange(0,16):
    plt.plot((2e-4*(1-l%16) +(data_NPX_filtered[l,:])), 'r')
    plt.plot((2e-4*(1-l%16) +(Rec_matrix_clc_filtered[l,:]*lsb_ADC/gain_NPX2RCS)), 'b')
plt.legend(ncol=4)    
plt.xlabel('Index')
plt.ylabel('Ampl($\mu$V)')
plt.grid()
    
Rec_matrix_clc_filtered_rescaled = Rec_matrix_clc_filtered*lsb_ADC/gain_NPX2RCS

 
# %%=============================================================================
# RMSE COMPUTATION
# =============================================================================
abs_error = data_NPX_filtered-Rec_matrix_clc_filtered_rescaled               # Calculate absolute error between original and reconstructed data
rmse_vs_ch = np.sqrt(np.mean((abs_error)**2,1))                              # Calculate RMSE for each channel
avg_rmse_128ch = np.mean(rmse_vs_ch)                                         # Calculate average RMSE across all the channels

print("AVG. RMSE (128chs):", avg_rmse_128ch)

### PLOT
plt.figure()
plt.plot(rmse_vs_ch*1e6)
# plt.title(f"Average RMSE ("+ np.round(t_FPGA[-1],2) + "s) vs Channel")
plt.title( f"Average RMSE ({t_ADC[-1]:.2f}s) vs Channel", fontsize = 16)
plt.grid(which = "both")
plt.xlabel('#Channel', fontsize = 14)
plt.ylabel('Ampl ($\mu$V)', fontsize = 14)
plt.text(0,4,"Avg. RMSE:" + str(np.round(avg_rmse_128ch*1e6,2))+"uV", fontsize = 12)

plt.figure()
plt.hist(rmse_vs_ch*1e6,70)
plt.title( f"Histogram average RMSE")
plt.grid(which = "both")
plt.xlabel('Avg. RMSE ($\mu$V)')
plt.ylabel('Count of RMSE values')
    
# %% PRINT RESULTS
# =============================================================================
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Compression ratio (with protocol):", np.round(NPX_datarate*time_interval_cut/nb_w_protocol,2), "(based on datarate)")
print("AVG. RMSE (128chs):", avg_rmse_128ch)
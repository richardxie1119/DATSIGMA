import numpy as np
import scipy.signal as sig
import scipy.stats as stats
from scipy.stats import median_absolute_deviation as mad
import pickle
from pyimzml.ImzMLWriter import ImzMLWriter
from pyImagingMSpec.inMemoryIMS import inMemoryIMS
from scipy.io import loadmat
import random
from tqdm import tqdm


# def proc(d):

#     """
#     the processing method for FT-ICR fid
#     """
#     d -= np.mean(d)
#     d = np.fft.ifft(np.fft.rfft(d))

#     d *= np.hamming(d.shape[1])   # hamming apodisation
#     #dzf = zeros( (4*d.size), dtype=complex)  # zero-fill 4 times to improve quality
#     #dzf[:d.size] = d[:]

#     sp = np.fft.fft(d)       # fourier transform
#     spmod = np.real(np.sqrt(sp*sp.conj()))  # and take modulus

#     return spmod

def proc(fid):

    """
    the processing method for FT-ICR fid
    """
    #d = fid.copy()
    # hamming apodisation
    if len(fid.shape) == 1: 
        fid_apod = fid*np.hamming(fid.size)
    if len(fid.shape) == 2:
        fid_apod = fid*np.hamming(fid.shape[1])

    #dzf = zeros( (4*d.size), dtype=complex)  # zero-fill 4 times to improve quality
    #dzf[:d.size] = d[:]

    sp = np.fft.rfft(fid_apod)       # fourier transform
    spmod = np.real(np.sqrt(sp*sp.conj()))  # and take modulus

    return spmod


def fticr_mass_axis(h, calib):
    """
    returns an array which will calibrate a FT-ICR experiment
    h_values : the frequency axis after fft
    calib : mass calibration parameters
    """

    h[h<0.1] = 0.1
    if calib[2] == 0:
        m = calib[0]/(calib[1] + h)
    else:
        delta = calib[0]**2 + 4*calib[2]*(calib[1] + h)
        m = 2*calib[2] / (np.sqrt(delta) - calib[0])

    return m

def mass2freq(m, calib):

    m = np.maximum(m, 1.0)

    if calib[2] == 0:
        h = calib[0] / m - calib[1]
    else:
        h = calib[0] / m + calib[2] / (m**2)  - calib[1]
        
    return h


def fid2spec(fid, m, mz_range):

    """
    converts a transient FID to a mass spectrum with a defined mass range
    """
    sp = proc(fid)
    
    mz_filter = (m < mz_range[1]) & (m > mz_range[0])

    if len(sp.shape) == 1:
        sp_return = sp[mz_filter]
    if len(sp.shape) == 2:
        sp_return = sp[:,mz_filter]

    return m[mz_filter], sp_return


def peak_detection(mz, spec, prominence, threshold):

    foundpeaks = sig.find_peaks(spec, prominence = prominence, height=threshold)
    tic = np.sum(spec)
    # mzs = mz[foundpeaks[0]]
    # mzs_sorted = mzs[np.argsort(mzs)]
    # intensity_sorted = spec[foundpeaks[0]][np.argsort(mzs)]

    return {'mzs':mz[foundpeaks[0]], 'intensity':spec[foundpeaks[0]],'tic':tic,'mz_index':foundpeaks[0]}



def extractMZFeatures(imzml_dataset, ppm, mz_range, feature_n = 0.05, mz_bins = []):
        
    if len(mz_bins) == 0:
        mz_bins = [mz_range[0]]
        while mz_bins[-1] < mz_range[1]:
            mz_bins.append(mz_bins[-1]+mz_bins[-1]*2*ppm*10**-6)
        
    print('number of mass bins {}'.format(len(mz_bins)))

    count = []
    for i in tqdm(range(len(mz_bins))):
        count.append(imzml_dataset.get_ion_image(mz_bins[i],ppm).xic_to_image(0).astype(bool).sum())
    
    mz_bins_filter = (np.array(count)>int(imzml_dataset.coords.shape[0]*feature_n))
    mz_bins_use = np.array(mz_bins)[mz_bins_filter]
                
    datacube = imzml_dataset.get_ion_image(mz_bins_use, ppm)
    
    datacube_array = [datacube.xic_to_image(i) for i in range(len(mz_bins_use))]
    datacube_array = np.concatenate(datacube_array,axis=1)
        
    return datacube_array, mz_bins_use, count

